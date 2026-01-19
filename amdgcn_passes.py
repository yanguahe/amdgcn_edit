#!/usr/bin/env python3
"""
AMDGCN Instruction Scheduling Passes

This module provides an LLVM-style optimization pass framework for instruction
reordering in AMDGCN assembly. Each pass operates on the AnalysisResult data
structure (CFG + DDGs) and can modify instruction order while respecting
data dependencies.

Usage:
    from amdgcn_passes import MoveInstructionPass, PassManager
    
    # Move instruction 5 in .LBB0_0 up by 1 position
    pass_ = MoveInstructionPass(
        block_label=".LBB0_0",
        instr_index=5,
        direction=-1  # -1 = up, +1 = down
    )
    
    pm = PassManager()
    pm.add_pass(pass_)
    success = pm.run_all(result)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union

from amdgcn_cfg import Instruction, BasicBlock, CFG
from amdgcn_ddg import (
    AnalysisResult,
    DDG,
    InstructionNode,
    PendingMemOp,
    parse_instruction_registers,
    build_ddg,
    is_lgkm_op,
    is_vm_op,
    SCC_WRITERS,
    SCC_READERS,
    SCC_ONLY_WRITERS,
    SCC_READ_WRITE,
    is_scc_only_writer,
    is_scc_reader,
    is_scc_writer,
    VCC_WRITERS,
    VCC_READERS,
    EXEC_WRITERS,
    EXEC_READERS,
)


# =============================================================================
# Pass Base Class and Manager
# =============================================================================

class Pass(ABC):
    """
    Abstract base class for optimization passes.
    
    Each pass operates on an AnalysisResult and returns whether it made any changes.
    Passes should be idempotent - running the same pass twice should have no effect
    if the first run was successful.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this pass."""
        pass
    
    @property
    def description(self) -> str:
        """Return a description of what this pass does."""
        return ""
    
    @abstractmethod
    def run(self, result: AnalysisResult) -> bool:
        """
        Run the pass on the analysis result.
        
        Args:
            result: The AnalysisResult to modify (CFG, DDGs, etc.)
            
        Returns:
            True if the pass made any changes, False otherwise.
        """
        pass


class PassManager:
    """
    Manages and executes a sequence of optimization passes.
    
    Passes are executed in the order they were added. After all passes run,
    the DDGs should be rebuilt to reflect the new instruction order.
    """
    
    def __init__(self):
        self.passes: List[Pass] = []
        self.verbose: bool = False
    
    def add_pass(self, pass_: Pass) -> None:
        """Add a pass to the manager."""
        self.passes.append(pass_)
    
    def clear(self) -> None:
        """Remove all passes."""
        self.passes.clear()
    
    def run_all(self, result: AnalysisResult, rebuild_ddg: bool = True) -> bool:
        """
        Run all passes in sequence.
        
        Args:
            result: The AnalysisResult to modify
            rebuild_ddg: If True, rebuild DDGs after passes complete
            
        Returns:
            True if any pass made changes, False otherwise.
        """
        any_changed = False
        
        for pass_ in self.passes:
            if self.verbose:
                print(f"Running pass: {pass_.name}")
            
            changed = pass_.run(result)
            any_changed = any_changed or changed
            
            if self.verbose:
                status = "modified" if changed else "no changes"
                print(f"  -> {status}")
        
        # Rebuild DDGs if any changes were made
        if any_changed and rebuild_ddg:
            self._rebuild_ddgs(result)
        
        return any_changed
    
    def _rebuild_ddgs(self, result: AnalysisResult) -> None:
        """Rebuild all DDGs after instruction modifications."""
        from amdgcn_ddg import generate_all_ddgs, compute_inter_block_deps
        
        # Rebuild DDGs
        new_ddgs, waitcnt_deps = generate_all_ddgs(result.cfg, enable_cross_block_waitcnt=True)
        result.ddgs = new_ddgs
        result.waitcnt_deps = waitcnt_deps
        
        # Recompute inter-block deps
        result.inter_block_deps = compute_inter_block_deps(result.cfg, new_ddgs)


# =============================================================================
# Dependency Analysis Helpers
# =============================================================================

def get_instruction_defs_uses(instr: Instruction) -> Tuple[Set[str], Set[str]]:
    """
    Get the registers defined (written) and used (read) by an instruction.
    
    Returns:
        (defs, uses) tuple of register sets
    """
    return parse_instruction_registers(instr)


def has_raw_dependency(instr_a: Instruction, instr_b: Instruction) -> Tuple[bool, Set[str]]:
    """
    Check if instruction A has a RAW (Read After Write) dependency on B.
    
    RAW: A reads a register that B writes.
    
    Returns:
        (has_dep, conflicting_regs) - whether dependency exists and which registers
    """
    defs_b, _ = get_instruction_defs_uses(instr_b)
    _, uses_a = get_instruction_defs_uses(instr_a)
    
    conflicts = defs_b & uses_a
    return (len(conflicts) > 0, conflicts)


def has_war_dependency(instr_a: Instruction, instr_b: Instruction) -> Tuple[bool, Set[str]]:
    """
    Check if instruction A has a WAR (Write After Read) dependency on B.
    
    WAR: A writes a register that B reads.
    
    Returns:
        (has_dep, conflicting_regs) - whether dependency exists and which registers
    """
    _, uses_b = get_instruction_defs_uses(instr_b)
    defs_a, _ = get_instruction_defs_uses(instr_a)
    
    conflicts = defs_a & uses_b
    return (len(conflicts) > 0, conflicts)


def has_waw_dependency(instr_a: Instruction, instr_b: Instruction) -> Tuple[bool, Set[str]]:
    """
    Check if instruction A has a WAW (Write After Write) dependency on B.
    
    WAW: A writes a register that B also writes.
    
    Returns:
        (has_dep, conflicting_regs) - whether dependency exists and which registers
    """
    defs_b, _ = get_instruction_defs_uses(instr_b)
    defs_a, _ = get_instruction_defs_uses(instr_a)
    
    conflicts = defs_a & defs_b
    return (len(conflicts) > 0, conflicts)


def can_ignore_scc_waw(instr_moving: Instruction, instr_stationary: Instruction) -> bool:
    """
    Check if a WAW-SCC dependency can be ignored when moving instr_moving past instr_stationary.
    
    WAW-SCC can be ignored if:
    - Both instructions write SCC
    - The moving instruction only writes SCC (doesn't read it)
    
    In this case, the moving instruction will simply overwrite SCC, and since it
    doesn't need the previous SCC value, there's no dependency.
    
    Example:
        [A] s_add_u32 s0, s1, s2     ; writes SCC (carry)
        [B] s_lshl_b64 s[4:5], s[4:5], 1  ; writes SCC (result != 0), doesn't read SCC
        
        B can move before A because B doesn't need A's SCC value.
        
    Counter-example:
        [A] s_add_u32 s0, s1, s2     ; writes SCC (carry)
        [B] s_addc_u32 s3, s4, s5    ; reads SCC, writes SCC
        
        B cannot move before A because B needs A's SCC value.
    
    Args:
        instr_moving: The instruction being moved
        instr_stationary: The instruction it's moving past
        
    Returns:
        True if the WAW-SCC dependency can be ignored
    """
    moving_opcode = instr_moving.opcode.lower()
    stationary_opcode = instr_stationary.opcode.lower()
    
    # Both must write SCC
    if not is_scc_writer(moving_opcode) or not is_scc_writer(stationary_opcode):
        return False
    
    # The moving instruction must only write SCC (not read it)
    if is_scc_reader(moving_opcode):
        return False
    
    # WAW-SCC can be ignored
    return True


def has_true_scc_dependency(instr_a: Instruction, instr_b: Instruction) -> Tuple[bool, str]:
    """
    Check if there's a true SCC dependency between instructions A and B.
    
    True SCC dependency exists when:
    - RAW-SCC: A reads SCC that B writes (A depends on B's SCC output)
    
    WAW-SCC is NOT a true dependency if A only writes SCC (doesn't read it).
    
    Args:
        instr_a: The instruction being moved
        instr_b: The instruction it's moving past
        
    Returns:
        (has_dep, dep_type) - whether dependency exists and its type ("RAW" or "")
    """
    opcode_a = instr_a.opcode.lower()
    opcode_b = instr_b.opcode.lower()
    
    # Check RAW-SCC: A reads SCC, B writes SCC
    if is_scc_reader(opcode_a) and is_scc_writer(opcode_b):
        return (True, "RAW-SCC")
    
    return (False, "")


def is_scc_tight_pair_start(block: BasicBlock, idx: int) -> bool:
    """
    Check if instruction at idx is the start of a tight SCC pair (s_add_u32 + s_addc_u32).
    
    A tight SCC pair is:
    - [idx] s_add_u32 (writes SCC)
    - [idx+1] s_addc_u32 (reads SCC from idx, writes SCC)
    
    These must be kept together.
    """
    if idx >= len(block.instructions) - 1:
        return False
    
    instr = block.instructions[idx]
    next_instr = block.instructions[idx + 1]
    
    opcode = instr.opcode.lower()
    next_opcode = next_instr.opcode.lower()
    
    # Check for add/addc or sub/subb pairs
    if opcode in ('s_add_u32', 's_add_i32') and next_opcode == 's_addc_u32':
        return True
    if opcode in ('s_sub_u32', 's_sub_i32') and next_opcode == 's_subb_u32':
        return True
    
    return False


def is_scc_tight_pair_end(block: BasicBlock, idx: int) -> bool:
    """
    Check if instruction at idx is the end of a tight SCC pair (s_addc_u32 after s_add_u32).
    Only checks for immediately adjacent pair.
    """
    if idx <= 0:
        return False
    
    instr = block.instructions[idx]
    prev_instr = block.instructions[idx - 1]
    
    opcode = instr.opcode.lower()
    prev_opcode = prev_instr.opcode.lower()
    
    # Check for add/addc or sub/subb pairs
    if opcode == 's_addc_u32' and prev_opcode in ('s_add_u32', 's_add_i32'):
        return True
    if opcode == 's_subb_u32' and prev_opcode in ('s_sub_u32', 's_sub_i32'):
        return True
    
    return False


def is_scc_pair_reader(opcode: str) -> bool:
    """Check if instruction is the 'reader' part of an SCC pair (s_addc_u32 or s_subb_u32)."""
    opcode = opcode.lower()
    return opcode in ('s_addc_u32', 's_subb_u32')


def is_scc_pair_writer(opcode: str) -> bool:
    """Check if instruction is the 'writer' part of an SCC pair (s_add_u32 or s_sub_u32)."""
    opcode = opcode.lower()
    return opcode in ('s_add_u32', 's_add_i32', 's_sub_u32', 's_sub_i32')


def find_scc_pair_start_separated(block: BasicBlock, reader_idx: int, max_distance: int = 10) -> int:
    """
    Find the start of an SCC pair, allowing for non-SCC instructions between.
    
    A "separated pair" example:
        [idx-2] s_add_u32  s0, ...    <- writes SCC (pair start)
        [idx-1] s_mul_i32  s1, ...    <- doesn't affect SCC
        [idx]   s_addc_u32 s2, ...    <- reads SCC (pair end, reader_idx)
    
    This function searches backwards from reader_idx to find the SCC writer,
    skipping instructions that don't read or write SCC.
    
    Args:
        block: The basic block
        reader_idx: Index of the SCC reader (s_addc_u32 or s_subb_u32)
        max_distance: Maximum distance to search backwards
        
    Returns:
        Index of the pair start (s_add_u32), or -1 if not found or pair is broken
    """
    if reader_idx <= 0:
        return -1
    
    instr_reader = block.instructions[reader_idx]
    reader_opcode = instr_reader.opcode.lower()
    
    # Must be s_addc_u32 or s_subb_u32
    if not is_scc_pair_reader(reader_opcode):
        return -1
    
    # Determine expected writer type
    if reader_opcode == 's_addc_u32':
        expected_writers = ('s_add_u32', 's_add_i32')
    else:  # s_subb_u32
        expected_writers = ('s_sub_u32', 's_sub_i32')
    
    # Search backwards for the SCC writer
    instructions_between = []
    for search_idx in range(reader_idx - 1, max(0, reader_idx - max_distance) - 1, -1):
        instr = block.instructions[search_idx]
        opcode = instr.opcode.lower()
        
        # Found the expected pair writer!
        if opcode in expected_writers:
            return search_idx
        
        # If we encounter another SCC writer/reader, the pair is broken
        if is_scc_writer(opcode) or is_scc_reader(opcode):
            return -1
        
        # This instruction doesn't affect SCC, continue searching
        instructions_between.append(search_idx)
    
    # Didn't find the pair writer within max_distance
    return -1


def find_scc_pair_start(block: BasicBlock, pair_end_idx: int) -> int:
    """
    Given the end of an SCC pair, find the start.
    First checks for tight pair, then for separated pair.
    Returns -1 if not a valid pair.
    """
    # First check tight pair (adjacent)
    if is_scc_tight_pair_end(block, pair_end_idx):
        return pair_end_idx - 1
    
    # Check for separated pair
    return find_scc_pair_start_separated(block, pair_end_idx)


def is_scc_separated_pair_end(block: BasicBlock, idx: int) -> bool:
    """
    Check if instruction at idx is the end of an SCC pair (tight or separated).
    """
    instr = block.instructions[idx]
    if not is_scc_pair_reader(instr.opcode):
        return False
    
    # Try to find the pair start
    pair_start = find_scc_pair_start(block, idx)
    return pair_start >= 0


def get_instructions_between_pair(block: BasicBlock, pair_start: int, pair_end: int) -> List[int]:
    """
    Get indices of instructions between pair_start and pair_end (exclusive).
    """
    return list(range(pair_start + 1, pair_end))


def can_chain_skip_scc_pair(
    block: BasicBlock,
    chain: List[int],
    pair_start: int,
    pair_end: int,
    original_chain: Optional[List[int]] = None
) -> bool:
    """
    Check if a chain can skip over an SCC pair (tight or separated) without breaking dependencies.
    
    The chain can skip if:
    1. No ORIGINAL chain instruction READS SCC (they only write or don't touch SCC)
       Note: Previously skipped pairs may contain s_addc_u32 which reads SCC, but that's
       from their own pair partner, not from this pair we're checking.
    2. No other data dependencies exist between chain and pair
    3. No data dependencies with instructions between pair_start and pair_end
    
    When chain skips the pair:
    - Chain moves to before pair_start
    - Pair (and instructions between) stay together
    - SCC from chain will be overwritten by pair's s_add_u32
    
    Args:
        block: The basic block
        chain: List of instruction indices in the chain (may include previously skipped pairs)
        pair_start: Index of s_add_u32
        pair_end: Index of s_addc_u32
        original_chain: The original chain before any pairs were added (for SCC reader check)
        
    Returns:
        True if chain can skip the pair
    """
    # Use original chain for SCC reader check (if provided)
    # This is important because accumulated chain may include s_addc_u32 from
    # previously skipped pairs, which read SCC from their own pair partner
    chain_for_scc_check = original_chain if original_chain is not None else chain
    
    # Check that no ORIGINAL chain instruction reads SCC
    for idx in chain_for_scc_check:
        instr = block.instructions[idx]
        if is_scc_reader(instr.opcode.lower()):
            return False
    
    # Get all instructions that are part of the pair region (including those between)
    pair_region_indices = list(range(pair_start, pair_end + 1))
    
    # Check for non-SCC data dependencies between chain and the entire pair region
    # Here we use the full accumulated chain, not just original
    for region_idx in pair_region_indices:
        region_instr = block.instructions[region_idx]
        region_defs, region_uses = get_instruction_defs_uses(region_instr)
        # Remove SCC from consideration
        region_defs_no_scc = region_defs - {'scc'}
        region_uses_no_scc = region_uses - {'scc'}
        
        for chain_idx in chain:
            chain_instr = block.instructions[chain_idx]
            chain_defs, chain_uses = get_instruction_defs_uses(chain_instr)
            chain_defs_no_scc = chain_defs - {'scc'}
            chain_uses_no_scc = chain_uses - {'scc'}
            
            # RAW: chain reads what region writes (excluding SCC)
            if region_defs_no_scc & chain_uses_no_scc:
                return False
            
            # WAR: chain writes what region reads (excluding SCC)
            if chain_defs_no_scc & region_uses_no_scc:
                return False
    
    return True


def is_register_live_after(
    block: BasicBlock,
    reg: str,
    after_index: int
) -> bool:
    """
    Check if a register is live (used) after a given instruction index in the block.
    
    Args:
        block: The basic block
        reg: Register name to check
        after_index: Check liveness after this instruction index
        
    Returns:
        True if the register is used by any instruction after after_index
    """
    for i in range(after_index + 1, len(block.instructions)):
        _, uses = get_instruction_defs_uses(block.instructions[i])
        if reg in uses:
            return True
    
    # Also consider live-out (register might be used in successor blocks)
    # For now, conservatively assume it's live if it's written
    # A more precise analysis would check the DDG's live_out set
    return False


def is_register_live_after_with_ddg(
    ddg: DDG,
    reg: str,
    after_index: int
) -> bool:
    """
    Check if a register is live after a given instruction index, using DDG info.
    
    This is more accurate than is_register_live_after as it uses the DDG's
    live_out set for cross-block liveness.
    """
    # Check if used by any instruction after after_index in this block
    for i in range(after_index + 1, len(ddg.nodes)):
        if reg in ddg.nodes[i].uses:
            return True
    
    # Check if in live_out (used in successor blocks)
    return reg in ddg.live_out


# =============================================================================
# s_waitcnt Handling
# =============================================================================

def parse_waitcnt_operands(operands: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse s_waitcnt operands to extract vmcnt and lgkmcnt values.
    
    Examples:
        "vmcnt(0)" -> (0, None)
        "lgkmcnt(0)" -> (None, 0)
        "vmcnt(3) lgkmcnt(2)" -> (3, 2)
        
    Returns:
        (vmcnt, lgkmcnt) tuple, None if not specified
    """
    vmcnt = None
    lgkmcnt = None
    
    vm_match = re.search(r'vmcnt\((\d+)\)', operands)
    if vm_match:
        vmcnt = int(vm_match.group(1))
    
    lgkm_match = re.search(r'lgkmcnt\((\d+)\)', operands)
    if lgkm_match:
        lgkmcnt = int(lgkm_match.group(1))
    
    return vmcnt, lgkmcnt


def build_waitcnt_operands(vmcnt: Optional[int], lgkmcnt: Optional[int]) -> str:
    """
    Build s_waitcnt operand string from vmcnt and lgkmcnt values.
    
    Args:
        vmcnt: VM count value (0-63), None to omit
        lgkmcnt: LGKM count value (0-15), None to omit
        
    Returns:
        Formatted operand string like "vmcnt(0) lgkmcnt(0)"
    """
    parts = []
    if vmcnt is not None:
        parts.append(f"vmcnt({vmcnt})")
    if lgkmcnt is not None:
        parts.append(f"lgkmcnt({lgkmcnt})")
    
    return " ".join(parts)


def get_instruction_cycles(opcode: str) -> int:
    """
    Get the number of cycles required to issue an instruction.
    
    Cycle costs:
    - v_mfma_* instructions: 16 cycles
    - ds_swizzle_*, ds_write_* instructions: 8 cycles
    - v_exp_* instructions: 16 cycles
    - All other instructions: 4 cycles
    
    Args:
        opcode: The instruction opcode
        
    Returns:
        Number of cycles for the instruction
    """
    opcode_lower = opcode.lower()
    
    if opcode_lower.startswith('v_mfma'):
        return 16
    elif opcode_lower.startswith('ds_swizzle') or opcode_lower.startswith('ds_write'):
        return 8
    elif opcode_lower.startswith('v_exp'):
        return 16
    else:
        return 4


def update_waitcnt_instruction(
    instr: Instruction,
    vmcnt_delta: int = 0,
    lgkmcnt_delta: int = 0
) -> bool:
    """
    Update a s_waitcnt instruction by adjusting its counts.
    
    Args:
        instr: The s_waitcnt instruction to modify
        vmcnt_delta: Amount to add to vmcnt (can be negative)
        lgkmcnt_delta: Amount to add to lgkmcnt (can be negative)
        
    Returns:
        True if update was successful, False if would result in invalid count
    """
    if instr.opcode.lower() != 's_waitcnt':
        return False
    
    vmcnt, lgkmcnt = parse_waitcnt_operands(instr.operands)
    
    # Apply deltas
    if vmcnt is not None:
        vmcnt += vmcnt_delta
        if vmcnt < 0 or vmcnt > 63:
            return False
    
    if lgkmcnt is not None:
        lgkmcnt += lgkmcnt_delta
        if lgkmcnt < 0 or lgkmcnt > 15:
            return False
    
    # Rebuild operands
    new_operands = build_waitcnt_operands(vmcnt, lgkmcnt)
    instr.operands = new_operands
    
    # Update raw_line
    instr.raw_line = f"\ts_waitcnt {new_operands}"
    
    # NOTE: Caller should also update block.raw_lines[instr.address] if needed
    # for file regeneration to pick up the change
    
    return True


def sync_instruction_to_raw_lines(block: BasicBlock, instr: Instruction) -> None:
    """
    Sync an instruction's raw_line to the block's raw_lines dictionary.
    
    This should be called after modifying an instruction's raw_line to ensure
    file regeneration uses the updated content.
    """
    if instr.address in block.raw_lines:
        block.raw_lines[instr.address] = instr.raw_line + '\n'


# =============================================================================
# Move Instruction Pass
# =============================================================================

@dataclass
class MoveResult:
    """Result of attempting to move an instruction."""
    success: bool
    message: str = ""
    blocked_by: Optional[str] = None  # Reason if blocked
    waitcnt_updated: bool = False
    cascaded_moves: List[int] = field(default_factory=list)  # Indices of instructions that moved along
    displaced_pair: List[int] = field(default_factory=list)  # Indices of displaced instructions that should stay together (e.g., [s_waitcnt_idx, dependent_idx])
    skipped_pairs: List[tuple] = field(default_factory=list)  # SCC pairs that the chain passes through without including


def find_dependent_waitcnt(
    block: BasicBlock,
    ddg: Optional[DDG],
    instr_index: int
) -> Optional[int]:
    """
    Find the s_waitcnt instruction that the given instruction depends on.
    
    This checks if the instruction uses registers that are made available
    by a preceding s_waitcnt (via cross-block memory operations).
    
    Args:
        block: The basic block
        ddg: The DDG for the block (needed for cross-block info)
        instr_index: Index of the instruction to check
        
    Returns:
        Index of the dependent s_waitcnt, or None if no dependency
    """
    if ddg is None:
        return None
    
    instr = block.instructions[instr_index]
    _, uses = get_instruction_defs_uses(instr)
    
    if not uses:
        return None
    
    # Search backwards for s_waitcnt that makes any of our used registers available
    for i in range(instr_index - 1, -1, -1):
        prev_instr = block.instructions[i]
        if prev_instr.opcode.lower() == 's_waitcnt':
            # Check cross-block available registers
            cross_block_regs = ddg.waitcnt_cross_block_regs.get(i, set())
            if cross_block_regs & uses:
                return i
            
            # Check intra-block available registers
            if i < len(ddg.nodes):
                intra_block_regs = ddg.nodes[i].available_regs
                if intra_block_regs & uses:
                    return i
    
    return None


def find_dependency_chain(
    block: BasicBlock,
    ddg: Optional[DDG],
    instr_index: int,
    direction: int
) -> List[int]:
    """
    Find all instructions that the target instruction depends on (recursively).
    
    This builds a dependency chain by following RAW dependencies backwards.
    The chain includes all instructions that must move together with the target.
    
    Args:
        block: The basic block
        ddg: The DDG for the block
        instr_index: Index of the target instruction
        direction: -1 for up, +1 for down
        
    Returns:
        List of instruction indices in the dependency chain (sorted by position).
        The target instruction is always the last element when moving up.
    """
    if direction > 0:
        # For moving down, we don't need to pull predecessors
        return [instr_index]
    
    # For moving up: find all RAW predecessors recursively
    chain = set()
    to_process = [instr_index]
    
    while to_process:
        idx = to_process.pop()
        if idx in chain:
            continue
        chain.add(idx)
        
        instr = block.instructions[idx]
        _, uses = get_instruction_defs_uses(instr)
        
        if not uses:
            continue
        
        # Find which instruction defines the registers we use
        # Only look at instructions between the current chain minimum and idx
        chain_min = min(chain) if chain else idx
        for prev_idx in range(idx - 1, -1, -1):
            prev_instr = block.instructions[prev_idx]
            defs, _ = get_instruction_defs_uses(prev_instr)
            
            # Check if prev_instr defines any register we use
            if defs & uses:
                # Only add if it's not already processed and it's adjacent or
                # between the chain boundary and current position
                if prev_idx not in chain:
                    # Check if this is immediately before the chain minimum
                    # or if moving will cause a gap
                    if prev_idx == chain_min - 1 or prev_idx >= chain_min - 1:
                        to_process.append(prev_idx)
                    # If there's a gap, we need to include intermediate instructions
                    # that might have dependencies
    
    # Return sorted list (smallest index first)
    return sorted(chain)


def find_immediate_dependency_chain(
    block: BasicBlock,
    instr_index: int,
    direction: int,
    ddg: Optional[DDG] = None
) -> List[int]:
    """
    Find the immediate dependency chain - instructions that directly prevent
    the target from moving and must move with it.
    
    This includes:
    1. Instructions that the target depends on (RAW: target reads what prev writes)
    2. Instructions that would block the chain due to WAR (chain writes what prev reads)
    3. s_waitcnt instructions that chain instructions depend on for register availability
    
    Args:
        block: The basic block
        instr_index: Index of the target instruction
        direction: -1 for up, +1 for down
        ddg: Optional DDG for checking s_waitcnt dependencies
        
    Returns:
        List of instruction indices that must move together (sorted by position)
    """
    chain = [instr_index]
    
    if direction < 0:  # Moving up
        # First pass: find RAW dependencies (target reads what prev writes)
        current_idx = instr_index
        while current_idx > 0:
            prev_idx = current_idx - 1
            instr_curr = block.instructions[current_idx]
            instr_prev = block.instructions[prev_idx]
            
            _, uses_curr = get_instruction_defs_uses(instr_curr)
            defs_prev, _ = get_instruction_defs_uses(instr_prev)
            
            # Check if current instruction depends on prev (RAW)
            if defs_prev & uses_curr:
                # Need to include prev in the chain
                chain.insert(0, prev_idx)
                current_idx = prev_idx
            else:
                # No RAW dependency, stop here
                break
        
        # Second pass: extend chain to resolve WAR conflicts
        # When we try to move the chain, if any chain instruction writes
        # a register that the instruction before the chain reads, we have
        # a WAR conflict. We resolve it by including that instruction in the chain.
        # 
        # EXCEPTION: WAR-SCC can be ignored if the chain instruction ONLY writes SCC
        # (doesn't read it). In this case, the chain instruction will just overwrite
        # whatever SCC value was there, and since it doesn't need the previous value,
        # there's no real dependency.
        changed = True
        max_iterations = len(block.instructions)  # Prevent infinite loops
        iterations = 0
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            chain_start = min(chain)
            if chain_start == 0:
                break
            
            before_chain_idx = chain_start - 1
            instr_before = block.instructions[before_chain_idx]
            _, uses_before = get_instruction_defs_uses(instr_before)
            
            # Check if any chain instruction has WAR conflict with instr_before
            for idx in chain:
                instr_chain = block.instructions[idx]
                defs_chain, _ = get_instruction_defs_uses(instr_chain)
                
                # WAR: chain writes what before reads
                conflicts = defs_chain & uses_before
                if conflicts:
                    # Check if conflict is only SCC and can be ignored
                    if conflicts == {'scc'} and is_scc_only_writer(instr_chain.opcode.lower()):
                        # WAR-SCC with chain only writing SCC - can be ignored
                        continue
                    
                    # Need to include instr_before and its dependencies
                    if before_chain_idx not in chain:
                        chain.insert(0, before_chain_idx)
                        changed = True
                        
                        # Also need to find RAW dependencies of the newly added instruction
                        current_idx = before_chain_idx
                        while current_idx > 0:
                            prev_idx = current_idx - 1
                            instr_curr = block.instructions[current_idx]
                            instr_prev = block.instructions[prev_idx]
                            
                            _, uses_curr = get_instruction_defs_uses(instr_curr)
                            defs_prev, _ = get_instruction_defs_uses(instr_prev)
                            
                            # Also check if RAW is SCC-only and can be ignored
                            raw_conflicts = defs_prev & uses_curr
                            if raw_conflicts:
                                if raw_conflicts == {'scc'} and not is_scc_reader(instr_curr.opcode.lower()):
                                    # RAW-SCC but current doesn't read SCC - can skip
                                    break
                                if prev_idx not in chain:
                                    chain.insert(0, prev_idx)
                                current_idx = prev_idx
                            else:
                                break
                        break  # Restart the WAR check from the beginning
        
        # Third pass: include s_waitcnt that chain instructions depend on
        # If a chain instruction uses registers that become available after a s_waitcnt,
        # and that s_waitcnt is immediately before the chain, include it
        if ddg is not None:
            changed = True
            iterations = 0
            
            while changed and iterations < max_iterations:
                changed = False
                iterations += 1
                
                chain_start = min(chain)
                if chain_start == 0:
                    break
                
                before_chain_idx = chain_start - 1
                instr_before = block.instructions[before_chain_idx]
                
                if instr_before.opcode.lower() == 's_waitcnt':
                    # Check if any chain instruction depends on this s_waitcnt
                    cross_block_regs = ddg.waitcnt_cross_block_regs.get(before_chain_idx, set())
                    intra_block_regs = ddg.nodes[before_chain_idx].available_regs if before_chain_idx < len(ddg.nodes) else set()
                    all_avail_regs = cross_block_regs | intra_block_regs
                    
                    for idx in chain:
                        instr = block.instructions[idx]
                        _, uses = get_instruction_defs_uses(instr)
                        
                        if all_avail_regs & uses:
                            # Chain instruction depends on this s_waitcnt
                            if before_chain_idx not in chain:
                                chain.insert(0, before_chain_idx)
                                changed = True
                                break
    
    elif direction > 0:  # Moving down
        current_idx = instr_index
        while current_idx < len(block.instructions) - 1:
            next_idx = current_idx + 1
            instr_curr = block.instructions[current_idx]
            instr_next = block.instructions[next_idx]
            
            defs_curr, _ = get_instruction_defs_uses(instr_curr)
            _, uses_next = get_instruction_defs_uses(instr_next)
            
            # Check if next instruction depends on current (RAW)
            if defs_curr & uses_next:
                # Next depends on current, so if we're moving current down,
                # we need to take next with us
                chain.append(next_idx)
                current_idx = next_idx
            else:
                break
    
    return sorted(chain)


def find_all_dependent_waitcnts(
    block: BasicBlock,
    ddg: Optional[DDG],
    instr_index: int,
    direction: int
) -> List[int]:
    """
    Find all s_waitcnt instructions that must move with the given instruction.
    
    When moving up, this finds s_waitcnts between the target position and current
    position that the instruction depends on.
    
    Args:
        block: The basic block
        ddg: The DDG for the block
        instr_index: Index of the instruction to move
        direction: -1 for up, +1 for down
        
    Returns:
        List of s_waitcnt indices that must move along (in order)
    """
    if ddg is None:
        return []
    
    instr = block.instructions[instr_index]
    _, uses = get_instruction_defs_uses(instr)
    
    if not uses:
        return []
    
    dependent_waitcnts = []
    
    if direction < 0:  # Moving up
        # Check all s_waitcnts between current position and the instruction we're swapping with
        # For now, just check the immediate predecessor
        prev_idx = instr_index - 1
        if prev_idx >= 0:
            prev_instr = block.instructions[prev_idx]
            if prev_instr.opcode.lower() == 's_waitcnt':
                # Check if we depend on this waitcnt
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(prev_idx, set())
                intra_block_regs = ddg.nodes[prev_idx].available_regs if prev_idx < len(ddg.nodes) else set()
                all_avail_regs = cross_block_regs | intra_block_regs
                
                if all_avail_regs & uses:
                    dependent_waitcnts.append(prev_idx)
    
    return dependent_waitcnts


class MoveInstructionPass(Pass):
    """
    Pass that moves a single instruction up or down by a specified number of cycles.
    
    This pass works by moving OTHER instructions to make room for the target instruction,
    rather than moving the target instruction directly. This allows for cycle-based
    movement while respecting all dependency constraints.
    
    This pass respects all dependency constraints:
    - RAW (Read After Write): Consumer cannot move before producer
    - WAR (Write After Read): Writer cannot move before reader (of same reg)
    - WAW (Write After Write): Handled specially - WAW is traversable
    - Implicit register dependencies (SCC, VCC, EXEC)
    - s_waitcnt constraints with automatic count adjustment
    
    Moving UP n cycles:
    1. Find the dependency tree above AA (ABT = all instructions AA depends on)
    2. Move instructions from above ABT to below AA, closest first, until n cycles moved
    3. If not enough, move instructions from ABT gaps to below AA
    
    Moving DOWN n cycles:
    1. Find the dependency tree below AA (ACT = all instructions that depend on AA)
    2. Move instructions from below ACT to above AA, closest first, until n cycles moved
    3. If not enough, move instructions from ACT gaps to above AA
    
    Attributes:
        block_label: Label of the block containing the instruction
        instr_index: Index of the instruction to move (0-based)
        cycles: Number of cycles to move (positive = up, negative = down)
    """
    
    def __init__(
        self,
        block_label: str,
        instr_index: int,
        cycles: int,
        verbose: bool = False,
        frozen_boundary: int = 0,
        protected_instructions: Optional[List['Instruction']] = None
    ):
        """
        Initialize the pass.
        
        Args:
            block_label: Label of the block (e.g., ".LBB0_0")
            instr_index: Index of instruction to move
            cycles: Number of cycles to move (positive = move up, negative = move down)
            verbose: Print detailed information during execution
            frozen_boundary: Instructions at idx < frozen_boundary cannot be moved
                            and no instruction can be moved into this region
            protected_instructions: List of instruction objects that should never be moved
                                   (e.g., remaining target instructions when distributing)
        """
        self.block_label = block_label
        self.instr_index = instr_index
        self.cycles = cycles
        self.verbose = verbose
        self.frozen_boundary = frozen_boundary
        self.protected_instructions = protected_instructions or []
        self._last_result: Optional[MoveResult] = None
        self._total_cycles_moved: int = 0
        # Detailed tracking of why movement stopped (use sets of instruction object IDs to track unique instructions)
        self._blocked_by_frozen: Set[int] = set()  # Set of id(instruction) blocked by frozen boundary
        self._blocked_by_dependencies: Set[int] = set()  # Set of id(instruction) blocked by dependencies
        self._blocked_by_branch: Set[int] = set()  # Set of id(instruction) blocked by branch boundary
        self._blocked_by_barrier: Set[int] = set()  # Set of id(instruction) blocked by s_barrier
        self._blocked_by_protected: Set[int] = set()  # Set of id(instruction) blocked because protected
        self._no_candidates: bool = False  # No candidate instructions available
    
    @property
    def name(self) -> str:
        dir_str = "up" if self.cycles > 0 else "down"
        return f"MoveInstruction({self.block_label}[{self.instr_index}] {dir_str} {abs(self.cycles)} cycles)"
    
    @property
    def description(self) -> str:
        return f"Move instruction at index {self.instr_index} in block {self.block_label} by {self.cycles} cycles"
    
    @property
    def last_result(self) -> Optional[MoveResult]:
        """Get the result of the last run."""
        return self._last_result
    
    @property
    def total_cycles_moved(self) -> int:
        """Get the total cycles actually moved."""
        return self._total_cycles_moved
    
    @property
    def stop_reason(self) -> str:
        """Get the reason why movement stopped before reaching target cycles."""
        reasons = []
        if len(self._blocked_by_frozen) > 0:
            reasons.append(f"frozen_boundary({len(self._blocked_by_frozen)})")
        if len(self._blocked_by_dependencies) > 0:
            reasons.append(f"dependencies({len(self._blocked_by_dependencies)})")
        if len(self._blocked_by_branch) > 0:
            reasons.append(f"branch_boundary({len(self._blocked_by_branch)})")
        if len(self._blocked_by_barrier) > 0:
            reasons.append(f"s_barrier({len(self._blocked_by_barrier)})")
        if len(self._blocked_by_protected) > 0:
            reasons.append(f"protected_instructions({len(self._blocked_by_protected)})")
        if self._no_candidates:
            reasons.append("no_candidate_instructions")
        
        if not reasons:
            return "reached target"
        return ", ".join(reasons)
    
    def run(self, result: AnalysisResult) -> bool:
        """
        Execute the pass.
        
        Returns:
            True if any instruction was moved, False otherwise.
            
        Raises:
            SchedulingVerificationError: If the optimization violates any dependency constraints.
        """
        # Import verification module
        from amdgcn_verify import build_global_ddg, verify_optimization
        
        # Validate block exists
        if self.block_label not in result.cfg.blocks:
            self._last_result = MoveResult(
                success=False,
                message=f"Block '{self.block_label}' not found"
            )
            return False
        
        block = result.cfg.blocks[self.block_label]
        ddg = result.ddgs.get(self.block_label)
        
        # Validate instruction index
        if self.instr_index < 0 or self.instr_index >= len(block.instructions):
            self._last_result = MoveResult(
                success=False,
                message=f"Invalid instruction index {self.instr_index} (block has {len(block.instructions)} instructions)"
            )
            return False
        
        if self.cycles == 0:
            self._last_result = MoveResult(
                success=True,
                message="No movement requested (cycles=0)"
            )
            return False
        
        # Capture original dependency graph BEFORE any changes
        original_gdg = build_global_ddg(result.cfg, result.ddgs)
        
        self._total_cycles_moved = 0
        
        # Get the target instruction reference (for tracking after moves)
        target_instr = block.instructions[self.instr_index]
        
        
        if self.cycles > 0:
            # Move UP by n cycles
            success = self._move_up_by_cycles(block, ddg, result, target_instr)
        else:
            # Move DOWN by |n| cycles
            success = self._move_down_by_cycles(block, ddg, result, target_instr)
        
        if self.verbose:
            print(f"  Total cycles moved: {self._total_cycles_moved} / {abs(self.cycles)} requested")
        
        self._last_result = MoveResult(
            success=success,
            message=f"Moved {self._total_cycles_moved} cycles"
        )
        
        # Mandatory verification: raises SchedulingVerificationError on failure
        verify_optimization(original_gdg, result.cfg)
        
        return success
    
    # =========================================================================
    # New cycle-based movement methods
    # =========================================================================
    
    def _find_upward_dependency_tree(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        target_idx: int
    ) -> Tuple[Set[int], int]:
        """
        Find the upward dependency tree (ABT) for the target instruction.
        
        Starting from target_idx, trace all instructions that the target depends on
        (directly or transitively) going upward in the block.
        
        Args:
            block: The basic block
            ddg: The DDG for the block
            target_idx: Index of the target instruction (AA)
            
        Returns:
            (abt_indices, bb_idx) - Set of indices in the dependency tree (including AA),
                                    and the index of the topmost instruction (BB)
        """
        abt = {target_idx}
        to_process = [target_idx]
        
        while to_process:
            idx = to_process.pop()
            if idx <= 0:
                continue
            
            instr = block.instructions[idx]
            _, uses = get_instruction_defs_uses(instr)
            
            # Find which instructions above define the registers we use
            for prev_idx in range(idx - 1, -1, -1):
                prev_instr = block.instructions[prev_idx]
                defs, _ = get_instruction_defs_uses(prev_instr)
                
                # Check if prev_instr defines any register we use (RAW dependency)
                raw_conflicts = defs & uses
                if raw_conflicts:
                    # Check if conflict is only SCC and can be ignored
                    if raw_conflicts == {'scc'} and not is_scc_reader(instr.opcode.lower()):
                        # Instruction doesn't read SCC, so this RAW-SCC doesn't matter
                        continue
                    
                    if prev_idx not in abt:
                        abt.add(prev_idx)
                        to_process.append(prev_idx)
                    # Found the definition for these registers, stop looking further for them
                    uses = uses - defs
                    if not uses:
                        break
            
            # Also check for s_waitcnt dependencies (AVAIL)
            if ddg is not None:
                for prev_idx in range(idx - 1, -1, -1):
                    prev_instr = block.instructions[prev_idx]
                    if prev_instr.opcode.lower() == 's_waitcnt':
                        cross_block_regs = ddg.waitcnt_cross_block_regs.get(prev_idx, set())
                        intra_block_regs = ddg.nodes[prev_idx].available_regs if prev_idx < len(ddg.nodes) else set()
                        all_avail_regs = cross_block_regs | intra_block_regs
                        
                        instr_for_check = block.instructions[idx]
                        _, instr_uses = get_instruction_defs_uses(instr_for_check)
                        
                        if all_avail_regs & instr_uses:
                            if prev_idx not in abt:
                                abt.add(prev_idx)
                                to_process.append(prev_idx)
                            break
        
        bb_idx = min(abt) if abt else target_idx
        return abt, bb_idx
    
    def _find_downward_dependency_tree(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        target_idx: int
    ) -> Tuple[Set[int], int]:
        """
        Find the downward dependency tree (ACT) for the target instruction.
        
        Starting from target_idx, trace all instructions that depend on the target
        (directly or transitively) going downward in the block.
        
        Args:
            block: The basic block
            ddg: The DDG for the block
            target_idx: Index of the target instruction (AA)
            
        Returns:
            (act_indices, cc_idx) - Set of indices in the dependency tree (including AA),
                                    and the index of the bottommost instruction (CC)
        """
        act = {target_idx}
        to_process = [target_idx]
        max_idx = len(block.instructions) - 1
        
        while to_process:
            idx = to_process.pop()
            if idx >= max_idx:
                continue
            
            instr = block.instructions[idx]
            defs, _ = get_instruction_defs_uses(instr)
            
            # Find which instructions below use the registers we define
            for next_idx in range(idx + 1, len(block.instructions)):
                next_instr = block.instructions[next_idx]
                _, uses = get_instruction_defs_uses(next_instr)
                
                # Check if next_instr uses any register we define (RAW dependency)
                raw_conflicts = defs & uses
                if raw_conflicts:
                    # Check if conflict is only SCC and can be ignored
                    if raw_conflicts == {'scc'} and not is_scc_reader(next_instr.opcode.lower()):
                        # Next instruction doesn't read SCC, so this RAW-SCC doesn't matter
                        continue
                    
                    if next_idx not in act:
                        act.add(next_idx)
                        to_process.append(next_idx)
        
        cc_idx = max(act) if act else target_idx
        return act, cc_idx
    
    def _get_abt_gaps(
        self,
        abt: Set[int],
        bb_idx: int,
        target_idx: int
    ) -> List[int]:
        """
        Find the gaps (non-dependency instructions) within the ABT range.
        
        Returns indices of instructions between bb_idx and target_idx that are NOT in ABT,
        sorted by distance from target_idx (closest first).
        """
        gaps = []
        for idx in range(bb_idx, target_idx + 1):
            if idx not in abt:
                gaps.append(idx)
        # Sort by distance from target (closest first)
        gaps.sort(key=lambda x: target_idx - x)
        return gaps
    
    def _get_act_gaps(
        self,
        act: Set[int],
        target_idx: int,
        cc_idx: int
    ) -> List[int]:
        """
        Find the gaps (non-dependency instructions) within the ACT range.
        
        Returns indices of instructions between target_idx and cc_idx that are NOT in ACT,
        sorted by distance from target_idx (closest first).
        """
        gaps = []
        for idx in range(target_idx, cc_idx + 1):
            if idx not in act:
                gaps.append(idx)
        # Sort by distance from target (closest first)
        gaps.sort(key=lambda x: x - target_idx)
        return gaps
    
    def _can_move_single_instruction_down(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        from_idx: int,
        to_idx: int,
        protected_indices: Set[int]
    ) -> bool:
        """
        Check if a single instruction at from_idx can be moved down to to_idx.
        
        This checks all intermediate dependencies without actually moving.
        
        Args:
            block: The basic block
            ddg: The DDG
            from_idx: Current index of the instruction
            to_idx: Target index (must be > from_idx)
            protected_indices: Indices that cannot be crossed/modified
            
        Returns:
            True if the move is possible
        """
        if from_idx >= to_idx:
            return True
        
        if from_idx in protected_indices:
            return False
        
        instr_a = block.instructions[from_idx]
        defs_a, uses_a = get_instruction_defs_uses(instr_a)
        opcode_a = instr_a.opcode.lower()
        
        # s_barrier is a boundary - it cannot be moved
        if opcode_a == 's_barrier':
            return False
        
        # Check all instructions we would pass
        for check_idx in range(from_idx + 1, to_idx + 1):
            if check_idx in protected_indices:
                return False
            
            instr_b = block.instructions[check_idx]
            defs_b, uses_b = get_instruction_defs_uses(instr_b)
            opcode_b = instr_b.opcode.lower()
            
            # s_barrier is a boundary - cannot cross it
            if opcode_b == 's_barrier':
                return False
            
            # RAW: B reads what A writes -> BLOCKED (A must stay before B)
            raw_conflicts = defs_a & uses_b
            if raw_conflicts:
                if raw_conflicts == {'scc'} and not is_scc_reader(opcode_b):
                    pass  # SCC-only conflict, B doesn't read SCC
                else:
                    return False
            
            # WAR: B writes what A reads -> BLOCKED (A reads, B writes same reg)
            war_conflicts = defs_b & uses_a
            if war_conflicts:
                if war_conflicts == {'scc'} and is_scc_only_writer(opcode_b):
                    pass  # SCC-only conflict, B only writes SCC
                else:
                    return False
            
            # WAW: Both write same register -> Usually OK (WAW traversable)
            # But check if B's result is needed later
            waw_conflicts = defs_a & defs_b
            if waw_conflicts:
                if waw_conflicts == {'scc'}:
                    # WAW-SCC is always traversable
                    pass
                else:
                    # For non-SCC WAW, check if B's result is live
                    # We'll be conservative and allow it (WAW traversable rule)
                    pass
            
            # Check s_waitcnt constraints
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                # vm_op/lgkm_op (A) moves DOWN past s_waitcnt (B): B's counter DECREASES by 1
                # Requirement: counter - 1 >= 0 (i.e., counter > 0 or counter >= 1)
                # If counter <= 0, cannot move (would become negative)
                if is_vm_op(opcode_a) and vmcnt is not None and vmcnt <= 0:
                    return False
                if is_lgkm_op(opcode_a) and lgkmcnt is not None and lgkmcnt <= 0:
                    return False
            
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                # s_waitcnt (A) moves DOWN past vm_op/lgkm_op (B): A's counter INCREASES by 1
                # Requirement: counter + 1 <= max (vmcnt <= 63, lgkmcnt <= 15)
                # If counter >= max, cannot move (would exceed limit)
                if is_vm_op(opcode_b) and vmcnt is not None and vmcnt >= 63:
                    return False
                if is_lgkm_op(opcode_b) and lgkmcnt is not None and lgkmcnt >= 15:
                    return False
                
                # Check s_waitcnt AVAIL dependency: if B depends on A (s_waitcnt) result,
                # A cannot move past B (A needs to stay before B)
                if ddg is not None:
                    cross_block_regs = ddg.waitcnt_cross_block_regs.get(from_idx, set())
                    intra_block_regs = ddg.nodes[from_idx].available_regs if from_idx < len(ddg.nodes) else set()
                    all_avail_regs = cross_block_regs | intra_block_regs
                    if all_avail_regs & uses_b:
                        # B depends on s_waitcnt's available registers
                        # s_waitcnt (A) cannot move past B
                        return False
            
            # Don't cross branch/terminator
            if instr_b.is_branch or instr_b.is_terminator:
                return False
        
        return True
    
    def _can_move_single_instruction_up(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        from_idx: int,
        to_idx: int,
        protected_indices: Set[int]
    ) -> bool:
        """
        Check if a single instruction at from_idx can be moved up to to_idx.
        
        This checks all intermediate dependencies without actually moving.
        
        Args:
            block: The basic block
            ddg: The DDG
            from_idx: Current index of the instruction
            to_idx: Target index (must be < from_idx)
            protected_indices: Indices that cannot be crossed/modified
            
        Returns:
            True if the move is possible
        """
        if from_idx <= to_idx:
            return True
        
        if from_idx in protected_indices:
            return False
        
        instr_a = block.instructions[from_idx]
        defs_a, uses_a = get_instruction_defs_uses(instr_a)
        opcode_a = instr_a.opcode.lower()
        
        # s_barrier is a boundary - it cannot be moved
        if opcode_a == 's_barrier':
            return False
        
        # Check all instructions we would pass
        for check_idx in range(from_idx - 1, to_idx - 1, -1):
            if check_idx in protected_indices:
                return False
            
            instr_b = block.instructions[check_idx]
            defs_b, uses_b = get_instruction_defs_uses(instr_b)
            opcode_b = instr_b.opcode.lower()
            
            # s_barrier is a boundary - cannot cross it
            if opcode_b == 's_barrier':
                return False
            
            # RAW: A reads what B writes -> BLOCKED (A depends on B)
            raw_conflicts = defs_b & uses_a
            if raw_conflicts:
                if raw_conflicts == {'scc'} and not is_scc_reader(opcode_a):
                    pass  # SCC-only conflict, A doesn't read SCC
                else:
                    return False
            
            # WAR: A writes what B reads -> BLOCKED
            war_conflicts = defs_a & uses_b
            if war_conflicts:
                if war_conflicts == {'scc'} and is_scc_only_writer(opcode_a):
                    pass  # SCC-only conflict, A only writes SCC
                else:
                    return False
            
            # WAW: Both write same register -> Usually OK (WAW traversable)
            waw_conflicts = defs_a & defs_b
            if waw_conflicts:
                if waw_conflicts == {'scc'}:
                    pass  # WAW-SCC is always traversable
                else:
                    pass  # WAW traversable rule
            
            # Check s_waitcnt constraints
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                # vm_op/lgkm_op (A) moves UP past s_waitcnt (B): B's counter INCREASES by 1
                # Requirement: counter + 1 <= max (vmcnt <= 63, lgkmcnt <= 15)
                # If counter >= max, cannot move (would exceed limit)
                if is_vm_op(opcode_a) and vmcnt is not None and vmcnt >= 63:
                    return False
                if is_lgkm_op(opcode_a) and lgkmcnt is not None and lgkmcnt >= 15:
                    return False
            
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                # s_waitcnt (A) moves UP past vm_op/lgkm_op (B): A's counter DECREASES by 1
                # Requirement: counter - 1 >= 0 (i.e., counter > 0 or counter >= 1)
                # If counter <= 0, cannot move (would become negative)
                if is_vm_op(opcode_b) and vmcnt is not None and vmcnt <= 0:
                    return False
                if is_lgkm_op(opcode_b) and lgkmcnt is not None and lgkmcnt <= 0:
                    return False
            
            # Check s_waitcnt AVAIL dependency
            if opcode_b == 's_waitcnt' and ddg is not None:
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(check_idx, set())
                intra_block_regs = ddg.nodes[check_idx].available_regs if check_idx < len(ddg.nodes) else set()
                all_avail_regs = cross_block_regs | intra_block_regs
                if all_avail_regs & uses_a:
                    return False
        
        return True
    
    def _move_single_instruction_down_impl(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        from_idx: int,
        to_idx: int
    ) -> int:
        """
        Move a single instruction from from_idx down to to_idx using swaps.
        
        Updates s_waitcnt counts as needed.
        
        When A at position X swaps with B at position X+1:
        - After swap: B is at X, A is at X+1
        - If B is s_waitcnt and A is vm_op/lgkm_op:
          A moved from BEFORE s_waitcnt to AFTER s_waitcnt -> vmcnt/lgkmcnt DECREASE
        - If A is s_waitcnt and B is vm_op/lgkm_op:
          B moved from AFTER s_waitcnt to BEFORE s_waitcnt -> vmcnt/lgkmcnt INCREASE
        
        Returns:
            The final index of the moved instruction
        """
        current_idx = from_idx
        while current_idx < to_idx:
            instr_a = block.instructions[current_idx]
            instr_b = block.instructions[current_idx + 1]
            
            opcode_a = instr_a.opcode.lower()
            opcode_b = instr_b.opcode.lower()
            
            
            # Update s_waitcnt counts
            # Case 1: A (vm_op/lgkm_op) moves DOWN past B (s_waitcnt)
            # A goes from BEFORE s_waitcnt to AFTER -> DECREASE count
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                if is_vm_op(opcode_a) and vmcnt is not None:
                    update_waitcnt_instruction(instr_b, vmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_b)
                if is_lgkm_op(opcode_a) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_b, lgkmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_b)
            
            # Case 2: A (s_waitcnt) moves DOWN past B (vm_op/lgkm_op)
            # B goes from AFTER s_waitcnt to BEFORE -> INCREASE count
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                if is_vm_op(opcode_b) and vmcnt is not None:
                    update_waitcnt_instruction(instr_a, vmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_a)
                if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_a, lgkmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_a)
            
            # Swap
            self._swap_instructions(block, current_idx, current_idx + 1)
            current_idx += 1
        
        return current_idx
    
    def _move_single_instruction_up_impl(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        from_idx: int,
        to_idx: int
    ) -> int:
        """
        Move a single instruction from from_idx up to to_idx using swaps.
        
        Updates s_waitcnt counts as needed.
        
        When A at position X swaps with B at position X-1:
        - After swap: A is at X-1, B is at X
        - If B is s_waitcnt and A is vm_op/lgkm_op:
          A moved from AFTER s_waitcnt to BEFORE s_waitcnt -> vmcnt/lgkmcnt INCREASE
        - If A is s_waitcnt and B is vm_op/lgkm_op:
          B moved from BEFORE s_waitcnt to AFTER s_waitcnt -> vmcnt/lgkmcnt DECREASE
        
        Returns:
            The final index of the moved instruction
        """
        current_idx = from_idx
        while current_idx > to_idx:
            instr_a = block.instructions[current_idx]
            instr_b = block.instructions[current_idx - 1]
            
            opcode_a = instr_a.opcode.lower()
            opcode_b = instr_b.opcode.lower()
            
            
            # Update s_waitcnt counts
            # Case 1: A (vm_op/lgkm_op) moves UP past B (s_waitcnt)
            # A goes from AFTER s_waitcnt to BEFORE -> INCREASE count
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                if is_vm_op(opcode_a) and vmcnt is not None:
                    update_waitcnt_instruction(instr_b, vmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_b)
                if is_lgkm_op(opcode_a) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_b, lgkmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_b)
            
            # Case 2: A (s_waitcnt) moves UP past B (vm_op/lgkm_op)
            # B goes from BEFORE s_waitcnt to AFTER -> DECREASE count
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                if is_vm_op(opcode_b) and vmcnt is not None:
                    update_waitcnt_instruction(instr_a, vmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_a)
                if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_a, lgkmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_a)
            
            # Swap
            self._swap_instructions(block, current_idx, current_idx - 1)
            current_idx -= 1
        
        return current_idx
    
    def _find_instruction_index(self, block: BasicBlock, target_instr: Instruction) -> int:
        """Find the current index of an instruction by object identity."""
        for idx, instr in enumerate(block.instructions):
            if instr is target_instr:
                return idx
        return -1
    
    def _move_up_by_cycles(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        result: AnalysisResult,
        target_instr: Instruction
    ) -> bool:
        """
        Move the target instruction up by self.cycles cycles.
        
        This works by moving other instructions from above the dependency tree
        to below the target instruction.
        """
        cycles_to_move = self.cycles
        any_moved = False
        # Reset blocking sets
        self._blocked_by_frozen = set()
        self._blocked_by_dependencies = set()
        self._blocked_by_branch = set()
        self._blocked_by_barrier = set()
        self._blocked_by_protected = set()
        self._no_candidates = False
        
        while self._total_cycles_moved < cycles_to_move:
            # Get current position of target
            target_idx = self._find_instruction_index(block, target_instr)
            if target_idx < 0:
                break
            
            # Find the dependency tree above target
            abt, bb_idx = self._find_upward_dependency_tree(block, ddg, target_idx)
            
            if self.verbose:
                print(f"  ABT: {sorted(abt)}, BB at index {bb_idx}, target at {target_idx}")
            
            # Step 2: Move instructions from above BB to below target
            moved_in_phase2 = False
            
            # Find instructions above BB, sorted by distance from target (closest first)
            # Closest to target means highest index below bb_idx
            # IMPORTANT: Store instruction objects, not indices, because indices shift after moves!
            # Track reasons for skipping (use sets of id(instr) for unique counting)
            above_bb_instrs = []
            for idx in range(bb_idx - 1, -1, -1):
                instr = block.instructions[idx]
                if idx < self.frozen_boundary:
                    self._blocked_by_frozen.add(id(instr))
                    continue
                # Skip branch/terminator - they are boundaries
                if instr.is_branch or instr.is_terminator:
                    continue
                # Skip s_barrier - it's a synchronization boundary
                if instr.opcode.lower() == 's_barrier':
                    self._blocked_by_barrier.add(id(instr))
                    continue
                if instr in self.protected_instructions:
                    self._blocked_by_protected.add(id(instr))
                else:
                        above_bb_instrs.append(instr)
            
            for instr_to_move in above_bb_instrs:
                if self._total_cycles_moved >= cycles_to_move:
                    break
                
                # Re-find target position (may have shifted)
                current_target_idx = self._find_instruction_index(block, target_instr)
                if current_target_idx < 0:
                    break
                
                # Find current index of the instruction to move (indices shift after each move!)
                src_idx = self._find_instruction_index(block, instr_to_move)
                if src_idx < 0 or src_idx >= current_target_idx:
                    # Instruction not found or already below target
                    continue
                
                # Destination is just after the target (current_target_idx + 1)
                # But since we're moving from src_idx to after target_idx, and src_idx < target_idx,
                # after the move, target will shift left by 1, so dest_idx is current_target_idx
                dest_idx = current_target_idx
                
                # Don't pass protected_indices for checking ability to cross through ABT
                # We check regular dependencies instead - an instruction can pass through ABT
                # as long as it has no dependencies with those instructions
                if self._can_move_single_instruction_down(block, ddg, src_idx, dest_idx, set()):
                    # Move it
                    instr_cycles = get_instruction_cycles(instr_to_move.opcode)
                    self._move_single_instruction_down_impl(block, ddg, src_idx, dest_idx)
                    self._total_cycles_moved += instr_cycles
                    any_moved = True
                    moved_in_phase2 = True
                    
                    if self.verbose:
                        print(f"    Moved [{src_idx}] {instr_to_move.opcode} down to [{dest_idx}] (+{instr_cycles} cycles)")
                else:
                    # Blocked by dependencies
                    self._blocked_by_dependencies.add(id(instr_to_move))
            
            if self._total_cycles_moved >= cycles_to_move:
                break
            
            # Step 3: Move instructions from ABT gaps to below target
            if not moved_in_phase2 or self._total_cycles_moved < cycles_to_move:
                # Re-find target and ABT
                current_target_idx = self._find_instruction_index(block, target_instr)
                if current_target_idx < 0:
                    break
                
                current_abt, current_bb_idx = self._find_upward_dependency_tree(block, ddg, current_target_idx)
                gaps = self._get_abt_gaps(current_abt, current_bb_idx, current_target_idx)
                
                moved_from_gaps = False
                for gap_idx in gaps:
                    if self._total_cycles_moved >= cycles_to_move:
                        break
                    
                    # Find the current index of the gap instruction
                    # (gaps was computed before, need to re-locate)
                    if gap_idx >= len(block.instructions):
                        continue
                    
                    gap_instr = block.instructions[gap_idx]
                    
                    # Skip gap instructions in frozen region
                    if gap_idx < self.frozen_boundary:
                        self._blocked_by_frozen.add(id(gap_instr))
                        continue
                    
                    # Re-find positions
                    current_target_idx = self._find_instruction_index(block, target_instr)
                    if current_target_idx < 0:
                        break
                    
                    # Skip protected instructions
                    if gap_instr in self.protected_instructions:
                        self._blocked_by_protected.add(id(gap_instr))
                        continue
                    
                    current_gap_idx = self._find_instruction_index(block, gap_instr)
                    if current_gap_idx < 0 or current_gap_idx >= current_target_idx:
                        continue
                    
                    # Also skip if current position is in frozen region
                    if current_gap_idx < self.frozen_boundary:
                        self._blocked_by_frozen.add(id(gap_instr))
                        continue
                    
                    dest_idx = current_target_idx
                    # For gap instructions, don't use protected_indices either
                    if self._can_move_single_instruction_down(block, ddg, current_gap_idx, dest_idx, set()):
                        instr_cycles = get_instruction_cycles(gap_instr.opcode)
                        self._move_single_instruction_down_impl(block, ddg, current_gap_idx, dest_idx)
                        self._total_cycles_moved += instr_cycles
                        any_moved = True
                        moved_from_gaps = True
                        
                        if self.verbose:
                            print(f"    Moved gap [{current_gap_idx}] {gap_instr.opcode} down to [{dest_idx}] (+{instr_cycles} cycles)")
                    else:
                        self._blocked_by_dependencies.add(id(gap_instr))
                
                if not moved_from_gaps and not moved_in_phase2:
                    # No more instructions can be moved
                    if len(above_bb_instrs) == 0 and len(gaps) == 0:
                        self._no_candidates = True
                    break
        
        return any_moved
    
    def _move_down_by_cycles(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        result: AnalysisResult,
        target_instr: Instruction
    ) -> bool:
        """
        Move the target instruction down by |self.cycles| cycles.
        
        This works by moving other instructions from below the dependency tree
        to above the target instruction.
        """
        cycles_to_move = abs(self.cycles)
        any_moved = False
        # Reset blocking sets
        self._blocked_by_frozen = set()
        self._blocked_by_dependencies = set()
        self._blocked_by_branch = set()
        self._blocked_by_barrier = set()
        self._blocked_by_protected = set()
        self._no_candidates = False

        # Find branch/barrier boundary
        # s_barrier is a synchronization barrier that acts as a boundary line
        branch_boundary = len(block.instructions)
        boundary_is_barrier = False  # Track if boundary is s_barrier
        for i, instr in enumerate(block.instructions):
            if instr.is_branch or instr.is_terminator:
                branch_boundary = i
                boundary_is_barrier = False
                break
            if instr.opcode.lower() == 's_barrier':
                branch_boundary = i
                boundary_is_barrier = True
                break
        
        while self._total_cycles_moved < cycles_to_move:
            # Get current position of target
            target_idx = self._find_instruction_index(block, target_instr)
            if target_idx < 0:
                break
            
            # Find the dependency tree below target
            act, cc_idx = self._find_downward_dependency_tree(block, ddg, target_idx)
            
            if self.verbose:
                print(f"  ACT: {sorted(act)}, CC at index {cc_idx}, target at {target_idx}")
            
            # Step 2: Move instructions from below CC to above target
            moved_in_phase2 = False
            
            # Find instructions below CC, sorted by distance from target (closest first)
            # Closest to target means lowest index above cc_idx
            # IMPORTANT: Store instruction objects, not indices, because indices shift after moves!
            # Track reasons for skipping (use sets of id(instr) for unique counting)
            below_cc_instrs = []
            for idx in range(cc_idx + 1, len(block.instructions)):
                instr = block.instructions[idx]
                if idx >= branch_boundary:
                    # Count based on what type of boundary we hit
                    if boundary_is_barrier:
                        self._blocked_by_barrier.add(id(instr))
                    else:
                        self._blocked_by_branch.add(id(instr))
                    continue
                # Skip branch/terminator - they are boundaries
                if instr.is_branch or instr.is_terminator:
                    continue
                # Skip s_barrier - it's a synchronization boundary
                if instr.opcode.lower() == 's_barrier':
                    self._blocked_by_barrier.add(id(instr))
                    continue
                if instr in self.protected_instructions:
                    self._blocked_by_protected.add(id(instr))
                else:
                    below_cc_instrs.append(instr)
            
            for instr_to_move in below_cc_instrs:
                if self._total_cycles_moved >= cycles_to_move:
                    break
                
                # Re-find target position (may have shifted)
                current_target_idx = self._find_instruction_index(block, target_instr)
                if current_target_idx < 0:
                    break
                
                # Find current index of the instruction to move (indices shift after each move!)
                src_idx = self._find_instruction_index(block, instr_to_move)
                if src_idx < 0 or src_idx <= current_target_idx:
                    # Instruction not found or already above target
                    continue
                
                # Check if we can move this instruction to just above target
                # Respect frozen boundary - don't move instruction into frozen region
                dest_idx = max(current_target_idx, self.frozen_boundary)
                
                # If dest_idx is same or greater than src_idx, can't move (blocked by frozen boundary)
                if dest_idx >= src_idx:
                    self._blocked_by_frozen.add(id(instr_to_move))
                    continue
                
                # Don't pass protected_indices - check regular dependencies
                if self._can_move_single_instruction_up(block, ddg, src_idx, dest_idx, set()):
                    # Move it
                    instr_cycles = get_instruction_cycles(instr_to_move.opcode)
                    self._move_single_instruction_up_impl(block, ddg, src_idx, dest_idx)
                    self._total_cycles_moved += instr_cycles
                    any_moved = True
                    moved_in_phase2 = True
                    
                    if self.verbose:
                        print(f"    Moved [{src_idx}] {instr_to_move.opcode} up to [{dest_idx}] (+{instr_cycles} cycles)")
                else:
                    self._blocked_by_dependencies.add(id(instr_to_move))
            
            if self._total_cycles_moved >= cycles_to_move:
                break
            
            # Step 3: Move instructions from ACT gaps to above target
            if not moved_in_phase2 or self._total_cycles_moved < cycles_to_move:
                # Re-find target and ACT
                current_target_idx = self._find_instruction_index(block, target_instr)
                if current_target_idx < 0:
                    break
                
                current_act, current_cc_idx = self._find_downward_dependency_tree(block, ddg, current_target_idx)
                gaps = self._get_act_gaps(current_act, current_target_idx, current_cc_idx)
                
                moved_from_gaps = False
                for gap_idx in gaps:
                    if self._total_cycles_moved >= cycles_to_move:
                        break
                    
                    # Find the current index of the gap instruction
                    if gap_idx >= len(block.instructions):
                        continue
                    
                    gap_instr = block.instructions[gap_idx]
                    
                    # Re-find positions
                    current_target_idx = self._find_instruction_index(block, target_instr)
                    if current_target_idx < 0:
                        break
                    
                    # Skip protected instructions
                    if gap_instr in self.protected_instructions:
                        self._blocked_by_protected.add(id(gap_instr))
                        continue
                    
                    current_gap_idx = self._find_instruction_index(block, gap_instr)
                    if current_gap_idx < 0 or current_gap_idx <= current_target_idx:
                        continue
                    
                    # Respect frozen boundary - don't move instruction into frozen region
                    dest_idx = max(current_target_idx, self.frozen_boundary)
                    
                    # If dest_idx is same or greater than current_gap_idx, can't move (blocked by frozen boundary)
                    if dest_idx >= current_gap_idx:
                        self._blocked_by_frozen.add(id(gap_instr))
                        continue
                    
                    # For gap instructions, don't use protected_indices either
                    if self._can_move_single_instruction_up(block, ddg, current_gap_idx, dest_idx, set()):
                        instr_cycles = get_instruction_cycles(gap_instr.opcode)
                        self._move_single_instruction_up_impl(block, ddg, current_gap_idx, dest_idx)
                        self._total_cycles_moved += instr_cycles
                        any_moved = True
                        moved_from_gaps = True
                        
                        if self.verbose:
                            print(f"    Moved gap [{current_gap_idx}] {gap_instr.opcode} up to [{dest_idx}] (+{instr_cycles} cycles)")
                    else:
                        self._blocked_by_dependencies.add(id(gap_instr))
                
                if not moved_from_gaps and not moved_in_phase2:
                    # No more instructions can be moved
                    if len(below_cc_instrs) == 0 and len(gaps) == 0:
                        self._no_candidates = True
                    break
        
        return any_moved
    
    # =========================================================================
    # Legacy methods (kept for compatibility and internal use)
    # =========================================================================
    
    def _can_move_up(self, block: BasicBlock, ddg: Optional[DDG]) -> MoveResult:
        """
        Check if the instruction can be moved up (swap with previous instruction).
        
        Let A = instruction to move (at instr_index)
        Let B = instruction above (at instr_index - 1)
        
        After move: B will be after A
        
        Constraints:
        - RAW: A reads reg written by B -> BLOCKED
        - WAR: A writes reg read by B -> BLOCKED
        - WAW: A writes reg written by B -> OK (B will overwrite A's result)
        - s_waitcnt: Special handling
        - AVAIL: If A depends on B (s_waitcnt) for register availability -> CASCADE
        """
        idx = self.instr_index
        prev_idx = idx - 1
        
        instr_a = block.instructions[idx]      # Instruction to move
        instr_b = block.instructions[prev_idx] # Instruction above
        
        # Check if we need to cascade move with s_waitcnt
        cascaded = []
        if instr_b.opcode.lower() == 's_waitcnt' and ddg:
            _, uses_a = get_instruction_defs_uses(instr_a)
            cross_block_regs = ddg.waitcnt_cross_block_regs.get(prev_idx, set())
            intra_block_regs = ddg.nodes[prev_idx].available_regs if prev_idx < len(ddg.nodes) else set()
            all_avail_regs = cross_block_regs | intra_block_regs
            
            if all_avail_regs & uses_a:
                # A depends on the s_waitcnt - need to cascade
                cascaded.append(prev_idx)
                # The s_waitcnt will move with A, so we need to check if
                # s_waitcnt can move past the instruction before it
                if prev_idx > 0:
                    instr_before_waitcnt = block.instructions[prev_idx - 1]
                    # Check basic constraints for s_waitcnt moving up
                    # s_waitcnt doesn't define/use general registers, so RAW/WAR usually OK
                    # But need to check waitcnt count adjustments
                    waitcnt_check = self._check_waitcnt_move_up(
                        block, prev_idx, instr_b, instr_before_waitcnt
                    )
                    if not waitcnt_check.success:
                        return MoveResult(
                            success=False,
                            message=f"Cannot cascade s_waitcnt: {waitcnt_check.message}",
                            blocked_by="CASCADE_WAITCNT"
                        )
                
                # Now check if A can move past the instruction before s_waitcnt
                if prev_idx > 0:
                    instr_before_waitcnt = block.instructions[prev_idx - 1]
                    has_raw, raw_regs = has_raw_dependency(instr_a, instr_before_waitcnt)
                    if has_raw:
                        return MoveResult(
                            success=False,
                            message=f"RAW dependency (with cascade): {instr_a.opcode} reads {raw_regs} written by {instr_before_waitcnt.opcode}",
                            blocked_by="RAW"
                        )
                    
                    has_war, war_regs = has_war_dependency(instr_a, instr_before_waitcnt)
                    if has_war:
                        return MoveResult(
                            success=False,
                            message=f"WAR dependency (with cascade): {instr_a.opcode} writes {war_regs} read by {instr_before_waitcnt.opcode}",
                            blocked_by="WAR"
                        )
                
                return MoveResult(
                    success=True,
                    message="Move is legal (with s_waitcnt cascade)",
                    waitcnt_updated=False,
                    cascaded_moves=cascaded
                )
        
        # Check RAW: A reads what B writes -> try chain move
        has_raw, raw_regs = has_raw_dependency(instr_a, instr_b)
        if has_raw:
            # Try to build a dependency chain and move them together
            chain = find_immediate_dependency_chain(block, idx, self.direction, ddg)
            
            if len(chain) > 1:
                # We have a dependency chain - check if the whole chain can move
                chain_result = self._check_chain_move_up(block, ddg, chain)
                if chain_result.success:
                    return chain_result
            
            # Chain move not possible
            return MoveResult(
                success=False,
                message=f"RAW dependency: {instr_a.opcode} reads {raw_regs} written by {instr_b.opcode}",
                blocked_by="RAW"
            )
        
        # Check WAR: A writes what B reads -> BLOCKED
        has_war, war_regs = has_war_dependency(instr_a, instr_b)
        if has_war:
            return MoveResult(
                success=False,
                message=f"WAR dependency: {instr_a.opcode} writes {war_regs} read by {instr_b.opcode}",
                blocked_by="WAR"
            )
        
        # WAW: A writes what B writes -> OK (B will overwrite)
        # No blocking needed
        
        # Check s_waitcnt constraints
        waitcnt_result = self._check_waitcnt_move_up(block, idx, instr_a, instr_b)
        if not waitcnt_result.success:
            return waitcnt_result
        
        return MoveResult(
            success=True,
            message="Move is legal",
            waitcnt_updated=waitcnt_result.waitcnt_updated
        )
    
    def _check_chain_move_up(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        chain: List[int],
        max_chain_extension: int = 20  # Limit to prevent infinite loops
    ) -> MoveResult:
        """
        Check if a dependency chain can move up together.
        
        The chain is a list of instruction indices that depend on each other.
        We move the whole chain up by 1 position (swap the chain with the
        instruction before it).
        
        When the instruction before the chain has dependencies with chain
        instructions, we try to extend the chain to include it.
        
        Special handling for SCC:
        - WAW-SCC: If chain only writes SCC (doesn't read), can skip SCC writers
        - Tight pairs: s_add_u32+s_addc_u32 pairs are skipped as atomic units
        
        Args:
            block: The basic block
            ddg: The DDG
            chain: List of instruction indices in the chain (sorted ascending)
            max_chain_extension: Maximum number of times to extend the chain
            
        Returns:
            MoveResult indicating success or failure
        """
        if not chain:
            return MoveResult(success=False, message="Empty chain")
        
        chain = sorted(chain)
        # Keep track of original chain for dependency checks
        # When skipping SCC pairs, we only check original_chain's dependencies,
        # not the accumulated chain which would include skipped pairs
        original_chain = chain.copy()
        extensions = 0
        skipped_pairs = []  # Track SCC pairs that chain will skip over
        # Track the "virtual" position - where we're checking from
        virtual_check_start = min(chain)
        
        while extensions < max_chain_extension:
            if virtual_check_start == 0:
                return MoveResult(
                    success=False,
                    message="Chain is already at the beginning of the block"
                )
            
            # The instruction before the current check position
            before_chain_idx = virtual_check_start - 1
            instr_before = block.instructions[before_chain_idx]
            
            need_extension = False
            can_skip = False
            
            # Check if it's s_waitcnt - if any instruction in ORIGINAL chain depends on it
            if instr_before.opcode.lower() == 's_waitcnt' and ddg:
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(before_chain_idx, set())
                intra_block_regs = ddg.nodes[before_chain_idx].available_regs if before_chain_idx < len(ddg.nodes) else set()
                all_avail_regs = cross_block_regs | intra_block_regs
                
                # Check if any instruction in the ORIGINAL chain uses these registers
                for idx in original_chain:
                    instr = block.instructions[idx]
                    _, uses = get_instruction_defs_uses(instr)
                    if all_avail_regs & uses:
                        # The chain depends on s_waitcnt - include it in the chain
                        chain = [before_chain_idx] + chain
                        original_chain = [before_chain_idx] + original_chain
                        need_extension = True
                        extensions += 1
                        virtual_check_start = before_chain_idx
                        break
                
                if need_extension:
                    continue
            
            # Get defs/uses for instr_before
            defs_before, uses_before = get_instruction_defs_uses(instr_before)
            
            # Check for SCC pair (tight or separated: s_addc_u32 after s_add_u32)
            # If instr_before is end of a pair, check if ORIGINAL chain can skip the whole pair
            if is_scc_separated_pair_end(block, before_chain_idx):
                pair_start = find_scc_pair_start(block, before_chain_idx)
                if pair_start >= 0:
                    # Use original_chain for ALL dependency checks
                    if can_chain_skip_scc_pair(block, original_chain, pair_start, before_chain_idx, original_chain):
                        # Chain can skip over this pair (and instructions between)
                        # Record the skipped pair for later
                        skipped_pairs.append((pair_start, before_chain_idx))
                        # Update virtual check position to before the pair_start
                        # DO NOT add pair to chain - it stays in place
                        virtual_check_start = pair_start
                        can_skip = True
            
            if can_skip:
                # Successfully skipped a pair, continue checking for more pairs
                continue
            
            # Check RAW: any ORIGINAL chain instruction reads what instr_before writes?
            # Special handling: ignore SCC if chain only writes SCC
            has_scc_only_conflict = False
            
            for idx in original_chain:
                instr = block.instructions[idx]
                _, uses = get_instruction_defs_uses(instr)
                
                conflicts = defs_before & uses
                if conflicts:
                    # Check if conflict is only SCC and chain can ignore it
                    if conflicts == {'scc'}:
                        # Check if this is a RAW-SCC that matters
                        # It matters only if the chain instruction READS SCC
                        if is_scc_reader(instr.opcode.lower()):
                            # Chain instruction reads SCC - must extend
                            chain = [before_chain_idx] + chain
                            original_chain = [before_chain_idx] + original_chain
                            virtual_check_start = before_chain_idx
                            need_extension = True
                            extensions += 1
                            break
                        else:
                            # Chain instruction only writes SCC - can skip
                            has_scc_only_conflict = True
                            continue
                    else:
                        # Non-SCC conflict - must extend
                        chain = [before_chain_idx] + chain
                        original_chain = [before_chain_idx] + original_chain
                        virtual_check_start = before_chain_idx
                        need_extension = True
                        extensions += 1
                        break
            
            if need_extension:
                continue
            
            # Check WAR: any ORIGINAL chain instruction writes what instr_before reads?
            # Special handling: ignore SCC WAR if chain only writes SCC
            for idx in original_chain:
                instr = block.instructions[idx]
                defs, _ = get_instruction_defs_uses(instr)
                
                conflicts = defs & uses_before
                if conflicts:
                    # Check if conflict is only SCC
                    if conflicts == {'scc'}:
                        # WAR-SCC: chain writes SCC, instr_before reads SCC
                        # This can be ignored if chain only writes SCC (doesn't read)
                        # AND we're not inserting between a tight pair
                        if is_scc_only_writer(instr.opcode.lower()):
                            # Check if instr_before is part of an SCC pair (tight or separated)
                            if is_scc_separated_pair_end(block, before_chain_idx):
                                # It's the end of a pair - we should have handled this above
                                # If we're here, can_chain_skip_scc_pair returned False
                                # Must extend to include the entire pair region
                                pair_start = find_scc_pair_start(block, before_chain_idx)
                                if pair_start >= 0:
                                    # Include all instructions between pair_start and before_chain_idx
                                    pair_region = list(range(pair_start, before_chain_idx + 1))
                                    chain = pair_region + [idx for idx in chain if idx not in pair_region]
                                    chain = sorted(set(chain))
                                    original_chain = pair_region + [idx for idx in original_chain if idx not in pair_region]
                                    original_chain = sorted(set(original_chain))
                                    virtual_check_start = pair_start
                                    need_extension = True
                                    extensions += len(pair_region)
                                    break
                            else:
                                # Not a pair - can skip
                                continue
                        else:
                            # Chain reads SCC - must extend
                            chain = [before_chain_idx] + chain
                            original_chain = [before_chain_idx] + original_chain
                            virtual_check_start = before_chain_idx
                            need_extension = True
                            extensions += 1
                            break
                    else:
                        # Non-SCC conflict - must extend
                        chain = [before_chain_idx] + chain
                        original_chain = [before_chain_idx] + original_chain
                        virtual_check_start = before_chain_idx
                        need_extension = True
                        extensions += 1
                        break
            
            if need_extension:
                continue
            
            # No more extensions needed, chain can move past instr_before
            break
        
        if extensions >= max_chain_extension:
            return MoveResult(
                success=False,
                message=f"Chain extension limit reached ({max_chain_extension})",
                blocked_by="CHAIN_LIMIT"
            )
        
        # Re-check: at this point, chain should be able to move past instr_before
        before_chain_idx = virtual_check_start - 1
        if before_chain_idx < 0:
            return MoveResult(
                success=False,
                message="Extended chain is at the beginning of the block"
            )
        
        instr_before = block.instructions[before_chain_idx]
        defs_before, uses_before = get_instruction_defs_uses(instr_before)
        
        # Final check: make sure no dependency with instr_before
        # With special handling for SCC: WAW-SCC and WAR-SCC can be ignored
        # if the chain instruction only writes SCC (doesn't read it)
        # Use original_chain for final check since that's what will actually move
        for idx in original_chain:
            instr = block.instructions[idx]
            defs, uses = get_instruction_defs_uses(instr)
            
            # Check RAW
            raw_conflicts = defs_before & uses
            if raw_conflicts:
                # Check if conflict is only SCC and can be ignored
                if raw_conflicts == {'scc'} and not is_scc_reader(instr.opcode.lower()):
                    # Chain instruction doesn't read SCC, so this RAW-SCC doesn't matter
                    pass
                else:
                    return MoveResult(
                        success=False,
                        message=f"Chain blocked: {instr.opcode} reads registers written by {instr_before.opcode}",
                        blocked_by="RAW"
                    )
            
            # Check WAR
            war_conflicts = defs & uses_before
            if war_conflicts:
                # Check if conflict is only SCC and can be ignored
                if war_conflicts == {'scc'} and is_scc_only_writer(instr.opcode.lower()):
                    # Chain instruction only writes SCC, WAR-SCC can be ignored
                    pass
                else:
                    return MoveResult(
                        success=False,
                        message=f"Chain blocked: {instr.opcode} writes registers read by {instr_before.opcode}",
                        blocked_by="WAR"
                    )
        
        # Check s_waitcnt adjustments if any memory op in chain passes waitcnt or vice versa
        waitcnt_updated = False
        for idx in original_chain:
            instr = block.instructions[idx]
            opcode = instr.opcode.lower()
            
            if opcode == 's_waitcnt':
                # s_waitcnt in chain moving past instr_before
                opcode_before = instr_before.opcode.lower()
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr.operands)
                
                if is_vm_op(opcode_before) and vmcnt is not None:
                    if vmcnt >= 63:
                        return MoveResult(
                            success=False,
                            message="Cannot increment vmcnt beyond 63",
                            blocked_by="WAITCNT"
                        )
                    waitcnt_updated = True
                
                if is_lgkm_op(opcode_before) and lgkmcnt is not None:
                    if lgkmcnt >= 15:
                        return MoveResult(
                            success=False,
                            message="Cannot increment lgkmcnt beyond 15",
                            blocked_by="WAITCNT"
                        )
                    waitcnt_updated = True
        
        # If instr_before is s_waitcnt and any chain instruction is a memory op
        if instr_before.opcode.lower() == 's_waitcnt':
            vmcnt, lgkmcnt = parse_waitcnt_operands(instr_before.operands)
            
            for idx in original_chain:
                instr = block.instructions[idx]
                opcode = instr.opcode.lower()
                
                if is_vm_op(opcode) and vmcnt is not None:
                    if vmcnt <= 0:
                        return MoveResult(
                            success=False,
                            message=f"Cannot move {opcode} past s_waitcnt vmcnt(0)",
                            blocked_by="WAITCNT"
                        )
                    waitcnt_updated = True
                
                if is_lgkm_op(opcode) and lgkmcnt is not None:
                    if lgkmcnt <= 0:
                        return MoveResult(
                            success=False,
                            message=f"Cannot move {opcode} past s_waitcnt lgkmcnt(0)",
                            blocked_by="WAITCNT"
                        )
                    waitcnt_updated = True
        
        # Check if the instruction being displaced (instr_before) depends on a s_waitcnt
        # immediately before it. If so, they should move together.
        displaced_pair = []
        if before_chain_idx > 0 and ddg is not None:
            instr_before_before_idx = before_chain_idx - 1
            instr_before_before = block.instructions[instr_before_before_idx]
            
            if instr_before_before.opcode.lower() == 's_waitcnt':
                # Check if instr_before depends on this s_waitcnt (AVAIL dependency)
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(instr_before_before_idx, set())
                intra_block_regs = ddg.nodes[instr_before_before_idx].available_regs if instr_before_before_idx < len(ddg.nodes) else set()
                all_avail_regs = cross_block_regs | intra_block_regs
                
                _, uses_before = get_instruction_defs_uses(instr_before)
                
                if all_avail_regs & uses_before:
                    # instr_before depends on the s_waitcnt before it
                    # They should stay together when displaced
                    displaced_pair = [instr_before_before_idx, before_chain_idx]
        
        # For the move, we use original_chain (the actual instructions that move together)
        # skipped_pairs are regions that the chain passes through without including
        return MoveResult(
            success=True,
            message=f"Chain move is legal ({len(original_chain)} instructions, skipped {len(skipped_pairs)} SCC pairs)",
            waitcnt_updated=waitcnt_updated,
            cascaded_moves=original_chain,  # Store the actual moving chain
            displaced_pair=displaced_pair,  # Store any displaced pair that needs to stay together
            skipped_pairs=skipped_pairs  # Store SCC pairs that chain passes through
        )
    
    def _can_move_down(self, block: BasicBlock, ddg: Optional[DDG]) -> MoveResult:
        """
        Check if the instruction can be moved down (swap with next instruction).
        
        Let A = instruction to move (at instr_index)
        Let B = instruction below (at instr_index + 1)
        
        After move: A will be after B
        
        Constraints:
        - RAW: B reads reg written by A -> BLOCKED (A writes what B reads)
        - WAR: B writes reg read by A -> BLOCKED (A reads what B writes)
        - WAW: A writes reg written by B -> OK only if B's result is dead
        - s_waitcnt: Special handling
        """
        idx = self.instr_index
        next_idx = idx + 1
        
        instr_a = block.instructions[idx]      # Instruction to move
        instr_b = block.instructions[next_idx] # Instruction below
        
        # Check RAW: B reads what A writes -> BLOCKED
        # This is has_raw_dependency(B, A)
        has_raw, raw_regs = has_raw_dependency(instr_b, instr_a)
        if has_raw:
            return MoveResult(
                success=False,
                message=f"RAW dependency: {instr_b.opcode} reads {raw_regs} written by {instr_a.opcode}",
                blocked_by="RAW"
            )
        
        # Check WAR: B writes what A reads -> BLOCKED
        # This is has_war_dependency(B, A)
        has_war, war_regs = has_war_dependency(instr_b, instr_a)
        if has_war:
            return MoveResult(
                success=False,
                message=f"WAR dependency: {instr_b.opcode} writes {war_regs} read by {instr_a.opcode}",
                blocked_by="WAR"
            )
        
        # Check WAW: A writes what B writes
        has_waw, waw_regs = has_waw_dependency(instr_a, instr_b)
        if has_waw:
            # WAW is OK only if B's result is not used later
            for reg in waw_regs:
                if ddg and is_register_live_after_with_ddg(ddg, reg, next_idx):
                    return MoveResult(
                        success=False,
                        message=f"WAW dependency: {instr_b.opcode} writes {reg} which is used later",
                        blocked_by="WAW"
                    )
                elif not ddg and is_register_live_after(block, reg, next_idx):
                    return MoveResult(
                        success=False,
                        message=f"WAW dependency: {instr_b.opcode} writes {reg} which may be used later",
                        blocked_by="WAW"
                    )
        
        # Check s_waitcnt constraints
        waitcnt_result = self._check_waitcnt_move_down(block, idx, instr_a, instr_b)
        if not waitcnt_result.success:
            return waitcnt_result
        
        return MoveResult(
            success=True,
            message="Move is legal",
            waitcnt_updated=waitcnt_result.waitcnt_updated
        )
    
    def _check_waitcnt_move_up(
        self,
        block: BasicBlock,
        idx: int,
        instr_a: Instruction,
        instr_b: Instruction
    ) -> MoveResult:
        """
        Check s_waitcnt constraints when moving instruction A up past B.
        
        Cases:
        1. A is memory op, B is s_waitcnt: May need to decrement count
        2. A is s_waitcnt, B is memory op: May need to increment count
        3. Other cases: No waitcnt adjustment needed
        """
        opcode_a = instr_a.opcode.lower()
        opcode_b = instr_b.opcode.lower()
        
        # Case 1: Moving memory op up past s_waitcnt
        if opcode_b == 's_waitcnt':
            vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
            
            if is_vm_op(opcode_a) and vmcnt is not None:
                # Moving VM op before waitcnt -> need to decrement vmcnt
                if vmcnt <= 0:
                    return MoveResult(
                        success=False,
                        message=f"Cannot move {opcode_a} past s_waitcnt vmcnt(0)",
                        blocked_by="WAITCNT"
                    )
                # Will decrement in perform_move
                return MoveResult(success=True, waitcnt_updated=True)
            
            if is_lgkm_op(opcode_a) and lgkmcnt is not None:
                # Moving LGKM op before waitcnt -> need to decrement lgkmcnt
                if lgkmcnt <= 0:
                    return MoveResult(
                        success=False,
                        message=f"Cannot move {opcode_a} past s_waitcnt lgkmcnt(0)",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
        
        # Case 2: Moving s_waitcnt up past memory op
        if opcode_a == 's_waitcnt':
            vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
            
            if is_vm_op(opcode_b) and vmcnt is not None:
                # s_waitcnt moves before the memory op it was waiting for
                # Need to increment vmcnt (allow one more pending)
                if vmcnt >= 63:
                    return MoveResult(
                        success=False,
                        message="Cannot increment vmcnt beyond 63",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
            
            if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                if lgkmcnt >= 15:
                    return MoveResult(
                        success=False,
                        message="Cannot increment lgkmcnt beyond 15",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
        
        return MoveResult(success=True, waitcnt_updated=False)
    
    def _check_waitcnt_move_down(
        self,
        block: BasicBlock,
        idx: int,
        instr_a: Instruction,
        instr_b: Instruction
    ) -> MoveResult:
        """
        Check s_waitcnt constraints when moving instruction A down past B.
        
        Cases:
        1. A is memory op, B is s_waitcnt: May need to increment count
        2. A is s_waitcnt, B is memory op: May need to decrement count
        3. Other cases: No waitcnt adjustment needed
        """
        opcode_a = instr_a.opcode.lower()
        opcode_b = instr_b.opcode.lower()
        
        # Case 1: Moving memory op down past s_waitcnt
        if opcode_b == 's_waitcnt':
            vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
            
            if is_vm_op(opcode_a) and vmcnt is not None:
                # Memory op moves after waitcnt -> need to increment vmcnt
                if vmcnt >= 63:
                    return MoveResult(
                        success=False,
                        message="Cannot increment vmcnt beyond 63",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
            
            if is_lgkm_op(opcode_a) and lgkmcnt is not None:
                if lgkmcnt >= 15:
                    return MoveResult(
                        success=False,
                        message="Cannot increment lgkmcnt beyond 15",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
        
        # Case 2: Moving s_waitcnt down past memory op
        if opcode_a == 's_waitcnt':
            vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
            
            if is_vm_op(opcode_b) and vmcnt is not None:
                # s_waitcnt moves after new memory op
                # Need to decrement vmcnt (wait for one more)
                if vmcnt <= 0:
                    return MoveResult(
                        success=False,
                        message="Cannot decrement vmcnt below 0",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
            
            if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                if lgkmcnt <= 0:
                    return MoveResult(
                        success=False,
                        message="Cannot decrement lgkmcnt below 0",
                        blocked_by="WAITCNT"
                    )
                return MoveResult(success=True, waitcnt_updated=True)
        
        return MoveResult(success=True, waitcnt_updated=False)
    
    def _perform_move(
        self,
        block: BasicBlock,
        result: AnalysisResult,
        cascaded_moves: Optional[List[int]] = None,
        displaced_pair: Optional[List[int]] = None
    ) -> None:
        """
        Actually perform the instruction move and update raw_lines.
        
        Args:
            block: The basic block
            result: The analysis result
            cascaded_moves: List of instruction indices that should move along
                           (this is the dependency chain including the target)
            displaced_pair: List of instruction indices that should stay together
                           when displaced (e.g., [s_waitcnt_idx, dependent_idx])
        """
        cascaded_moves = cascaded_moves or []
        displaced_pair = displaced_pair or []
        
        if self.direction < 0 and cascaded_moves and len(cascaded_moves) > 1:
            # Moving up with a dependency chain
            # cascaded_moves is the full chain [i1, i2, ..., iN] sorted ascending
            # 
            # Normal case:
            # Before: [... B, i1, i2, ..., iN, ...]  (we want to move chain up)
            # After:  [... i1, i2, ..., iN, B, ...]
            #
            # With displaced_pair [s_waitcnt, B]:
            # Before: [... s_waitcnt, B, i1, i2, ..., iN, ...]
            # After:  [... i1, i2, ..., iN, s_waitcnt, B, ...]
            #
            # We do this by:
            # 1. Get the instruction(s) before the chain that need to be displaced
            # 2. Handle any s_waitcnt adjustments
            # 3. "Rotate" displaced instructions to the end of the chain by swapping
            
            chain = sorted(cascaded_moves)
            chain_start = min(chain)
            
            if chain_start > 0:
                # Determine what instructions need to be displaced
                if displaced_pair and len(displaced_pair) == 2:
                    # We have a pair (s_waitcnt, B) that should stay together
                    # They will both be rotated to the end
                    displaced_indices = sorted(displaced_pair)  # [s_waitcnt_idx, B_idx]
                else:
                    # Just the single instruction before the chain
                    before_chain_idx = chain_start - 1
                    displaced_indices = [before_chain_idx]
                
                # Handle s_waitcnt count adjustments
                # For each displaced instruction, if it's s_waitcnt and chain has mem ops,
                # or if chain has s_waitcnt and displaced has mem ops, adjust counts
                for disp_idx in displaced_indices:
                    disp_instr = block.instructions[disp_idx]
                    disp_opcode = disp_instr.opcode.lower()
                    
                    # Chain s_waitcnt moving past displaced mem op
                    for chain_idx in chain:
                        chain_instr = block.instructions[chain_idx]
                        if chain_instr.opcode.lower() == 's_waitcnt':
                            vmcnt, lgkmcnt = parse_waitcnt_operands(chain_instr.operands)
                            if is_vm_op(disp_opcode) and vmcnt is not None:
                                update_waitcnt_instruction(chain_instr, vmcnt_delta=+1)
                                sync_instruction_to_raw_lines(block, chain_instr)
                            if is_lgkm_op(disp_opcode) and lgkmcnt is not None:
                                update_waitcnt_instruction(chain_instr, lgkmcnt_delta=+1)
                                sync_instruction_to_raw_lines(block, chain_instr)
                    
                    # Displaced s_waitcnt moving AFTER chain mem ops
                    # (s_waitcnt was before chain, now it's after chain)
                    # This means mem ops now execute BEFORE s_waitcnt
                    # s_waitcnt should INCREMENT count to allow one more pending per mem op
                    if disp_opcode == 's_waitcnt':
                        vmcnt, lgkmcnt = parse_waitcnt_operands(disp_instr.operands)
                        for chain_idx in chain:
                            chain_instr = block.instructions[chain_idx]
                            chain_opcode = chain_instr.opcode.lower()
                            if is_vm_op(chain_opcode) and vmcnt is not None:
                                if vmcnt >= 63:
                                    # Cannot increment beyond max, but this is unusual
                                    pass
                                else:
                                    update_waitcnt_instruction(disp_instr, vmcnt_delta=+1)
                                    vmcnt += 1
                            if is_lgkm_op(chain_opcode) and lgkmcnt is not None:
                                if lgkmcnt >= 15:
                                    pass
                                else:
                                    update_waitcnt_instruction(disp_instr, lgkmcnt_delta=+1)
                                    lgkmcnt += 1
                        # Sync the updated s_waitcnt to raw_lines for file regeneration
                        sync_instruction_to_raw_lines(block, disp_instr)
                
                # Perform the rotation to move displaced instructions to the end of the chain
                # 
                # Example with displaced_indices = [33, 34] (s_waitcnt at 33, B at 34)
                # and chain = [35, 36, 37, 38, 39]:
                # 
                # Original: [33=s_waitcnt, 34=B, 35=c1, 36=c2, 37=c3, 38=c4, 39=c5]
                # We want:  [c1, c2, c3, c4, c5, s_waitcnt, B] at positions [33-39]
                #
                # Algorithm: 
                # - First, bubble B (at 34) to position 39
                # - Then, bubble s_waitcnt (now at 33) to position 38
                # This gives us: [c1, c2, c3, c4, c5, s_waitcnt, B]
                #
                # We process displaced_indices in REVERSE order (last element first)
                # so that they end up in the correct order at the end.
                
                working_end = max(chain)
                for i, _ in enumerate(reversed(displaced_indices)):
                    # Process from the last displaced instruction to the first
                    # After each iteration, the current displaced instruction is at
                    # the highest index of the remaining displaced group
                    current_pos = max(displaced_indices) - i
                    target_pos = working_end - i
                    
                    # Bubble the instruction at current_pos to target_pos
                    while current_pos < target_pos:
                        self._swap_instructions(block, current_pos, current_pos + 1)
                        current_pos += 1
                
                return
        
        elif self.direction < 0 and cascaded_moves and len(cascaded_moves) == 1:
            # Special case: single cascaded instruction (e.g., s_waitcnt only)
            idx = self.instr_index
            waitcnt_idx = cascaded_moves[0]
            
            if waitcnt_idx == idx - 1 and waitcnt_idx > 0:
                before_waitcnt_idx = waitcnt_idx - 1
                
                instr_waitcnt = block.instructions[waitcnt_idx]
                instr_b = block.instructions[before_waitcnt_idx]
                
                # Move s_waitcnt up past instr_b
                opcode_b = instr_b.opcode.lower()
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_waitcnt.operands)
                if is_vm_op(opcode_b) and vmcnt is not None:
                    update_waitcnt_instruction(instr_waitcnt, vmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_waitcnt)
                if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_waitcnt, lgkmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_waitcnt)
                
                # Swap s_waitcnt with B
                self._swap_instructions(block, waitcnt_idx, before_waitcnt_idx)
                
                # Now swap A with B (A is at idx, B is at waitcnt_idx)
                self._swap_instructions(block, idx, waitcnt_idx)
                
                return
        
        # Standard move (no cascade)
        idx = self.instr_index
        new_idx = idx + self.direction
        
        instr_a = block.instructions[idx]
        instr_b = block.instructions[new_idx]
        
        opcode_a = instr_a.opcode.lower()
        opcode_b = instr_b.opcode.lower()
        
        # Update s_waitcnt if needed
        if self.direction < 0:  # Moving up
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                if is_vm_op(opcode_a) and vmcnt is not None:
                    update_waitcnt_instruction(instr_b, vmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_b)
                if is_lgkm_op(opcode_a) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_b, lgkmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_b)
            
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                if is_vm_op(opcode_b) and vmcnt is not None:
                    update_waitcnt_instruction(instr_a, vmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_a)
                if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_a, lgkmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_a)
        
        else:  # Moving down
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                if is_vm_op(opcode_a) and vmcnt is not None:
                    update_waitcnt_instruction(instr_b, vmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_b)
                if is_lgkm_op(opcode_a) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_b, lgkmcnt_delta=+1)
                    sync_instruction_to_raw_lines(block, instr_b)
            
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                if is_vm_op(opcode_b) and vmcnt is not None:
                    update_waitcnt_instruction(instr_a, vmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_a)
                if is_lgkm_op(opcode_b) and lgkmcnt is not None:
                    update_waitcnt_instruction(instr_a, lgkmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_a)
        
        # Swap instructions
        self._swap_instructions(block, idx, new_idx)
    
    def _swap_instructions(self, block: BasicBlock, idx1: int, idx2: int) -> None:
        """
        Swap two instructions in the block and update raw_lines.
        """
        instr_a = block.instructions[idx1]
        instr_b = block.instructions[idx2]
        
        # Swap in the list
        block.instructions[idx1], block.instructions[idx2] = \
            block.instructions[idx2], block.instructions[idx1]
        
        # Update raw_lines - swap the line numbers
        addr_a = instr_a.address
        addr_b = instr_b.address
        
        if addr_a in block.raw_lines and addr_b in block.raw_lines:
            # Swap raw line contents
            block.raw_lines[addr_a], block.raw_lines[addr_b] = \
                block.raw_lines[addr_b], block.raw_lines[addr_a]
            
            # Update instruction addresses
            instr_a.address = addr_b
            instr_b.address = addr_a


# =============================================================================
# Distribute Instruction Pass
# =============================================================================

class DistributeInstructionPass(Pass):
    """
    Pass that distributes a specified instruction type evenly across a basic block
    based on cycle timing.
    
    This pass:
    1. Finds all instructions of the target opcode in the block
    2. Distributes the first K instructions evenly based on cycle positions
    3. Moves remaining (M-K) instructions to the end of the block
    4. Respects branch/terminator boundaries (cannot move past them)
    
    Attributes:
        block_label: Label of the block to modify
        target_opcode: Exact opcode to match (e.g., "global_load_dwordx4")
        distribute_count: K, number of instructions to distribute evenly
        verbose: Print detailed information during execution
    """
    
    def __init__(
        self,
        block_label: str,
        target_opcode: str,
        distribute_count: int,
        verbose: bool = False
    ):
        """
        Initialize the pass.
        
        Args:
            block_label: Label of the block (e.g., ".LBB0_0")
            target_opcode: Exact opcode to match (e.g., "global_load_dwordx4")
            distribute_count: K, number of instructions to distribute evenly
            verbose: Print detailed information during execution
        """
        self.block_label = block_label
        self.target_opcode = target_opcode
        self.distribute_count = distribute_count
        self.verbose = verbose
        self._moved_count = 0
    
    @property
    def name(self) -> str:
        return f"DistributeInstruction({self.block_label}, {self.target_opcode}, K={self.distribute_count})"
    
    @property
    def description(self) -> str:
        return f"Distribute {self.target_opcode} instructions evenly in block {self.block_label}"
    
    def _find_target_instructions(self, block: BasicBlock) -> List[int]:
        """Find all indices of target instructions in the block."""
        indices = []
        for i, instr in enumerate(block.instructions):
            if instr.opcode == self.target_opcode:
                indices.append(i)
        return indices
    
    def _calculate_block_cycles(self, block: BasicBlock) -> int:
        """Calculate total cycles for all instructions in the block."""
        total = 0
        for instr in block.instructions:
            total += get_instruction_cycles(instr.opcode)
        return total
    
    def _calculate_ideal_cycles(self, total_cycles: int, k: int) -> List[int]:
        """
        Calculate ideal cycle positions for K evenly distributed instructions.
        
        For K instructions over N cycles:
        - Interval = N / (K + 1)
        - Position i (1-indexed) = i * interval
        
        Args:
            total_cycles: Total cycles N in the block
            k: Number of instructions to distribute
            
        Returns:
            List of ideal cycle positions for each instruction
        """
        if k <= 0:
            return []
        
        interval = total_cycles / (k + 1)
        return [int(i * interval) for i in range(1, k + 1)]
    
    def _find_branch_boundary(self, block: BasicBlock) -> int:
        """
        Find index of first branch/terminator/s_barrier instruction.
        Instructions cannot be moved past this boundary.
        
        s_barrier is a synchronization barrier that acts as a boundary line -
        instructions before and after it cannot cross the boundary.
        
        Returns:
            Index of first branch/terminator/s_barrier, or len(instructions) if none
        """
        for i, instr in enumerate(block.instructions):
            if instr.is_branch or instr.is_terminator:
                return i
            if instr.opcode.lower() == 's_barrier':
                return i
        return len(block.instructions)
    
    def _cycle_to_index(self, block: BasicBlock, target_cycle: int) -> int:
        """
        Convert a target cycle position to an instruction index.
        
        Finds the instruction that starts at or just after the target cycle.
        
        Args:
            block: The basic block
            target_cycle: Target cycle position
            
        Returns:
            Instruction index corresponding to the target cycle
        """
        cumulative = 0
        for i, instr in enumerate(block.instructions):
            if cumulative >= target_cycle:
                return i
            cumulative += get_instruction_cycles(instr.opcode)
        return len(block.instructions) - 1
    
    def _get_instruction_cycle_position(self, block: BasicBlock, index: int) -> int:
        """
        Get the cycle position (start cycle) of an instruction at given index.
        
        Args:
            block: The basic block
            index: Instruction index
            
        Returns:
            Starting cycle of the instruction
        """
        cumulative = 0
        for i, instr in enumerate(block.instructions):
            if i == index:
                return cumulative
            cumulative += get_instruction_cycles(instr.opcode)
        return cumulative
    
    def _find_nth_target(self, block: BasicBlock, n: int) -> int:
        """
        Find the index of the nth target instruction (0-indexed).
        
        Args:
            block: The basic block
            n: Which target instruction (0-indexed)
            
        Returns:
            Index of the nth target instruction, or -1 if not found
        """
        count = 0
        for i, instr in enumerate(block.instructions):
            if instr.opcode == self.target_opcode:
                if count == n:
                    return i
                count += 1
        return -1
    
    def _move_instruction_toward(
        self,
        result: AnalysisResult,
        current_idx: int,
        target_idx: int,
        branch_boundary: int,
        frozen_boundary: int = 0,
        protected_instructions: Optional[List[Instruction]] = None
    ) -> int:
        """
        Move an instruction toward a target position using cycle-based MoveInstructionPass.
        
        This method calculates the cycle difference between current and target positions,
        then uses MoveInstructionPass with the appropriate cycle count to move the instruction.
        
        Args:
            result: The AnalysisResult to modify
            current_idx: Current index of the instruction
            target_idx: Target index to move toward
            branch_boundary: Index of branch/terminator (cannot move past)
            frozen_boundary: Instructions at idx < frozen_boundary cannot be moved
            protected_instructions: List of instruction objects that should never be moved
            
        Returns:
            Final index of the instruction after all moves
        """
        block = result.cfg.blocks[self.block_label]
        
        # Store the instruction object reference for accurate position tracking
        target_instr = block.instructions[current_idx]
        
        if current_idx == target_idx:
            return current_idx  # Already at target
        
        # Clamp target to valid range considering branch boundary
        if current_idx < target_idx:
            # Moving down: can't go past branch_boundary - 1
            target_idx = min(target_idx, branch_boundary - 1)
        else:
            # Moving up: can always go up (no upper constraint other than 0)
            target_idx = max(target_idx, 0)
        
        if current_idx == target_idx:
            return current_idx  # Already at (clamped) target
        
        # Calculate cycle difference
        current_cycle = self._get_instruction_cycle_position(block, current_idx)
        target_cycle = self._get_instruction_cycle_position(block, target_idx)
        cycle_diff = target_cycle - current_cycle
        
        if cycle_diff == 0:
            return current_idx
        
        # Determine cycles to move:
        # - cycle_diff > 0: target is at higher cycle (further down), so move down (negative cycles)
        # - cycle_diff < 0: target is at lower cycle (further up), so move up (positive cycles)
        # 
        # MoveInstructionPass convention: positive cycles = move up, negative cycles = move down
        cycles_to_move = -cycle_diff  # Negate because we want to move toward target
        
        if self.verbose:
            direction_str = "up" if cycles_to_move > 0 else "down"
            print(f"    Cycle-based move: {abs(cycles_to_move)} cycles {direction_str}")
        
        # Use the new cycle-based MoveInstructionPass
        move_pass = MoveInstructionPass(
            self.block_label,
            current_idx,
            cycles_to_move,
            verbose=False,
            frozen_boundary=frozen_boundary,
            protected_instructions=protected_instructions
        )
        
        success = move_pass.run(result)
        
        if success:
            self._moved_count += 1
            if self.verbose and move_pass.total_cycles_moved > 0:
                print(f"    Actually moved: {move_pass.total_cycles_moved} cycles")
                # Show reason if didn't reach target
                if move_pass.total_cycles_moved < abs(cycles_to_move):
                    print(f"    Stop reason: {move_pass.stop_reason}")
        else:
            if self.verbose:
                print(f"    Move blocked: {move_pass.last_result.message if move_pass.last_result else 'Unknown'}")
                if move_pass.stop_reason:
                    print(f"    Stop reason: {move_pass.stop_reason}")
        
        # Return the actual final position
        return self._find_instruction_index(block, target_instr)
    
    def _find_instruction_index(self, block: BasicBlock, target_instr: Instruction) -> int:
        """
        Find the current index of an instruction in the block.
        
        Args:
            block: The basic block to search
            target_instr: The instruction object to find
            
        Returns:
            The index of the instruction, or -1 if not found
        """
        for idx, instr in enumerate(block.instructions):
            if instr is target_instr:
                return idx
        return -1
    
    def _find_root_blocking_dependency(
        self,
        block: BasicBlock,
        chain: List[int],
        ddg: Optional[DDG] = None
    ) -> str:
        """
        Find the root cause of why a chain cannot move further up.
        
        Traces through the dependency chain to find the ultimate blocker
        (the instruction that the chain truly depends on).
        
        Args:
            block: The basic block
            chain: List of instruction indices in the chain
            ddg: Optional DDG for additional context
            
        Returns:
            A string describing the root blocking dependency
        """
        if not chain:
            return "Empty chain"
        
        chain_start = min(chain)
        if chain_start == 0:
            return "Chain is at the beginning of the block"
        
        before_chain_idx = chain_start - 1
        instr_before = block.instructions[before_chain_idx]
        defs_before, uses_before = get_instruction_defs_uses(instr_before)
        
        # Find which chain instruction depends on instr_before
        for chain_idx in sorted(chain):
            chain_instr = block.instructions[chain_idx]
            chain_defs, chain_uses = get_instruction_defs_uses(chain_instr)
            
            # Check RAW: chain reads what before writes
            raw_conflicts = defs_before & chain_uses
            # Ignore SCC-only conflicts for SCC-only writers
            if raw_conflicts:
                if raw_conflicts == {'scc'} and is_scc_only_writer(chain_instr.opcode.lower()):
                    continue
                return (
                    f"[{before_chain_idx}] {instr_before.opcode} writes {raw_conflicts} "
                    f"needed by [{chain_idx}] {chain_instr.opcode}"
                )
            
            # Check WAR: chain writes what before reads
            war_conflicts = chain_defs & uses_before
            if war_conflicts:
                if war_conflicts == {'scc'} and is_scc_only_writer(chain_instr.opcode.lower()):
                    continue
                return (
                    f"[{chain_idx}] {chain_instr.opcode} writes {war_conflicts} "
                    f"read by [{before_chain_idx}] {instr_before.opcode}"
                )
        
        # Check if it's an SCC pair that can't be skipped
        if is_scc_separated_pair_end(block, before_chain_idx):
            pair_start = find_scc_pair_start(block, before_chain_idx)
            if pair_start >= 0:
                pair_start_instr = block.instructions[pair_start]
                pair_end_instr = block.instructions[before_chain_idx]
                return (
                    f"SCC pair [{pair_start}] {pair_start_instr.opcode} + "
                    f"[{before_chain_idx}] {pair_end_instr.opcode} cannot be skipped "
                    f"(chain contains SCC reader or has data dependency with pair)"
                )
        
        return f"Unknown dependency with [{before_chain_idx}] {instr_before.opcode}"
    
    def run(self, result: AnalysisResult) -> bool:
        """
        Execute the pass to distribute instructions evenly.
        
        Returns:
            True if any changes were made, False otherwise.
            
        Raises:
            SchedulingVerificationError: If the optimization violates any dependency constraints.
        """
        # Import verification module
        from amdgcn_verify import build_global_ddg, verify_optimization
        
        # Validate block exists
        if self.block_label not in result.cfg.blocks:
            if self.verbose:
                print(f"Block '{self.block_label}' not found")
            return False
        
        # Capture original dependency graph BEFORE any changes
        original_gdg = build_global_ddg(result.cfg, result.ddgs)
        
        block = result.cfg.blocks[self.block_label]
        self._moved_count = 0
        
        # 1. Find all target instructions
        target_indices = self._find_target_instructions(block)
        M = len(target_indices)
        
        if M == 0:
            if self.verbose:
                print(f"No {self.target_opcode} instructions found in {self.block_label}")
            return False
        
        K = min(self.distribute_count, M)
        
        if self.verbose:
            print(f"Found {M} {self.target_opcode} instructions, distributing {K} evenly")
        
        # 2. Calculate total cycles
        total_cycles = self._calculate_block_cycles(block)
        
        if self.verbose:
            print(f"Block total cycles: {total_cycles}")
        
        # 3. Calculate ideal cycle positions for K instructions
        ideal_cycles = self._calculate_ideal_cycles(total_cycles, K)
        
        if self.verbose:
            print(f"Ideal cycle positions: {ideal_cycles}")
        
        # 4. Find branch boundary
        branch_boundary = self._find_branch_boundary(block)
        
        if self.verbose:
            print(f"Branch boundary at index: {branch_boundary}")
        
        # 5. Collect all target instruction objects for protection
        # This ensures when moving one target, we don't accidentally move others
        all_target_instrs = []
        for i in range(M):
            idx = self._find_nth_target(block, i)
            if idx >= 0:
                all_target_instrs.append(block.instructions[idx])
        
        # 6. Move first K instructions to ideal positions (in order)
        # Track frozen boundary: after each instruction is placed, freeze that position and above
        frozen_boundary = 0
        
        for i in range(K):
            # Re-find the nth target instruction (index may have changed)
            current_idx = self._find_nth_target(block, i)
            if current_idx < 0:
                if self.verbose:
                    print(f"  Could not find target instruction {i}")
                continue
            
            ideal_cycle = ideal_cycles[i]
            target_idx = self._cycle_to_index(block, ideal_cycle)
            
            # Don't move past already-placed instructions
            # (they should stay in order)
            if i > 0:
                prev_target_idx = self._find_nth_target(block, i - 1)
                if prev_target_idx >= 0 and target_idx <= prev_target_idx:
                    target_idx = prev_target_idx + 1
            
            # Ensure target is at least at frozen_boundary (can't place into frozen region)
            target_idx = max(target_idx, frozen_boundary)
            
            # Create protected list: all remaining target instructions (i+1, i+2, ..., M-1)
            # These should never be moved when moving instruction i
            protected_instrs = all_target_instrs[i+1:]
            
            if self.verbose:
                current_cycle = self._get_instruction_cycle_position(block, current_idx)
                print(f"  [{i}] Moving from idx={current_idx} (cycle={current_cycle}) to idx={target_idx} (ideal cycle={ideal_cycle})")
            
            final_idx = self._move_instruction_toward(result, current_idx, target_idx, branch_boundary, frozen_boundary, protected_instrs)
            
            if self.verbose:
                final_cycle = self._get_instruction_cycle_position(block, final_idx)
                print(f"      Final position: idx={final_idx} (cycle={final_cycle})")
            
            # Update frozen boundary: everything at or above final_idx is now frozen
            frozen_boundary = final_idx + 1
        
        # 7. Move remaining M-K instructions toward the end
        if M > K:
            if self.verbose:
                print(f"Moving {M - K} remaining instructions to block end")
            
            for i in range(K, M):
                current_idx = self._find_nth_target(block, i)
                if current_idx < 0:
                    continue
                
                # Target is just before branch boundary
                target_idx = branch_boundary - 1
                
                # Ensure target is at least at frozen_boundary
                target_idx = max(target_idx, frozen_boundary)
                
                # Create protected list: all remaining target instructions
                protected_instrs = all_target_instrs[i+1:]
                
                if self.verbose:
                    print(f"  [{i}] Moving from idx={current_idx} to end (target={target_idx})")
                
                final_idx = self._move_instruction_toward(result, current_idx, target_idx, branch_boundary, frozen_boundary, protected_instrs)
                
                if self.verbose:
                    print(f"      Final position: idx={final_idx}")
                
                # Update frozen boundary for remaining instructions
                frozen_boundary = final_idx + 1
        
        if self.verbose:
            print(f"Total moves made: {self._moved_count}")
            
            # Print final summary of all target instruction positions
            # This is the ACTUAL final position after all moves, which may differ
            # from intermediate "Final position" logs due to subsequent moves
            print(f"\n=== Final {self.target_opcode} positions (after all moves) ===")
            total_error = 0
            target_count = 0
            for i, instr in enumerate(block.instructions):
                if instr.opcode == self.target_opcode:
                    cycle = self._get_instruction_cycle_position(block, i)
                    # Get destination register for identification
                    dest = instr.raw_line.split(',')[0].split()[-1] if instr.raw_line else '?'
                    
                    # Get ideal cycle and calculate error
                    if target_count < len(ideal_cycles):
                        ideal = ideal_cycles[target_count]
                        error = abs(cycle - ideal)
                        total_error += error
                        print(f"  [{target_count}] idx={i}, dest={dest}, current_cycle={cycle}, ideal_cycle={ideal}, error={error}")
                    else:
                        # Instructions beyond K don't have ideal positions
                        print(f"  [{target_count}] idx={i}, dest={dest}, current_cycle={cycle}, ideal_cycle=N/A (beyond K)")
                    target_count += 1
            print(f"-----------------------------------------------")
            print(f"  Total absolute error: {total_error}")
            print(f"  Average error per instruction: {total_error / K if K > 0 else 0:.1f}")
            print(f"===============================================\n")
        
        # Mandatory verification: raises SchedulingVerificationError on failure
        verify_optimization(original_gdg, result.cfg)
        
        return self._moved_count > 0


# =============================================================================
# Utility Functions
# =============================================================================

def apply_passes(result: AnalysisResult, passes: List[Pass], verbose: bool = False) -> bool:
    """
    Convenience function to apply a list of passes to an analysis result.
    
    Args:
        result: The AnalysisResult to modify
        passes: List of Pass instances to run
        verbose: Print progress information
        
    Returns:
        True if any pass made changes
    """
    pm = PassManager()
    pm.verbose = verbose
    for p in passes:
        pm.add_pass(p)
    return pm.run_all(result)


def move_instruction(
    result: AnalysisResult,
    block_label: str,
    instr_index: int,
    cycles: int,
    verbose: bool = False,
    frozen_boundary: int = 0,
    protected_instructions: Optional[List['Instruction']] = None
) -> MoveResult:
    """
    Convenience function to move a single instruction by a specified number of cycles.
    
    Args:
        result: The AnalysisResult to modify
        block_label: Label of the block
        instr_index: Index of instruction to move
        cycles: Number of cycles to move (positive = move up, negative = move down)
        verbose: Print progress information
        frozen_boundary: Instructions at idx < frozen_boundary cannot be moved
        protected_instructions: List of instruction objects that should never be moved
        
    Returns:
        MoveResult with success status and message
    """
    pass_ = MoveInstructionPass(
        block_label=block_label,
        instr_index=instr_index,
        cycles=cycles,
        verbose=verbose,
        frozen_boundary=frozen_boundary,
        protected_instructions=protected_instructions
    )
    
    pm = PassManager()
    pm.verbose = verbose
    pm.add_pass(pass_)
    pm.run_all(result)
    
    return pass_.last_result or MoveResult(success=False, message="Unknown error")


def distribute_instructions(
    result: AnalysisResult,
    block_label: str,
    target_opcode: str,
    distribute_count: int,
    verbose: bool = False
) -> bool:
    """
    Convenience function to distribute instructions evenly across a basic block.
    
    This function distributes the first K instances of the target instruction type
    evenly across the block based on cycle timing, then moves remaining instructions
    to the end of the block.
    
    Args:
        result: The AnalysisResult to modify
        block_label: Label of the block (e.g., ".LBB0_0")
        target_opcode: Exact opcode to match (e.g., "global_load_dwordx4")
        distribute_count: K, number of instructions to distribute evenly
        verbose: Print progress information
        
    Returns:
        True if any changes were made, False otherwise
    """
    pass_ = DistributeInstructionPass(
        block_label=block_label,
        target_opcode=target_opcode,
        distribute_count=distribute_count,
        verbose=verbose
    )
    
    pm = PassManager()
    pm.verbose = verbose
    pm.add_pass(pass_)
    return pm.run_all(result)


# =============================================================================
# Main (for testing)
# =============================================================================

def main():
    """Test the pass framework."""
    import argparse
    from amdgcn_ddg import load_analysis_from_json, save_analysis_to_json
    
    parser = argparse.ArgumentParser(
        description='Test AMDGCN instruction scheduling passes'
    )
    parser.add_argument('input', help='Input analysis JSON file')
    parser.add_argument('--block', '-b', required=True,
                       help='Block label (e.g., .LBB0_0)')
    parser.add_argument('--index', '-i', type=int, required=True,
                       help='Instruction index to move')
    parser.add_argument('--cycles', '-c', type=int, required=True,
                       help='Cycles to move: positive for up, negative for down')
    parser.add_argument('--output', '-o', help='Output JSON file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load analysis
    print(f"Loading {args.input}...")
    result = load_analysis_from_json(args.input)
    
    # Get instruction info before move
    if args.block in result.cfg.blocks:
        block = result.cfg.blocks[args.block]
        if 0 <= args.index < len(block.instructions):
            instr = block.instructions[args.index]
            print(f"Instruction to move: [{args.index}] {instr.opcode} {instr.operands}")
    
    # Try to move
    dir_str = "up" if args.cycles > 0 else "down"
    print(f"Attempting to move instruction {args.index} {dir_str} by {abs(args.cycles)} cycles...")
    
    move_result = move_instruction(
        result,
        args.block,
        args.index,
        args.cycles,
        verbose=args.verbose
    )
    
    if move_result.success:
        print(f"Success: {move_result.message}")
        if move_result.waitcnt_updated:
            print("  s_waitcnt instruction was updated")
        
        # Save if output specified
        if args.output:
            save_analysis_to_json(
                result.cfg, result.ddgs,
                result.inter_block_deps, result.waitcnt_deps,
                args.output
            )
    else:
        print(f"Failed: {move_result.message}")
        if move_result.blocked_by:
            print(f"  Blocked by: {move_result.blocked_by}")
    
    return 0 if move_result.success else 1


if __name__ == '__main__':
    exit(main())

