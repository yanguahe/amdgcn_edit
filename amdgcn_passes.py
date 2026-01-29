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
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union, Callable, Generator

from amdgcn_cfg import Instruction, BasicBlock, CFG
from amdgcn_ddg import (
    AnalysisResult,
    DDG,
    InstructionNode,
    PendingMemOp,
    parse_instruction_registers,
    build_ddg,
    generate_all_ddgs,
    compute_inter_block_deps,
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
from amdgcn_verify import build_global_ddg, verify_optimization
from amdgcn_latency import (
    load_hardware_info,
    get_instruction_cycles as get_instruction_cycles_from_config,
    get_instr_cycles,
    is_mfma_instruction,
    get_mfma_dst_registers,
    get_instruction_src_registers,
    check_register_overlap,
    get_required_latency,
    check_move_preserves_latency,
    check_move_side_effects_on_latency,
    calculate_latency_nops_for_move,
    insert_latency_nops,
    LatencyNopsResult,
    InsertLatencyNopsPass,
)

# Enable unbuffered output for real-time printing
print = functools.partial(print, flush=True)


# =============================================================================
# Base Data Classes for Pass Executors
# =============================================================================

@dataclass
class StepResult:
    """Result of a single step execution."""
    step_num: int
    success: bool
    cycles_moved: int = 0
    move_count: int = 0       # Number of individual instruction moves
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleMoveInfo:
    """Information about a single instruction move (for Level-2 debugging)."""
    move_num: int             # 1-indexed move number within the step
    instruction_idx: int      # Current index of the moved instruction
    direction: str            # "up" or "down"
    cycles_this_move: int     # Cycles moved in this single move
    total_cycles_so_far: int  # Cumulative cycles moved in this step
    target_cycles: int        # Total cycles requested for the step


@dataclass
class MoveYieldInfo:
    """Information yielded after each logical move unit in iter_moves()."""
    move_num: int              # 1-indexed logical move number
    move_type: str             # "normal" | "scc_pair" | "recursive"
    swaps_in_this_move: int    # Physical swap count in this logical move
    target_new_idx: int        # Target instruction's new index after move
    cycles_this_move: int      # Cycles moved in this logical unit
    total_cycles_moved: int    # Cumulative cycles moved
    target_cycles: int         # Total cycles requested
    blocked: bool = False      # True if movement is now blocked
    block_reason: str = ""     # Reason for blocking (if blocked)


@dataclass
class DistributeContext:
    """Context for distribute operation with all precomputed parameters."""
    block: BasicBlock
    block_label: str
    target_opcode: str
    target_indices: List[int]        # Initial indices of all target instructions
    all_target_instrs: List['Instruction']  # All target instruction objects
    K: int                           # Number of instructions to distribute
    M: int                           # Total target instructions found
    ideal_cycles: List[int]          # Ideal cycle position for each instruction
    total_cycles: int                # Block total cycles
    branch_boundary: int             # Branch boundary index


@dataclass
class RegisterReplaceContext:
    """Context for register replacement operation."""
    range_start: int
    range_end: int
    segments: List['RegisterSegment']
    alignments: List[int]
    target_opcodes: Set[str]
    register_mapping: Dict[str, str] = field(default_factory=dict)  # old_reg -> new_reg


# =============================================================================
# LDS Synchronization Order Constraint Helper
# =============================================================================

def get_lds_sync_priority(opcode: str, operands: str = "") -> int:
    """
    Get the priority of an LDS synchronization instruction.
    Returns 0 for non-LDS synchronization instructions.
    
    Correct order for LDS inter-thread synchronization pattern:
      ds_write_*            ; Phase 1: All threads issue LDS writes
      s_waitcnt lgkmcnt(sx) ; Phase 2: Wait for LDS operations to complete
      s_barrier             ; Phase 3: Synchronize all threads
      ds_read_*             ; Phase 4: Safely read data written by other threads
    
    Priority: ds_write(1) < s_waitcnt lgkmcnt(2) < s_barrier(3) < ds_read(4)
    
    Args:
        opcode: Instruction opcode
        operands: Instruction operands (used to check if s_waitcnt contains lgkmcnt)
        
    Returns:
        Priority value (1-4), 0 for non-LDS synchronization instructions
    """
    opcode_lower = opcode.lower()
    
    if opcode_lower.startswith('ds_write'):
        return 1
    if opcode_lower == 's_waitcnt':
        # Only s_waitcnt with lgkmcnt participates in LDS synchronization
        if 'lgkmcnt' in operands.lower():
            return 2
    if opcode_lower == 's_barrier':
        return 3
    if opcode_lower.startswith('ds_read'):
        return 4
    
    return 0  # Not an LDS synchronization instruction


def is_exec_save_instruction(opcode: str) -> bool:
    """
    Check if instruction saves exec register (start of exec save/restore pair).
    
    These instructions save exec to a register and modify exec:
    - s_and_saveexec_b64, s_or_saveexec_b64, s_xor_saveexec_b64, etc.
    """
    opcode_lower = opcode.lower()
    return 'saveexec' in opcode_lower


def is_exec_restore_instruction(opcode: str, operands: str) -> bool:
    """
    Check if instruction restores exec register (end of exec save/restore pair).
    
    Pattern: s_or_b64 exec, exec, s[x:y] (restores exec from saved value)
    """
    opcode_lower = opcode.lower()
    if opcode_lower in ('s_or_b64', 's_mov_b64'):
        # Check if operands start with 'exec' as destination
        operands_lower = operands.lower().strip()
        if operands_lower.startswith('exec'):
            return True
    return False


def find_exec_save_restore_ranges(block) -> list:
    """
    Find all exec save/restore pair ranges in a block.
    
    Returns a list of (save_idx, restore_idx) tuples representing ranges
    where instructions should move together.
    
    New rule: For each s_and_saveexec_b64 (or similar saveexec), find the FIRST
    s_or_b64 below it to form a pair. Instructions within the pair (inclusive)
    must move together as a unit.
    """
    ranges = []
    n = len(block.instructions)
    used_restore_indices = set()  # Track already-paired s_or_b64
    
    for idx, instr in enumerate(block.instructions):
        opcode_lower = instr.opcode.lower()
        
        # Check for saveexec instructions (s_and_saveexec_b64, s_or_saveexec_b64, etc.)
        if is_exec_save_instruction(opcode_lower):
            # Find first s_or_b64 below this saveexec
            for restore_idx in range(idx + 1, n):
                restore_instr = block.instructions[restore_idx]
                restore_opcode = restore_instr.opcode.lower()
                if restore_opcode == 's_or_b64':
                    if restore_idx not in used_restore_indices:
                        ranges.append((idx, restore_idx))
                        used_restore_indices.add(restore_idx)
                        break
    
    return ranges


def is_in_exec_save_restore_range(block, idx: int, ranges: list = None) -> tuple:
    """
    Check if an instruction is inside an exec save/restore range.
    
    Returns:
        (is_in_range, range_start, range_end) or (False, -1, -1)
    """
    if ranges is None:
        ranges = find_exec_save_restore_ranges(block)
    
    for save_idx, restore_idx in ranges:
        if save_idx <= idx <= restore_idx:
            return (True, save_idx, restore_idx)
    
    return (False, -1, -1)


def find_unpaired_saveexec_indices(block) -> set:
    """
    Find saveexec instructions that don't have a paired s_or_b64 below.
    
    If a saveexec doesn't have a matching s_or_b64 below it, then instructions
    at or below this saveexec position cannot move up (including the saveexec itself).
    
    Returns:
        Set of indices where unpaired saveexec instructions are located.
    """
    ranges = find_exec_save_restore_ranges(block)
    paired_save_indices = {r[0] for r in ranges}
    
    unpaired = set()
    for idx, instr in enumerate(block.instructions):
        opcode_lower = instr.opcode.lower()
        if is_exec_save_instruction(opcode_lower):
            if idx not in paired_save_indices:
                unpaired.add(idx)
    
    return unpaired


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


def find_scc_writer_for_reader(block: BasicBlock, reader_idx: int, max_distance: int = 20) -> int:
    """
    Find the matching SCC writer above an SCC reader instruction.
    
    For carry chain instructions (s_addc_u32, s_subb_u32), only match with
    their specific pair writers (s_add_u32, s_sub_u32), not any SCC writer.
    
    For other SCC readers (s_cselect_*, s_cbranch_scc*), match with ANY SCC writer.
    
    The SCC pair [writer_idx, reader_idx] should be treated as an atomic unit
    when moving instructions - they cannot be separated.
    
    Args:
        block: The basic block
        reader_idx: Index of the SCC reader instruction
        max_distance: Maximum distance to search backwards
        
    Returns:
        Index of the SCC writer, or -1 if not found within max_distance
    """
    if reader_idx <= 0:
        return -1
    
    instr_reader = block.instructions[reader_idx]
    reader_opcode = instr_reader.opcode.lower()
    
    # Must be an SCC reader
    if not is_scc_reader(reader_opcode):
        return -1
    
    # For carry chain instructions, only match with specific pair writers
    # s_addc_u32 must pair with s_add_u32 or s_add_i32
    # s_subb_u32 must pair with s_sub_u32 or s_sub_i32
    if reader_opcode == 's_addc_u32':
        expected_writers = {'s_add_u32', 's_add_i32'}
    elif reader_opcode == 's_subb_u32':
        expected_writers = {'s_sub_u32', 's_sub_i32'}
    else:
        # For s_cselect_*, s_cbranch_scc*, match any SCC writer
        expected_writers = None
    
    # Search backwards for the matching SCC writer
    for search_idx in range(reader_idx - 1, max(0, reader_idx - max_distance) - 1, -1):
        instr = block.instructions[search_idx]
        opcode = instr.opcode.lower()
        
        if expected_writers is not None:
            # Carry chain: only match specific pair writers
            if opcode in expected_writers:
                return search_idx
            # If we encounter another SCC writer/reader, the pair is broken
            # because the SCC value would be corrupted
            if is_scc_writer(opcode) or is_scc_reader(opcode):
                return -1
        else:
            # Generic SCC reader: match any SCC writer
            if is_scc_writer(opcode):
                return search_idx
    
    # Didn't find the matching writer within max_distance
    return -1


def get_scc_pair_range(block: BasicBlock, idx: int) -> Optional[tuple]:
    """
    Check if instruction at idx is part of an SCC pair and return the pair range.
    
    An SCC pair consists of:
    - A writer instruction that writes SCC (s_add_u32, s_cmp_*, etc.)
    - A reader instruction that reads SCC (s_addc_u32, s_cselect_*, s_cbranch_scc*)
    - Any instructions in between (that don't affect SCC)
    
    When moving instructions, the entire pair range should be treated as atomic.
    
    Args:
        block: The basic block
        idx: Index to check
        
    Returns:
        Tuple (start_idx, end_idx) if idx is part of an SCC pair, None otherwise
    """
    instr = block.instructions[idx]
    opcode = instr.opcode.lower()
    
    # Case 1: idx is an SCC reader - find the writer above
    if is_scc_reader(opcode):
        writer_idx = find_scc_writer_for_reader(block, idx)
        if writer_idx >= 0:
            return (writer_idx, idx)
        return None
    
    # Case 2: idx might be between a writer and reader, or is a writer itself
    # Search forwards to find an SCC reader that uses this writer
    if is_scc_writer(opcode):
        # Search forward for an SCC reader (up to max_distance)
        max_distance = 20
        for search_idx in range(idx + 1, min(len(block.instructions), idx + max_distance)):
            search_instr = block.instructions[search_idx]
            search_opcode = search_instr.opcode.lower()
            
            if is_scc_reader(search_opcode):
                # Found an SCC reader - verify it's paired with this writer
                writer_idx = find_scc_writer_for_reader(block, search_idx)
                if writer_idx == idx:
                    return (idx, search_idx)
                # Found a reader paired with a different writer
                return None
            
            # If we encounter another SCC writer, this one doesn't have a reader
            if is_scc_writer(search_opcode):
                return None
    
    # Case 3: idx might be between a writer and reader
    # Search backwards for a writer, then verify there's a reader
    for writer_search_idx in range(idx - 1, max(0, idx - 20) - 1, -1):
        writer_instr = block.instructions[writer_search_idx]
        writer_opcode = writer_instr.opcode.lower()
        
        if is_scc_writer(writer_opcode):
            # Found a potential writer - search forward for its reader
            for reader_search_idx in range(writer_search_idx + 1, min(len(block.instructions), writer_search_idx + 20)):
                reader_instr = block.instructions[reader_search_idx]
                reader_opcode = reader_instr.opcode.lower()
                
                if is_scc_reader(reader_opcode):
                    # Found a reader - verify it uses this writer
                    actual_writer = find_scc_writer_for_reader(block, reader_search_idx)
                    if actual_writer == writer_search_idx and writer_search_idx <= idx <= reader_search_idx:
                        return (writer_search_idx, reader_search_idx)
                    break
                
                if is_scc_writer(reader_opcode):
                    # Another writer before finding reader - original writer has no reader
                    break
            break
        
        if is_scc_reader(writer_opcode):
            # Found an SCC reader before finding a writer - idx is not in a pair
            break
    
    return None


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
    
    This function delegates to the JSON-based configuration in amdgcn_latency.py.
    
    Cycle costs are loaded from gfx942_hardware_info.json:
    - v_mfma_* instructions: 16 cycles
    - ds_swizzle_*, ds_write_* instructions: 8 cycles
    - v_exp_* instructions: 16 cycles
    - All other instructions: 4 cycles (default)
    
    Args:
        opcode: The instruction opcode
        
    Returns:
        Number of cycles for the instruction
    """
    return get_instruction_cycles_from_config(opcode)


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


def compute_waitcnt_available_regs(block: BasicBlock, waitcnt_idx: int) -> Set[str]:
    """
    Dynamically compute the registers made available by an s_waitcnt instruction.
    
    This function computes available_regs based on the CURRENT block state,
    not relying on pre-computed DDG data which may be stale after instruction moves.
    
    For an s_waitcnt with lgkmcnt(N), the available registers are the destination
    registers of all LGKM operations (ds_read_*, etc.) that precede the waitcnt,
    up to the N-th most recent one. If N=0, all preceding LGKM ops are waited for.
    
    Similarly for vmcnt(N) and VM operations (global_load_*, etc.).
    
    Args:
        block: The basic block
        waitcnt_idx: Index of the s_waitcnt instruction in the block
        
    Returns:
        Set of register names that become available after this s_waitcnt
    """
    if waitcnt_idx >= len(block.instructions):
        return set()
    
    waitcnt_instr = block.instructions[waitcnt_idx]
    if waitcnt_instr.opcode.lower() != 's_waitcnt':
        return set()
    
    vmcnt, lgkmcnt = parse_waitcnt_operands(waitcnt_instr.operands)
    
    available_regs = set()
    
    # Collect LGKM operations before the waitcnt
    if lgkmcnt is not None:
        lgkm_ops = []
        for i in range(waitcnt_idx - 1, -1, -1):
            instr = block.instructions[i]
            if is_lgkm_op(instr.opcode.lower()):
                defs, _ = get_instruction_defs_uses(instr)
                lgkm_ops.append((i, defs))
        
        # lgkmcnt(N) means wait for all but the N most recent LGKM ops
        # So if lgkmcnt=0, all LGKM ops are waited for (all their results are available)
        # If lgkmcnt=1, the most recent LGKM op is not waited for
        ops_to_wait = lgkm_ops[lgkmcnt:] if lgkmcnt < len(lgkm_ops) else []
        for _, defs in ops_to_wait:
            available_regs.update(defs)
    
    # Collect VM operations before the waitcnt
    if vmcnt is not None:
        vm_ops = []
        for i in range(waitcnt_idx - 1, -1, -1):
            instr = block.instructions[i]
            if is_vm_op(instr.opcode.lower()):
                defs, _ = get_instruction_defs_uses(instr)
                vm_ops.append((i, defs))
        
        # vmcnt(N) means wait for all but the N most recent VM ops
        ops_to_wait = vm_ops[vmcnt:] if vmcnt < len(vm_ops) else []
        for _, defs in ops_to_wait:
            available_regs.update(defs)
    
    return available_regs


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
            
            # Check intra-block available registers using dynamic computation
            intra_block_regs = compute_waitcnt_available_regs(block, i)
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
                    # Use dynamic computation for accurate available_regs
                    cross_block_regs = ddg.waitcnt_cross_block_regs.get(before_chain_idx, set()) if ddg else set()
                    intra_block_regs = compute_waitcnt_available_regs(block, before_chain_idx)
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
                # Use dynamic computation for accurate available_regs
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(prev_idx, set()) if ddg else set()
                intra_block_regs = compute_waitcnt_available_regs(block, prev_idx)
                all_avail_regs = cross_block_regs | intra_block_regs
                
                if all_avail_regs & uses:
                    dependent_waitcnts.append(prev_idx)
    
    return dependent_waitcnts


# =============================================================================
# MoveExecutor - Modular interface for instruction movement with callbacks
# =============================================================================

class MoveExecutor:
    """
    Executor for instruction movement that supports single-instruction move tracking.
    Enables Level-2 debugging by exposing each individual instruction move.
    
    This class wraps MoveInstructionPass and provides:
    - Per-move callbacks for debugging
    - Incremental movement with progress tracking
    - Clean interface for external tools
    """
    
    def __init__(
        self,
        result: AnalysisResult,
        block_label: str,
        instr_index: int,
        frozen_boundary: int = 0,
        protected_instructions: Optional[List['Instruction']] = None,
        auto_insert_nops: bool = True,
        verbose: bool = False,
        on_swap_callback: Optional[Callable[['BasicBlock', int, int, 'Instruction'], None]] = None
    ):
        """
        Initialize the move executor.
        
        Args:
            result: The AnalysisResult to modify
            block_label: Label of the block (e.g., ".LBB0_0")
            instr_index: Index of instruction to move
            frozen_boundary: Instructions at idx < frozen_boundary cannot be moved
            protected_instructions: List of instruction objects that should never be moved
            auto_insert_nops: If True, automatically insert s_nop for latency constraints
            verbose: Print detailed information during execution
            on_swap_callback: Optional callback called after each swap (block, idx1, idx2, moved_instr)
        """
        self.result = result
        self.block_label = block_label
        self.instr_index = instr_index
        self.frozen_boundary = frozen_boundary
        self.protected_instructions = protected_instructions or []
        self.auto_insert_nops = auto_insert_nops
        self.verbose = verbose
        self.on_swap_callback = on_swap_callback
        
        # Validate block exists
        if block_label not in result.cfg.blocks:
            raise ValueError(f"Block '{block_label}' not found")
        
        self.block = result.cfg.blocks[block_label]
        
        # Validate instruction index
        if instr_index < 0 or instr_index >= len(self.block.instructions):
            raise ValueError(f"Invalid instruction index {instr_index}")
        
        # Store target instruction reference
        self.target_instr = self.block.instructions[instr_index]
        
        # Tracking state
        self._total_cycles_moved = 0
        self._move_count = 0
        self._stop_reason = ""
    
    @property
    def total_cycles_moved(self) -> int:
        """Get total cycles actually moved."""
        return self._total_cycles_moved
    
    @property
    def move_count(self) -> int:
        """Get number of individual instruction moves."""
        return self._move_count
    
    @property
    def stop_reason(self) -> str:
        """Get reason why movement stopped."""
        return self._stop_reason
    
    def _find_current_index(self) -> int:
        """Find the current index of the target instruction."""
        for idx, instr in enumerate(self.block.instructions):
            if instr is self.target_instr:
                return idx
        return -1
    
    def move_single_step(
        self,
        direction: str,
        max_cycles: int = 4
    ) -> Tuple[int, bool]:
        """
        Move instruction by a single step (one instruction worth of cycles).
        
        Args:
            direction: "up" or "down"
            max_cycles: Maximum cycles to move in this step
            
        Returns:
            (cycles_moved, can_continue) - cycles moved in this step, and whether more moves possible
        """
        cycles = max_cycles if direction == "up" else -max_cycles
        
        current_idx = self._find_current_index()
        if current_idx < 0:
            return 0, False
        
        move_pass = MoveInstructionPass(
            self.block_label,
            current_idx,
            cycles,
            verbose=self.verbose,
            frozen_boundary=self.frozen_boundary,
            protected_instructions=self.protected_instructions,
            auto_insert_nops=self.auto_insert_nops,
            on_swap_callback=self.on_swap_callback
        )
        
        try:
            move_pass.run(self.result)
        except Exception:
            # Move failed (verification error)
            self._stop_reason = "verification_failed"
            return 0, False
        
        cycles_moved = move_pass.total_cycles_moved
        
        if cycles_moved > 0:
            self._move_count += 1
            self._total_cycles_moved += cycles_moved
        
        # Check if we can continue - continue as long as we made progress
        # The previous logic was too strict: it required stop_reason == "reached target"
        # But MoveInstructionPass may set blocking reasons even when it moved the requested cycles
        can_continue = cycles_moved > 0
        
        if cycles_moved == 0:
            self._stop_reason = move_pass.stop_reason
        
        return cycles_moved, can_continue
    
    def move_by_cycles(
        self,
        cycles: int,
        on_single_move: Optional[callable] = None
    ) -> int:
        """
        Move instruction by specified cycles, with optional per-move callback.
        
        Args:
            cycles: Total cycles to move (positive=up, negative=down)
            on_single_move: Called after each single-instruction move.
                           Signature: (move_info: SingleMoveInfo) -> bool
                           Return False to stop moving.
                           
        Returns:
            Total cycles actually moved
        """
        if cycles == 0:
            return 0
        
        direction = "up" if cycles > 0 else "down"
        target_cycles = abs(cycles)
        
        self._total_cycles_moved = 0
        self._move_count = 0
        self._stop_reason = ""
        
        while self._total_cycles_moved < target_cycles:
            remaining = target_cycles - self._total_cycles_moved
            step_cycles = min(4, remaining)  # Move at most 4 cycles at a time
            
            cycles_moved, can_continue = self.move_single_step(direction, step_cycles)
            
            if cycles_moved > 0 and on_single_move is not None:
                move_info = SingleMoveInfo(
                    move_num=self._move_count,
                    instruction_idx=self._find_current_index(),
                    direction=direction,
                    cycles_this_move=cycles_moved,
                    total_cycles_so_far=self._total_cycles_moved,
                    target_cycles=target_cycles
                )
                
                # Call the callback, stop if it returns False
                if not on_single_move(move_info):
                    self._stop_reason = "callback_stopped"
                    break
            
            if not can_continue:
                break
        
        if self._total_cycles_moved >= target_cycles:
            self._stop_reason = "reached_target"
        
        return self._total_cycles_moved


class MoveInstructionPass(Pass):
    """
    Pass that moves a single instruction up or down by a specified number of cycles.
    
    Moving UP n cycles:
    1. For each step, check if target AA can swap with instruction above (AAB1)
    2. If no conflict: swap AA and AAB1
    3. If RAW dependency exists (AA reads what AAB1 writes), recursively move AAB1 up first
    4. If WAR/WAW conflict: stop movement
    5. Continue until target cycle position reached or movement blocked
    
    Moving DOWN n cycles:
    1. For each step, check if target AA can swap with instruction below (AAF1)
    2. Only move if no register conflicts exist
    3. Stop immediately if any conflict (no recursive movement)
    
    Constraints respected:
    - RAW/WAR/WAW dependencies (including SCC register)
    - s_waitcnt counter adjustments
    - exec save/restore ranges (treated as atomic units)
    - Latency requirements
    - Branch boundaries (cannot cross s_branch*, s_cbranch*)
    - Saveexec boundaries (cannot cross saveexec instructions)
    
    NOT considered:
    - s_barrier (no longer acts as barrier)
    
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
        protected_instructions: Optional[List['Instruction']] = None,
        auto_insert_nops: bool = True,
        on_swap_callback: Optional[Callable] = None
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
            auto_insert_nops: If True (default), automatically insert s_nop instructions
                             when a move would violate MFMA latency constraints.
                             If False, such moves are blocked.
            on_swap_callback: Optional callback function called after each swap.
                             Signature: (block, from_idx, to_idx, instr) -> None
                             Can be used for debugging/verification.
        """
        self.block_label = block_label
        self.instr_index = instr_index
        self.cycles = cycles
        self.verbose = verbose
        self.frozen_boundary = frozen_boundary
        self.protected_instructions = protected_instructions or []
        self.auto_insert_nops = auto_insert_nops
        self.on_swap_callback = on_swap_callback
        self._last_result: Optional[MoveResult] = None
        self._total_cycles_moved: int = 0
        # Detailed tracking of why movement stopped (use sets of instruction object IDs to track unique instructions)
        self._blocked_by_frozen: Set[int] = set()  # Set of id(instruction) blocked by frozen boundary
        self._blocked_by_dependencies: Set[int] = set()  # Set of id(instruction) blocked by dependencies
        self._blocked_by_branch: Set[int] = set()  # Set of id(instruction) blocked by branch boundary
        self._blocked_by_saveexec: Set[int] = set()  # Set of id(instruction) blocked by saveexec
        self._blocked_by_latency: Set[int] = set()  # Set of id(instruction) blocked by latency constraints
        self._no_candidates: bool = False  # No candidate instructions available
        # Track inserted nops for reporting
        self._inserted_nops: List[Tuple[int, int]] = []  # [(position, count), ...]
    
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
        if len(self._blocked_by_saveexec) > 0:
            reasons.append(f"saveexec({len(self._blocked_by_saveexec)})")
        if len(self._blocked_by_latency) > 0:
            reasons.append(f"latency({len(self._blocked_by_latency)})")
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
    
    def iter_moves(self, result: AnalysisResult) -> Generator[MoveYieldInfo, None, None]:
        """
        Generator that yields after each logical move unit.
        
        Unlike run() which executes all moves at once, this method yields control
        after each logical move unit, allowing the caller to inspect/test intermediate
        states.
        
        Logical move units:
        - normal: Single instruction swap
        - scc_pair: Skip entire SCC pair as atomic unit
        - recursive: Recursive move of blocking instruction (UP direction only)
        
        Args:
            result: The AnalysisResult to modify
            
        Yields:
            MoveYieldInfo with details of each logical move
            
        Raises:
            ValueError: If block or instruction index is invalid
            
        Note:
            This method does NOT perform verification after each move.
            Caller is responsible for verification if needed.
        """
        # Validate block exists
        if self.block_label not in result.cfg.blocks:
            return
        
        block = result.cfg.blocks[self.block_label]
        ddg = result.ddgs.get(self.block_label)
        
        # Validate instruction index
        if self.instr_index < 0 or self.instr_index >= len(block.instructions):
            return
        
        if self.cycles == 0:
            return
        
        self._total_cycles_moved = 0
        
        # Get the target instruction reference (for tracking after moves)
        target_instr = block.instructions[self.instr_index]
        
        cycles_to_move = abs(self.cycles)
        
        if self.cycles > 0:
            # Move UP by n cycles
            yield from self._iter_move_up(block, ddg, result, target_instr, cycles_to_move)
        else:
            # Move DOWN by |n| cycles
            yield from self._iter_move_down(block, ddg, result, target_instr, cycles_to_move)
        
        if self.verbose:
            print(f"  Total cycles moved: {self._total_cycles_moved} / {cycles_to_move} requested")
    
    # =========================================================================
    # Instruction movement methods
    # =========================================================================
    
    def _is_boundary_instruction(self, opcode: str) -> bool:
        """
        Check if instruction is a movement boundary.
        Instructions cannot be moved across these boundaries.
        """
        op_lower = opcode.lower()
        # Branch instructions
        if op_lower.startswith('s_branch') or op_lower.startswith('s_cbranch'):
            return True
        # Saveexec instructions
        if 'saveexec' in op_lower or 'wrexec' in op_lower:
            return True
        return False
    
    def _can_skip_scc_pair(
        self,
        block: BasicBlock,
        target_instr: Instruction,
        pair_start: int,
        pair_end: int
    ) -> bool:
        """
        Check if target instruction can skip over an SCC pair without conflicts.
        
        An SCC pair [pair_start, pair_end] consists of:
        - pair_start: SCC writer (e.g., s_add_u32, s_cmp_*)
        - pair_end: SCC reader (e.g., s_addc_u32, s_cselect_*, s_cbranch_scc*)
        - Any instructions in between (that don't affect SCC)
        
        The target can skip the pair if:
        1. Target doesn't read SCC (otherwise it would read wrong SCC value)
        2. No data dependencies exist between target and any instruction in the pair
        3. No protected instructions in the pair
        
        Args:
            block: The basic block
            target_instr: The instruction trying to skip the pair
            pair_start: Start index of the SCC pair (writer)
            pair_end: End index of the SCC pair (reader)
            
        Returns:
            True if target can safely skip the entire SCC pair
        """
        target_defs, target_uses = get_instruction_defs_uses(target_instr)
        target_opcode = target_instr.opcode.lower()
        
        # Rule 1: Target must not read SCC (would read wrong value after skip)
        if is_scc_reader(target_opcode):
            return False
        
        # Check each instruction in the SCC pair range
        for idx in range(pair_start, pair_end + 1):
            pair_instr = block.instructions[idx]
            
            # Check if instruction is protected
            if pair_instr in self.protected_instructions:
                return False
            
            # Check for boundary/branch instructions
            pair_opcode = pair_instr.opcode.lower()
            if self._is_boundary_instruction(pair_opcode):
                return False
            if pair_instr.is_branch or pair_instr.is_terminator:
                return False
            
            pair_defs, pair_uses = get_instruction_defs_uses(pair_instr)
            
            # Rule 2: No RAW dependency (target reads what pair writes, excluding SCC)
            raw_conflicts = pair_defs & target_uses
            if raw_conflicts:
                # SCC conflict is OK if target doesn't read SCC (already checked above)
                if raw_conflicts != {'scc'}:
                    return False
            
            # Rule 3: No WAR dependency (target writes what pair reads, excluding SCC)
            war_conflicts = target_defs & pair_uses
            if war_conflicts:
                # SCC conflict is OK if target only writes SCC
                if not (war_conflicts == {'scc'} and is_scc_only_writer(target_opcode)):
                    return False
            
            # Rule 4: WAW conflicts are generally OK (can be traversed)
            # But we should be careful with non-SCC WAW
            waw_conflicts = target_defs & pair_defs
            if waw_conflicts and waw_conflicts != {'scc'}:
                return False
        
        return True
    
    def _can_skip_scc_pair_down(
        self,
        block: BasicBlock,
        target_instr: Instruction,
        target_idx: int,
        pair_start: int,
        pair_end: int
    ) -> bool:
        """
        Check if target instruction can skip over an SCC pair when moving DOWN.
        
        Similar to _can_skip_scc_pair but checks for downward movement constraints.
        
        Args:
            block: The basic block
            target_instr: The instruction trying to skip the pair
            target_idx: Current index of the target instruction
            pair_start: Start index of the SCC pair (writer)
            pair_end: End index of the SCC pair (reader)
            
        Returns:
            True if target can safely skip the entire SCC pair
        """
        target_defs, target_uses = get_instruction_defs_uses(target_instr)
        target_opcode = target_instr.opcode.lower()
        
        # Rule 1: Target must not read SCC (would read wrong value after skip)
        if is_scc_reader(target_opcode):
            return False
        
        # Check each instruction in the SCC pair range
        for idx in range(pair_start, pair_end + 1):
            pair_instr = block.instructions[idx]
            
            # Check if instruction is protected
            if pair_instr in self.protected_instructions:
                return False
            
            # Check for boundary/branch instructions
            pair_opcode = pair_instr.opcode.lower()
            if self._is_boundary_instruction(pair_opcode):
                return False
            if pair_instr.is_branch or pair_instr.is_terminator:
                return False
            
            pair_defs, pair_uses = get_instruction_defs_uses(pair_instr)
            
            # Rule 2: For downward movement, check reversed dependencies
            # RAW: pair reads what target writes -> stop
            raw_conflicts = target_defs & pair_uses
            if raw_conflicts:
                if not (raw_conflicts == {'scc'} and not is_scc_reader(pair_opcode)):
                    return False
            
            # Rule 3: WAR: pair writes what target reads -> stop
            war_conflicts = pair_defs & target_uses
            if war_conflicts:
                if not (war_conflicts == {'scc'} and is_scc_only_writer(pair_opcode)):
                    return False
            
            # Rule 4: WAW conflicts - generally OK but be careful
            waw_conflicts = target_defs & pair_defs
            if waw_conflicts and waw_conflicts != {'scc'}:
                # For safety, block on non-SCC WAW
                return False
        
        return True
    
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
        
        Simplified logic: only check conflicts with each instruction we pass.
        No recursive movement for downward direction.
        
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
        
        # Check all instructions we would pass
        for check_idx in range(from_idx + 1, to_idx + 1):
            # Skip instructions in protected_indices (they move together with the instruction)
            if check_idx in protected_indices:
                continue
            
            instr_b = block.instructions[check_idx]
            defs_b, uses_b = get_instruction_defs_uses(instr_b)
            opcode_b = instr_b.opcode.lower()
            
            # CRITICAL: Check if B is the SCC writer of a pair (e.g., s_add_u32)
            # and its reader is below. If A moves down past B, it would enter the pair.
            if is_scc_pair_writer(opcode_b):
                # Check if there's a reader below that pairs with this writer
                pair_end_idx = -1
                for search_idx in range(check_idx + 1, min(len(block.instructions), check_idx + 20)):
                    search_opcode = block.instructions[search_idx].opcode.lower()
                    if is_scc_pair_reader(search_opcode):
                        # Verify this reader pairs with B
                        found_writer = find_scc_pair_start(block, search_idx)
                        if found_writer == check_idx:
                            pair_end_idx = search_idx
                            break
                        break
                    if is_scc_writer(search_opcode):
                        break
                
                if pair_end_idx > check_idx:
                    # B is part of a valid SCC pair [check_idx, pair_end_idx]
                    # A cannot move past B because it would enter the pair
                    return False
            
            # Boundary instruction check: cannot cross branch/saveexec
            if self._is_boundary_instruction(opcode_b):
                self._blocked_by_saveexec.add(id(instr_b))
                return False
            
            # Cannot cross branch/terminator
            if instr_b.is_branch or instr_b.is_terminator:
                self._blocked_by_branch.add(id(instr_b))
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
            waw_conflicts = defs_a & defs_b
            if waw_conflicts:
                if waw_conflicts == {'scc'}:
                    pass  # WAW-SCC is always traversable
                else:
                    pass  # WAW traversable rule
            
            # Check s_waitcnt constraints
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                # vm_op/lgkm_op (A) moves DOWN past s_waitcnt (B): B's counter DECREASES by 1
                if is_vm_op(opcode_a) and vmcnt is not None and vmcnt <= 0:
                    return False
                if is_lgkm_op(opcode_a) and lgkmcnt is not None and lgkmcnt <= 0:
                    return False
            
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                # s_waitcnt (A) moves DOWN past vm_op/lgkm_op (B): A's counter INCREASES by 1
                if is_vm_op(opcode_b) and vmcnt is not None and vmcnt >= 63:
                    return False
                if is_lgkm_op(opcode_b) and lgkmcnt is not None and lgkmcnt >= 15:
                    return False
                
                # Check s_waitcnt AVAIL dependency
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(from_idx, set()) if ddg else set()
                intra_block_regs = compute_waitcnt_available_regs(block, from_idx)
                all_avail_regs = cross_block_regs | intra_block_regs
                if all_avail_regs & uses_b:
                    return False
        
        # Check MFMA latency constraints
        if not check_move_preserves_latency(block, from_idx, to_idx):
            if not self.auto_insert_nops:
                self._blocked_by_latency.add(id(instr_a))
                return False
        
        if not check_move_side_effects_on_latency(block, from_idx, to_idx):
            if not self.auto_insert_nops:
                self._blocked_by_latency.add(id(instr_a))
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
        
        Simplified logic: only check conflicts with each instruction we pass.
        
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
        
        # Check all instructions we would pass
        for check_idx in range(from_idx - 1, to_idx - 1, -1):
            # Skip instructions in protected_indices (they move together with the instruction)
            if check_idx in protected_indices:
                continue
            
            instr_b = block.instructions[check_idx]
            defs_b, uses_b = get_instruction_defs_uses(instr_b)
            opcode_b = instr_b.opcode.lower()
            
            # CRITICAL: Check if B is the SCC reader of a pair (e.g., s_addc_u32)
            # and A is not part of the same pair. If so, A cannot move past B
            # because it would enter the SCC pair and corrupt the dependency.
            if is_scc_pair_reader(opcode_b):
                # Find the SCC writer for this reader
                pair_start = find_scc_pair_start(block, check_idx)
                if pair_start >= 0:
                    # B is part of a valid SCC pair [pair_start, check_idx]
                    # A (at from_idx) is trying to move up past B
                    # After the swap, A would be at check_idx and B at check_idx+1
                    # This would put A inside the pair if pair_start < check_idx
                    # We must block this to preserve the SCC pair integrity
                    return False
            
            # Boundary instruction check: cannot cross branch/saveexec
            if self._is_boundary_instruction(opcode_b):
                self._blocked_by_saveexec.add(id(instr_b))
                return False
            
            # Cannot cross branch/terminator
            if instr_b.is_branch or instr_b.is_terminator:
                self._blocked_by_branch.add(id(instr_b))
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
                if is_vm_op(opcode_a) and vmcnt is not None and vmcnt >= 63:
                    return False
                if is_lgkm_op(opcode_a) and lgkmcnt is not None and lgkmcnt >= 15:
                    return False
            
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                # s_waitcnt (A) moves UP past vm_op/lgkm_op (B): A's counter DECREASES by 1
                if is_vm_op(opcode_b) and vmcnt is not None and vmcnt <= 0:
                    return False
                if is_lgkm_op(opcode_b) and lgkmcnt is not None and lgkmcnt <= 0:
                    return False
            
            # Check s_waitcnt AVAIL dependency
            if opcode_b == 's_waitcnt':
                cross_block_regs = ddg.waitcnt_cross_block_regs.get(check_idx, set()) if ddg else set()
                intra_block_regs = compute_waitcnt_available_regs(block, check_idx)
                all_avail_regs = cross_block_regs | intra_block_regs
                if all_avail_regs & uses_a:
                    return False
        
        # Check MFMA latency constraints
        if not check_move_preserves_latency(block, from_idx, to_idx):
            if not self.auto_insert_nops:
                self._blocked_by_latency.add(id(instr_a))
                return False
        
        if not check_move_side_effects_on_latency(block, from_idx, to_idx):
            if not self.auto_insert_nops:
                self._blocked_by_latency.add(id(instr_a))
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
        # === New design: Pre-move s_waitcnt adjustment for memory ops moving down ===
        # When A (vm_op/lgkm_op) moves down, find the s_waitcnt sA that waits for A,
        # count how many same-type memory ops A will pass through (k),
        # then decrement sA's counter by k.
        instr_a = block.instructions[from_idx]
        opcode_a_initial = instr_a.opcode.lower()
        is_vm = is_vm_op(opcode_a_initial)
        is_lgkm = is_lgkm_op(opcode_a_initial)
        
        if is_vm or is_lgkm:
            # Find sA: the first s_waitcnt below A that waits for A
            sA_idx = -1
            for i in range(from_idx + 1, len(block.instructions)):
                if block.instructions[i].opcode.lower() == 's_waitcnt':
                    vmcnt_i, lgkmcnt_i = parse_waitcnt_operands(block.instructions[i].operands)
                    # Count memory ops between A and this s_waitcnt
                    vm_count = 0
                    lgkm_count = 0
                    for j in range(from_idx + 1, i):
                        op_j = block.instructions[j].opcode.lower()
                        if is_vm_op(op_j):
                            vm_count += 1
                        if is_lgkm_op(op_j):
                            lgkm_count += 1
                    
                    # A is the (count + 1)-th op. If counter <= count, A is waited for.
                    waits_for_a = False
                    if is_vm and vmcnt_i is not None and vmcnt_i <= vm_count:
                        waits_for_a = True
                    if is_lgkm and lgkmcnt_i is not None and lgkmcnt_i <= lgkm_count:
                        waits_for_a = True
                    
                    if waits_for_a:
                        sA_idx = i
                        break
            
            # If we found sA, count how many same-type memory ops A will pass through
            # from from_idx to min(to_idx, sA_idx - 1)
            if sA_idx > from_idx:
                # Count same-type memory ops between from_idx and to_idx (exclusive of A)
                # that are also before sA
                end_idx = min(to_idx, sA_idx - 1)
                k_vm = 0
                k_lgkm = 0
                for j in range(from_idx + 1, end_idx + 1):
                    op_j = block.instructions[j].opcode.lower()
                    if is_vm and is_vm_op(op_j):
                        k_vm += 1
                    if is_lgkm and is_lgkm_op(op_j):
                        k_lgkm += 1
                
                # Update sA's counter: N - k
                if k_vm > 0 or k_lgkm > 0:
                    vmcnt_sA, lgkmcnt_sA = parse_waitcnt_operands(block.instructions[sA_idx].operands)
                    if is_vm and k_vm > 0 and vmcnt_sA is not None:
                        new_vmcnt = vmcnt_sA - k_vm
                        assert new_vmcnt >= 0, f"s_waitcnt at idx {sA_idx} would have vmcnt={new_vmcnt} < 0 after adjustment"
                        update_waitcnt_instruction(block.instructions[sA_idx], vmcnt_delta=-k_vm)
                        sync_instruction_to_raw_lines(block, block.instructions[sA_idx])
                    if is_lgkm and k_lgkm > 0 and lgkmcnt_sA is not None:
                        new_lgkmcnt = lgkmcnt_sA - k_lgkm
                        assert new_lgkmcnt >= 0, f"s_waitcnt at idx {sA_idx} would have lgkmcnt={new_lgkmcnt} < 0 after adjustment"
                        update_waitcnt_instruction(block.instructions[sA_idx], lgkmcnt_delta=-k_lgkm)
                        sync_instruction_to_raw_lines(block, block.instructions[sA_idx])
        
        # === Now do the actual swaps ===
        current_idx = from_idx
        while current_idx < to_idx:
            instr_a = block.instructions[current_idx]
            instr_b = block.instructions[current_idx + 1]
            
            opcode_a = instr_a.opcode.lower()
            opcode_b = instr_b.opcode.lower()
            
            
            # Update s_waitcnt counts
            # Case 1: A (vm_op/lgkm_op) moves DOWN past B (s_waitcnt)
            # A goes from BEFORE s_waitcnt to AFTER -> DECREASE count
            # SAFETY CHECK: Do not decrease if counter would become negative
            if opcode_b == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_b.operands)
                if is_vm_op(opcode_a) and vmcnt is not None and vmcnt > 0:
                    update_waitcnt_instruction(instr_b, vmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_b)
                if is_lgkm_op(opcode_a) and lgkmcnt is not None and lgkmcnt > 0:
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
        # === New design: Pre-move s_waitcnt adjustment for memory ops moving up ===
        # When A (vm_op/lgkm_op) moves up, find the s_waitcnt sA that waits for A,
        # then for all s_waitcnt instructions between A and sA (excluding sA),
        # decrement their counters by 1.
        instr_a = block.instructions[from_idx]
        opcode_a_initial = instr_a.opcode.lower()
        is_vm = is_vm_op(opcode_a_initial)
        is_lgkm = is_lgkm_op(opcode_a_initial)
        
        if is_vm or is_lgkm:
            # Find sA: the first s_waitcnt below A that waits for A
            sA_idx = -1
            for i in range(from_idx + 1, len(block.instructions)):
                if block.instructions[i].opcode.lower() == 's_waitcnt':
                    vmcnt_i, lgkmcnt_i = parse_waitcnt_operands(block.instructions[i].operands)
                    # Check if this s_waitcnt waits for A (not just skips it)
                    # Count memory ops between A and this s_waitcnt
                    vm_count = 0
                    lgkm_count = 0
                    for j in range(from_idx + 1, i):
                        op_j = block.instructions[j].opcode.lower()
                        if is_vm_op(op_j):
                            vm_count += 1
                        if is_lgkm_op(op_j):
                            lgkm_count += 1
                    
                    # A is the (count + 1)-th op. If counter < count + 1, A is waited for.
                    # For vm_op: vmcnt <= vm_count means A is waited for
                    # For lgkm_op: lgkmcnt <= lgkm_count means A is waited for
                    waits_for_a = False
                    if is_vm and vmcnt_i is not None and vmcnt_i <= vm_count:
                        waits_for_a = True
                    if is_lgkm and lgkmcnt_i is not None and lgkmcnt_i <= lgkm_count:
                        waits_for_a = True
                    
                    if waits_for_a:
                        sA_idx = i
                        break
            
            # Now find all s_waitcnt between A and sA (excluding sA), decrement their counters
            if sA_idx > from_idx + 1:
                for i in range(from_idx + 1, sA_idx):
                    if block.instructions[i].opcode.lower() == 's_waitcnt':
                        vmcnt_i, lgkmcnt_i = parse_waitcnt_operands(block.instructions[i].operands)
                        # Assert counter > 0 and decrement
                        if is_vm and vmcnt_i is not None:
                            assert vmcnt_i > 0, f"s_waitcnt at idx {i} has vmcnt=0, cannot decrement"
                            update_waitcnt_instruction(block.instructions[i], vmcnt_delta=-1)
                            sync_instruction_to_raw_lines(block, block.instructions[i])
                        if is_lgkm and lgkmcnt_i is not None:
                            assert lgkmcnt_i > 0, f"s_waitcnt at idx {i} has lgkmcnt=0, cannot decrement"
                            update_waitcnt_instruction(block.instructions[i], lgkmcnt_delta=-1)
                            sync_instruction_to_raw_lines(block, block.instructions[i])
        
        # === Now do the actual swaps ===
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
            # SAFETY CHECK: Do not decrease if counter would become negative
            if opcode_a == 's_waitcnt':
                vmcnt, lgkmcnt = parse_waitcnt_operands(instr_a.operands)
                if is_vm_op(opcode_b) and vmcnt is not None and vmcnt > 0:
                    update_waitcnt_instruction(instr_a, vmcnt_delta=-1)
                    sync_instruction_to_raw_lines(block, instr_a)
                if is_lgkm_op(opcode_b) and lgkmcnt is not None and lgkmcnt > 0:
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
        
        New algorithm:
        1. Record target's read/write register sets (rreg_set, wreg_set)
        2. For each upward move, check relationship with instruction above (AAB1):
           - If no conflict: swap AA and AAB1
           - If RAW dependency (rreg_set intersects AAB1's wreg): 
             recursively move AAB1 up first
           - If WAR/WAW conflict: stop
        3. Repeat until target cycle position reached or blocked
        
        Uses predecessor's cycles for accumulation.
        """
        cycles_to_move = self.cycles
        any_moved = False
        
        # Reset blocking sets
        self._blocked_by_frozen = set()
        self._blocked_by_dependencies = set()
        self._blocked_by_branch = set()
        self._blocked_by_saveexec = set()
        self._blocked_by_latency = set()
        self._no_candidates = False
        
        while self._total_cycles_moved < cycles_to_move:
            # Get current position of target
            target_idx = self._find_instruction_index(block, target_instr)
            if target_idx < 0:
                break
            
            # Cannot move if at the beginning of the block
            if target_idx <= 0:
                self._no_candidates = True
                break
            
            # Cannot move into frozen region
            if target_idx <= self.frozen_boundary:
                self._blocked_by_frozen.add(id(target_instr))
                break
            
            # Get registers for target instruction (AA)
            aa_defs, aa_uses = get_instruction_defs_uses(target_instr)
            aa_rreg_set = aa_uses  # Read registers
            aa_wreg_set = aa_defs  # Write registers
            
            # Get instruction above (AAB1)
            aab1_idx = target_idx - 1
            aab1_instr = block.instructions[aab1_idx]
            aab1_opcode = aab1_instr.opcode.lower()
            
            # Check for boundary instructions
            if self._is_boundary_instruction(aab1_opcode):
                self._blocked_by_saveexec.add(id(aab1_instr))
                break
            
            if aab1_instr.is_branch or aab1_instr.is_terminator:
                self._blocked_by_branch.add(id(aab1_instr))
                break
            
            # Skip protected instructions
            if aab1_instr in self.protected_instructions:
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # Check if AAB1 is part of an SCC pair - if so, treat the entire pair as atomic
            scc_pair_range = get_scc_pair_range(block, aab1_idx)
            if scc_pair_range is not None:
                pair_start, pair_end = scc_pair_range
                # Target is trying to move past an SCC pair - need to skip the entire pair
                # Check if target instruction can skip the entire pair without conflicts
                if not self._can_skip_scc_pair(block, target_instr, pair_start, pair_end):
                    self._blocked_by_dependencies.add(id(aab1_instr))
                    if self.verbose:
                        print(f"    Blocked by SCC pair [{pair_start}]-[{pair_end}]")
                    break
                
                # Move target past the entire SCC pair
                pair_cycles = sum(get_instr_cycles(block.instructions[i]) for i in range(pair_start, pair_end + 1))
                
                # Overshoot prevention
                error_if_stop = cycles_to_move - self._total_cycles_moved
                error_if_move = abs(cycles_to_move - (self._total_cycles_moved + pair_cycles))
                
                if error_if_move > error_if_stop:
                    if self.verbose:
                        print(f"    Overshoot prevention: stop at error={error_if_stop}, skip SCC pair would give error={error_if_move}")
                    break
                
                # Move target past the entire SCC pair
                for i in range(pair_end - pair_start + 1):
                    current_target_idx = self._find_instruction_index(block, target_instr)
                    if current_target_idx <= pair_start:
                        break
                    self._move_single_instruction_up_impl(block, ddg, current_target_idx, current_target_idx - 1)
                
                self._total_cycles_moved += pair_cycles
                any_moved = True
                
                if self.verbose:
                    print(f"    Skipped SCC pair [{pair_start}]-[{pair_end}] (+{pair_cycles} cycles)")
                continue
            
            # Get registers for AAB1
            aab1_defs, aab1_uses = get_instruction_defs_uses(aab1_instr)
            
            # Check conflicts
            # WAR: AA writes what AAB1 reads -> stop
            war_conflicts = aa_wreg_set & aab1_uses
            if war_conflicts and not (war_conflicts == {'scc'} and is_scc_only_writer(target_instr.opcode.lower())):
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # WAW: AA writes what AAB1 writes -> stop  
            waw_conflicts = aa_wreg_set & aab1_defs
            if waw_conflicts and not (waw_conflicts == {'scc'}):
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # RAW: AA reads what AAB1 writes -> need to recursively move AAB1 first
            raw_conflicts = aa_rreg_set & aab1_defs
            if raw_conflicts:
                if raw_conflicts == {'scc'} and not is_scc_reader(target_instr.opcode.lower()):
                    pass  # SCC-only conflict, AA doesn't read SCC
                else:
                    # Try to recursively move AAB1 up first
                    moved_aab1 = self._try_move_instruction_up_recursive(
                        block, ddg, result, aab1_instr, depth=0, max_depth=50
                    )
                    if not moved_aab1:
                        self._blocked_by_dependencies.add(id(aab1_instr))
                        break
                    # After moving AAB1, continue the loop to try moving target again
                    continue
            
            # Check s_waitcnt constraints
            if not self._can_move_single_instruction_up(block, ddg, target_idx, target_idx - 1, set()):
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # No conflict - swap AA and AAB1
            # Use predecessor's cycles (AAB1's cycles)
            predecessor_cycles = get_instr_cycles(aab1_instr)
            
            # Overshoot prevention: compare error_if_stop vs error_if_move
            error_if_stop = cycles_to_move - self._total_cycles_moved
            error_if_move = abs(cycles_to_move - (self._total_cycles_moved + predecessor_cycles))
            
            if error_if_move > error_if_stop:
                # Making this move would increase the error - stop here
                if self.verbose:
                    print(f"    Overshoot prevention: stop at error={error_if_stop}, move would give error={error_if_move}")
                break
            
            # Perform the swap with s_waitcnt adjustment
            self._move_single_instruction_up_impl(block, ddg, target_idx, target_idx - 1)
            self._total_cycles_moved += predecessor_cycles
            any_moved = True
            
            if self.verbose:
                print(f"    Moved target [{target_idx}] {target_instr.opcode} up past [{aab1_idx}] {aab1_instr.opcode} (+{predecessor_cycles} cycles)")
        
        return any_moved
    
    def _iter_move_up(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        result: AnalysisResult,
        target_instr: Instruction,
        cycles_to_move: int
    ) -> Generator[MoveYieldInfo, None, None]:
        """
        Generator version of _move_up_by_cycles that yields after each logical move unit.
        
        Logical move units:
        - normal: Single instruction swap
        - scc_pair: Skip entire SCC pair as atomic unit
        - recursive: Recursive move of blocking instruction (yields once after all recursive moves)
        
        Args:
            block: The basic block
            ddg: The DDG for the block
            result: The analysis result
            target_instr: The instruction to move
            cycles_to_move: Total cycles to move (positive value)
            
        Yields:
            MoveYieldInfo with details of each logical move
        """
        move_num = 0
        
        # Reset blocking sets
        self._blocked_by_frozen = set()
        self._blocked_by_dependencies = set()
        self._blocked_by_branch = set()
        self._blocked_by_saveexec = set()
        self._blocked_by_latency = set()
        self._no_candidates = False
        
        while self._total_cycles_moved < cycles_to_move:
            # Get current position of target
            target_idx = self._find_instruction_index(block, target_instr)
            if target_idx < 0:
                break
            
            # Cannot move if at the beginning of the block
            if target_idx <= 0:
                self._no_candidates = True
                break
            
            # Cannot move into frozen region
            if target_idx <= self.frozen_boundary:
                self._blocked_by_frozen.add(id(target_instr))
                break
            
            # Get registers for target instruction (AA)
            aa_defs, aa_uses = get_instruction_defs_uses(target_instr)
            aa_rreg_set = aa_uses  # Read registers
            aa_wreg_set = aa_defs  # Write registers
            
            # Get instruction above (AAB1)
            aab1_idx = target_idx - 1
            aab1_instr = block.instructions[aab1_idx]
            aab1_opcode = aab1_instr.opcode.lower()
            
            # Check for boundary instructions
            if self._is_boundary_instruction(aab1_opcode):
                self._blocked_by_saveexec.add(id(aab1_instr))
                break
            
            if aab1_instr.is_branch or aab1_instr.is_terminator:
                self._blocked_by_branch.add(id(aab1_instr))
                break
            
            # Skip protected instructions
            if aab1_instr in self.protected_instructions:
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # Check if AAB1 is part of an SCC pair - if so, treat the entire pair as atomic
            scc_pair_range = get_scc_pair_range(block, aab1_idx)
            if scc_pair_range is not None:
                pair_start, pair_end = scc_pair_range
                # Target is trying to move past an SCC pair - need to skip the entire pair
                # Check if target instruction can skip the entire pair without conflicts
                if not self._can_skip_scc_pair(block, target_instr, pair_start, pair_end):
                    self._blocked_by_dependencies.add(id(aab1_instr))
                    if self.verbose:
                        print(f"    Blocked by SCC pair [{pair_start}]-[{pair_end}]")
                    break
                
                # Move target past the entire SCC pair
                pair_cycles = sum(get_instr_cycles(block.instructions[i]) for i in range(pair_start, pair_end + 1))
                
                # Overshoot prevention
                error_if_stop = cycles_to_move - self._total_cycles_moved
                error_if_move = abs(cycles_to_move - (self._total_cycles_moved + pair_cycles))
                
                if error_if_move > error_if_stop:
                    if self.verbose:
                        print(f"    Overshoot prevention: stop at error={error_if_stop}, skip SCC pair would give error={error_if_move}")
                    break
                
                # Move target past the entire SCC pair
                swaps_count = 0
                for i in range(pair_end - pair_start + 1):
                    current_target_idx = self._find_instruction_index(block, target_instr)
                    if current_target_idx <= pair_start:
                        break
                    self._move_single_instruction_up_impl(block, ddg, current_target_idx, current_target_idx - 1)
                    swaps_count += 1
                
                self._total_cycles_moved += pair_cycles
                move_num += 1
                
                if self.verbose:
                    print(f"    Skipped SCC pair [{pair_start}]-[{pair_end}] (+{pair_cycles} cycles)")
                
                # Yield after SCC pair skip
                new_target_idx = self._find_instruction_index(block, target_instr)
                yield MoveYieldInfo(
                    move_num=move_num,
                    move_type="scc_pair",
                    swaps_in_this_move=swaps_count,
                    target_new_idx=new_target_idx,
                    cycles_this_move=pair_cycles,
                    total_cycles_moved=self._total_cycles_moved,
                    target_cycles=cycles_to_move
                )
                continue
            
            # Get registers for AAB1
            aab1_defs, aab1_uses = get_instruction_defs_uses(aab1_instr)
            
            # Check conflicts
            # WAR: AA writes what AAB1 reads -> stop
            war_conflicts = aa_wreg_set & aab1_uses
            if war_conflicts and not (war_conflicts == {'scc'} and is_scc_only_writer(target_instr.opcode.lower())):
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # WAW: AA writes what AAB1 writes -> stop
            waw_conflicts = aa_wreg_set & aab1_defs
            if waw_conflicts and not (waw_conflicts == {'scc'}):
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # RAW: AA reads what AAB1 writes -> need to recursively move AAB1 first
            raw_conflicts = aa_rreg_set & aab1_defs
            if raw_conflicts:
                if raw_conflicts == {'scc'} and not is_scc_reader(target_instr.opcode.lower()):
                    pass  # SCC-only conflict, AA doesn't read SCC
                else:
                    # Try to recursively move AAB1 up first
                    moved_aab1 = self._try_move_instruction_up_recursive(
                        block, ddg, result, aab1_instr, depth=0, max_depth=50
                    )
                    if not moved_aab1:
                        self._blocked_by_dependencies.add(id(aab1_instr))
                        break
                    
                    # Yield after recursive move (the recursive move itself is atomic)
                    move_num += 1
                    new_target_idx = self._find_instruction_index(block, target_instr)
                    yield MoveYieldInfo(
                        move_num=move_num,
                        move_type="recursive",
                        swaps_in_this_move=1,  # Recursive move counts as 1 logical unit
                        target_new_idx=new_target_idx,
                        cycles_this_move=0,  # Recursive moves don't count toward target cycles
                        total_cycles_moved=self._total_cycles_moved,
                        target_cycles=cycles_to_move
                    )
                    # After moving AAB1, continue the loop to try moving target again
                    continue
            
            # Check s_waitcnt constraints
            if not self._can_move_single_instruction_up(block, ddg, target_idx, target_idx - 1, set()):
                self._blocked_by_dependencies.add(id(aab1_instr))
                break
            
            # No conflict - swap AA and AAB1
            # Use predecessor's cycles (AAB1's cycles)
            predecessor_cycles = get_instr_cycles(aab1_instr)
            
            # Overshoot prevention: compare error_if_stop vs error_if_move
            error_if_stop = cycles_to_move - self._total_cycles_moved
            error_if_move = abs(cycles_to_move - (self._total_cycles_moved + predecessor_cycles))
            
            if error_if_move > error_if_stop:
                # Making this move would increase the error - stop here
                if self.verbose:
                    print(f"    Overshoot prevention: stop at error={error_if_stop}, move would give error={error_if_move}")
                break
            
            # Perform the swap with s_waitcnt adjustment
            self._move_single_instruction_up_impl(block, ddg, target_idx, target_idx - 1)
            self._total_cycles_moved += predecessor_cycles
            move_num += 1
            
            if self.verbose:
                print(f"    Moved target [{target_idx}] {target_instr.opcode} up past [{aab1_idx}] {aab1_instr.opcode} (+{predecessor_cycles} cycles)")
            
            # Yield after normal swap
            new_target_idx = self._find_instruction_index(block, target_instr)
            yield MoveYieldInfo(
                move_num=move_num,
                move_type="normal",
                swaps_in_this_move=1,
                target_new_idx=new_target_idx,
                cycles_this_move=predecessor_cycles,
                total_cycles_moved=self._total_cycles_moved,
                target_cycles=cycles_to_move
            )
    
    def _try_move_instruction_up_recursive(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        result: AnalysisResult,
        instr: Instruction,
        depth: int = 0,
        max_depth: int = 50
    ) -> bool:
        """
        Recursively try to move an instruction up.
        If blocked by dependency, first try to move the blocking instruction up.
        
        Args:
            block: The basic block
            ddg: The DDG
            result: The analysis result
            instr: The instruction to move
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            True if the instruction was successfully moved up
        """
        if depth >= max_depth:
            return False
        
        # Get current position
        instr_idx = self._find_instruction_index(block, instr)
        if instr_idx < 0:
            return False
        
        # Cannot move if at the beginning or in frozen region
        if instr_idx <= 0 or instr_idx <= self.frozen_boundary:
            return False
        
        # IMPORTANT: If this instruction is part of an SCC pair, we cannot move it independently
        # The SCC pair must be kept together as an atomic unit
        scc_pair_range = get_scc_pair_range(block, instr_idx)
        if scc_pair_range is not None:
            # If this is the SCC reader (end of pair), don't move it without the writer
            # If this is the SCC writer (start of pair), don't move it without the reader
            # For simplicity, disallow independent movement of any instruction in an SCC pair
            return False
        
        # Get instruction above
        above_idx = instr_idx - 1
        above_instr = block.instructions[above_idx]
        above_opcode = above_instr.opcode.lower()
        
        # Check for boundary instructions
        if self._is_boundary_instruction(above_opcode):
            return False
        
        if above_instr.is_branch or above_instr.is_terminator:
            return False
        
        # Skip protected instructions
        if above_instr in self.protected_instructions:
            return False
        
        # Get registers
        instr_defs, instr_uses = get_instruction_defs_uses(instr)
        above_defs, above_uses = get_instruction_defs_uses(above_instr)
        
        # Check conflicts
        # WAR: instr writes what above reads -> cannot move
        war_conflicts = instr_defs & above_uses
        if war_conflicts and not (war_conflicts == {'scc'} and is_scc_only_writer(instr.opcode.lower())):
            return False
        
        # WAW: instr writes what above writes -> cannot move
        waw_conflicts = instr_defs & above_defs
        if waw_conflicts and not (waw_conflicts == {'scc'}):
            return False
        
        # RAW: instr reads what above writes -> try to move above first
        raw_conflicts = instr_uses & above_defs
        if raw_conflicts:
            if raw_conflicts == {'scc'} and not is_scc_reader(instr.opcode.lower()):
                pass  # SCC-only conflict, instr doesn't read SCC
            else:
                # Recursively try to move above instruction first
                moved_above = self._try_move_instruction_up_recursive(
                    block, ddg, result, above_instr, depth + 1, max_depth
                )
                if not moved_above:
                    return False
                # After moving, try again (indices may have changed)
                return self._try_move_instruction_up_recursive(
                    block, ddg, result, instr, depth, max_depth
                )
        
        # Check s_waitcnt constraints
        if not self._can_move_single_instruction_up(block, ddg, instr_idx, instr_idx - 1, set()):
            return False
        
        # No conflict - perform the swap
        self._move_single_instruction_up_impl(block, ddg, instr_idx, instr_idx - 1)
        
        if self.verbose:
            print(f"      Recursive: moved [{instr_idx}] {instr.opcode} up past [{above_idx}] {above_instr.opcode}")
        
        return True
    
    def _move_down_by_cycles(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        result: AnalysisResult,
        target_instr: Instruction
    ) -> bool:
        """
        Move the target instruction down by |self.cycles| cycles.
        
        New algorithm:
        1. Record target's read/write register sets (rreg_set, wreg_set)
        2. For each downward move, check relationship with instruction below (AAF1):
           - If no conflict: swap AA and AAF1
           - Otherwise: stop (no recursive movement for down direction)
        3. Repeat until target cycle position reached or blocked
        
        Uses successor's cycles for accumulation.
        """
        cycles_to_move = abs(self.cycles)
        any_moved = False
        
        # Reset blocking sets
        self._blocked_by_frozen = set()
        self._blocked_by_dependencies = set()
        self._blocked_by_branch = set()
        self._blocked_by_saveexec = set()
        self._blocked_by_latency = set()
        self._no_candidates = False
        
        # Find branch boundary
        n = len(block.instructions)
        branch_boundary = n
        for i, instr in enumerate(block.instructions):
            if instr.is_branch or instr.is_terminator:
                branch_boundary = i
                break
        
        while self._total_cycles_moved < cycles_to_move:
            # Get current position of target
            target_idx = self._find_instruction_index(block, target_instr)
            if target_idx < 0:
                break
            
            # Cannot move if at the end of the block
            if target_idx >= n - 1 or target_idx >= branch_boundary - 1:
                self._no_candidates = True
                break
            
            # Get registers for target instruction (AA)
            aa_defs, aa_uses = get_instruction_defs_uses(target_instr)
            aa_rreg_set = aa_uses  # Read registers
            aa_wreg_set = aa_defs  # Write registers
            
            # Get instruction below (AAF1)
            aaf1_idx = target_idx + 1
            aaf1_instr = block.instructions[aaf1_idx]
            aaf1_opcode = aaf1_instr.opcode.lower()
            
            # Check for boundary instructions
            if self._is_boundary_instruction(aaf1_opcode):
                self._blocked_by_saveexec.add(id(aaf1_instr))
                break
            
            if aaf1_instr.is_branch or aaf1_instr.is_terminator:
                self._blocked_by_branch.add(id(aaf1_instr))
                break
            
            # Skip protected instructions
            if aaf1_instr in self.protected_instructions:
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # Check if AAF1 is part of an SCC pair - if so, treat the entire pair as atomic
            scc_pair_range = get_scc_pair_range(block, aaf1_idx)
            if scc_pair_range is not None:
                pair_start, pair_end = scc_pair_range
                # Target is trying to move past an SCC pair - need to skip the entire pair
                # For downward movement, we need to check if target can skip the pair
                if not self._can_skip_scc_pair_down(block, target_instr, target_idx, pair_start, pair_end):
                    self._blocked_by_dependencies.add(id(aaf1_instr))
                    if self.verbose:
                        print(f"    Blocked by SCC pair [{pair_start}]-[{pair_end}]")
                    break
                
                # Calculate cycles for the entire pair
                pair_cycles = sum(get_instr_cycles(block.instructions[i]) for i in range(pair_start, pair_end + 1))
                
                # Overshoot prevention
                error_if_stop = cycles_to_move - self._total_cycles_moved
                error_if_move = abs(cycles_to_move - (self._total_cycles_moved + pair_cycles))
                
                if error_if_move > error_if_stop:
                    if self.verbose:
                        print(f"    Overshoot prevention: stop at error={error_if_stop}, skip SCC pair would give error={error_if_move}")
                    break
                
                # Move target past the entire SCC pair
                for i in range(pair_end - pair_start + 1):
                    current_target_idx = self._find_instruction_index(block, target_instr)
                    if current_target_idx > pair_end:
                        break
                    self._move_single_instruction_down_impl(block, ddg, current_target_idx, current_target_idx + 1)
                
                self._total_cycles_moved += pair_cycles
                any_moved = True
                
                if self.verbose:
                    print(f"    Skipped SCC pair [{pair_start}]-[{pair_end}] (+{pair_cycles} cycles)")
                continue
            
            # Get registers for AAF1
            aaf1_defs, aaf1_uses = get_instruction_defs_uses(aaf1_instr)
            
            # Check conflicts - for downward movement, any conflict stops movement (no recursion)
            # RAW: AAF1 reads what AA writes -> stop
            raw_conflicts = aa_wreg_set & aaf1_uses
            if raw_conflicts and not (raw_conflicts == {'scc'} and not is_scc_reader(aaf1_opcode)):
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # WAR: AAF1 writes what AA reads -> stop
            war_conflicts = aaf1_defs & aa_rreg_set
            if war_conflicts and not (war_conflicts == {'scc'} and is_scc_only_writer(aaf1_opcode)):
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # WAW: Both write same register -> usually OK (traversable)
            # But we check anyway for non-SCC registers
            waw_conflicts = aa_wreg_set & aaf1_defs
            if waw_conflicts and not (waw_conflicts == {'scc'}):
                # For non-SCC WAW, check if AAF1's result is live after
                # We'll be conservative and allow it (WAW traversable rule)
                pass
            
            # Check s_waitcnt constraints
            if not self._can_move_single_instruction_down(block, ddg, target_idx, target_idx + 1, set()):
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # No conflict - swap AA and AAF1
            # Use successor's cycles (AAF1's cycles)
            successor_cycles = get_instr_cycles(aaf1_instr)
            
            # Overshoot prevention: compare error_if_stop vs error_if_move
            error_if_stop = cycles_to_move - self._total_cycles_moved
            error_if_move = abs(cycles_to_move - (self._total_cycles_moved + successor_cycles))
            
            if error_if_move > error_if_stop:
                # Making this move would increase the error - stop here
                if self.verbose:
                    print(f"    Overshoot prevention: stop at error={error_if_stop}, move would give error={error_if_move}")
                break
            
            # Perform the swap with s_waitcnt adjustment
            self._move_single_instruction_down_impl(block, ddg, target_idx, target_idx + 1)
            self._total_cycles_moved += successor_cycles
            any_moved = True
            
            if self.verbose:
                print(f"    Moved target [{target_idx}] {target_instr.opcode} down past [{aaf1_idx}] {aaf1_instr.opcode} (+{successor_cycles} cycles)")
        
        return any_moved
    
    def _iter_move_down(
        self,
        block: BasicBlock,
        ddg: Optional[DDG],
        result: AnalysisResult,
        target_instr: Instruction,
        cycles_to_move: int
    ) -> Generator[MoveYieldInfo, None, None]:
        """
        Generator version of _move_down_by_cycles that yields after each logical move unit.
        
        Logical move units:
        - normal: Single instruction swap
        - scc_pair: Skip entire SCC pair as atomic unit
        
        Args:
            block: The basic block
            ddg: The DDG for the block
            result: The analysis result
            target_instr: The instruction to move
            cycles_to_move: Total cycles to move (positive value)
            
        Yields:
            MoveYieldInfo with details of each logical move
        """
        move_num = 0
        
        # Reset blocking sets
        self._blocked_by_frozen = set()
        self._blocked_by_dependencies = set()
        self._blocked_by_branch = set()
        self._blocked_by_saveexec = set()
        self._blocked_by_latency = set()
        self._no_candidates = False
        
        # Find branch boundary
        n = len(block.instructions)
        branch_boundary = n
        for i, instr in enumerate(block.instructions):
            if instr.is_branch or instr.is_terminator:
                branch_boundary = i
                break
        
        while self._total_cycles_moved < cycles_to_move:
            # Get current position of target
            target_idx = self._find_instruction_index(block, target_instr)
            if target_idx < 0:
                break
            
            # Cannot move if at the end of the block
            if target_idx >= n - 1 or target_idx >= branch_boundary - 1:
                self._no_candidates = True
                break
            
            # Get registers for target instruction (AA)
            aa_defs, aa_uses = get_instruction_defs_uses(target_instr)
            aa_rreg_set = aa_uses  # Read registers
            aa_wreg_set = aa_defs  # Write registers
            
            # Get instruction below (AAF1)
            aaf1_idx = target_idx + 1
            aaf1_instr = block.instructions[aaf1_idx]
            aaf1_opcode = aaf1_instr.opcode.lower()
            
            # Check for boundary instructions
            if self._is_boundary_instruction(aaf1_opcode):
                self._blocked_by_saveexec.add(id(aaf1_instr))
                break
            
            if aaf1_instr.is_branch or aaf1_instr.is_terminator:
                self._blocked_by_branch.add(id(aaf1_instr))
                break
            
            # Skip protected instructions
            if aaf1_instr in self.protected_instructions:
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # Check if AAF1 is part of an SCC pair - if so, treat the entire pair as atomic
            scc_pair_range = get_scc_pair_range(block, aaf1_idx)
            if scc_pair_range is not None:
                pair_start, pair_end = scc_pair_range
                # Target is trying to move past an SCC pair - need to skip the entire pair
                # For downward movement, we need to check if target can skip the pair
                if not self._can_skip_scc_pair_down(block, target_instr, target_idx, pair_start, pair_end):
                    self._blocked_by_dependencies.add(id(aaf1_instr))
                    if self.verbose:
                        print(f"    Blocked by SCC pair [{pair_start}]-[{pair_end}]")
                    break
                
                # Calculate cycles for the entire pair
                pair_cycles = sum(get_instr_cycles(block.instructions[i]) for i in range(pair_start, pair_end + 1))
                
                # Overshoot prevention
                error_if_stop = cycles_to_move - self._total_cycles_moved
                error_if_move = abs(cycles_to_move - (self._total_cycles_moved + pair_cycles))
                
                if error_if_move > error_if_stop:
                    if self.verbose:
                        print(f"    Overshoot prevention: stop at error={error_if_stop}, skip SCC pair would give error={error_if_move}")
                    break
                
                # Move target past the entire SCC pair
                swaps_count = 0
                for i in range(pair_end - pair_start + 1):
                    current_target_idx = self._find_instruction_index(block, target_instr)
                    if current_target_idx > pair_end:
                        break
                    self._move_single_instruction_down_impl(block, ddg, current_target_idx, current_target_idx + 1)
                    swaps_count += 1
                
                self._total_cycles_moved += pair_cycles
                move_num += 1
                
                if self.verbose:
                    print(f"    Skipped SCC pair [{pair_start}]-[{pair_end}] (+{pair_cycles} cycles)")
                
                # Yield after SCC pair skip
                new_target_idx = self._find_instruction_index(block, target_instr)
                yield MoveYieldInfo(
                    move_num=move_num,
                    move_type="scc_pair",
                    swaps_in_this_move=swaps_count,
                    target_new_idx=new_target_idx,
                    cycles_this_move=pair_cycles,
                    total_cycles_moved=self._total_cycles_moved,
                    target_cycles=cycles_to_move
                )
                continue
            
            # Get registers for AAF1
            aaf1_defs, aaf1_uses = get_instruction_defs_uses(aaf1_instr)
            
            # Check conflicts - for downward movement, any conflict stops movement (no recursion)
            # RAW: AAF1 reads what AA writes -> stop
            raw_conflicts = aa_wreg_set & aaf1_uses
            if raw_conflicts and not (raw_conflicts == {'scc'} and not is_scc_reader(aaf1_opcode)):
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # WAR: AAF1 writes what AA reads -> stop
            war_conflicts = aaf1_defs & aa_rreg_set
            if war_conflicts and not (war_conflicts == {'scc'} and is_scc_only_writer(aaf1_opcode)):
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # WAW: Both write same register -> usually OK (traversable)
            # But we check anyway for non-SCC registers
            waw_conflicts = aa_wreg_set & aaf1_defs
            if waw_conflicts and not (waw_conflicts == {'scc'}):
                # For non-SCC WAW, check if AAF1's result is live after
                # We'll be conservative and allow it (WAW traversable rule)
                pass
            
            # Check s_waitcnt constraints
            if not self._can_move_single_instruction_down(block, ddg, target_idx, target_idx + 1, set()):
                self._blocked_by_dependencies.add(id(aaf1_instr))
                break
            
            # No conflict - swap AA and AAF1
            # Use successor's cycles (AAF1's cycles)
            successor_cycles = get_instr_cycles(aaf1_instr)
            
            # Overshoot prevention: compare error_if_stop vs error_if_move
            error_if_stop = cycles_to_move - self._total_cycles_moved
            error_if_move = abs(cycles_to_move - (self._total_cycles_moved + successor_cycles))
            
            if error_if_move > error_if_stop:
                # Making this move would increase the error - stop here
                if self.verbose:
                    print(f"    Overshoot prevention: stop at error={error_if_stop}, move would give error={error_if_move}")
                break
            
            # Perform the swap with s_waitcnt adjustment
            self._move_single_instruction_down_impl(block, ddg, target_idx, target_idx + 1)
            self._total_cycles_moved += successor_cycles
            move_num += 1
            
            if self.verbose:
                print(f"    Moved target [{target_idx}] {target_instr.opcode} down past [{aaf1_idx}] {aaf1_instr.opcode} (+{successor_cycles} cycles)")
            
            # Yield after normal swap
            new_target_idx = self._find_instruction_index(block, target_instr)
            yield MoveYieldInfo(
                move_num=move_num,
                move_type="normal",
                swaps_in_this_move=1,
                target_new_idx=new_target_idx,
                cycles_this_move=successor_cycles,
                total_cycles_moved=self._total_cycles_moved,
                target_cycles=cycles_to_move
            )
    
    # =========================================================================
    # Waitcnt adjustment methods
    # =========================================================================
    
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
        Calls the on_swap_callback if set.
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
        
        # Call the hook if registered
        if self.on_swap_callback:
            self.on_swap_callback(block, idx1, idx2, instr_a)


# =============================================================================
# DistributeStepExecutor - Modular interface for instruction distribution
# =============================================================================

class DistributeStepExecutor:
    """
    Step executor for DistributeInstructionPass.
    Exposes clean interface for debug_distribute_pass.py.
    
    This class provides:
    - Context creation with all precomputed parameters
    - Step-by-step execution with callbacks
    - Per-move callbacks for Level-2 debugging
    """
    
    def __init__(self, result: AnalysisResult, context: DistributeContext):
        """
        Initialize the step executor.
        
        Args:
            result: The AnalysisResult to modify
            context: Precomputed distribute context
        """
        self.result = result
        self.ctx = context
        self.frozen_boundary = 0
        self._step_results: List[StepResult] = []
    
    @staticmethod
    def get_cumulative_cycle(block: BasicBlock, idx: int) -> int:
        """
        Calculate cumulative cycle count up to and including instruction at idx.
        
        This is the canonical implementation - use this instead of duplicating.
        """
        total = 0
        for i in range(idx + 1):
            total += get_instruction_cycles(block.instructions[i].opcode)
        return total
    
    @staticmethod
    def cycle_to_index(block: BasicBlock, target_cycle: int) -> int:
        """
        Convert target cycle to instruction index.
        
        Finds the first instruction whose cumulative cycle reaches or exceeds target_cycle.
        """
        total = 0
        for i, instr in enumerate(block.instructions):
            cycles = get_instruction_cycles(instr.opcode)
            total += cycles
            if total >= target_cycle:
                return i
        return len(block.instructions) - 1
    
    @staticmethod
    def _find_branch_boundary(block: BasicBlock) -> int:
        """
        Find index of first branch/terminator instruction.
        s_barrier is no longer considered as a boundary.
        """
        n = len(block.instructions)
        
        # Find the branch/terminator position
        for i, instr in enumerate(block.instructions):
            if instr.is_branch or instr.is_terminator:
                return i
        
        return n
    
    @staticmethod
    def _calculate_ideal_cycles(total_cycles: int, k: int) -> List[int]:
        """Calculate ideal cycle positions for K evenly distributed instructions."""
        if k <= 0:
            return []
        
        if k == 1:
            return [total_cycles // 2]
        
        cycle_spacing = total_cycles / k
        return [int(cycle_spacing * (i + 1)) for i in range(k)]
    
    @staticmethod
    def create_context(
        result: AnalysisResult,
        block_label: str,
        target_opcode: str,
        distribute_count: int
    ) -> Optional[DistributeContext]:
        """
        Create distribute context with all precomputed parameters.
        
        Args:
            result: The AnalysisResult
            block_label: Label of the block to modify
            target_opcode: Opcode to distribute (e.g., "global_load_dwordx4")
            distribute_count: K, number of instructions to distribute
            
        Returns:
            DistributeContext if successful, None if block not found or no targets
        """
        if block_label not in result.cfg.blocks:
            return None
        
        block = result.cfg.blocks[block_label]
        
        # Find all target instructions
        target_indices = []
        all_target_instrs = []
        for i, instr in enumerate(block.instructions):
            if instr.opcode == target_opcode:
                target_indices.append(i)
                all_target_instrs.append(instr)
        
        M = len(target_indices)
        if M == 0:
            return None
        
        K = min(distribute_count, M)
        
        # Find branch boundary
        branch_boundary = DistributeStepExecutor._find_branch_boundary(block)
        
        # Calculate total cycles up to branch_boundary
        total_cycles = 0
        for i in range(branch_boundary):
            total_cycles += get_instr_cycles(block.instructions[i])
        
        # Calculate ideal cycle positions
        ideal_cycles = DistributeStepExecutor._calculate_ideal_cycles(total_cycles, K)
        
        return DistributeContext(
            block=block,
            block_label=block_label,
            target_opcode=target_opcode,
            target_indices=target_indices,
            all_target_instrs=all_target_instrs,
            K=K,
            M=M,
            ideal_cycles=ideal_cycles,
            total_cycles=total_cycles,
            branch_boundary=branch_boundary
        )
    
    def find_nth_target_index(self, n: int) -> int:
        """Find current index of the n-th target instruction (0-indexed)."""
        count = 0
        for idx, instr in enumerate(self.ctx.block.instructions):
            if instr.opcode == self.ctx.target_opcode:
                if count == n:
                    return idx
                count += 1
        return -1
    
    @staticmethod
    def get_start_cycle(block: BasicBlock, idx: int) -> int:
        """
        Get the START cycle of an instruction (cumulative cycles BEFORE this instruction).
        This matches _get_instruction_cycle_position in DistributeInstructionPass.
        """
        cumulative = 0
        for i, instr in enumerate(block.instructions):
            if i == idx:
                return cumulative
            cumulative += get_instr_cycles(instr)
        return cumulative
    
    def get_execution_order(self) -> List[int]:
        """
        Get the correct execution order for steps, matching DistributeInstructionPass.run().
        
        Two-phase strategy:
        - Phase 1: Instructions that need to move UP (current_cycle > ideal_cycle) - forward order
        - Phase 2: Instructions that need to move DOWN (current_cycle <= ideal_cycle) - REVERSE order
        
        Returns:
            List of step numbers in the order they should be executed.
        """
        needs_move_up, needs_move_down = self._classify_moves()
        
        # Build execution order: UP steps first (forward), then DOWN steps (reverse)
        execution_order = [step for step, _ in needs_move_up]
        execution_order.extend([step for step, _ in reversed(needs_move_down)])
        
        return execution_order
    
    def _classify_moves(self) -> tuple:
        """
        Classify steps into UP and DOWN moves.
        
        Returns:
            Tuple of (needs_move_up, needs_move_down), each is a list of (step_num, ideal_cycle).
        """
        needs_move_up = []    # (step_num, ideal_cycle)
        needs_move_down = []  # (step_num, ideal_cycle)
        
        for i in range(self.ctx.K):
            current_idx = self.find_nth_target_index(i)
            if current_idx < 0:
                continue
            
            ideal_cycle = self.ctx.ideal_cycles[i]
            current_cycle = self.get_start_cycle(self.ctx.block, current_idx)
            
            if current_cycle > ideal_cycle:
                # Need to move up (current position is after ideal)
                needs_move_up.append((i, ideal_cycle))
            else:
                # Need to move down or stay (current position is at or before ideal)
                needs_move_down.append((i, ideal_cycle))
        
        return needs_move_up, needs_move_down
    
    def is_move_up(self, step_num: int) -> bool:
        """
        Check if a step is an UP move (needs to move instruction upward).
        
        Args:
            step_num: Step number (0-indexed)
            
        Returns:
            True if this step is an UP move, False if DOWN move.
        """
        current_idx = self.find_nth_target_index(step_num)
        if current_idx < 0:
            return False
        
        ideal_cycle = self.ctx.ideal_cycles[step_num]
        current_cycle = self.get_start_cycle(self.ctx.block, current_idx)
        
        return current_cycle > ideal_cycle
    
    def get_execution_order(self) -> List[int]:
        """
        Get the execution order of steps matching DistributeInstructionPass.run().
        
        The order is:
        1. UP moves in forward order (step 0, 1, ...)
        2. DOWN moves in reverse order (step K-1, K-2, ...)
        
        Returns:
            List of step numbers in execution order
        """
        needs_move_up, needs_move_down = self._classify_moves()
        
        # UP moves in forward order
        up_steps = [step_num for step_num, _ in needs_move_up]
        
        # DOWN moves in reverse order
        down_steps = [step_num for step_num, _ in reversed(needs_move_down)]
        
        return up_steps + down_steps
    
    def get_execution_index(self, step_num: int) -> int:
        """
        Get the execution index (position in execution order) for a step.
        
        Args:
            step_num: The step number (0-indexed)
            
        Returns:
            The position in execution order (0-indexed), or -1 if not found
        """
        execution_order = self.get_execution_order()
        try:
            return execution_order.index(step_num)
        except ValueError:
            return -1
    
    def get_steps_before(self, step_num: int) -> List[int]:
        """
        Get all steps that should execute before the given step.
        
        Args:
            step_num: The step number (0-indexed)
            
        Returns:
            List of step numbers that execute before this step
        """
        execution_order = self.get_execution_order()
        try:
            exec_idx = execution_order.index(step_num)
            return execution_order[:exec_idx]
        except ValueError:
            return []
    
    def get_step_params(self, step_num: int) -> Dict[str, Any]:
        """
        Get computed parameters for a step (for debugging/logging).
        
        Returns dict with: current_idx, target_idx, current_cycle, target_cycle, cycles_to_move
        """
        current_idx = self.find_nth_target_index(step_num)
        if current_idx < 0:
            return {"error": "target_not_found"}
        
        # Calculate target_idx from ideal_cycle
        target_idx = self.cycle_to_index(self.ctx.block, self.ctx.ideal_cycles[step_num])
        
        # Ensure target_idx respects ordering constraints
        if step_num > 0:
            prev_idx = self.find_nth_target_index(step_num - 1)
            if prev_idx >= 0 and target_idx <= prev_idx:
                target_idx = prev_idx + 1
        
        target_idx = max(target_idx, self.frozen_boundary)
        
        # Calculate cycles using START cycle to match _move_instruction_toward behavior
        # START cycle = cumulative cycles BEFORE the instruction
        current_cycle = self.get_start_cycle(self.ctx.block, current_idx)
        ideal_cycle = self.ctx.ideal_cycles[step_num]
        target_cycle = self.get_start_cycle(self.ctx.block, target_idx)
        
        # Use the same calculation as _move_instruction_toward:
        # cycles_to_move = current_cycle - ideal_cycle
        # positive = move up, negative = move down
        cycles_to_move = current_cycle - ideal_cycle
        
        return {
            "step_num": step_num,
            "current_idx": current_idx,
            "target_idx": target_idx,
            "current_cycle": current_cycle,
            "target_cycle": target_cycle,
            "ideal_cycle": ideal_cycle,
            "cycles_to_move": cycles_to_move
        }
    
    def execute_step(
        self,
        step_num: int,
        on_before_move: Optional[callable] = None,
        on_after_move: Optional[callable] = None
    ) -> StepResult:
        """
        Execute a single distribution step.
        
        Args:
            step_num: Step number (0-indexed)
            on_before_move: Called before the move with step params
            on_after_move: Called after the move with step result
            
        Returns:
            StepResult with execution details
        """
        params = self.get_step_params(step_num)
        
        if "error" in params:
            return StepResult(
                step_num=step_num,
                success=False,
                message=params.get("error", "Unknown error")
            )
        
        current_idx = params["current_idx"]
        target_idx = params["target_idx"]
        cycles_to_move = params["cycles_to_move"]
        
        # Check if this is an UP move BEFORE executing (state will change after move)
        is_up_move = self.is_move_up(step_num)
        
        if on_before_move:
            on_before_move(params)
        
        cycles_moved = 0
        move_count = 0
        
        if cycles_to_move != 0:
            # Get protected instructions based on move direction
            # This matches DistributeInstructionPass.run() behavior:
            # - UP moves: protect instructions after current step (not yet processed)
            # - DOWN moves: protect instructions before current step (already processed)
            if is_up_move:
                protected_instrs = self.ctx.all_target_instrs[step_num + 1:]
            else:
                protected_instrs = self.ctx.all_target_instrs[:step_num]
            
            move_pass = MoveInstructionPass(
                self.ctx.block_label,
                current_idx,
                cycles_to_move,
                verbose=False,
                frozen_boundary=self.frozen_boundary,
                protected_instructions=protected_instrs
            )
            move_pass.run(self.result)
            cycles_moved = move_pass.total_cycles_moved
            move_count = 1 if cycles_moved > 0 else 0
        
        # Update frozen boundary only for UP moves (matching DistributeInstructionPass.run())
        # DOWN moves should not update frozen_boundary
        if is_up_move:
            new_idx = self.find_nth_target_index(step_num)
            if new_idx >= 0:
                self.frozen_boundary = new_idx + 1
        
        result = StepResult(
            step_num=step_num,
            success=True,
            cycles_moved=cycles_moved,
            move_count=move_count,
            details=params
        )
        
        self._step_results.append(result)
        
        if on_after_move:
            on_after_move(result)
        
        return result
    
    def execute_step_with_moves(
        self,
        step_num: int,
        on_single_move: Optional[callable] = None
    ) -> StepResult:
        """
        Execute step with per-move callbacks for Level-2 debugging.
        
        Args:
            step_num: Step number (0-indexed)
            on_single_move: Called after each single instruction move.
                           Signature: (move_info: SingleMoveInfo) -> bool
                           Return False to stop execution.
                           
        Returns:
            StepResult with execution details
        """
        params = self.get_step_params(step_num)
        
        if "error" in params:
            return StepResult(
                step_num=step_num,
                success=False,
                message=params.get("error", "Unknown error")
            )
        
        current_idx = params["current_idx"]
        cycles_to_move = params["cycles_to_move"]
        
        # Check if this is an UP move BEFORE executing
        is_up_move = self.is_move_up(step_num)
        
        if cycles_to_move == 0:
            return StepResult(
                step_num=step_num,
                success=True,
                cycles_moved=0,
                move_count=0,
                message="No move needed"
            )
        
        # Get protected instructions based on move direction
        # This matches DistributeInstructionPass.run() behavior:
        # - UP moves: protect instructions after current step (not yet processed)
        # - DOWN moves: protect instructions before current step (already processed)
        if is_up_move:
            protected_instrs = self.ctx.all_target_instrs[step_num + 1:]
        else:
            protected_instrs = self.ctx.all_target_instrs[:step_num]
        
        # Get target instruction for tracking
        target_instr = self.ctx.block.instructions[current_idx]
        
        # Use MoveExecutor for per-move callbacks
        executor = MoveExecutor(
            self.result,
            self.ctx.block_label,
            current_idx,
            frozen_boundary=self.frozen_boundary,
            protected_instructions=protected_instrs,
            verbose=False
        )
        
        total_cycles = executor.move_by_cycles(cycles_to_move, on_single_move=on_single_move)
        
        # Update frozen boundary only for UP moves (matching DistributeInstructionPass.run())
        if is_up_move:
            new_idx = self.find_nth_target_index(step_num)
            if new_idx >= 0:
                self.frozen_boundary = new_idx + 1
        
        result = StepResult(
            step_num=step_num,
            success=True,
            cycles_moved=total_cycles,
            move_count=executor.move_count,
            details=params
        )
        
        self._step_results.append(result)
        return result
    
    def execute_step_with_iter(
        self,
        step_num: int,
        on_move: Optional[Callable[[MoveYieldInfo], bool]] = None
    ) -> StepResult:
        """
        Execute step with per-logical-move callbacks using MoveInstructionPass.iter_moves().
        
        This method provides fine-grained control over instruction movement, yielding
        after each logical move unit (normal swap, SCC pair skip, or recursive move).
        
        Args:
            step_num: Step number (0-indexed)
            on_move: Called after each logical move unit.
                     Signature: (move_info: MoveYieldInfo) -> bool
                     Return False to stop execution early.
                     
        Returns:
            StepResult with execution details
        """
        params = self.get_step_params(step_num)
        
        if "error" in params:
            return StepResult(
                step_num=step_num,
                success=False,
                message=params.get("error", "Unknown error")
            )
        
        current_idx = params["current_idx"]
        cycles_to_move = params["cycles_to_move"]
        
        # Check if this is an UP move BEFORE executing (state will change after move)
        is_up_move = self.is_move_up(step_num)
        
        if cycles_to_move == 0:
            return StepResult(
                step_num=step_num,
                success=True,
                cycles_moved=0,
                move_count=0,
                message="No move needed"
            )
        
        # Get protected instructions based on move direction
        # This matches DistributeInstructionPass.run() behavior:
        # - UP moves: protect instructions after current step (not yet processed)
        # - DOWN moves: protect instructions before current step (already processed)
        if is_up_move:
            protected_instrs = self.ctx.all_target_instrs[step_num + 1:]
        else:
            protected_instrs = self.ctx.all_target_instrs[:step_num]
        
        # Create MoveInstructionPass for iteration
        move_pass = MoveInstructionPass(
            self.ctx.block_label,
            current_idx,
            cycles_to_move,
            verbose=False,
            frozen_boundary=self.frozen_boundary,
            protected_instructions=protected_instrs
        )
        
        # Iterate through moves, calling callback after each
        move_count = 0
        last_move_info = None
        stopped_early = False
        
        for move_info in move_pass.iter_moves(self.result):
            move_count += 1
            last_move_info = move_info
            
            if on_move is not None:
                if not on_move(move_info):
                    stopped_early = True
                    break
        
        # Get total cycles moved
        total_cycles = move_pass.total_cycles_moved
        
        # Update frozen boundary only for UP moves (matching DistributeInstructionPass.run())
        if is_up_move:
            new_idx = self.find_nth_target_index(step_num)
            if new_idx >= 0:
                self.frozen_boundary = new_idx + 1
        
        result = StepResult(
            step_num=step_num,
            success=True,
            cycles_moved=total_cycles,
            move_count=move_count,
            message="Stopped early by callback" if stopped_early else "",
            details=params
        )
        
        self._step_results.append(result)
        return result
    
    def execute_all_steps(
        self,
        on_step_complete: Optional[callable] = None
    ) -> List[StepResult]:
        """
        Execute all K steps with optional callback after each step.
        
        Args:
            on_step_complete: Called after each step completes.
                             Signature: (result: StepResult) -> bool
                             Return False to stop execution.
                             
        Returns:
            List of StepResult for each step
        """
        results = []
        
        for i in range(self.ctx.K):
            result = self.execute_step(i)
            results.append(result)
            
            if on_step_complete:
                if not on_step_complete(result):
                    break
        
        return results


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
            total += get_instr_cycles(instr)
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
        
        if k == 1:
            return [total_cycles // 2]
        
        # Match debug_distribute_pass.py: cycle_spacing = total_cycles / k
        cycle_spacing = total_cycles / k
        return [int(cycle_spacing * (i + 1)) for i in range(k)]
    
    def _find_branch_boundary(self, block: BasicBlock) -> int:
        """
        Find index of first branch/terminator instruction.
        s_barrier is no longer considered as a boundary.
        
        Returns:
            Index of first branch/terminator, or len(instructions) if none
        """
        n = len(block.instructions)
        
        for i, instr in enumerate(block.instructions):
            if instr.is_branch or instr.is_terminator:
                return i
        
        return n
    
    def _cycle_to_index(self, block: BasicBlock, target_cycle: int) -> int:
        """
        Convert a target cycle position to an instruction index.
        
        Matches debug_distribute_pass.py implementation: finds the first instruction
        whose cumulative cycle (after adding its cycles) reaches or exceeds target_cycle.
        
        Args:
            block: The basic block
            target_cycle: Target cycle position
            
        Returns:
            Instruction index corresponding to the target cycle
        """
        total = 0
        for i, instr in enumerate(block.instructions):
            cycles = get_instr_cycles(instr)
            total += cycles
            if total >= target_cycle:
                return i
        return len(block.instructions) - 1
    
    def _get_instruction_cycle_position(self, block: BasicBlock, index: int) -> int:
        """
        Get the cycle position (start cycle) of an instruction at given index.
        
        Args:
            block: The basic block
            index: Instruction index
            
        Returns:
            Starting cycle of the instruction (cumulative cycles BEFORE this instruction)
        """
        cumulative = 0
        for i, instr in enumerate(block.instructions):
            if i == index:
                return cumulative
            cumulative += get_instr_cycles(instr)
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
        protected_instructions: Optional[List[Instruction]] = None,
        ideal_cycle: Optional[int] = None
    ) -> int:
        """
        Move an instruction toward a target position using cycle-based MoveInstructionPass.
        
        This method calculates the cycle difference between current and ideal cycle positions,
        then uses MoveInstructionPass with the appropriate cycle count to move the instruction.
        
        Uses START cycle (cumulative cycles BEFORE the instruction) for consistency with
        error reporting.
        
        Args:
            result: The AnalysisResult to modify
            current_idx: Current index of the instruction
            target_idx: Target index to move toward (used for clamping and direction)
            branch_boundary: Index of branch/terminator (cannot move past)
            frozen_boundary: Instructions at idx < frozen_boundary cannot be moved
            protected_instructions: List of instruction objects that should never be moved
            ideal_cycle: The ideal cycle position to move to. If provided, cycles_to_move
                        is calculated directly from this instead of from target_idx.
            
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
        
        # Get current cycle using START cycle (cumulative BEFORE the instruction)
        current_cycle = self._get_instruction_cycle_position(block, current_idx)
        
        # Calculate cycles_to_move
        if ideal_cycle is not None:
            # Use ideal_cycle directly for more accurate movement
            cycles_to_move = current_cycle - ideal_cycle
        else:
            # Fall back to target_idx-based calculation
            target_cycle = self._get_instruction_cycle_position(block, target_idx)
            cycles_to_move = current_cycle - target_cycle
        
        if cycles_to_move == 0:
            return current_idx
        
        # MoveInstructionPass convention: positive cycles = move up, negative cycles = move down
        # cycles_to_move > 0: current is after ideal, need to move up (positive cycles)
        # cycles_to_move < 0: current is before ideal, need to move down (negative cycles)
        
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
        
        # Get actual final position and cycle
        final_idx = self._find_instruction_index(block, target_instr)
        final_cycle = self._get_instruction_cycle_position(block, final_idx)
        
        if success and self.verbose:
            if move_pass.total_cycles_moved > 0:
                print(f"    Actually moved: {move_pass.total_cycles_moved} cycles")
                # Show reason if didn't reach target
                if move_pass.total_cycles_moved < abs(cycles_to_move):
                    print(f"    Stop reason: {move_pass.stop_reason}")
                    
                    # Overshoot prevention: compare error_if_stop vs error_if_move
                    if ideal_cycle is not None:
                        error_now = abs(final_cycle - ideal_cycle)
                        # If we could move one more step, what would be the error?
                        # This is already handled in the MoveInstructionPass
                        print(f"    Final error: {error_now} cycles from ideal")
        elif not success and self.verbose:
            print(f"    Move blocked: {move_pass.last_result.message if move_pass.last_result else 'Unknown'}")
            if move_pass.stop_reason:
                print(f"    Stop reason: {move_pass.stop_reason}")
        
        return final_idx
    
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
            print(f"Found {M} {self.target_opcode} instructions in {self.block_label}")
        
        # 2. Find branch boundary first (needed for total_cycles calculation)
        branch_boundary = self._find_branch_boundary(block)
        
        # 3. Calculate total cycles up to branch_boundary (matches debug_distribute_pass.py)
        # Use cumulative cycle INCLUDING the instruction at branch_boundary - 1
        if branch_boundary > 0:
            total_cycles = 0
            for i in range(branch_boundary):
                total_cycles += get_instr_cycles(block.instructions[i])
        else:
            total_cycles = 0
        
        if self.verbose:
            print(f"Block total cycles: {total_cycles}, branch boundary: {branch_boundary}")
        
        # 4. Calculate ideal cycle positions for K instructions
        ideal_cycles = self._calculate_ideal_cycles(total_cycles, K)
        
        if self.verbose:
            print(f"Ideal cycle positions: {ideal_cycles[:5]}{'...' if len(ideal_cycles) > 5 else ''}")
        
        if self.verbose:
            print(f"Branch boundary at index: {branch_boundary}")
        
        # 5. Collect all target instruction objects for protection
        # This ensures when moving one target, we don't accidentally move others
        all_target_instrs = []
        for i in range(M):
            idx = self._find_nth_target(block, i)
            if idx >= 0:
                all_target_instrs.append(block.instructions[idx])
        
        # 6. Two-phase distribution strategy
        # Phase 1: Process instructions that need to move UP (in forward order: 0, 1, 2, ...)
        # Phase 2: Process instructions that need to move DOWN (in reverse order: K-1, K-2, ...)
        # 
        # Rationale: When moving down, subsequent target instructions block the path.
        # By processing from the furthest first (reverse order), each instruction
        # clears the way for the ones before it.
        
        # First, classify each instruction as needing to move up or down
        needs_move_up = []    # (instruction_index, ideal_cycle, target_idx)
        needs_move_down = []  # (instruction_index, ideal_cycle, target_idx)
        
        for i in range(K):
            current_idx = self._find_nth_target(block, i)
            if current_idx < 0:
                continue
            
            ideal_cycle = ideal_cycles[i]
            current_cycle = self._get_instruction_cycle_position(block, current_idx)
            target_idx = self._cycle_to_index(block, ideal_cycle)
            
            if current_cycle > ideal_cycle:
                # Need to move up (current position is after ideal)
                needs_move_up.append((i, ideal_cycle, target_idx))
            else:
                # Need to move down or stay (current position is at or before ideal)
                needs_move_down.append((i, ideal_cycle, target_idx))
        
        if self.verbose:
            print(f"  Instructions needing move UP: {[x[0] for x in needs_move_up]}")
            print(f"  Instructions needing move DOWN: {[x[0] for x in needs_move_down]}")
        
        frozen_boundary = 0
        
        # Phase 1: Process instructions that need to move UP (forward order)
        # These are processed first because moving up doesn't block other targets
        for i, ideal_cycle, _ in needs_move_up:
            current_idx = self._find_nth_target(block, i)
            if current_idx < 0:
                if self.verbose:
                    print(f"  Could not find target instruction {i}")
                continue
            
            target_idx = self._cycle_to_index(block, ideal_cycle)
            
            # Ensure target_idx respects ordering constraints
            if i > 0:
                prev_target_idx = self._find_nth_target(block, i - 1)
                if prev_target_idx >= 0 and target_idx <= prev_target_idx:
                    target_idx = prev_target_idx + 1
            
            # Ensure target is at least at frozen_boundary
            target_idx = max(target_idx, frozen_boundary)
            
            # Protected: all other target instructions except current one
            protected_instrs = [instr for j, instr in enumerate(all_target_instrs) if j != i]
            
            if self.verbose:
                current_cycle = self._get_instruction_cycle_position(block, current_idx)
                print(f"  [{i}] (UP) Moving from idx={current_idx} (cycle={current_cycle}) to idx={target_idx} (ideal cycle={ideal_cycle})")
            
            original_idx = current_idx
            final_idx = self._move_instruction_toward(result, current_idx, target_idx, branch_boundary, frozen_boundary, protected_instrs, ideal_cycle)
            
            # Track if any move happened
            if final_idx != original_idx:
                self._moved_count += 1
            
            # Use _find_nth_target to get actual position by count order
            actual_final_idx = self._find_nth_target(block, i)
            
            if self.verbose:
                final_cycle = self._get_instruction_cycle_position(block, actual_final_idx)
                print(f"      Final position: idx={actual_final_idx} (cycle={final_cycle})")
            
            # Update frozen boundary only for move-up instructions
            frozen_boundary = actual_final_idx + 1
        
        # Phase 2: Process instructions that need to move DOWN (reverse order)
        # By processing from furthest to closest, each instruction clears the path
        # for the ones that need to move less far
        # Track already-processed DOWN instructions to protect them from chain movement
        down_processed_instrs = []
        
        for i, ideal_cycle, _ in reversed(needs_move_down):
            current_idx = self._find_nth_target(block, i)
            if current_idx < 0:
                if self.verbose:
                    print(f"  Could not find target instruction {i}")
                continue
            
            target_idx = self._cycle_to_index(block, ideal_cycle)
            
            # For downward moves, ensure we don't go past branch boundary
            target_idx = min(target_idx, branch_boundary - 1)
            
            # Ensure target_idx respects ordering with subsequent instructions
            # (instructions after this one should stay after it)
            if i < K - 1:
                next_target_idx = self._find_nth_target(block, i + 1)
                if next_target_idx >= 0 and target_idx >= next_target_idx:
                    target_idx = next_target_idx - 1
            
            # Ensure we don't move below frozen boundary
            target_idx = max(target_idx, frozen_boundary)
            
            # Protected: instructions BEFORE this one (0..i-1) AND already-processed DOWN instructions
            # This prevents chain movement from affecting instructions that have already been placed
            protected_instrs = all_target_instrs[:i] + down_processed_instrs
            
            if self.verbose:
                current_cycle = self._get_instruction_cycle_position(block, current_idx)
                print(f"  [{i}] (DOWN) Moving from idx={current_idx} (cycle={current_cycle}) to idx={target_idx} (ideal cycle={ideal_cycle})")
            
            original_idx = current_idx
            final_idx = self._move_instruction_toward(result, current_idx, target_idx, branch_boundary, frozen_boundary, protected_instrs, ideal_cycle)
            
            # Track if any move happened
            if final_idx != original_idx:
                self._moved_count += 1
            
            # Add this instruction to processed list for protection in subsequent moves
            down_processed_instrs.append(all_target_instrs[i])
            
            if self.verbose:
                # Use _find_nth_target to get actual position by count order (not object reference)
                # This is more accurate because chain movement may reorder target instructions
                actual_final_idx = self._find_nth_target(block, i)
                final_cycle = self._get_instruction_cycle_position(block, actual_final_idx)
                print(f"      Final position: idx={actual_final_idx} (cycle={final_cycle})")
        
        # 7. Move remaining M-K instructions toward the end (in reverse order since they're all moving down)
        if M > K:
            if self.verbose:
                print(f"Moving {M - K} remaining instructions to block end (in reverse order)")
            
            for i in reversed(range(K, M)):
                current_idx = self._find_nth_target(block, i)
                if current_idx < 0:
                    continue
                
                # Target is just before branch boundary
                target_idx = branch_boundary - 1
                
                # Ensure we don't go below already-placed instructions (next higher index that was placed)
                if i < M - 1:
                    next_target_idx = self._find_nth_target(block, i + 1)
                    if next_target_idx >= 0 and target_idx >= next_target_idx:
                        target_idx = next_target_idx - 1
                
                # Protected: only instructions before this one (0..i-1), not those after
                protected_instrs = all_target_instrs[:i]
                
                if self.verbose:
                    print(f"  [{i}] Moving from idx={current_idx} to end (target={target_idx})")
                
                original_idx = current_idx
                final_idx = self._move_instruction_toward(result, current_idx, target_idx, branch_boundary, frozen_boundary, protected_instrs)
                
                # Track if any move happened
                if final_idx != original_idx:
                    self._moved_count += 1
                
                if self.verbose:
                    print(f"      Final position: idx={final_idx}")
        
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
    verbose: bool = False,
    auto_fix_latency: bool = True
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
        auto_fix_latency: If True, automatically run InsertLatencyNopsPass after
                         distribution to fix any MFMA latency violations
        
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
    
    # # Add latency fix pass to ensure MFMA latency constraints are satisfied
    # if auto_fix_latency:
    #     pm.add_pass(InsertLatencyNopsPass())
    
    return pm.run_all(result)


# =============================================================================
# Register Replace Pass
# =============================================================================

@dataclass
class RegisterSegment:
    """Represents a contiguous segment of registers."""
    prefix: str  # 'v', 'a', or 's'
    start: int   # Starting index
    count: int   # Number of registers
    
    def get_registers(self) -> List[str]:
        """Get list of register names in this segment."""
        return [f"{self.prefix}{i}" for i in range(self.start, self.start + self.count)]
    
    def __str__(self) -> str:
        if self.count == 1:
            return f"{self.prefix}{self.start}"
        return f"{self.prefix}[{self.start}:{self.start + self.count - 1}]"


def parse_register_segment(reg_str: str) -> Optional[RegisterSegment]:
    """
    Parse a register segment string into a RegisterSegment.
    
    Supports formats:
    - "v40" -> single register
    - "v[40:45]" -> range of registers (inclusive)
    - "s[37:40]" -> range of registers (inclusive)
    
    Args:
        reg_str: Register segment string
        
    Returns:
        RegisterSegment or None if parsing fails
    """
    reg_str = reg_str.strip().lower()
    
    # Match range format: v[40:45], s[37:40], a[0:3]
    range_match = re.match(r'^([vsa])\[(\d+):(\d+)\]$', reg_str)
    if range_match:
        prefix = range_match.group(1)
        start = int(range_match.group(2))
        end = int(range_match.group(3))
        count = end - start + 1
        return RegisterSegment(prefix=prefix, start=start, count=count)
    
    # Match single register: v40, s37, a0
    single_match = re.match(r'^([vsa])(\d+)$', reg_str)
    if single_match:
        prefix = single_match.group(1)
        start = int(single_match.group(2))
        return RegisterSegment(prefix=prefix, start=start, count=1)
    
    return None


def find_aligned_free_registers(
    fgpr_set: Set[str],
    prefix: str,
    count: int,
    alignment: int
) -> Optional[int]:
    """
    Find the smallest aligned starting index from free registers.
    
    Args:
        fgpr_set: Set of free register names (e.g., {'v91', 'v92', 'v93', ...})
        prefix: Register prefix ('v', 'a', or 's')
        count: Number of consecutive registers needed
        alignment: Starting index must be divisible by this value
        
    Returns:
        Starting index if found, None otherwise
    """
    # Extract indices from the free set that match the prefix
    free_indices = sorted([
        int(r[1:]) for r in fgpr_set 
        if r.startswith(prefix) and r[1:].isdigit()
    ])
    
    if len(free_indices) < count:
        return None
    
    # Find the smallest aligned starting index with 'count' consecutive registers
    for start_idx in free_indices:
        # Check alignment
        if alignment > 1 and start_idx % alignment != 0:
            continue
        
        # Check if we have 'count' consecutive registers starting from start_idx
        consecutive = True
        for i in range(count):
            if (start_idx + i) not in free_indices:
                consecutive = False
                break
        
        if consecutive:
            return start_idx
    
    return None


# =============================================================================
# RegisterReplaceExecutor - Modular interface for register replacement
# =============================================================================

class RegisterReplaceExecutor:
    """
    Step executor for RegisterReplacePass.
    Provides per-instruction callbacks for debugging.
    """
    
    def __init__(self, result: AnalysisResult, context: RegisterReplaceContext):
        """
        Initialize the register replace executor.
        
        Args:
            result: The AnalysisResult to modify
            context: Precomputed register replace context
        """
        self.result = result
        self.ctx = context
        self._instructions_modified = 0
        self._instructions_skipped = 0
    
    @property
    def instructions_modified(self) -> int:
        """Get count of instructions modified."""
        return self._instructions_modified
    
    @property
    def instructions_skipped(self) -> int:
        """Get count of instructions skipped (not in target_opcodes)."""
        return self._instructions_skipped
    
    @staticmethod
    def create_context(
        result: AnalysisResult,
        range_start: int,
        range_end: int,
        registers_to_replace: List[str],
        alignments: Optional[List[int]] = None,
        target_opcodes: Optional[List[str]] = None
    ) -> Optional[RegisterReplaceContext]:
        """
        Create register replace context and compute register mapping.
        
        Args:
            result: The AnalysisResult
            range_start: Starting instruction address (global ID, inclusive)
            range_end: Ending instruction address (global ID, inclusive)
            registers_to_replace: List of register segment strings
            alignments: List of alignment values
            target_opcodes: List of opcodes to apply replacement to
            
        Returns:
            RegisterReplaceContext if successful, None on error
        """
        # Parse register segments
        segments = []
        for reg_str in registers_to_replace:
            segment = parse_register_segment(reg_str)
            if segment is None:
                return None
            segments.append(segment)
        
        if alignments is None:
            alignments = [1] * len(segments)
        
        if len(alignments) != len(segments):
            return None
        
        # Normalize target_opcodes
        target_opcodes_set = set(op.lower() for op in target_opcodes) if target_opcodes else set()
        
        # Get FGPR from CFG
        cfg = result.cfg
        if cfg.fgpr is None:
            from amdgcn_ddg import compute_register_statistics, compute_fgpr
            stats = compute_register_statistics(result.ddgs)
            fgpr_info = compute_fgpr(stats)
            cfg.fgpr = fgpr_info.to_dict()
            cfg.register_stats = stats.to_dict()
        
        fgpr_data = cfg.fgpr
        fgpr_v = set(fgpr_data['full_free']['vgpr'])
        fgpr_a = set(fgpr_data['full_free']['agpr'])
        fgpr_s = set(fgpr_data['full_free']['sgpr'])
        
        # Compute register mapping
        register_mapping = {}
        used_new_regs = set()
        
        for i, segment in enumerate(segments):
            alignment = alignments[i]
            
            # Select the appropriate FGPR set
            if segment.prefix == 'v':
                fgpr_set = fgpr_v - used_new_regs
            elif segment.prefix == 'a':
                fgpr_set = fgpr_a - used_new_regs
            else:  # 's'
                fgpr_set = fgpr_s - used_new_regs
            
            # Find aligned starting index
            new_start = find_aligned_free_registers(
                fgpr_set, segment.prefix, segment.count, alignment
            )
            
            if new_start is None:
                return None  # No free registers available
            
            # Build mapping for each register in the segment
            for j in range(segment.count):
                old_reg = f"{segment.prefix}{segment.start + j}"
                new_reg = f"{segment.prefix}{new_start + j}"
                register_mapping[old_reg] = new_reg
                used_new_regs.add(new_reg)
        
        return RegisterReplaceContext(
            range_start=range_start,
            range_end=range_end,
            segments=segments,
            alignments=alignments,
            target_opcodes=target_opcodes_set,
            register_mapping=register_mapping
        )
    
    def _replace_registers_in_string(self, text: str) -> str:
        """Replace registers in a string according to the mapping."""
        result = text
        
        # Sort by length descending to replace longer register names first
        # This prevents v10 from being replaced before v100
        sorted_mapping = sorted(
            self.ctx.register_mapping.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for old_reg, new_reg in sorted_mapping:
            # Use word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(old_reg) + r'\b'
            result = re.sub(pattern, new_reg, result)
        
        return result
    
    def replace_in_instruction(
        self,
        block_label: str,
        instr_idx: int
    ) -> bool:
        """
        Replace registers in a single instruction.
        
        Args:
            block_label: Label of the block
            instr_idx: Index of instruction in the block
            
        Returns:
            True if the instruction was modified
        """
        block = self.result.cfg.blocks[block_label]
        instr = block.instructions[instr_idx]
        
        # Check if instruction is in range
        if instr.address < self.ctx.range_start or instr.address > self.ctx.range_end:
            return False
        
        # Check if instruction matches target_opcodes (if specified)
        if self.ctx.target_opcodes and instr.opcode.lower() not in self.ctx.target_opcodes:
            self._instructions_skipped += 1
            return False
        
        # Replace in operands
        original_operands = instr.operands
        new_operands = self._replace_registers_in_string(original_operands)
        
        if new_operands != original_operands:
            instr.operands = new_operands
            
            # Update raw_line if present
            if instr.raw_line:
                instr.raw_line = self._replace_registers_in_string(instr.raw_line)
            
            self._instructions_modified += 1
            return True
        
        return False
    
    def execute_all(
        self,
        on_instruction: Optional[callable] = None
    ) -> int:
        """
        Execute replacement on all instructions in range.
        
        Args:
            on_instruction: Callback after each instruction.
                           Signature: (block_label: str, instr_idx: int, modified: bool) -> None
                           
        Returns:
            Number of instructions modified
        """
        self._instructions_modified = 0
        self._instructions_skipped = 0
        
        for block_label, block in self.result.cfg.blocks.items():
            for idx, instr in enumerate(block.instructions):
                # Check if in range
                if instr.address < self.ctx.range_start or instr.address > self.ctx.range_end:
                    continue
                
                modified = self.replace_in_instruction(block_label, idx)
                
                if on_instruction:
                    on_instruction(block_label, idx, modified)
        
        return self._instructions_modified


class RegisterReplacePass(Pass):
    """
    Pass that replaces a set of registers with free registers within a specified instruction range.
    
    This pass:
    1. Parses register segments from RPRS (Registers to Replace Set)
    2. Finds aligned free registers from FGPR (Free GPR set)
    3. Creates a mapping from old registers to new registers
    4. Replaces all occurrences in instructions within the specified range
    
    The pass does NOT change instruction order, only modifies operands.
    This means it's compatible with verify_optimization() which checks
    instruction ordering constraints.
    
    Attributes:
        range_start: Starting instruction address (global ID, inclusive)
        range_end: Ending instruction address (global ID, inclusive)
        registers_to_replace: List of register segment strings (e.g., ["v[40:45]", "s[37:40]"])
        alignments: List of alignment requirements for each segment
        target_opcodes: List of opcodes to apply replacement to (if empty, apply to all)
        verbose: Print detailed information during execution
    """
    
    def __init__(
        self,
        range_start: int,
        range_end: int,
        registers_to_replace: List[str],
        alignments: Optional[List[int]] = None,
        target_opcodes: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize the pass.
        
        Args:
            range_start: Starting instruction address (global ID, inclusive)
            range_end: Ending instruction address (global ID, inclusive)
            registers_to_replace: List of register segment strings
            alignments: List of alignment values (default: [1] * len(registers_to_replace))
            target_opcodes: List of opcodes to apply replacement to (e.g., ["v_lshl_add_u64", "global_load_dwordx4"]).
                           If empty or None, replacement applies to all instructions in range.
            verbose: Print detailed information
        """
        self.range_start = range_start
        self.range_end = range_end
        self.registers_to_replace = registers_to_replace
        self.alignments = alignments or [1] * len(registers_to_replace)
        # Normalize opcodes to lowercase for case-insensitive matching
        self.target_opcodes: Set[str] = set(op.lower() for op in target_opcodes) if target_opcodes else set()
        self.verbose = verbose
        
        # Computed during run
        self._register_mapping: Dict[str, str] = {}  # old_reg -> new_reg
        self._instructions_modified: int = 0
        self._instructions_skipped: int = 0  # Count of non-target instructions
        self._error_message: Optional[str] = None
    
    @property
    def name(self) -> str:
        return "RegisterReplacePass"
    
    @property
    def description(self) -> str:
        desc = f"Replace registers {self.registers_to_replace} in range [{self.range_start}, {self.range_end}]"
        if self.target_opcodes:
            desc += f" (only opcodes: {list(self.target_opcodes)})"
        return desc
    
    @property
    def register_mapping(self) -> Dict[str, str]:
        """Get the computed register mapping (old -> new)."""
        return self._register_mapping
    
    @property
    def error_message(self) -> Optional[str]:
        """Get error message if run failed."""
        return self._error_message
    
    def run(self, result: AnalysisResult) -> bool:
        """
        Execute the register replacement pass.
        
        Args:
            result: The AnalysisResult to modify
            
        Returns:
            True if any changes were made, False otherwise
        """
        self._register_mapping = {}
        self._instructions_modified = 0
        self._error_message = None
        
        cfg = result.cfg
        
        # Step 1: Parse register segments
        segments = []
        for reg_str in self.registers_to_replace:
            segment = parse_register_segment(reg_str)
            if segment is None:
                self._error_message = f"Invalid register segment: {reg_str}"
                if self.verbose:
                    print(f"Error: {self._error_message}")
                return False
            segments.append(segment)
        
        if len(self.alignments) != len(segments):
            self._error_message = f"Alignment count ({len(self.alignments)}) doesn't match segment count ({len(segments)})"
            if self.verbose:
                print(f"Error: {self._error_message}")
            return False
        
        if self.verbose:
            print(f"Parsed {len(segments)} register segments:")
            for i, seg in enumerate(segments):
                print(f"  [{i}] {seg} (alignment={self.alignments[i]})")
        
        # Step 2: Get FGPR from CFG
        if cfg.fgpr is None:
            # Compute FGPR if not already computed
            from amdgcn_ddg import compute_register_statistics, compute_fgpr
            stats = compute_register_statistics(result.ddgs)
            fgpr_info = compute_fgpr(stats)
            cfg.fgpr = fgpr_info.to_dict()
            cfg.register_stats = stats.to_dict()
        
        fgpr_data = cfg.fgpr
        fgpr_v = set(fgpr_data['full_free']['vgpr'])
        fgpr_a = set(fgpr_data['full_free']['agpr'])
        fgpr_s = set(fgpr_data['full_free']['sgpr'])
        
        if self.verbose:
            print(f"Free registers available: VGPR={len(fgpr_v)}, AGPR={len(fgpr_a)}, SGPR={len(fgpr_s)}")
        
        # Step 3: Find aligned free registers for each segment and build mapping
        used_new_regs = set()  # Track newly allocated registers
        
        for i, segment in enumerate(segments):
            alignment = self.alignments[i]
            
            # Select the appropriate FGPR set
            if segment.prefix == 'v':
                fgpr_set = fgpr_v - used_new_regs
            elif segment.prefix == 'a':
                fgpr_set = fgpr_a - used_new_regs
            else:  # 's'
                fgpr_set = fgpr_s - used_new_regs
            
            # Find aligned starting index
            new_start = find_aligned_free_registers(
                fgpr_set, segment.prefix, segment.count, alignment
            )
            
            if new_start is None:
                self._error_message = (
                    f"Insufficient free {segment.prefix.upper()}GPRs for segment {segment}: "
                    f"need {segment.count} consecutive registers with alignment {alignment}"
                )
                if self.verbose:
                    print(f"Error: {self._error_message}")
                return False
            
            # Build mapping for this segment
            old_regs = segment.get_registers()
            new_regs = [f"{segment.prefix}{new_start + j}" for j in range(segment.count)]
            
            for old_reg, new_reg in zip(old_regs, new_regs):
                self._register_mapping[old_reg] = new_reg
                used_new_regs.add(new_reg)
            
            if self.verbose:
                print(f"Segment {segment} -> {segment.prefix}[{new_start}:{new_start + segment.count - 1}]")
        
        if self.verbose:
            print(f"Register mapping ({len(self._register_mapping)} registers):")
            for old, new in sorted(self._register_mapping.items(), 
                                   key=lambda x: (x[0][0], int(x[0][1:]) if x[0][1:].isdigit() else 0)):
                print(f"  {old} -> {new}")
        
        # Step 4: Replace registers in instructions within the range
        for block_label in cfg.block_order:
            block = cfg.blocks[block_label]
            for instr in block.instructions:
                # Check if instruction is within the range
                if not (self.range_start <= instr.address <= self.range_end):
                    continue
                
                # Check if instruction opcode matches target_opcodes (if specified)
                if self.target_opcodes:
                    if instr.opcode.lower() not in self.target_opcodes:
                        self._instructions_skipped += 1
                        continue
                
                # Replace registers in operands
                modified = self._replace_registers_in_instruction(instr)
                if modified:
                    self._instructions_modified += 1
        
        if self.verbose:
            print(f"Modified {self._instructions_modified} instructions")
            if self._instructions_skipped > 0:
                print(f"Skipped {self._instructions_skipped} non-target instructions")
        
        # Step 5: Update DDG nodes to reflect register changes
        self._update_ddg_registers(result)
        
        return self._instructions_modified > 0
    
    def _replace_registers_in_instruction(self, instr: Instruction) -> bool:
        """
        Replace registers in a single instruction.
        
        Returns True if any replacement was made.
        """
        modified = False
        new_operands = instr.operands
        new_raw_line = instr.raw_line if instr.raw_line else ""
        
        # Sort by length (longer first) to avoid partial replacements
        # e.g., replace v40 before v4
        sorted_mappings = sorted(
            self._register_mapping.items(),
            key=lambda x: -len(x[0])
        )
        
        for old_reg, new_reg in sorted_mappings:
            # Replace in operands - use word boundary matching
            # Match patterns like v40, v[40:41], etc.
            old_pattern = re.compile(
                r'\b' + re.escape(old_reg) + r'(?=[\s,\]\):]|$)',
                re.IGNORECASE
            )
            
            if old_pattern.search(new_operands):
                new_operands = old_pattern.sub(new_reg, new_operands)
                modified = True
            
            # Also replace in raw_line if present
            if new_raw_line and old_pattern.search(new_raw_line):
                new_raw_line = old_pattern.sub(new_reg, new_raw_line)
        
        # Also handle register range format like v[40:45]
        # Need to update both start and end indices in ranges
        for old_reg, new_reg in sorted_mappings:
            old_idx = int(old_reg[1:])
            new_idx = int(new_reg[1:])
            prefix = old_reg[0]
            
            # Pattern for range like v[40:45] where old_idx is start or end
            def replace_range_idx(match):
                start = int(match.group(1))
                end = int(match.group(2))
                
                # Check if this range overlaps with our old registers
                new_start = start
                new_end = end
                
                if start == old_idx:
                    new_start = new_idx
                if end == old_idx:
                    new_end = new_idx
                
                if new_start != start or new_end != end:
                    return f"{prefix}[{new_start}:{new_end}]"
                return match.group(0)
            
            range_pattern = re.compile(
                r'\b' + prefix + r'\[(\d+):(\d+)\]',
                re.IGNORECASE
            )
            
            if range_pattern.search(new_operands):
                new_operands = range_pattern.sub(replace_range_idx, new_operands)
                modified = True
            
            if new_raw_line and range_pattern.search(new_raw_line):
                new_raw_line = range_pattern.sub(replace_range_idx, new_raw_line)
        
        if modified:
            instr.operands = new_operands
            if instr.raw_line:
                instr.raw_line = new_raw_line
        
        return modified
    
    def _update_ddg_registers(self, result: AnalysisResult) -> None:
        """Update DDG nodes to reflect register changes."""
        for block_label, ddg in result.ddgs.items():
            for node in ddg.nodes:
                # Only update nodes within the range
                if not (self.range_start <= node.instr.address <= self.range_end):
                    continue
                
                # Update defs
                new_defs = set()
                for reg in node.defs:
                    reg_lower = reg.lower()
                    if reg_lower in self._register_mapping:
                        new_defs.add(self._register_mapping[reg_lower])
                    else:
                        new_defs.add(reg)
                node.defs = new_defs
                
                # Update uses
                new_uses = set()
                for reg in node.uses:
                    reg_lower = reg.lower()
                    if reg_lower in self._register_mapping:
                        new_uses.add(self._register_mapping[reg_lower])
                    else:
                        new_uses.add(reg)
                node.uses = new_uses
                
                # Update available_regs
                new_avail = set()
                for reg in node.available_regs:
                    reg_lower = reg.lower()
                    if reg_lower in self._register_mapping:
                        new_avail.add(self._register_mapping[reg_lower])
                    else:
                        new_avail.add(reg)
                node.available_regs = new_avail


def replace_registers(
    result: AnalysisResult,
    range_start: int,
    range_end: int,
    registers_to_replace: List[str],
    alignments: Optional[List[int]] = None,
    target_opcodes: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[bool, Dict[str, str]]:
    """
    Convenience function to replace registers in an instruction range.
    
    Args:
        result: The AnalysisResult to modify
        range_start: Starting instruction address (global ID, inclusive)
        range_end: Ending instruction address (global ID, inclusive)
        registers_to_replace: List of register segment strings (e.g., ["v[40:45]", "s[37:40]"])
        alignments: List of alignment values (default: [1] * len(registers_to_replace))
        target_opcodes: List of opcodes to apply replacement to (e.g., ["v_lshl_add_u64", "global_load_dwordx4"]).
                       If empty or None, replacement applies to all instructions in range.
        verbose: Print progress information
        
    Returns:
        Tuple of (success, register_mapping)
    """
    pass_ = RegisterReplacePass(
        range_start=range_start,
        range_end=range_end,
        registers_to_replace=registers_to_replace,
        alignments=alignments,
        target_opcodes=target_opcodes,
        verbose=verbose
    )
    
    success = pass_.run(result)
    return success, pass_.register_mapping


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

