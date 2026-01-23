#!/usr/bin/env python3
"""
AMDGCN Instruction Scheduling Verification Module

This module provides verification for instruction scheduling passes to ensure
that the optimized code maintains all data dependencies and respects barrier
constraints.

Key Features:
- Verifies RAW (Read After Write) dependencies are preserved
- Verifies barrier constraints (s_barrier, s_cbranch_*) are not violated
- Supports cross-basic-block instruction movement verification
- Uses global line numbers (Instruction.address) as unique identifiers

Usage:
    # Standalone verification
    python amdgcn_verify.py original.amdgcn optimized.amdgcn
    
    # Programmatic use (automatically called by passes)
    from amdgcn_verify import build_global_ddg, verify_optimization
    original_gdg = build_global_ddg(cfg, ddgs)
    # ... perform optimization ...
    verify_optimization(original_gdg, optimized_cfg)  # Raises on failure
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from amdgcn_cfg import AMDGCNParser
from amdgcn_ddg import generate_all_ddgs, is_vm_op, is_lgkm_op


# =============================================================================
# Exceptions
# =============================================================================

class SchedulingVerificationError(Exception):
    """
    Exception raised when instruction scheduling verification fails.
    
    Contains detailed information about which constraints were violated.
    """
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Scheduling verification failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InstructionInfo:
    """
    Information about a single instruction for verification purposes.
    
    Attributes:
        address: Global line number in original source (unique identifier)
        block_label: Label of the containing basic block
        position: Index within the basic block
        opcode: Instruction mnemonic
        operands: Operand string
        defs: Set of registers written by this instruction
        uses: Set of registers read by this instruction
        is_barrier: True if this is a barrier instruction (s_barrier, s_cbranch_*)
        raw_line: Original line text
    """
    address: int
    block_label: str
    position: int
    opcode: str
    operands: str
    defs: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)
    is_barrier: bool = False
    raw_line: str = ""
    
    def __hash__(self):
        return hash(self.address)
    
    def __eq__(self, other):
        if isinstance(other, InstructionInfo):
            return self.address == other.address
        return False
    
    def __repr__(self):
        return f"InstructionInfo(addr={self.address}, op={self.opcode}, block={self.block_label})"


@dataclass
class BarrierRegion:
    """
    Represents a barrier and the instructions that must stay on each side.
    
    Attributes:
        barrier_addr: Address of the barrier instruction
        barrier_opcode: Opcode of the barrier (s_barrier, s_cbranch_*, etc.)
        before_addrs: Set of instruction addresses that must execute before the barrier
        after_addrs: Set of instruction addresses that must execute after the barrier
    """
    barrier_addr: int
    barrier_opcode: str
    before_addrs: Set[int] = field(default_factory=set)
    after_addrs: Set[int] = field(default_factory=set)


@dataclass
class AvailDependency:
    """
    Represents an AVAIL dependency: user_instr depends on s_waitcnt for data availability.
    
    Attributes:
        waitcnt_addr: Address of the s_waitcnt instruction
        user_addr: Address of the instruction using the available registers
        needed_regs: Set of registers that user_instr needs from s_waitcnt
    """
    waitcnt_addr: int
    user_addr: int
    needed_regs: Set[str] = field(default_factory=set)


@dataclass
class GlobalDependencyGraph:
    """
    Global dependency graph spanning all basic blocks.
    
    This structure captures all data dependencies and barrier constraints
    that must be preserved during instruction scheduling.
    
    Attributes:
        instructions: Mapping from address to InstructionInfo
        raw_edges: Set of RAW dependency edges as (writer_addr, reader_addr, register)
        barrier_regions: List of BarrierRegion objects
        avail_deps: List of AVAIL dependencies (s_waitcnt → user_instr with needed registers)
        block_order: Ordered list of block labels (for determining execution order)
    """
    instructions: Dict[int, InstructionInfo] = field(default_factory=dict)
    raw_edges: Set[Tuple[int, int, str]] = field(default_factory=set)
    barrier_regions: List[BarrierRegion] = field(default_factory=list)
    avail_deps: List[AvailDependency] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)
    
    def get_instruction_count(self) -> int:
        """Return total number of instructions."""
        return len(self.instructions)
    
    def get_raw_edge_count(self) -> int:
        """Return total number of RAW dependency edges."""
        return len(self.raw_edges)
    
    def get_barrier_count(self) -> int:
        """Return total number of barrier instructions."""
        return len(self.barrier_regions)
    
    def get_avail_dep_count(self) -> int:
        """Return total number of AVAIL dependencies."""
        return len(self.avail_deps)


@dataclass
class VerificationResult:
    """
    Result of verification, containing success status and any errors.
    
    Attributes:
        success: True if verification passed
        raw_violations: List of RAW dependency violations
        barrier_violations: List of barrier constraint violations
        avail_violations: List of AVAIL dependency violations (s_waitcnt data availability)
    """
    success: bool = True
    raw_violations: List[str] = field(default_factory=list)
    barrier_violations: List[str] = field(default_factory=list)
    avail_violations: List[str] = field(default_factory=list)
    
    def get_all_errors(self) -> List[str]:
        """Get all errors as a single list."""
        errors = []
        errors.extend(self.raw_violations)
        errors.extend(self.barrier_violations)
        errors.extend(self.avail_violations)
        return errors


# =============================================================================
# Barrier Instruction Detection
# =============================================================================

# Conditional branch instructions that act as barriers
CONDITIONAL_BRANCHES = {
    's_cbranch_scc0',
    's_cbranch_scc1',
    's_cbranch_vccz',
    's_cbranch_vccnz',
    's_cbranch_execz',
    's_cbranch_execnz',
    's_cbranch_cdbgsys',
    's_cbranch_cdbguser',
    's_cbranch_cdbgsys_or_user',
    's_cbranch_cdbgsys_and_user',
}

# Unconditional branch instructions
UNCONDITIONAL_BRANCHES = {
    's_branch',
    's_setpc_b64',
}

# Terminator instructions
TERMINATOR_INSTRUCTIONS = {
    's_endpgm',
    's_endpgm_saved',
}


def is_barrier_instruction(opcode: str, is_branch: bool = False, is_terminator: bool = False) -> bool:
    """
    Check if an instruction is a barrier that cannot be crossed.
    
    Barrier instructions include:
    - s_barrier (synchronization barrier)
    - s_cbranch_* (conditional branches)
    - s_branch (unconditional branch)
    - s_endpgm (program terminator)
    
    Args:
        opcode: The instruction opcode
        is_branch: Whether the instruction is marked as a branch
        is_terminator: Whether the instruction is marked as a terminator
        
    Returns:
        True if this is a barrier instruction
    """
    opcode_lower = opcode.lower()
    
    # Explicit barrier
    if opcode_lower == 's_barrier':
        return True
    
    # Conditional branches
    if opcode_lower in CONDITIONAL_BRANCHES:
        return True
    
    # Unconditional branches
    if opcode_lower in UNCONDITIONAL_BRANCHES:
        return True
    
    # Terminators
    if opcode_lower in TERMINATOR_INSTRUCTIONS:
        return True
    
    # Use flags if available
    if is_branch or is_terminator:
        return True
    
    return False


# =============================================================================
# Global DDG Construction
# =============================================================================

def build_global_ddg(cfg: 'CFG', ddgs: Dict[str, 'DDG']) -> GlobalDependencyGraph:
    """
    Build a global dependency graph from CFG and per-block DDGs.
    
    This function extracts:
    1. All instruction information with their addresses
    2. RAW dependency edges from DDGs
    3. Barrier regions with before/after instruction sets
    4. WAIT dependency edges from DDGs
    
    Args:
        cfg: The Control Flow Graph
        ddgs: Dictionary mapping block labels to their DDGs
        
    Returns:
        GlobalDependencyGraph containing all dependencies and constraints
    """
    gdg = GlobalDependencyGraph()
    gdg.block_order = list(cfg.block_order) if hasattr(cfg, 'block_order') else list(cfg.blocks.keys())
    
    # Pass 1: Collect all instruction information
    for block_label in gdg.block_order:
        if block_label not in cfg.blocks:
            continue
        block = cfg.blocks[block_label]
        ddg = ddgs.get(block_label)
        
        for pos, instr in enumerate(block.instructions):
            # Get defs and uses from DDG if available
            defs = set()
            uses = set()
            if ddg and pos < len(ddg.nodes):
                node = ddg.nodes[pos]
                defs = node.defs.copy()
                uses = node.uses.copy()
            
            info = InstructionInfo(
                address=instr.address,
                block_label=block_label,
                position=pos,
                opcode=instr.opcode,
                operands=instr.operands,
                defs=defs,
                uses=uses,
                is_barrier=is_barrier_instruction(
                    instr.opcode, 
                    getattr(instr, 'is_branch', False),
                    getattr(instr, 'is_terminator', False)
                ),
                raw_line=getattr(instr, 'raw_line', '')
            )
            gdg.instructions[instr.address] = info
    
    # Pass 2: Extract RAW dependency edges from DDGs
    for block_label, ddg in ddgs.items():
        if not ddg or not ddg.nodes:
            continue
        
        for edge in ddg.edges:
            from_id, to_id, dep_type = edge
            
            # Only process RAW dependencies
            if dep_type.startswith("RAW:"):
                reg = dep_type.split(":", 1)[1]
                
                # Get addresses from nodes
                if from_id < len(ddg.nodes) and to_id < len(ddg.nodes):
                    writer_addr = ddg.nodes[from_id].instr.address
                    reader_addr = ddg.nodes[to_id].instr.address
                    gdg.raw_edges.add((writer_addr, reader_addr, reg))
            
            # Extract AVAIL dependencies (s_waitcnt -> user_instr)
            # AVAIL edges represent: user_instr uses registers that s_waitcnt made available
            elif dep_type.startswith("AVAIL:"):
                reg = dep_type.split(":", 1)[1]
                
                if from_id < len(ddg.nodes) and to_id < len(ddg.nodes):
                    waitcnt_addr = ddg.nodes[from_id].instr.address
                    user_addr = ddg.nodes[to_id].instr.address
                    
                    # Check if we already have an AvailDependency for this pair
                    existing_dep = None
                    for dep in gdg.avail_deps:
                        if dep.waitcnt_addr == waitcnt_addr and dep.user_addr == user_addr:
                            existing_dep = dep
                            break
                    
                    if existing_dep:
                        existing_dep.needed_regs.add(reg)
                    else:
                        gdg.avail_deps.append(AvailDependency(
                            waitcnt_addr=waitcnt_addr,
                            user_addr=user_addr,
                            needed_regs={reg}
                        ))
    
    # Pass 3: Build barrier regions
    for block_label in gdg.block_order:
        if block_label not in cfg.blocks:
            continue
        block = cfg.blocks[block_label]
        
        for pos, instr in enumerate(block.instructions):
            if is_barrier_instruction(
                instr.opcode,
                getattr(instr, 'is_branch', False),
                getattr(instr, 'is_terminator', False)
            ):
                region = BarrierRegion(
                    barrier_addr=instr.address,
                    barrier_opcode=instr.opcode
                )
                
                # Instructions before the barrier in this block
                for i in range(pos):
                    region.before_addrs.add(block.instructions[i].address)
                
                # Instructions after the barrier in this block
                for i in range(pos + 1, len(block.instructions)):
                    region.after_addrs.add(block.instructions[i].address)
                
                gdg.barrier_regions.append(region)
    
    return gdg


# =============================================================================
# Position Comparison
# =============================================================================

def build_position_map(cfg: 'CFG') -> Dict[int, Tuple[str, int]]:
    """
    Build a mapping from instruction addresses to their current positions.
    
    Args:
        cfg: The (potentially optimized) CFG
        
    Returns:
        Dictionary mapping address -> (block_label, position_in_block)
    """
    position_map: Dict[int, Tuple[str, int]] = {}
    
    block_order = list(cfg.block_order) if hasattr(cfg, 'block_order') else list(cfg.blocks.keys())
    
    for block_label in block_order:
        if block_label not in cfg.blocks:
            continue
        block = cfg.blocks[block_label]
        
        for pos, instr in enumerate(block.instructions):
            position_map[instr.address] = (block_label, pos)
    
    return position_map


def get_block_order_index(block_label: str, block_order: List[str]) -> int:
    """
    Get the index of a block in the block order.
    
    Args:
        block_label: The block label to find
        block_order: The ordered list of block labels
        
    Returns:
        Index of the block, or -1 if not found
    """
    try:
        return block_order.index(block_label)
    except ValueError:
        return -1


def is_before(
    pos_a: Tuple[str, int],
    pos_b: Tuple[str, int],
    block_order: List[str]
) -> bool:
    """
    Determine if position A is executed before position B.
    
    For same-block comparison: compare positions directly.
    For cross-block comparison: use block order (assumes linear execution for now).
    
    Args:
        pos_a: (block_label, position) for instruction A
        pos_b: (block_label, position) for instruction B
        block_order: Ordered list of block labels
        
    Returns:
        True if A is executed before B
    """
    block_a, idx_a = pos_a
    block_b, idx_b = pos_b
    
    if block_a == block_b:
        return idx_a < idx_b
    
    # Cross-block: compare block order
    order_a = get_block_order_index(block_a, block_order)
    order_b = get_block_order_index(block_b, block_order)
    
    if order_a == -1 or order_b == -1:
        # Block not found in order, assume A is before B if A's index is smaller
        return order_a < order_b
    
    return order_a < order_b


# =============================================================================
# Verification
# =============================================================================

def verify_optimization(
    original_gdg: GlobalDependencyGraph,
    optimized_cfg: 'CFG',
    barrier_crossing_opcodes: Optional[Set[str]] = None
) -> None:
    """
    Verify that the optimized CFG maintains all dependencies from the original.
    
    This function checks:
    1. All RAW dependencies are preserved (writer before reader)
    2. All barrier constraints are preserved (no crossing, unless allowed)
    3. All WAIT dependencies are preserved
    
    Args:
        original_gdg: The global dependency graph from before optimization
        optimized_cfg: The CFG after optimization
        barrier_crossing_opcodes: Set of opcodes that are allowed to cross s_barrier.
                                 Instructions with these opcodes won't trigger barrier
                                 violations when they cross a barrier.
        
    Raises:
        SchedulingVerificationError: If any constraint is violated
    """
    result = VerificationResult()
    barrier_crossing_opcodes = barrier_crossing_opcodes or set()
    
    # Build position map for optimized code
    new_positions = build_position_map(optimized_cfg)
    block_order = list(optimized_cfg.block_order) if hasattr(optimized_cfg, 'block_order') else list(optimized_cfg.blocks.keys())
    
    # Check 1: Verify RAW dependencies
    for writer_addr, reader_addr, reg in original_gdg.raw_edges:
        # Skip if instruction was removed (shouldn't happen normally)
        if writer_addr not in new_positions:
            result.raw_violations.append(
                f"RAW violation: writer instruction @line{writer_addr} not found in optimized code"
            )
            result.success = False
            continue
        if reader_addr not in new_positions:
            result.raw_violations.append(
                f"RAW violation: reader instruction @line{reader_addr} not found in optimized code"
            )
            result.success = False
            continue
        
        writer_pos = new_positions[writer_addr]
        reader_pos = new_positions[reader_addr]
        
        if not is_before(writer_pos, reader_pos, block_order):
            # Get instruction info for better error message
            writer_info = original_gdg.instructions.get(writer_addr)
            reader_info = original_gdg.instructions.get(reader_addr)
            
            writer_desc = f"@line{writer_addr}"
            reader_desc = f"@line{reader_addr}"
            
            if writer_info:
                writer_desc = f"[{writer_info.opcode}] @line{writer_addr}"
            if reader_info:
                reader_desc = f"[{reader_info.opcode}] @line{reader_addr}"
            
            result.raw_violations.append(
                f"RAW violation: {writer_desc} must be before {reader_desc} (reg: {reg})"
            )
            result.success = False
    
    # Check 2: Verify barrier constraints
    for region in original_gdg.barrier_regions:
        barrier_addr = region.barrier_addr
        
        # Skip if barrier was removed
        if barrier_addr not in new_positions:
            result.barrier_violations.append(
                f"Barrier violation: barrier [{region.barrier_opcode}] @line{barrier_addr} not found"
            )
            result.success = False
            continue
        
        barrier_pos = new_positions[barrier_addr]
        
        # Check that "before" instructions are still before the barrier
        for before_addr in region.before_addrs:
            # Skip synthetic instructions (address <= 0)
            # These are inserted instructions (like s_nop) that don't need barrier checking
            if before_addr <= 0:
                continue
            
            if before_addr not in new_positions:
                continue  # Instruction might have been in a different block
            
            before_pos = new_positions[before_addr]
            
            if not is_before(before_pos, barrier_pos, block_order):
                before_info = original_gdg.instructions.get(before_addr)
                
                # Skip barrier check if this opcode is allowed to cross barriers
                if before_info and before_info.opcode in barrier_crossing_opcodes:
                    continue
                
                before_desc = f"@line{before_addr}"
                if before_info:
                    before_desc = f"[{before_info.opcode}] @line{before_addr}"
                
                result.barrier_violations.append(
                    f"Barrier violation: {before_desc} crossed after [{region.barrier_opcode}] @line{barrier_addr}"
                )
                result.success = False
        
        # Check that "after" instructions are still after the barrier
        for after_addr in region.after_addrs:
            # Skip synthetic instructions (address <= 0)
            if after_addr <= 0:
                continue
            
            if after_addr not in new_positions:
                continue  # Instruction might have been in a different block
            
            after_pos = new_positions[after_addr]
            
            if not is_before(barrier_pos, after_pos, block_order):
                after_info = original_gdg.instructions.get(after_addr)
                
                # Skip barrier check if this opcode is allowed to cross barriers
                if after_info and after_info.opcode in barrier_crossing_opcodes:
                    continue
                
                after_desc = f"@line{after_addr}"
                if after_info:
                    after_desc = f"[{after_info.opcode}] @line{after_addr}"
                
                result.barrier_violations.append(
                    f"Barrier violation: {after_desc} crossed before [{region.barrier_opcode}] @line{barrier_addr}"
                )
                result.success = False
    
    # Check 3: Verify s_waitcnt data availability (AVAIL verification)
    # 
    # This is an INDEPENDENT verification that does not rely on pre-extracted AVAIL edges.
    # Instead, it directly verifies that in the optimized code, every instruction that uses
    # memory operation results has a preceding s_waitcnt that waits for those results.
    #
    # Why not use original_gdg.avail_deps?
    # - AVAIL edges in the DDG are based on instruction ORDER when the DDG was built
    # - But instruction.address is the ORIGINAL line number from source file
    # - When DDG was built from modified code, these two don't match
    # - This causes false positives in verification
    #
    # Instead, we rebuild the DDG for optimized code and verify it's internally consistent:
    # - For each instruction that uses memory operation results (RAW dependency on mem op)
    # - There should be a s_waitcnt before it that has the needed registers in available_regs
    
    optimized_ddgs, _ = generate_all_ddgs(optimized_cfg, enable_cross_block_waitcnt=False)
    
    for block_label, ddg in optimized_ddgs.items():
        if not ddg or not ddg.nodes:
            continue
        
        # Build map: node_id -> set of available_regs from all preceding s_waitcnt
        # This tells us which memory operation results are available at each instruction
        cumulative_available: Dict[int, Set[str]] = {}
        available_so_far: Set[str] = set()
        
        for node in ddg.nodes:
            cumulative_available[node.node_id] = available_so_far.copy()
            if node.instr.opcode.lower() == 's_waitcnt':
                available_so_far = available_so_far.union(node.available_regs)
        
        # Check each instruction: if it uses a register written by a preceding memory operation,
        # that register must be in cumulative_available (meaning some s_waitcnt waited for it)
        #
        # Key insight: we only check if the MOST RECENT writer of the register is a memory op
        # If a non-memory op wrote the register after the memory op, no s_waitcnt is needed.
        
        # Track the most recent writer for each register as we scan through instructions
        recent_writer: Dict[str, Any] = {}  # reg -> (node, is_mem_op)
        
        for node in ddg.nodes:
            # Check uses before updating recent_writer
            for used_reg in node.uses:
                if used_reg in recent_writer:
                    writer_node, is_mem_op = recent_writer[used_reg]
                    
                    if is_mem_op:
                        # This instruction uses a register from a preceding memory operation
                        available_at_user = cumulative_available.get(node.node_id, set())
                        
                        if used_reg not in available_at_user:
                            # Violation: no s_waitcnt between memory op and user made this register available
                            result.avail_violations.append(
                                f"AVAIL coverage violation in {block_label}:\n"
                                f"  User: [{node.instr.opcode}] @line{node.instr.address} (idx={node.node_id}) uses {used_reg}\n"
                                f"  Memory op: [{writer_node.instr.opcode}] @line{writer_node.instr.address} (idx={writer_node.node_id}) writes {used_reg}\n"
                                f"  No s_waitcnt before user made {used_reg} available\n"
                                f"  Available registers at user: {available_at_user}"
                            )
                            result.success = False
            
            # Update recent_writer for registers defined by this instruction
            is_mem_op = is_vm_op(node.instr.opcode) or is_lgkm_op(node.instr.opcode)
            for def_reg in node.defs:
                recent_writer[def_reg] = (node, is_mem_op)
    
    # Raise exception if any violations found
    if not result.success:
        raise SchedulingVerificationError(result.get_all_errors())


def verify_and_report(
    original_gdg: GlobalDependencyGraph,
    optimized_cfg: 'CFG',
    verbose: bool = True,
    barrier_crossing_opcodes: Optional[Set[str]] = None
) -> VerificationResult:
    """
    Verify optimization and return detailed result (without raising exception).
    
    This is useful for CLI tools that want to report results without exceptions.
    
    Args:
        original_gdg: The global dependency graph from before optimization
        optimized_cfg: The CFG after optimization
        verbose: If True, print progress information
        barrier_crossing_opcodes: Set of opcodes that are allowed to cross s_barrier
        
    Returns:
        VerificationResult with success status and any errors
    """
    result = VerificationResult()
    barrier_crossing_opcodes = barrier_crossing_opcodes or set()
    
    new_positions = build_position_map(optimized_cfg)
    block_order = list(optimized_cfg.block_order) if hasattr(optimized_cfg, 'block_order') else list(optimized_cfg.blocks.keys())
    
    if verbose:
        print(f"Verifying {original_gdg.get_instruction_count()} instructions...")
        print(f"  RAW edges: {original_gdg.get_raw_edge_count()}")
        print(f"  Barriers: {original_gdg.get_barrier_count()}")
        if barrier_crossing_opcodes:
            print(f"  Barrier crossing opcodes: {barrier_crossing_opcodes}")
    
    # Same verification logic as verify_optimization but collect results
    for writer_addr, reader_addr, reg in original_gdg.raw_edges:
        if writer_addr not in new_positions or reader_addr not in new_positions:
            result.raw_violations.append(f"Missing instruction in RAW edge: {writer_addr} -> {reader_addr}")
            result.success = False
            continue
        
        writer_pos = new_positions[writer_addr]
        reader_pos = new_positions[reader_addr]
        
        if not is_before(writer_pos, reader_pos, block_order):
            writer_info = original_gdg.instructions.get(writer_addr)
            reader_info = original_gdg.instructions.get(reader_addr)
            
            writer_desc = f"[{writer_info.opcode}]" if writer_info else ""
            reader_desc = f"[{reader_info.opcode}]" if reader_info else ""
            
            result.raw_violations.append(
                f"RAW: {writer_desc}@line{writer_addr} must be before {reader_desc}@line{reader_addr} (reg: {reg})"
            )
            result.success = False
    
    for region in original_gdg.barrier_regions:
        barrier_addr = region.barrier_addr
        if barrier_addr not in new_positions:
            result.barrier_violations.append(f"Barrier @line{barrier_addr} not found")
            result.success = False
            continue
        
        barrier_pos = new_positions[barrier_addr]
        
        for before_addr in region.before_addrs:
            # Skip synthetic instructions (address <= 0)
            if before_addr <= 0:
                continue
            if before_addr not in new_positions:
                continue
            before_pos = new_positions[before_addr]
            if not is_before(before_pos, barrier_pos, block_order):
                before_info = original_gdg.instructions.get(before_addr)
                # Skip if opcode is allowed to cross barrier
                if before_info and before_info.opcode in barrier_crossing_opcodes:
                    continue
                result.barrier_violations.append(
                    f"@line{before_addr} crossed after barrier @line{barrier_addr}"
                )
                result.success = False
        
        for after_addr in region.after_addrs:
            # Skip synthetic instructions (address <= 0)
            if after_addr <= 0:
                continue
            if after_addr not in new_positions:
                continue
            after_pos = new_positions[after_addr]
            if not is_before(barrier_pos, after_pos, block_order):
                after_info = original_gdg.instructions.get(after_addr)
                # Skip if opcode is allowed to cross barrier
                if after_info and after_info.opcode in barrier_crossing_opcodes:
                    continue
                result.barrier_violations.append(
                    f"@line{after_addr} crossed before barrier @line{barrier_addr}"
                )
                result.success = False
    
    return result


# =============================================================================
# CLI Interface
# =============================================================================

def compare_before_after(original_path: str, optimized_path: str, verbose: bool = True) -> bool:
    """
    Compare original and optimized AMDGCN files for correctness.
    
    Args:
        original_path: Path to original .amdgcn file
        optimized_path: Path to optimized .amdgcn file
        verbose: Print detailed output
        
    Returns:
        True if verification passed, False otherwise
    """
    if verbose:
        print(f"=== AMDGCN Scheduling Verification ===")
        print(f"Original: {original_path}")
        print(f"Optimized: {optimized_path}")
        print()
    
    # Parse original file
    if verbose:
        print("Parsing original file...")
    parser = AMDGCNParser()
    original_cfg = parser.parse_file(original_path)
    original_ddgs, _ = generate_all_ddgs(original_cfg)
    
    # Build global DDG from original
    if verbose:
        print("Building dependency graph...")
    original_gdg = build_global_ddg(original_cfg, original_ddgs)
    
    # Parse optimized file
    if verbose:
        print("Parsing optimized file...")
    optimized_cfg = parser.parse_file(optimized_path)
    
    # Verify
    if verbose:
        print("Verifying constraints...")
        print()
    
    result = verify_and_report(original_gdg, optimized_cfg, verbose=False)
    
    # Print results
    if verbose:
        print(f"=== Verification Report ===")
        print(f"Instructions: {original_gdg.get_instruction_count()}")
        print(f"RAW Dependencies: {original_gdg.get_raw_edge_count()}")
        print(f"Barriers: {original_gdg.get_barrier_count()}")
        print()
    
    if result.success:
        if verbose:
            print("✓ All RAW dependencies preserved")
            print("✓ All barrier constraints respected")
            print()
            print("Result: PASS - Optimization is correct")
        return True
    else:
        if verbose:
            if result.raw_violations:
                print(f"✗ RAW violations ({len(result.raw_violations)}):")
                for v in result.raw_violations[:10]:  # Show first 10
                    print(f"    {v}")
                if len(result.raw_violations) > 10:
                    print(f"    ... and {len(result.raw_violations) - 10} more")
            
            if result.barrier_violations:
                print(f"✗ Barrier violations ({len(result.barrier_violations)}):")
                for v in result.barrier_violations[:10]:
                    print(f"    {v}")
                if len(result.barrier_violations) > 10:
                    print(f"    ... and {len(result.barrier_violations) - 10} more")
            
            if result.wait_violations:
                print(f"✗ WAIT violations ({len(result.wait_violations)}):")
                for v in result.wait_violations[:10]:
                    print(f"    {v}")
            
            print()
            print("Result: FAIL - Optimization has errors")
        return False


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify AMDGCN instruction scheduling correctness"
    )
    parser.add_argument(
        "original",
        help="Path to original .amdgcn file"
    )
    parser.add_argument(
        "optimized",
        help="Path to optimized .amdgcn file"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - only print errors"
    )
    
    args = parser.parse_args()
    
    success = compare_before_after(
        args.original,
        args.optimized,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

