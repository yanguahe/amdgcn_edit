#!/usr/bin/env python3
"""
AMDGCN Data Dependency Graph (DDG) Generator

This module extends the CFG parser to generate Data Dependency Graphs
for each basic block, showing instruction-level dependencies.

Features:
- Intra-block DDG: data dependencies within each basic block
- Inter-block dependencies: live-in/live-out data flow between blocks

Usage:
    python amdgcn_ddg.py <input.amdgcn> [--output-dir <dir>]
"""

import re
import os
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from pathlib import Path

from amdgcn_cfg import (
    AMDGCNParser, CFG, BasicBlock, Instruction,
    escape_dot_string, truncate_instruction
)


# =============================================================================
# AMDGCN Register Patterns
# =============================================================================

# SGPR (Scalar General Purpose Register) patterns
# s0, s1, s[0:1], s[0:3], etc.
SGPR_PATTERN = re.compile(r'\bs(\d+|\[\d+:\d+\])')

# VGPR (Vector General Purpose Register) patterns  
# v0, v1, v[0:1], v[0:3], etc.
VGPR_PATTERN = re.compile(r'\bv(\d+|\[\d+:\d+\])')

# AGPR (Accumulator General Purpose Register) patterns
# a0, a1, a[0:3], etc.
AGPR_PATTERN = re.compile(r'\ba(\d+|\[\d+:\d+\])')

# VCC (Vector Condition Code) - special register
VCC_PATTERN = re.compile(r'\bvcc\b')

# EXEC (Execution Mask) - special register
EXEC_PATTERN = re.compile(r'\bexec\b')

# SCC (Scalar Condition Code) - implicitly set by many scalar ops
SCC_PATTERN = re.compile(r'\bscc\b')


# =============================================================================
# Instruction Classification for Register Usage
# =============================================================================

# Instructions that implicitly write to SCC
# Based on AMD GCN ISA documentation:
# - Arithmetic with carry/overflow: s_add_*, s_addc_*, s_sub_*, s_subb_*, s_addk_i32, s_absdiff_i32
# - Comparison: s_cmp_*, s_cmpk_*, s_bitcmp*
# - Bitwise/logical (result != 0): s_and_*, s_or_*, s_xor_*, s_andn2_*, s_orn2_*, s_nand_*, s_nor_*, s_xnor_*, s_not_*
# - Shift (result != 0): s_lshl_*, s_lshr_*, s_ashr_*
# - Bit field (result != 0): s_bfe_*, s_bcnt*
# - Other: s_abs_i32, s_wqm_*, s_quadmask_*, s_*_saveexec_*, s_andn*_wrexec_*
#
# NOTE: s_mul_i32, s_mul_hi_*, s_mulk_i32 do NOT write SCC!
# NOTE: s_mov_*, s_cmov_*, s_cselect_* do NOT write SCC!
SCC_WRITERS = {
    # Arithmetic with carry/overflow
    's_add_i32', 's_add_u32', 's_addc_u32',
    's_sub_i32', 's_sub_u32', 's_subb_u32',
    's_add_lsh1_u32', 's_add_lsh2_u32', 's_add_lsh3_u32', 's_add_lsh4_u32',
    's_absdiff_i32', 's_addk_i32',
    
    # Comparison
    's_cmp_eq_i32', 's_cmp_eq_u32', 's_cmp_lg_i32', 's_cmp_lg_u32',
    's_cmp_gt_i32', 's_cmp_gt_u32', 's_cmp_ge_i32', 's_cmp_ge_u32',
    's_cmp_lt_i32', 's_cmp_lt_u32', 's_cmp_le_i32', 's_cmp_le_u32',
    's_cmp_eq_u64', 's_cmp_lg_u64',
    's_cmpk_eq_i32', 's_cmpk_ne_i32', 's_cmpk_lg_i32',
    's_cmpk_gt_i32', 's_cmpk_ge_i32', 's_cmpk_lt_i32', 's_cmpk_le_i32',
    's_cmpk_eq_u32', 's_cmpk_ne_u32', 's_cmpk_lg_u32',
    's_cmpk_gt_u32', 's_cmpk_ge_u32', 's_cmpk_lt_u32', 's_cmpk_le_u32',
    's_bitcmp0_b32', 's_bitcmp0_b64', 's_bitcmp1_b32', 's_bitcmp1_b64',
    
    # Min/Max (1 if S0 was min/max)
    's_min_i32', 's_min_u32', 's_max_i32', 's_max_u32',
    
    # Bitwise/logical (result != 0)
    's_and_b32', 's_and_b64', 's_or_b32', 's_or_b64', 's_xor_b32', 's_xor_b64',
    's_andn2_b32', 's_andn2_b64', 's_orn2_b32', 's_orn2_b64',
    's_nand_b32', 's_nand_b64', 's_nor_b32', 's_nor_b64',
    's_xnor_b32', 's_xnor_b64', 's_not_b32', 's_not_b64',
    
    # Shift (result != 0)
    's_lshl_b32', 's_lshl_b64', 's_lshr_b32', 's_lshr_b64',
    's_ashr_i32', 's_ashr_i64',
    
    # Bit field/count (result != 0)
    's_bfe_u32', 's_bfe_u64', 's_bfe_i32', 's_bfe_i64',
    's_bcnt0_i32_b32', 's_bcnt0_i32_b64', 's_bcnt1_i32_b32', 's_bcnt1_i32_b64',
    
    # Other (result != 0 or special)
    's_abs_i32',
    's_wqm_b32', 's_wqm_b64',
    's_quadmask_b32', 's_quadmask_b64',
    
    # Saveexec (exec != 0)
    's_and_saveexec_b64', 's_or_saveexec_b64', 's_xor_saveexec_b64',
    's_andn2_saveexec_b64', 's_orn2_saveexec_b64', 's_nand_saveexec_b64',
    's_nor_saveexec_b64', 's_xnor_saveexec_b64', 's_not_saveexec_b64',
    's_andn1_saveexec_b64', 's_orn1_saveexec_b64',
    's_andn1_wrexec_b64', 's_andn2_wrexec_b64',
}

# Instructions that do NOT write SCC (for clarity/documentation)
# These can be freely interleaved with s_add_u32/s_addc_u32 pairs
SCC_NON_WRITERS = {
    # Multiplication - no SCC
    's_mul_i32', 's_mul_hi_i32', 's_mul_hi_u32', 's_mulk_i32',
    
    # Data movement - no SCC
    's_mov_b32', 's_mov_b64', 's_cmov_b32', 's_cmov_b64',
    's_movk_i32', 's_cmovk_i32',
    's_movrels_b32', 's_movrels_b64', 's_movreld_b32', 's_movreld_b64',
    
    # Conditional select - no SCC (reads SCC but doesn't write)
    's_cselect_b32', 's_cselect_b64',
    
    # Pack - no SCC
    's_pack_ll_b32_b16', 's_pack_lh_b32_b16', 's_pack_hh_b32_b16',
    
    # Bit field mask/reverse - no SCC
    's_bfm_b32', 's_bfm_b64', 's_brev_b32', 's_brev_b64',
    's_bitset0_b32', 's_bitset0_b64', 's_bitset1_b32', 's_bitset1_b64',
    
    # Find bit - no SCC
    's_ff0_i32_b32', 's_ff0_i32_b64', 's_ff1_i32_b32', 's_ff1_i32_b64',
    's_flbit_i32_b32', 's_flbit_i32_b64', 's_flbit_i32', 's_flbit_i32_i64',
    
    # Sign extend - no SCC
    's_sext_i32_i8', 's_sext_i32_i16',
}

# Instructions that read SCC (and may also write it)
SCC_READERS = {
    's_cbranch_scc0', 's_cbranch_scc1',
    's_addc_u32', 's_subb_u32',  # Read carry in, also write carry out
    's_cselect_b32', 's_cselect_b64',  # Conditional select based on SCC (don't write SCC)
}

# Instructions that ONLY write SCC (don't read it)
# These can move past other SCC writers without breaking dependencies (WAW is OK)
# Key insight: s_add_u32 writes SCC but doesn't read it, so it can move past other SCC writers
SCC_ONLY_WRITERS = {
    # Arithmetic that produce carry/overflow (don't consume carry)
    's_add_i32', 's_add_u32',  # Note: s_addc_u32 READS SCC, so not here
    's_sub_i32', 's_sub_u32',  # Note: s_subb_u32 READS SCC, so not here
    's_add_lsh1_u32', 's_add_lsh2_u32', 's_add_lsh3_u32', 's_add_lsh4_u32',
    's_absdiff_i32', 's_addk_i32',
    
    # Comparison (only produce result in SCC)
    's_cmp_eq_i32', 's_cmp_eq_u32', 's_cmp_lg_i32', 's_cmp_lg_u32',
    's_cmp_gt_i32', 's_cmp_gt_u32', 's_cmp_ge_i32', 's_cmp_ge_u32',
    's_cmp_lt_i32', 's_cmp_lt_u32', 's_cmp_le_i32', 's_cmp_le_u32',
    's_cmp_eq_u64', 's_cmp_lg_u64',
    's_cmpk_eq_i32', 's_cmpk_ne_i32', 's_cmpk_lg_i32',
    's_cmpk_gt_i32', 's_cmpk_ge_i32', 's_cmpk_lt_i32', 's_cmpk_le_i32',
    's_cmpk_eq_u32', 's_cmpk_ne_u32', 's_cmpk_lg_u32',
    's_cmpk_gt_u32', 's_cmpk_ge_u32', 's_cmpk_lt_u32', 's_cmpk_le_u32',
    's_bitcmp0_b32', 's_bitcmp0_b64', 's_bitcmp1_b32', 's_bitcmp1_b64',
    
    # Min/Max (produce result != 0 in SCC)
    's_min_i32', 's_min_u32', 's_max_i32', 's_max_u32',
    
    # Bitwise/logical (produce result != 0 in SCC)
    's_and_b32', 's_and_b64', 's_or_b32', 's_or_b64', 's_xor_b32', 's_xor_b64',
    's_andn2_b32', 's_andn2_b64', 's_orn2_b32', 's_orn2_b64',
    's_nand_b32', 's_nand_b64', 's_nor_b32', 's_nor_b64',
    's_xnor_b32', 's_xnor_b64', 's_not_b32', 's_not_b64',
    
    # Shift (produce result != 0 in SCC)
    's_lshl_b32', 's_lshl_b64', 's_lshr_b32', 's_lshr_b64',
    's_ashr_i32', 's_ashr_i64',
    
    # Bit field/count (produce result != 0 in SCC)
    's_bfe_u32', 's_bfe_u64', 's_bfe_i32', 's_bfe_i64',
    's_bcnt0_i32_b32', 's_bcnt0_i32_b64', 's_bcnt1_i32_b32', 's_bcnt1_i32_b64',
    
    # Other (produce various results in SCC)
    's_abs_i32',
    's_wqm_b32', 's_wqm_b64',
    's_quadmask_b32', 's_quadmask_b64',
    
    # Saveexec (produce exec != 0 in SCC)
    's_and_saveexec_b64', 's_or_saveexec_b64', 's_xor_saveexec_b64',
    's_andn2_saveexec_b64', 's_orn2_saveexec_b64', 's_nand_saveexec_b64',
    's_nor_saveexec_b64', 's_xnor_saveexec_b64', 's_not_saveexec_b64',
    's_andn1_saveexec_b64', 's_orn1_saveexec_b64',
    's_andn1_wrexec_b64', 's_andn2_wrexec_b64',
}

# Instructions that both READ and WRITE SCC
# These need the incoming SCC value and also produce a new SCC value
SCC_READ_WRITE = {
    's_addc_u32',  # Add with carry: reads carry in, writes carry out
    's_subb_u32',  # Sub with borrow: reads borrow in, writes borrow out
}


def is_scc_only_writer(opcode: str) -> bool:
    """Check if instruction only writes SCC (doesn't read it)."""
    return opcode.lower() in SCC_ONLY_WRITERS


def is_scc_reader(opcode: str) -> bool:
    """Check if instruction reads SCC."""
    return opcode.lower() in SCC_READERS


def is_scc_writer(opcode: str) -> bool:
    """Check if instruction writes SCC."""
    return opcode.lower() in SCC_WRITERS


def is_dead_scc_write(instructions: List, instr_index: int) -> bool:
    """
    Check if an instruction's SCC write is dead (never read).
    
    An SCC write is dead if:
    1. The instruction writes SCC (regardless of whether it also reads SCC)
    2. The next instruction that has any SCC behavior only writes SCC (doesn't read it)
    
    This means the SCC value produced by this instruction is never used
    before being overwritten.
    
    Example 1:
        [47] s_lshl_b64 s[8:9], s[8:9], 1     ; writes SCC (dead)
        ... (no SCC read/write)
        [51] s_lshl_b64 s[0:1], s[0:1], 1     ; writes SCC (overwrites before read)
        
    Example 2:
        [19] s_addc_u32 s9, s73, s9           ; reads and writes SCC (SCC write is dead)
        ... (no SCC read)
        [23] s_cmpk_gt_i32 s23, 0x7f          ; writes SCC (overwrites before read)
        
    In both cases, the SCC write is dead.
    
    Args:
        instructions: List of Instruction objects in the block
        instr_index: Index of the instruction to check
        
    Returns:
        True if the SCC write is dead and should be removed from defs
    """
    if instr_index >= len(instructions):
        return False
    
    instr = instructions[instr_index]
    opcode = instr.opcode.lower()
    
    # Only check instructions that write SCC
    if not is_scc_writer(opcode):
        return False
    
    # Look forward for the next instruction that touches SCC
    for i in range(instr_index + 1, len(instructions)):
        next_instr = instructions[i]
        next_opcode = next_instr.opcode.lower()
        
        # Check if this instruction reads SCC
        if is_scc_reader(next_opcode):
            # Next SCC instruction reads SCC - write is not dead
            return False
        
        # Check if this instruction ONLY writes SCC (doesn't read it)
        # Use is_scc_only_writer, not is_scc_writer, because instructions
        # that both read and write SCC (like s_addc_u32) need the incoming SCC
        if is_scc_only_writer(next_opcode):
            # Next SCC instruction overwrites SCC without reading it
            # Current SCC write is dead
            return True
        
        # If next instruction both reads and writes SCC, the write is not dead
        if is_scc_writer(next_opcode):
            # This instruction reads SCC (it's in SCC_WRITERS but not SCC_ONLY_WRITERS)
            return False
    
    # No more SCC instructions found - SCC might be live-out
    # Be conservative and assume it's not dead
    return False

# Instructions that write to VCC
VCC_WRITERS = {
    'v_cmp_eq_f32', 'v_cmp_lt_f32', 'v_cmp_le_f32', 'v_cmp_gt_f32', 'v_cmp_ge_f32',
    'v_cmp_eq_i32', 'v_cmp_lt_i32', 'v_cmp_le_i32', 'v_cmp_gt_i32', 'v_cmp_ge_i32',
    'v_cmp_eq_u32', 'v_cmp_lt_u32', 'v_cmp_le_u32', 'v_cmp_gt_u32', 'v_cmp_ge_u32',
    'v_cmp_o_f32', 'v_cmp_u_f32',
    'v_add_co_u32', 'v_sub_co_u32', 'v_addc_co_u32', 'v_subb_co_u32',
    'v_div_scale_f32', 'v_div_scale_f64',
}

# Instructions that read VCC
VCC_READERS = {
    's_cbranch_vccz', 's_cbranch_vccnz',
    'v_cndmask_b32', 'v_addc_co_u32', 'v_subb_co_u32',
    'v_div_fmas_f32', 'v_div_fmas_f64',
}

# Instructions that write to EXEC
EXEC_WRITERS = {
    's_and_saveexec_b64', 's_or_saveexec_b64', 's_xor_saveexec_b64',
}

# Instructions that read EXEC
EXEC_READERS = {
    's_cbranch_execz', 's_cbranch_execnz',
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RegisterAccess:
    """Represents a register access (read or write)."""
    reg_name: str      # Normalized register name (e.g., "s0", "v[0:3]")
    is_write: bool     # True if write, False if read
    reg_type: str      # "sgpr", "vgpr", "agpr", "vcc", "exec", "scc"


@dataclass 
class InstructionNode:
    """Node in the Data Dependency Graph."""
    instr: Instruction
    node_id: int
    defs: Set[str] = field(default_factory=set)    # Registers written
    uses: Set[str] = field(default_factory=set)    # Registers read
    successors: List['InstructionNode'] = field(default_factory=list)
    predecessors: List['InstructionNode'] = field(default_factory=list)
    # For s_waitcnt: registers that become available after this instruction
    available_regs: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize instruction node to dictionary (excluding graph links)."""
        return {
            'node_id': self.node_id,
            'instr': self.instr.to_dict(),
            'defs': sorted(list(self.defs)),
            'uses': sorted(list(self.uses)),
            'available_regs': sorted(list(self.available_regs)),
            'successor_ids': [s.node_id for s in self.successors],
            'predecessor_ids': [p.node_id for p in self.predecessors],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], instr: Instruction) -> 'InstructionNode':
        """Deserialize instruction node from dictionary."""
        node = cls(
            instr=instr,
            node_id=data['node_id'],
            defs=set(data.get('defs', [])),
            uses=set(data.get('uses', [])),
            available_regs=set(data.get('available_regs', [])),
        )
        return node


@dataclass
class PendingMemOp:
    """
    Represents a pending memory operation waiting for data.
    Used to track the global state of memory operations across basic blocks.
    """
    regs: Set[str]           # Registers that will be defined when data arrives
    block_label: str         # Which block this operation originated from
    node_id: Optional[int]   # Node ID within the block (None if from predecessor)
    op_type: str             # 'lgkm' or 'vm'
    instr_text: str = ""     # Full instruction text for display
    
    def __hash__(self):
        return hash((self.block_label, self.node_id, self.op_type))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pending memory operation to dictionary."""
        return {
            'regs': sorted(list(self.regs)),
            'block_label': self.block_label,
            'node_id': self.node_id,
            'op_type': self.op_type,
            'instr_text': self.instr_text,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PendingMemOp':
        """Deserialize pending memory operation from dictionary."""
        return cls(
            regs=set(data.get('regs', [])),
            block_label=data['block_label'],
            node_id=data.get('node_id'),
            op_type=data['op_type'],
            instr_text=data.get('instr_text', ''),
        )


@dataclass
class DDG:
    """Data Dependency Graph for a basic block."""
    block_label: str
    nodes: List[InstructionNode] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)  # (from_id, to_id, dep_type)
    live_in: Set[str] = field(default_factory=set)   # Registers read before written
    live_out: Set[str] = field(default_factory=set)  # Registers written and potentially used later
    
    # Memory operation tracking for s_waitcnt dependencies
    lgkm_ops: List[InstructionNode] = field(default_factory=list)  # s_load_*, ds_* (all LDS instructions)
    vm_ops: List[InstructionNode] = field(default_factory=list)    # buffer_load, global_load
    
    # Pending memory ops queue at block entry (inherited from predecessors)
    # These are ORDERED - oldest operations first (FIFO queue)
    lgkm_pending_in: List[PendingMemOp] = field(default_factory=list)
    vm_pending_in: List[PendingMemOp] = field(default_factory=list)
    
    # Pending memory ops queue at block exit (to pass to successors)
    lgkm_pending_out: List[PendingMemOp] = field(default_factory=list)
    vm_pending_out: List[PendingMemOp] = field(default_factory=list)
    
    # Cross-block waitcnt availability edges: (waitcnt_node_id, user_node_id, reg)
    cross_block_avail_edges: List[Tuple[int, int, str]] = field(default_factory=list)
    # Registers waited by s_waitcnt from cross-block ops: waitcnt_node_id -> set of regs
    waitcnt_cross_block_regs: Dict[int, Set[str]] = field(default_factory=dict)
    # Cross-block memory ops waited by each s_waitcnt: waitcnt_node_id -> list of PendingMemOp
    waitcnt_cross_block_ops: Dict[int, List['PendingMemOp']] = field(default_factory=dict)
    
    def get_critical_path_length(self) -> int:
        """Calculate the length of the critical path."""
        if not self.nodes:
            return 0
        
        # Use topological sort to find longest path
        distances = {node.node_id: 0 for node in self.nodes}
        
        for node in self.nodes:
            for succ in node.successors:
                distances[succ.node_id] = max(
                    distances[succ.node_id],
                    distances[node.node_id] + 1
                )
        
        return max(distances.values()) if distances else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize DDG to dictionary."""
        return {
            'block_label': self.block_label,
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [[e[0], e[1], e[2]] for e in self.edges],
            'live_in': sorted(list(self.live_in)),
            'live_out': sorted(list(self.live_out)),
            'lgkm_ops_ids': [n.node_id for n in self.lgkm_ops],
            'vm_ops_ids': [n.node_id for n in self.vm_ops],
            'lgkm_pending_in': [op.to_dict() for op in self.lgkm_pending_in],
            'vm_pending_in': [op.to_dict() for op in self.vm_pending_in],
            'lgkm_pending_out': [op.to_dict() for op in self.lgkm_pending_out],
            'vm_pending_out': [op.to_dict() for op in self.vm_pending_out],
            'cross_block_avail_edges': [[e[0], e[1], e[2]] for e in self.cross_block_avail_edges],
            'waitcnt_cross_block_regs': {
                str(k): sorted(list(v)) for k, v in self.waitcnt_cross_block_regs.items()
            },
            'waitcnt_cross_block_ops': {
                str(k): [op.to_dict() for op in v] for k, v in self.waitcnt_cross_block_ops.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], block: BasicBlock) -> 'DDG':
        """Deserialize DDG from dictionary."""
        ddg = cls(block_label=data['block_label'])
        
        # Reconstruct nodes
        node_map: Dict[int, InstructionNode] = {}
        for node_data in data.get('nodes', []):
            node_id = node_data['node_id']
            # Find matching instruction from block
            if node_id < len(block.instructions):
                instr = block.instructions[node_id]
            else:
                # Fallback: create instruction from saved data
                instr = Instruction.from_dict(node_data['instr'])
            node = InstructionNode.from_dict(node_data, instr)
            ddg.nodes.append(node)
            node_map[node_id] = node
        
        # Reconstruct graph links
        for node_data in data.get('nodes', []):
            node_id = node_data['node_id']
            node = node_map.get(node_id)
            if node:
                for succ_id in node_data.get('successor_ids', []):
                    if succ_id in node_map:
                        node.successors.append(node_map[succ_id])
                for pred_id in node_data.get('predecessor_ids', []):
                    if pred_id in node_map:
                        node.predecessors.append(node_map[pred_id])
        
        # Edges
        ddg.edges = [tuple(e) for e in data.get('edges', [])]
        
        # Live-in/out
        ddg.live_in = set(data.get('live_in', []))
        ddg.live_out = set(data.get('live_out', []))
        
        # Memory ops (reference nodes by ID)
        for nid in data.get('lgkm_ops_ids', []):
            if nid in node_map:
                ddg.lgkm_ops.append(node_map[nid])
        for nid in data.get('vm_ops_ids', []):
            if nid in node_map:
                ddg.vm_ops.append(node_map[nid])
        
        # Pending memory ops
        ddg.lgkm_pending_in = [PendingMemOp.from_dict(d) for d in data.get('lgkm_pending_in', [])]
        ddg.vm_pending_in = [PendingMemOp.from_dict(d) for d in data.get('vm_pending_in', [])]
        ddg.lgkm_pending_out = [PendingMemOp.from_dict(d) for d in data.get('lgkm_pending_out', [])]
        ddg.vm_pending_out = [PendingMemOp.from_dict(d) for d in data.get('vm_pending_out', [])]
        
        # Cross-block avail edges
        ddg.cross_block_avail_edges = [tuple(e) for e in data.get('cross_block_avail_edges', [])]
        
        # Waitcnt cross-block regs
        ddg.waitcnt_cross_block_regs = {
            int(k): set(v) for k, v in data.get('waitcnt_cross_block_regs', {}).items()
        }
        
        # Waitcnt cross-block ops
        ddg.waitcnt_cross_block_ops = {
            int(k): [PendingMemOp.from_dict(op) for op in v] 
            for k, v in data.get('waitcnt_cross_block_ops', {}).items()
        }
        
        return ddg


@dataclass
class InterBlockDep:
    """Dependency between basic blocks."""
    from_block: str
    to_block: str
    registers: Set[str]  # Registers that flow between blocks
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'from_block': self.from_block,
            'to_block': self.to_block,
            'registers': sorted(list(self.registers)),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterBlockDep':
        """Deserialize from dictionary."""
        return cls(
            from_block=data['from_block'],
            to_block=data['to_block'],
            registers=set(data.get('registers', [])),
        )


@dataclass
class WaitcntInterBlockDep:
    """Cross-block memory dependency for s_waitcnt."""
    from_block: str
    to_block: str
    lgkm_count: int  # Number of LGKM ops from this predecessor
    vm_count: int    # Number of VM ops from this predecessor
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'from_block': self.from_block,
            'to_block': self.to_block,
            'lgkm_count': self.lgkm_count,
            'vm_count': self.vm_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaitcntInterBlockDep':
        """Deserialize from dictionary."""
        return cls(
            from_block=data['from_block'],
            to_block=data['to_block'],
            lgkm_count=data.get('lgkm_count', 0),
            vm_count=data.get('vm_count', 0),
        )


# =============================================================================
# Register Parsing
# =============================================================================

def expand_register_range(reg_str: str) -> List[str]:
    """
    Expand a register range into individual registers.
    e.g., "s[0:3]" -> ["s0", "s1", "s2", "s3"]
    """
    match = re.match(r'([sva])(\d+)$', reg_str)
    if match:
        return [reg_str]
    
    match = re.match(r'([sva])\[(\d+):(\d+)\]', reg_str)
    if match:
        prefix = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        return [f"{prefix}{i}" for i in range(start, end + 1)]
    
    return [reg_str]


def parse_operand_registers(operand: str) -> Tuple[Set[str], str]:
    """
    Parse a single operand and return the registers it references.
    Returns (set of register names, register type).
    """
    regs = set()
    reg_type = "unknown"
    
    # Skip if it's clearly an immediate value or offset keyword
    if operand in ('off', 'offen', 'idxen', 'offset'):
        return regs, reg_type
    
    # Check for SGPR
    for match in SGPR_PATTERN.finditer(operand):
        reg_part = match.group(1)
        if reg_part.startswith('['):
            regs.update(expand_register_range(f"s{reg_part}"))
        else:
            regs.add(f"s{reg_part}")
        reg_type = "sgpr"
    
    # Check for VGPR
    for match in VGPR_PATTERN.finditer(operand):
        reg_part = match.group(1)
        if reg_part.startswith('['):
            regs.update(expand_register_range(f"v{reg_part}"))
        else:
            regs.add(f"v{reg_part}")
        reg_type = "vgpr"
    
    # Check for AGPR
    for match in AGPR_PATTERN.finditer(operand):
        reg_part = match.group(1)
        if reg_part.startswith('['):
            regs.update(expand_register_range(f"a{reg_part}"))
        else:
            regs.add(f"a{reg_part}")
        reg_type = "agpr"
    
    # Check for VCC
    if VCC_PATTERN.search(operand):
        regs.add("vcc")
        reg_type = "vcc"
    
    # Check for EXEC
    if EXEC_PATTERN.search(operand):
        regs.add("exec")
        reg_type = "exec"
    
    return regs, reg_type


def parse_instruction_registers(instr: Instruction) -> Tuple[Set[str], Set[str]]:
    """
    Parse an instruction to determine which registers it defines (writes)
    and which it uses (reads).
    
    AMDGCN instruction format: opcode dst, src1, src2, ...
    - First operand is typically the destination (written)
    - Remaining operands are sources (read)
    
    Returns: (defs, uses)
    """
    opcode = instr.opcode.lower()
    operands = instr.operands
    
    defs = set()
    uses = set()
    
    # Split operands by comma, handling nested brackets
    operand_list = []
    current = ""
    bracket_depth = 0
    for char in operands:
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == ',' and bracket_depth == 0:
            operand_list.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        operand_list.append(current.strip())
    
    # ==== MFMA Instructions ====
    # v_mfma_f32_16x16x16_bf16 a[0:3], v[88:89], v[56:57], 0
    # Format: dst (AGPR), src0, src1, src2 (accumulator or 0)
    if opcode.startswith('v_mfma'):
        if operand_list:
            # First operand is always destination (AGPR)
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            # Rest are sources
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== Global/Buffer Load Instructions ====
    # global_load_dwordx4 v[172:175], v[36:37], off
    # Format: dst, addr, offset
    if opcode.startswith('global_load') or opcode.startswith('buffer_load'):
        if operand_list:
            # First operand is destination
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            # Rest are sources (address registers)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== Global/Buffer Store Instructions ====
    # global_store_dwordx4 v[36:37], v[88:91], off
    # Format: addr, data, offset - all are reads
    if opcode.startswith('global_store') or opcode.startswith('buffer_store'):
        for op in operand_list:
            src_regs, _ = parse_operand_registers(op)
            uses.update(src_regs)
        return defs, uses
    
    # ==== Scalar Load Instructions ====
    # s_load_dwordx2 s[2:3], s[0:1], 0x0
    # Format: dst, base, offset
    if opcode.startswith('s_load'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== DS Read Instructions ====
    # ds_read_b128 v[56:59], v14
    # Format: dst, addr
    if opcode.startswith('ds_read'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== DS Write Instructions ====
    # ds_write_b128 v6, v[2:5]
    # Format: addr, data - all are reads
    if opcode.startswith('ds_write'):
        for op in operand_list:
            src_regs, _ = parse_operand_registers(op)
            uses.update(src_regs)
        return defs, uses
    
    # ==== DS Permute/Swizzle Instructions ====
    # ds_bpermute_b32 v53, v28, v36
    # ds_swizzle_b32 v53, v36 offset:...
    # Format: dst, addr/src, src
    if opcode.startswith('ds_bpermute') or opcode.startswith('ds_swizzle'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== ACCVGPR Read/Write ====
    # v_accvgpr_read_b32 v36, a0
    if opcode.startswith('v_accvgpr_read'):
        if len(operand_list) >= 2:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            src_regs, _ = parse_operand_registers(operand_list[1])
            uses.update(src_regs)
        return defs, uses
    
    # v_accvgpr_write_b32 a0, v36
    if opcode.startswith('v_accvgpr_write'):
        if len(operand_list) >= 2:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            src_regs, _ = parse_operand_registers(operand_list[1])
            uses.update(src_regs)
        return defs, uses
    
    # ==== No-op and Sync Instructions ====
    if opcode in ('s_waitcnt', 's_barrier', 's_nop', 's_endpgm', 's_endpgm_saved'):
        return defs, uses
    
    # ==== Branch Instructions ====
    if opcode.startswith('s_branch') or opcode.startswith('s_cbranch'):
        if opcode in SCC_READERS:
            uses.add("scc")
        if opcode in VCC_READERS:
            uses.add("vcc")
        if opcode in EXEC_READERS:
            uses.add("exec")
        return defs, uses
    
    # ==== Scalar Compare Instructions ====
    # s_cmp_lt_i32 s69, s17 - writes SCC
    if opcode.startswith('s_cmp') or opcode.startswith('s_cmpk'):
        defs.add("scc")
        for op in operand_list:
            src_regs, _ = parse_operand_registers(op)
            uses.update(src_regs)
        return defs, uses
    
    # ==== Vector Compare Instructions ====
    # v_cmp_gt_i32_e64 s[24:25], s17, v46 - explicit dest
    # v_cmp_eq_u32_e32 vcc, 0, v1 - implicit VCC dest
    if opcode.startswith('v_cmp'):
        # Check if there's an explicit destination (e64 encoding with SGPR dest)
        if '_e64' in opcode and operand_list and operand_list[0].startswith('s'):
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        else:
            # Implicit VCC destination
            defs.add("vcc")
            for op in operand_list:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== Conditional Move ====
    # v_cndmask_b32_e64 v37, v33, v36, s[0:1]
    # v_cndmask_b32_e32 v4, v14, v9, vcc - implicit VCC
    if opcode.startswith('v_cndmask'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        # If no explicit mask operand, uses VCC
        if '_e32' in opcode or (len(operand_list) <= 3):
            uses.add("vcc")
        return defs, uses
    
    # ==== Save Exec Instructions ====
    # s_and_saveexec_b64 s[0:1], vcc
    if opcode.startswith('s_') and 'saveexec' in opcode:
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        defs.add("exec")
        uses.add("exec")
        return defs, uses
    
    # ==== V_READFIRSTLANE / V_READLANE ====
    # v_readfirstlane_b32 s8, v200
    if opcode.startswith('v_readfirstlane') or opcode.startswith('v_readlane'):
        if len(operand_list) >= 2:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== V_WRITELANE ====
    # v_writelane_b32 v0, s0, 0
    if opcode.startswith('v_writelane'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # ==== PK (Packed) Instructions ====
    # v_pk_fma_f32 v[16:17], v[16:17], v[22:23], v[38:39] op_sel_hi:[1,0,1]
    # v_pk_mul_f32 v[4:5], v[0:1], v[16:17] op_sel_hi:[0,1]
    if opcode.startswith('v_pk_'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                # Strip op_sel modifiers before parsing registers
                # e.g., "v[38:39] op_sel_hi:[1,0,1]" -> "v[38:39]"
                op_clean = op.split(' op_sel')[0].strip() if ' op_sel' in op else op
                src_regs, _ = parse_operand_registers(op_clean)
                uses.update(src_regs)
        return defs, uses
    
    # ==== Division Instructions ====
    # v_div_scale_f32 v0, s[6:7], v37, v37, 1.0 - writes both VGPR and SGPR (VCC-like)
    if opcode.startswith('v_div_scale'):
        if len(operand_list) >= 2:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            dest_regs2, _ = parse_operand_registers(operand_list[1])
            defs.update(dest_regs2)
            for op in operand_list[2:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        return defs, uses
    
    # v_div_fmas_f32 v0, v0, v1, v3 - reads VCC
    if opcode.startswith('v_div_fmas'):
        if operand_list:
            dest_regs, _ = parse_operand_registers(operand_list[0])
            defs.update(dest_regs)
            for op in operand_list[1:]:
                src_regs, _ = parse_operand_registers(op)
                uses.update(src_regs)
        uses.add("vcc")
        return defs, uses
    
    # ==== Default: First operand is dest, rest are sources ====
    # This handles most ALU instructions like:
    # v_add_f32, v_mul_f32, v_fma_f32, v_fmac_f32, s_mov_b32, etc.
    if operand_list:
        # First operand is destination
        dest_regs, _ = parse_operand_registers(operand_list[0])
        defs.update(dest_regs)
        
        # Remaining operands are sources
        for op in operand_list[1:]:
            src_regs, _ = parse_operand_registers(op)
            uses.update(src_regs)
    
    # Add implicit register accesses based on opcode
    if opcode in SCC_WRITERS:
        defs.add("scc")
    if opcode in SCC_READERS:
        uses.add("scc")
    if opcode in VCC_WRITERS:
        defs.add("vcc")
    if opcode in VCC_READERS:
        uses.add("vcc")
    
    return defs, uses


# =============================================================================
# s_waitcnt Parsing
# =============================================================================

def parse_waitcnt_operands(operands: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse s_waitcnt operands to extract vmcnt and lgkmcnt values.
    
    Examples:
        "vmcnt(0)"           -> (0, None)
        "lgkmcnt(0)"         -> (None, 0)
        "vmcnt(2) lgkmcnt(1)" -> (2, 1)
        "0"                   -> (0, 0)  # Wait for all
    
    Returns:
        (vmcnt, lgkmcnt) - None means the counter is not specified
    """
    vmcnt = None
    lgkmcnt = None
    
    # Check for explicit vmcnt(N)
    vm_match = re.search(r'vmcnt\s*\(\s*(\d+)\s*\)', operands)
    if vm_match:
        vmcnt = int(vm_match.group(1))
    
    # Check for explicit lgkmcnt(N)
    lgkm_match = re.search(r'lgkmcnt\s*\(\s*(\d+)\s*\)', operands)
    if lgkm_match:
        lgkmcnt = int(lgkm_match.group(1))
    
    # If operand is just "0", it means wait for all
    if operands.strip() == '0':
        vmcnt = 0
        lgkmcnt = 0
    
    return vmcnt, lgkmcnt


def is_lgkm_op(opcode: str) -> bool:
    """
    Check if instruction is an LGKM memory operation.
    
    LGKM operations include:
    - s_load_* : Scalar memory load instructions
    - s_store_* : Scalar memory store instructions
    - ds_*     : All LDS (Local Data Share) instructions
    
    These are monitored by lgkmcnt in s_waitcnt.
    """
    op_lower = opcode.lower()
    return (op_lower.startswith('s_load_') or 
            op_lower.startswith('s_store_') or 
            op_lower.startswith('ds_'))


def is_vm_op(opcode: str) -> bool:
    """
    Check if instruction is a VM memory operation.
    
    VM operations include:
    - buffer_load_* : Buffer load instructions
    - buffer_store_* : Buffer store instructions
    - global_load_* : Global load instructions
    - global_store_* : Global store instructions
    
    These are monitored by vmcnt in s_waitcnt.
    """
    op_lower = opcode.lower()
    return (op_lower.startswith('buffer_load') or 
            op_lower.startswith('buffer_store') or
            op_lower.startswith('global_load') or
            op_lower.startswith('global_store'))


# =============================================================================
# Cycle Detection for Cross-Block Waitcnt Dependencies
# =============================================================================

def detect_back_edge_predecessors(cfg: 'CFG', block_label: str) -> Set[str]:
    """
    Detect predecessors that are part of a back-edge (cycle) to this block.
    
    A predecessor is a back-edge predecessor if starting from that predecessor
    and recursively following its predecessors, we can reach back to block_label.
    
    This is used to exclude cyclic dependencies when computing s_waitcnt
    cross-block dependencies.
    
    Args:
        cfg: The control flow graph
        block_label: The target block label
        
    Returns:
        Set of predecessor labels that form back-edges to this block
    """
    block = cfg.blocks.get(block_label)
    if not block:
        return set()
    
    predecessors = block.predecessors if hasattr(block, 'predecessors') else []
    
    # For blocks with in-degree <= 1, no cycle handling needed
    if len(predecessors) <= 1:
        return set()
    
    back_edge_preds = set()
    
    for pred_label in predecessors:
        # Check if following this predecessor leads back to block_label
        if can_reach_block(cfg, pred_label, block_label, set()):
            back_edge_preds.add(pred_label)
    
    return back_edge_preds


def can_reach_block(cfg: 'CFG', start_label: str, target_label: str, visited: Set[str]) -> bool:
    """
    Check if we can reach target_label by recursively following predecessors from start_label.
    
    Args:
        cfg: The control flow graph
        start_label: Starting block label
        target_label: Target block label we're trying to reach
        visited: Set of already visited blocks (to prevent infinite recursion)
        
    Returns:
        True if target_label is reachable via predecessors
    """
    if start_label == target_label:
        return True
    
    if start_label in visited:
        return False
    
    visited.add(start_label)
    
    block = cfg.blocks.get(start_label)
    if not block:
        return False
    
    predecessors = block.predecessors if hasattr(block, 'predecessors') else []
    
    for pred_label in predecessors:
        if can_reach_block(cfg, pred_label, target_label, visited.copy()):
            return True
    
    return False


# =============================================================================
# DDG Construction
# =============================================================================

def build_ddg(block: BasicBlock,
              lgkm_pending_in: Optional[List[PendingMemOp]] = None,
              vm_pending_in: Optional[List[PendingMemOp]] = None) -> DDG:
    """
    Build a Data Dependency Graph for a basic block.
    
    Dependencies are created for:
    - RAW (Read After Write): instruction B reads register that A wrote
    - WAW (Write After Write): instruction B writes register that A wrote
    - WAIT: s_waitcnt dependencies on previous memory operations
    - AVAIL: registers that become available after s_waitcnt
    
    The pending memory operation queues are ORDERED (FIFO):
    - Oldest operations are at the front (index 0)
    - New operations are appended at the end
    - s_waitcnt clears operations from the front
    
    For example:
    - Queue has 8 VM ops: [op0, op1, op2, op3, op4, op5, op6, op7]
    - s_waitcnt vmcnt(3) means allow 3 ops pending
    - Need to wait for 8 - 3 = 5 oldest ops: op0, op1, op2, op3, op4
    - Remaining queue: [op5, op6, op7]
    
    Args:
        block: The basic block to analyze
        lgkm_pending_in: Pending LGKM ops from predecessors (FIFO order)
        vm_pending_in: Pending VM ops from predecessors (FIFO order)
    """
    ddg = DDG(block_label=block.label)
    
    # Initialize pending queues from predecessors
    lgkm_pending_in = lgkm_pending_in or []
    vm_pending_in = vm_pending_in or []
    
    # Store copies for the DDG
    ddg.lgkm_pending_in = [PendingMemOp(op.regs.copy(), op.block_label, op.node_id, op.op_type, op.instr_text) 
                          for op in lgkm_pending_in]
    ddg.vm_pending_in = [PendingMemOp(op.regs.copy(), op.block_label, op.node_id, op.op_type, op.instr_text) 
                        for op in vm_pending_in]
    
    # Track which registers are used before being defined (live-in)
    defined_in_block = set()
    
    # Create nodes for all instructions
    nodes = []
    for i, instr in enumerate(block.instructions):
        defs, uses = parse_instruction_registers(instr)
        
        # Optimize: Remove dead SCC writes
        # If an instruction writes SCC but doesn't read it, and the next
        # instruction that touches SCC also only writes it (doesn't read),
        # then the SCC write is dead and should be removed from defs.
        if 'scc' in defs and is_dead_scc_write(block.instructions, i):
            defs = defs - {'scc'}
        
        node = InstructionNode(
            instr=instr,
            node_id=i,
            defs=defs,
            uses=uses
        )
        nodes.append(node)
        
        # Track live-in: used before defined
        for reg in uses:
            if reg not in defined_in_block:
                ddg.live_in.add(reg)
        
        # Track defined registers
        defined_in_block.update(defs)
        
        # Track memory operations
        opcode = instr.opcode.lower()
        if is_lgkm_op(opcode):
            ddg.lgkm_ops.append(node)
        elif is_vm_op(opcode):
            ddg.vm_ops.append(node)
    
    ddg.nodes = nodes
    
    # Live-out: all registers defined in the block that might be used later
    ddg.live_out = defined_in_block.copy()
    
    # Build dependency edges
    # Track last writer for each register (for RAW dependencies)
    last_writer: Dict[str, InstructionNode] = {}
    
    # Working copies of pending queues (these will be modified as we process)
    # Each element is either a PendingMemOp (cross-block) or an InstructionNode (intra-block)
    lgkm_queue: List[Union[PendingMemOp, InstructionNode]] = list(lgkm_pending_in)
    vm_queue: List[Union[PendingMemOp, InstructionNode]] = list(vm_pending_in)
    
    for node in nodes:
        opcode = node.instr.opcode.lower()
        
        # RAW dependencies: current instruction reads what previous wrote
        for reg in node.uses:
            if reg in last_writer:
                writer = last_writer[reg]
                if writer not in node.predecessors:
                    node.predecessors.append(writer)
                    writer.successors.append(node)
                    ddg.edges.append((writer.node_id, node.node_id, f"RAW:{reg}"))
        
        # Note: WAW (Write After Write) dependencies are NOT created.
        # Only RAW (Read After Write) dependencies matter for correctness.
        # WAW edges are unnecessary because:
        #   - If instruction A writes reg and instruction B also writes reg,
        #     B will simply overwrite A's value
        #   - The ordering constraint only matters when someone READS the value
        
        # Handle s_waitcnt dependencies
        if opcode == 's_waitcnt':
            vmcnt, lgkmcnt = parse_waitcnt_operands(node.instr.operands)
            
            # Collect registers that become available after this waitcnt
            available_regs: Set[str] = set()
            # Collect cross-block registers that become available
            cross_block_avail_regs: Set[str] = set()
            
            # Track cross-block ops waited by this s_waitcnt
            cross_block_waited_ops: List[PendingMemOp] = []
            
            # LGKM dependencies: wait for oldest (queue_len - lgkmcnt) LGKM ops
            if lgkmcnt is not None and len(lgkm_queue) > lgkmcnt:
                wait_count = len(lgkm_queue) - lgkmcnt
                
                for pending_op in lgkm_queue[:wait_count]:
                    if isinstance(pending_op, InstructionNode):
                        # Intra-block memory op
                        if pending_op not in node.predecessors:
                            node.predecessors.append(pending_op)
                            pending_op.successors.append(node)
                            ddg.edges.append((pending_op.node_id, node.node_id, f"WAIT:lgkm"))
                        available_regs.update(pending_op.defs)
                    else:
                        # Cross-block memory op (PendingMemOp)
                        cross_block_avail_regs.update(pending_op.regs)
                        cross_block_waited_ops.append(pending_op)
                
                # Remove waited-for ops from queue
                lgkm_queue = lgkm_queue[wait_count:]
            
            # VM dependencies: wait for oldest (queue_len - vmcnt) VM ops
            if vmcnt is not None and len(vm_queue) > vmcnt:
                wait_count = len(vm_queue) - vmcnt
                
                for pending_op in vm_queue[:wait_count]:
                    if isinstance(pending_op, InstructionNode):
                        # Intra-block memory op
                        if pending_op not in node.predecessors:
                            node.predecessors.append(pending_op)
                            pending_op.successors.append(node)
                            ddg.edges.append((pending_op.node_id, node.node_id, f"WAIT:vm"))
                        available_regs.update(pending_op.defs)
                    else:
                        # Cross-block memory op (PendingMemOp)
                        cross_block_avail_regs.update(pending_op.regs)
                        cross_block_waited_ops.append(pending_op)
                
                # Remove waited-for ops from queue
                vm_queue = vm_queue[wait_count:]
            
            # Store available registers for this waitcnt node for later use
            node.available_regs = available_regs
            
            # Store cross-block available registers for visualization
            if cross_block_avail_regs:
                ddg.waitcnt_cross_block_regs[node.node_id] = cross_block_avail_regs
            
            # Store cross-block waited ops for visualization
            if cross_block_waited_ops:
                ddg.waitcnt_cross_block_ops[node.node_id] = cross_block_waited_ops
        
        # Update last writer
        for reg in node.defs:
            last_writer[reg] = node
        
        # Add memory ops to queue after processing (for next s_waitcnt)
        if is_lgkm_op(opcode):
            lgkm_queue.append(node)
        elif is_vm_op(opcode):
            vm_queue.append(node)
    
    # Build pending_out lists from remaining queue
    for item in lgkm_queue:
        if isinstance(item, PendingMemOp):
            # Pass through cross-block op
            ddg.lgkm_pending_out.append(item)
        else:
            # Convert intra-block node to PendingMemOp
            # Build full instruction text: opcode operands
            instr = item.instr
            instr_text = f"[{item.node_id}] {instr.opcode}"
            if instr.operands:
                instr_text += f" {instr.operands}"
            ddg.lgkm_pending_out.append(PendingMemOp(
                regs=item.defs.copy(),
                block_label=block.label,
                node_id=item.node_id,
                op_type='lgkm',
                instr_text=instr_text
            ))
    
    for item in vm_queue:
        if isinstance(item, PendingMemOp):
            # Pass through cross-block op
            ddg.vm_pending_out.append(item)
        else:
            # Convert intra-block node to PendingMemOp
            # Build full instruction text: opcode operands
            instr = item.instr
            instr_text = f"[{item.node_id}] {instr.opcode}"
            if instr.operands:
                instr_text += f" {instr.operands}"
            ddg.vm_pending_out.append(PendingMemOp(
                regs=item.defs.copy(),
                block_label=block.label,
                node_id=item.node_id,
                op_type='vm',
                instr_text=instr_text
            ))
    
    # Third pass: Create AVAIL edges from s_waitcnt to instructions that use
    # the registers that became available after the waitcnt
    for i, node in enumerate(nodes):
        if node.instr.opcode.lower() == 's_waitcnt':
            # Handle intra-block available registers
            if node.available_regs:
                avail_regs_copy = node.available_regs.copy()
                for j in range(i + 1, len(nodes)):
                    later_node = nodes[j]
                    # Check if this instruction uses any of the available registers
                    used_avail_regs = avail_regs_copy & later_node.uses
                    if used_avail_regs:
                        # Create AVAIL edge for each used register
                        for reg in used_avail_regs:
                            ddg.edges.append((node.node_id, later_node.node_id, f"AVAIL:{reg}"))
                    
                    # Stop if the later instruction redefines any of the available registers
                    redefined = avail_regs_copy & later_node.defs
                    if redefined:
                        avail_regs_copy = avail_regs_copy - redefined
                        if not avail_regs_copy:
                            break
            
            # Handle cross-block available registers (from predecessor memory ops)
            if node.node_id in ddg.waitcnt_cross_block_regs:
                cross_avail_regs = ddg.waitcnt_cross_block_regs[node.node_id].copy()
                for j in range(i + 1, len(nodes)):
                    later_node = nodes[j]
                    # Check if this instruction uses any of the cross-block available registers
                    used_cross_regs = cross_avail_regs & later_node.uses
                    if used_cross_regs:
                        # Create XAVAIL edge for cross-block available registers
                        for reg in used_cross_regs:
                            ddg.cross_block_avail_edges.append((node.node_id, later_node.node_id, reg))
                    
                    # Stop if the later instruction redefines any of the available registers
                    redefined = cross_avail_regs & later_node.defs
                    if redefined:
                        cross_avail_regs = cross_avail_regs - redefined
                        if not cross_avail_regs:
                            break
    
    return ddg


def compute_inter_block_deps(cfg: CFG, ddgs: Dict[str, DDG]) -> List[InterBlockDep]:
    """
    Compute data dependencies between basic blocks.
    
    A dependency exists from block A to block B if:
    - B is a successor of A in the CFG
    - A's live-out registers intersect with B's live-in registers
    """
    inter_deps = []
    
    for label, block in cfg.blocks.items():
        ddg = ddgs.get(label)
        if not ddg:
            continue
        
        for succ_label in block.successors:
            succ_ddg = ddgs.get(succ_label)
            if not succ_ddg:
                continue
            
            # Find registers that flow from this block to successor
            flowing_regs = ddg.live_out & succ_ddg.live_in
            
            if flowing_regs:
                inter_deps.append(InterBlockDep(
                    from_block=label,
                    to_block=succ_label,
                    registers=flowing_regs
                ))
    
    return inter_deps


def compute_cross_block_pending_ops(
    cfg: CFG, 
    ddgs: Dict[str, DDG]
) -> Tuple[Dict[str, List[PendingMemOp]], Dict[str, List[PendingMemOp]], List[WaitcntInterBlockDep]]:
    """
    Compute pending memory operation queues for each basic block.
    
    For each basic block, compute the incoming pending memory operations
    from predecessor blocks (in FIFO order), excluding back-edge predecessors
    for blocks with in-degree > 1 to break cycles.
    
    The queues maintain the correct order:
    - Operations from earlier blocks come first
    - Operations within a block are in program order
    
    Returns:
        lgkm_pending: Dict mapping block label to incoming LGKM pending ops list
        vm_pending: Dict mapping block label to incoming VM pending ops list
        waitcnt_deps: List of cross-block waitcnt dependencies
    """
    lgkm_pending: Dict[str, List[PendingMemOp]] = {}
    vm_pending: Dict[str, List[PendingMemOp]] = {}
    waitcnt_deps: List[WaitcntInterBlockDep] = []
    
    # Initialize with empty lists
    for label in cfg.blocks:
        lgkm_pending[label] = []
        vm_pending[label] = []
    
    # Propagate pending ops through the CFG
    # Use iterative approach until stable
    changed = True
    iterations = 0
    max_iterations = len(cfg.blocks) * 2  # Prevent infinite loops
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for label, block in cfg.blocks.items():
            ddg = ddgs.get(label)
            if not ddg:
                continue
            
            # Get predecessors, excluding back-edges for blocks with in-degree > 1
            predecessors = block.predecessors if hasattr(block, 'predecessors') else []
            
            if len(predecessors) > 1:
                # Detect and exclude back-edge predecessors
                back_edge_preds = detect_back_edge_predecessors(cfg, label)
                valid_preds = [p for p in predecessors if p not in back_edge_preds]
            else:
                valid_preds = predecessors
            
            # Collect pending ops from valid predecessors
            # When multiple predecessors exist (in-degree > 1), take the MINIMUM
            # pending count from all predecessors. This is conservative because
            # at a control flow merge point, we don't know which path was taken,
            # so we assume the path with fewest pending operations.
            new_lgkm_pending: List[PendingMemOp] = []
            new_vm_pending: List[PendingMemOp] = []
            first_pred = True
            
            for pred_label in valid_preds:
                pred_ddg = ddgs.get(pred_label)
                if pred_ddg:
                    if first_pred:
                        # Initialize with first predecessor's pending ops
                        new_lgkm_pending = [PendingMemOp(op.regs.copy(), op.block_label, op.node_id, op.op_type, op.instr_text) 
                                           for op in pred_ddg.lgkm_pending_out]
                        new_vm_pending = [PendingMemOp(op.regs.copy(), op.block_label, op.node_id, op.op_type, op.instr_text) 
                                         for op in pred_ddg.vm_pending_out]
                        first_pred = False
                    else:
                        # Take MINIMUM: use smaller queue length from predecessors
                        if len(pred_ddg.lgkm_pending_out) < len(new_lgkm_pending):
                            new_lgkm_pending = [PendingMemOp(op.regs.copy(), op.block_label, op.node_id, op.op_type, op.instr_text) 
                                               for op in pred_ddg.lgkm_pending_out]
                        if len(pred_ddg.vm_pending_out) < len(new_vm_pending):
                            new_vm_pending = [PendingMemOp(op.regs.copy(), op.block_label, op.node_id, op.op_type, op.instr_text) 
                                             for op in pred_ddg.vm_pending_out]
            
            # Check if values changed
            old_lgkm_len = len(lgkm_pending[label])
            old_vm_len = len(vm_pending[label])
            
            if len(new_lgkm_pending) != old_lgkm_len or len(new_vm_pending) != old_vm_len:
                changed = True
                lgkm_pending[label] = new_lgkm_pending
                vm_pending[label] = new_vm_pending
    
    # Build waitcnt inter-block dependencies
    for label, block in cfg.blocks.items():
        predecessors = block.predecessors if hasattr(block, 'predecessors') else []
        
        if len(predecessors) > 1:
            back_edge_preds = detect_back_edge_predecessors(cfg, label)
            valid_preds = [p for p in predecessors if p not in back_edge_preds]
        else:
            valid_preds = predecessors
        
        for pred_label in valid_preds:
            pred_ddg = ddgs.get(pred_label)
            if pred_ddg:
                lgkm_count = len(pred_ddg.lgkm_pending_out)
                vm_count = len(pred_ddg.vm_pending_out)
                if lgkm_count > 0 or vm_count > 0:
                    waitcnt_deps.append(WaitcntInterBlockDep(
                        from_block=pred_label,
                        to_block=label,
                        lgkm_count=lgkm_count,
                        vm_count=vm_count
                    ))
    
    return lgkm_pending, vm_pending, waitcnt_deps


# =============================================================================
# DOT Generation for DDG
# =============================================================================

def generate_ddg_dot(ddg: DDG, max_label_len: int = 50, show_live_nodes: bool = True) -> str:
    """
    Generate DOT format for a single basic block's DDG.
    
    Args:
        ddg: The data dependency graph
        max_label_len: Maximum length for instruction labels
        show_live_nodes: Whether to show Live-In/Live-Out nodes for cross-block data flow
    """
    lines = []
    lines.append(f'digraph "DDG_{ddg.block_label}" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, fontname="Courier", fontsize=9];')
    lines.append('    edge [fontname="Courier", fontsize=8];')
    lines.append('    graph [ranksep=0.4, nodesep=0.3];')
    lines.append('')
    
    # Add graph title with memory operation info
    mem_info = ""
    lgkm_in_count = len(ddg.lgkm_pending_in)
    vm_in_count = len(ddg.vm_pending_in)
    lgkm_out_count = len(ddg.lgkm_pending_out)
    vm_out_count = len(ddg.vm_pending_out)
    
    if lgkm_in_count > 0 or vm_in_count > 0:
        mem_info = f"\\n[Incoming: lgkm={lgkm_in_count}, vm={vm_in_count}]"
    if lgkm_out_count > 0 or vm_out_count > 0:
        mem_info += f"\\n[Outgoing: lgkm={lgkm_out_count}, vm={vm_out_count}]"
    
    lgkm_count = len(ddg.lgkm_ops)
    vm_count = len(ddg.vm_ops)
    if lgkm_count > 0 or vm_count > 0:
        mem_info += f"\\n[MemOps: lgkm={lgkm_count}, vm={vm_count}]"
    
    lines.append(f'    label="{ddg.block_label}{mem_info}";')
    lines.append('    labelloc=t;')
    lines.append('')
    
    # Track first user of each live-in register and last definer of each live-out register
    first_user: Dict[str, int] = {}  # reg -> node_id
    last_definer: Dict[str, int] = {}  # reg -> node_id
    
    defined_so_far = set()
    for node in ddg.nodes:
        # First user of live-in registers
        for reg in node.uses:
            if reg in ddg.live_in and reg not in defined_so_far and reg not in first_user:
                first_user[reg] = node.node_id
        # Track definitions
        defined_so_far.update(node.defs)
        # Last definer of live-out registers
        for reg in node.defs:
            if reg in ddg.live_out:
                last_definer[reg] = node.node_id
    
    # ==== Live-In Nodes ====
    if show_live_nodes and ddg.live_in:
        lines.append('    // Live-In nodes (registers from predecessors)')
        lines.append('    subgraph cluster_live_in {')
        lines.append('        label="LIVE-IN (from predecessors)";')
        lines.append('        style=dashed;')
        lines.append('        color=forestgreen;')
        lines.append('        fontcolor=forestgreen;')
        lines.append('        node [shape=ellipse, style=filled, fillcolor=palegreen, fontsize=8];')
        
        # Group live-in registers by type
        sgpr_in = sorted([r for r in ddg.live_in if r.startswith('s') and r not in ('scc',)])
        vgpr_in = sorted([r for r in ddg.live_in if r.startswith('v') and r not in ('vcc',)])
        agpr_in = sorted([r for r in ddg.live_in if r.startswith('a')])
        special_in = sorted([r for r in ddg.live_in if r in ('scc', 'vcc', 'exec')])
        
        # Create grouped nodes for readability
        if sgpr_in:
            sgpr_label = ", ".join(sgpr_in[:8])
            if len(sgpr_in) > 8:
                sgpr_label += f"... (+{len(sgpr_in)-8})"
            lines.append(f'        live_in_sgpr [label="SGPR\\n{sgpr_label}"];')
        if vgpr_in:
            vgpr_label = ", ".join(vgpr_in[:8])
            if len(vgpr_in) > 8:
                vgpr_label += f"... (+{len(vgpr_in)-8})"
            lines.append(f'        live_in_vgpr [label="VGPR\\n{vgpr_label}"];')
        if agpr_in:
            agpr_label = ", ".join(agpr_in[:8])
            if len(agpr_in) > 8:
                agpr_label += f"... (+{len(agpr_in)-8})"
            lines.append(f'        live_in_agpr [label="AGPR\\n{agpr_label}"];')
        if special_in:
            lines.append(f'        live_in_special [label="Special\\n{", ".join(special_in)}"];')
        
        lines.append('    }')
        lines.append('')
    
    # ==== Instruction Nodes ====
    lines.append('    // Instruction nodes')
    for node in ddg.nodes:
        # Create instruction label
        instr_text = truncate_instruction(node.instr, max_label_len)
        instr_text = escape_dot_string(instr_text)
        
        # Show defs and uses
        defs_str = ",".join(sorted(node.defs)[:5])
        if len(node.defs) > 5:
            defs_str += "..."
        uses_str = ",".join(sorted(node.uses)[:5])
        if len(node.uses) > 5:
            uses_str += "..."
        
        # Color based on instruction type
        color = "white"
        opcode_lower = node.instr.opcode.lower()
        if is_lgkm_op(opcode_lower):
            color = "lightskyblue"  # LGKM operations (s_load_*, ds_*)
        elif is_vm_op(opcode_lower):
            color = "khaki"  # VM operations (buffer_load_*, global_load_*)
        elif opcode_lower.startswith(('s_store', 'buffer_store', 'global_store')):
            color = "lightyellow"  # Memory store
        elif node.instr.opcode.startswith('v_mfma'):
            color = "lightgreen"  # Matrix operation
        elif node.instr.opcode.startswith(('s_waitcnt', 's_barrier')):
            color = "lightgray"  # Synchronization
        elif node.instr.is_branch or node.instr.is_terminator:
            color = "lightcoral"  # Control flow
        
        # Build label
        label = f"[{node.node_id}] {instr_text}\\nW:{defs_str}\\nR:{uses_str}"
        
        # For s_waitcnt, add cross-block waited instructions
        if opcode_lower == 's_waitcnt' and node.node_id in ddg.waitcnt_cross_block_ops:
            waited_ops = ddg.waitcnt_cross_block_ops[node.node_id]
            if waited_ops:
                label += "\\n--- WAIT for cross-block ops ---"
                for op in waited_ops:
                    # Display full instruction text from the source block
                    if op.instr_text:
                        # Escape and truncate if needed
                        op_instr = escape_dot_string(op.instr_text)
                        if len(op_instr) > 60:
                            op_instr = op_instr[:57] + "..."
                        label += f"\\n{op.block_label}: {op_instr}"
                    else:
                        # Fallback if no instruction text available
                        regs_str = ",".join(sorted(op.regs)[:4])
                        if len(op.regs) > 4:
                            regs_str += f"...(+{len(op.regs)-4})"
                        op_type = "LGKM" if op.op_type == 'lgkm' else "VM"
                        label += f"\\n{op.block_label}: [{op.node_id}] {op_type} -> {regs_str}"
        
        lines.append(f'    n{node.node_id} [label="{label}", style=filled, fillcolor={color}];')
    
    lines.append('')
    
    # ==== Live-Out Nodes ====
    if show_live_nodes and ddg.live_out:
        lines.append('    // Live-Out nodes (registers to successors)')
        lines.append('    subgraph cluster_live_out {')
        lines.append('        label="LIVE-OUT (to successors)";')
        lines.append('        style=dashed;')
        lines.append('        color=firebrick;')
        lines.append('        fontcolor=firebrick;')
        lines.append('        node [shape=ellipse, style=filled, fillcolor=lightsalmon, fontsize=8];')
        
        # Group live-out registers by type
        sgpr_out = sorted([r for r in ddg.live_out if r.startswith('s') and r not in ('scc',)])
        vgpr_out = sorted([r for r in ddg.live_out if r.startswith('v') and r not in ('vcc',)])
        agpr_out = sorted([r for r in ddg.live_out if r.startswith('a')])
        special_out = sorted([r for r in ddg.live_out if r in ('scc', 'vcc', 'exec')])
        
        if sgpr_out:
            sgpr_label = ", ".join(sgpr_out[:8])
            if len(sgpr_out) > 8:
                sgpr_label += f"... (+{len(sgpr_out)-8})"
            lines.append(f'        live_out_sgpr [label="SGPR\\n{sgpr_label}"];')
        if vgpr_out:
            vgpr_label = ", ".join(vgpr_out[:8])
            if len(vgpr_out) > 8:
                vgpr_label += f"... (+{len(vgpr_out)-8})"
            lines.append(f'        live_out_vgpr [label="VGPR\\n{vgpr_label}"];')
        if agpr_out:
            agpr_label = ", ".join(agpr_out[:8])
            if len(agpr_out) > 8:
                agpr_label += f"... (+{len(agpr_out)-8})"
            lines.append(f'        live_out_agpr [label="AGPR\\n{agpr_label}"];')
        if special_out:
            lines.append(f'        live_out_special [label="Special\\n{", ".join(special_out)}"];')
        
        lines.append('    }')
        lines.append('')
    
    # ==== Edges from Live-In to first users ====
    # Collect registers that are waited by s_waitcnt (from cross-block memory ops)
    cross_block_waited_regs: Set[str] = set()
    for waited_regs in ddg.waitcnt_cross_block_regs.values():
        cross_block_waited_regs.update(waited_regs)
    
    if show_live_nodes and ddg.live_in:
        lines.append('    // Edges from Live-In to first users')
        
        # For each register in live_in that is NOT waited by s_waitcnt,
        # draw an edge from LIVE-IN directly to the first user with register label
        for reg, node_id in first_user.items():
            # Skip registers that are handled by cross-block waitcnt
            if reg in cross_block_waited_regs:
                continue
            
            # Determine the source node based on register type
            if reg.startswith('s') and reg not in ('scc',):
                source = 'live_in_sgpr'
            elif reg.startswith('v') and reg not in ('vcc',):
                source = 'live_in_vgpr'
            elif reg.startswith('a'):
                source = 'live_in_agpr'
            elif reg in ('scc', 'vcc', 'exec'):
                source = 'live_in_special'
            else:
                continue
            
            lines.append(f'    {source} -> n{node_id} [color=forestgreen, style=dashed, label="{reg}"];')
        
        lines.append('')
    
    # ==== Internal DDG edges ====
    lines.append('    // Internal dependency edges')
    for from_id, to_id, dep_type in ddg.edges:
        # Color edges by dependency type
        if dep_type.startswith("RAW"):
            edge_style = 'color=blue'
        elif dep_type.startswith("WAW"):
            edge_style = 'color=red, style=dashed'
        elif dep_type.startswith("WAIT"):
            # Waitcnt dependencies - orange/purple for visibility
            if "lgkm" in dep_type:
                edge_style = 'color=darkorange, style=bold, penwidth=2'
            else:  # vm
                edge_style = 'color=purple, style=bold, penwidth=2'
        elif dep_type.startswith("AVAIL"):
            # Available register edges from s_waitcnt - green for data availability
            edge_style = 'color=green, style=bold, penwidth=1.5'
        else:
            edge_style = 'color=gray'
        
        # Shorten dependency label
        reg = dep_type.split(":")[1] if ":" in dep_type else ""
        if len(reg) > 10:
            reg = reg[:10] + "..."
        
        lines.append(f'    n{from_id} -> n{to_id} [{edge_style}, label="{reg}"];')
    
    lines.append('')
    
    # ==== Cross-block waitcnt edges: LIVE-IN -> s_waitcnt -> users ====
    has_cross_block_pending = ddg.lgkm_pending_in or ddg.vm_pending_in
    if show_live_nodes and has_cross_block_pending:
        lines.append('    // Cross-block waitcnt edges (from predecessor memory ops)')
        
        # Find which s_waitcnt instructions wait for cross-block ops
        for waitcnt_id, waited_regs in ddg.waitcnt_cross_block_regs.items():
            # Add edges from LIVE-IN to s_waitcnt with register labels
            for reg in waited_regs:
                if reg.startswith('s') and reg not in ('scc',):
                    lines.append(f'    live_in_sgpr -> n{waitcnt_id} [color=darkorange, style=bold, penwidth=2, label="{reg}"];')
                elif reg.startswith('v') and reg not in ('vcc',):
                    lines.append(f'    live_in_vgpr -> n{waitcnt_id} [color=purple, style=bold, penwidth=2, label="{reg}"];')
                elif reg.startswith('a'):
                    lines.append(f'    live_in_agpr -> n{waitcnt_id} [color=darkorange, style=bold, penwidth=2, label="{reg}"];')
        
        # Add cross-block AVAIL edges: s_waitcnt -> users
        for from_id, to_id, reg in ddg.cross_block_avail_edges:
            lines.append(f'    n{from_id} -> n{to_id} [color=cyan, style=bold, penwidth=1.5, label="{reg}"];')
        
        lines.append('')
    
    # ==== Edges from last definers to Live-Out ====
    if show_live_nodes and ddg.live_out:
        lines.append('    // Edges from last definers to Live-Out')
        
        # For each register in live_out, draw an edge from the last definer to LIVE-OUT with register label
        for reg, node_id in last_definer.items():
            # Determine the target node based on register type
            if reg.startswith('s') and reg not in ('scc',):
                target = 'live_out_sgpr'
            elif reg.startswith('v') and reg not in ('vcc',):
                target = 'live_out_vgpr'
            elif reg.startswith('a'):
                target = 'live_out_agpr'
            elif reg in ('scc', 'vcc', 'exec'):
                target = 'live_out_special'
            else:
                continue
            
            lines.append(f'    n{node_id} -> {target} [color=firebrick, style=dashed, label="{reg}"];')
    
    lines.append('}')
    return '\n'.join(lines)


def generate_combined_cfg_ddg_dot(cfg: CFG, ddgs: Dict[str, DDG], inter_deps: List[InterBlockDep]) -> str:
    """
    Generate a combined DOT file with CFG as main graph and DDGs as subgraphs.
    Includes inter-block data dependencies.
    """
    lines = []
    lines.append(f'digraph "CFG_DDG_{escape_dot_string(cfg.name)}" {{')
    lines.append('    compound=true;')
    lines.append('    rankdir=TB;')
    lines.append('    fontname="Courier";')
    lines.append('    node [fontname="Courier", fontsize=10];')
    lines.append('    edge [fontname="Courier", fontsize=9];')
    lines.append('')
    
    # Create subgraph for each basic block with its DDG
    for label, block in cfg.blocks.items():
        cluster_id = label.replace('.', '_').replace('-', '_')
        ddg = ddgs.get(label)
        
        lines.append(f'    subgraph cluster_{cluster_id} {{')
        lines.append(f'        label="{escape_dot_string(label)} ({len(block.instructions)} instr)";')
        
        # Color the cluster
        if label == cfg.entry_block:
            lines.append('        style=filled;')
            lines.append('        fillcolor=lightgreen;')
        elif block.get_terminator() and block.get_terminator().is_terminator:
            lines.append('        style=filled;')
            lines.append('        fillcolor=lightcoral;')
        else:
            lines.append('        style=filled;')
            lines.append('        fillcolor=white;')
        
        if ddg and ddg.nodes:
            # Add DDG nodes inside the cluster
            for node in ddg.nodes:
                instr_text = truncate_instruction(node.instr, 40)
                instr_text = escape_dot_string(instr_text)
                
                # Color based on instruction type
                color = "white"
                opcode_lower = node.instr.opcode.lower()
                if is_lgkm_op(opcode_lower):
                    color = "lightskyblue"  # LGKM operations (s_load_*, ds_*)
                elif is_vm_op(opcode_lower):
                    color = "khaki"  # VM operations (buffer_load_*, global_load_*)
                elif opcode_lower.startswith(('s_store', 'buffer_store', 'global_store')):
                    color = "lightyellow"  # Memory store
                elif node.instr.opcode.startswith('v_mfma'):
                    color = "palegreen"
                elif node.instr.opcode.startswith(('s_waitcnt', 's_barrier')):
                    color = "lightgray"
                
                lines.append(f'        {cluster_id}_n{node.node_id} [label="{instr_text}", shape=box, style=filled, fillcolor={color}, fontsize=8];')
            
            # Add DDG edges inside the cluster
            for from_id, to_id, dep_type in ddg.edges:
                if dep_type.startswith("RAW"):
                    edge_style = 'color=blue'
                elif dep_type.startswith("WAW"):
                    edge_style = 'color=red, style=dashed'
                else:
                    edge_style = 'color=gray'
                lines.append(f'        {cluster_id}_n{from_id} -> {cluster_id}_n{to_id} [{edge_style}];')
        else:
            # Empty block - add placeholder node
            lines.append(f'        {cluster_id}_empty [label="(empty)", shape=none];')
        
        lines.append('    }')
        lines.append('')
    
    # Add CFG edges between clusters (control flow)
    lines.append('    // CFG control flow edges')
    for label, block in cfg.blocks.items():
        from_cluster = label.replace('.', '_').replace('-', '_')
        ddg = ddgs.get(label)
        
        # Find the last node in this cluster for the edge source
        if ddg and ddg.nodes:
            from_node = f"{from_cluster}_n{ddg.nodes[-1].node_id}"
        else:
            from_node = f"{from_cluster}_empty"
        
        for succ in block.successors:
            to_cluster = succ.replace('.', '_').replace('-', '_')
            succ_ddg = ddgs.get(succ)
            
            # Find the first node in the successor cluster
            if succ_ddg and succ_ddg.nodes:
                to_node = f"{to_cluster}_n{succ_ddg.nodes[0].node_id}"
            else:
                to_node = f"{to_cluster}_empty"
            
            lines.append(f'    {from_node} -> {to_node} [ltail=cluster_{from_cluster}, lhead=cluster_{to_cluster}, color=black, penwidth=2];')
    
    lines.append('}')
    return '\n'.join(lines)


def generate_cfg_with_inter_deps_dot(cfg: CFG, ddgs: Dict[str, DDG], inter_deps: List[InterBlockDep], output_dir: str) -> str:
    """
    Generate a CFG showing inter-block data dependencies.
    Each block shows live-in/live-out and edges show flowing registers.
    """
    lines = []
    lines.append(f'digraph "{escape_dot_string(cfg.name)}" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=record, fontname="Courier", fontsize=10];')
    lines.append('    edge [fontname="Courier", fontsize=8];')
    lines.append('')
    
    for label, block in cfg.blocks.items():
        node_id = label.replace('.', '_').replace('-', '_')
        ddg = ddgs.get(label)
        
        instr_count = len(block.instructions)
        critical_path = ddg.get_critical_path_length() if ddg else 0
        
        # Build label with live-in/live-out
        live_in_count = len(ddg.live_in) if ddg else 0
        live_out_count = len(ddg.live_out) if ddg else 0
        
        label_text = f"{{{label}|{instr_count} instr, CP:{critical_path}|in:{live_in_count} out:{live_out_count}}}"
        
        # Color coding
        color = ""
        if label == cfg.entry_block:
            color = ', style=filled, fillcolor=lightgreen'
        elif block.get_terminator() and block.get_terminator().is_terminator:
            color = ', style=filled, fillcolor=lightcoral'
        
        # Add URL to DDG SVG
        ddg_filename = f"ddg_{node_id}.svg"
        url_attr = f', URL="{ddg_filename}"'
        
        lines.append(f'    {node_id} [label="{label_text}"{color}{url_attr}];')
    
    lines.append('')
    
    # Add edges with data dependency info
    lines.append('    // Control flow edges with data dependencies')
    
    # Create a lookup for inter-block deps
    dep_lookup = {}
    for dep in inter_deps:
        key = (dep.from_block, dep.to_block)
        dep_lookup[key] = dep.registers
    
    for label, block in cfg.blocks.items():
        from_id = label.replace('.', '_').replace('-', '_')
        for succ in block.successors:
            to_id = succ.replace('.', '_').replace('-', '_')
            
            # Check for data dependencies
            key = (label, succ)
            if key in dep_lookup:
                regs = dep_lookup[key]
                reg_count = len(regs)
                sample_regs = ", ".join(sorted(regs)[:3])
                if reg_count > 3:
                    sample_regs += "..."
                edge_label = f"{reg_count} regs\\n{sample_regs}"
                lines.append(f'    {from_id} -> {to_id} [label="{edge_label}", color=blue];')
            else:
                lines.append(f'    {from_id} -> {to_id};')
    
    lines.append('}')
    return '\n'.join(lines)


def generate_cfg_with_waitcnt_deps_dot(
    cfg: CFG, 
    ddgs: Dict[str, DDG], 
    waitcnt_deps: List[WaitcntInterBlockDep]
) -> str:
    """
    Generate a CFG showing cross-block s_waitcnt memory dependencies.
    
    Each block shows:
    - Number of LGKM and VM operations
    - Incoming/outgoing memory operation counts
    
    Edges show the number of memory operations flowing between blocks.
    """
    lines = []
    lines.append(f'digraph "{escape_dot_string(cfg.name)}_waitcnt" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=record, fontname="Courier", fontsize=10];')
    lines.append('    edge [fontname="Courier", fontsize=8];')
    lines.append('')
    lines.append('    // Legend')
    lines.append('    subgraph cluster_legend {')
    lines.append('        label="Legend";')
    lines.append('        style=dashed;')
    lines.append('        legend_lgkm [label="LGKM: s_load, ds_*", shape=box, style=filled, fillcolor=lightblue];')
    lines.append('        legend_vm [label="VM: buffer_load, global_load", shape=box, style=filled, fillcolor=lightyellow];')
    lines.append('    }')
    lines.append('')
    
    # Build waitcnt dep lookup
    waitcnt_lookup: Dict[Tuple[str, str], WaitcntInterBlockDep] = {}
    for dep in waitcnt_deps:
        waitcnt_lookup[(dep.from_block, dep.to_block)] = dep
    
    for label, block in cfg.blocks.items():
        node_id = label.replace('.', '_').replace('-', '_')
        ddg = ddgs.get(label)
        
        if ddg:
            lgkm_count = len(ddg.lgkm_ops)
            vm_count = len(ddg.vm_ops)
            lgkm_in = len(ddg.lgkm_pending_in)
            vm_in = len(ddg.vm_pending_in)
            lgkm_out = len(ddg.lgkm_pending_out)
            vm_out = len(ddg.vm_pending_out)
        else:
            lgkm_count = vm_count = lgkm_in = vm_in = lgkm_out = vm_out = 0
        
        # Build detailed label
        label_text = f"{{{label}|LGKM: {lgkm_count}, VM: {vm_count}|in: lgkm={lgkm_in}, vm={vm_in}|out: lgkm={lgkm_out}, vm={vm_out}}}"
        
        # Color based on memory operation density
        if lgkm_count > 0 or vm_count > 0:
            if lgkm_count > vm_count:
                color = ', style=filled, fillcolor=lightblue'
            else:
                color = ', style=filled, fillcolor=lightyellow'
        else:
            color = ''
        
        # Mark entry/exit blocks
        if label == cfg.entry_block:
            color = ', style=filled, fillcolor=lightgreen'
        elif block.get_terminator() and block.get_terminator().is_terminator:
            color = ', style=filled, fillcolor=lightcoral'
        
        lines.append(f'    {node_id} [label="{label_text}"{color}];')
    
    lines.append('')
    lines.append('    // Edges with waitcnt dependencies')
    
    for label, block in cfg.blocks.items():
        from_id = label.replace('.', '_').replace('-', '_')
        for succ in block.successors:
            to_id = succ.replace('.', '_').replace('-', '_')
            
            # Check for waitcnt dependencies
            key = (label, succ)
            if key in waitcnt_lookup:
                dep = waitcnt_lookup[key]
                edge_parts = []
                if dep.lgkm_count > 0:
                    edge_parts.append(f"lgkm:{dep.lgkm_count}")
                if dep.vm_count > 0:
                    edge_parts.append(f"vm:{dep.vm_count}")
                edge_label = "\\n".join(edge_parts)
                
                # Color based on which type is dominant
                if dep.lgkm_count > dep.vm_count:
                    edge_color = "darkorange"
                elif dep.vm_count > dep.lgkm_count:
                    edge_color = "purple"
                else:
                    edge_color = "gray"
                
                lines.append(f'    {from_id} -> {to_id} [label="{edge_label}", color={edge_color}, penwidth=2];')
            else:
                lines.append(f'    {from_id} -> {to_id};')
    
    lines.append('}')
    return '\n'.join(lines)


def generate_summary_cfg_with_ddg_links(cfg: CFG, ddgs: Dict[str, DDG], output_dir: str) -> str:
    """
    Generate a CFG where each block is clickable to view its DDG.
    """
    lines = []
    lines.append(f'digraph "{escape_dot_string(cfg.name)}" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, fontname="Courier", fontsize=10];')
    lines.append('    edge [fontname="Courier", fontsize=9];')
    lines.append('')
    
    for label, block in cfg.blocks.items():
        node_id = label.replace('.', '_').replace('-', '_')
        ddg = ddgs.get(label)
        
        instr_count = len(block.instructions)
        critical_path = ddg.get_critical_path_length() if ddg else 0
        
        # Build label
        label_text = f"{label}\\n{instr_count} instr\\nCP: {critical_path}"
        
        # Color coding
        color = ""
        if label == cfg.entry_block:
            color = ', style=filled, fillcolor=lightgreen'
        elif block.get_terminator() and block.get_terminator().is_terminator:
            color = ', style=filled, fillcolor=lightcoral'
        
        # Add URL to DDG SVG
        ddg_filename = f"ddg_{node_id}.svg"
        url_attr = f', URL="{ddg_filename}"'
        
        lines.append(f'    {node_id} [label="{label_text}"{color}{url_attr}];')
    
    lines.append('')
    
    # Add edges
    for label, block in cfg.blocks.items():
        from_id = label.replace('.', '_').replace('-', '_')
        for succ in block.successors:
            to_id = succ.replace('.', '_').replace('-', '_')
            lines.append(f'    {from_id} -> {to_id};')
    
    lines.append('}')
    return '\n'.join(lines)


# =============================================================================
# Main Functions
# =============================================================================

def generate_all_ddgs(cfg: CFG, enable_cross_block_waitcnt: bool = True) -> Tuple[Dict[str, DDG], List[WaitcntInterBlockDep]]:
    """
    Generate DDGs for all basic blocks in the CFG.
    
    Args:
        cfg: The control flow graph
        enable_cross_block_waitcnt: If True, compute cross-block s_waitcnt dependencies
        
    Returns:
        ddgs: Dictionary of DDGs for each block
        waitcnt_deps: List of cross-block waitcnt dependencies
    """
    # First pass: build DDGs without cross-block info
    ddgs = {}
    for label, block in cfg.blocks.items():
        ddg = build_ddg(block)
        ddgs[label] = ddg
    
    waitcnt_deps = []
    
    if enable_cross_block_waitcnt:
        # Iteratively compute cross-block pending operations
        # Multiple iterations are needed because:
        # - Block A's outgoing pending depends on its incoming pending
        # - Block B's incoming pending depends on A's outgoing pending
        # So we iterate until stable
        max_iterations = len(cfg.blocks) * 2
        for iteration in range(max_iterations):
            # Compute cross-block pending operations based on current DDGs
            lgkm_pending, vm_pending, waitcnt_deps = compute_cross_block_pending_ops(cfg, ddgs)
            
            # Rebuild DDGs with cross-block pending operation queues
            changed = False
            for label, block in cfg.blocks.items():
                lgkm_ops = lgkm_pending.get(label, [])
                vm_ops = vm_pending.get(label, [])
                
                old_ddg = ddgs[label]
                old_lgkm_in = len(old_ddg.lgkm_pending_in)
                old_vm_in = len(old_ddg.vm_pending_in)
                
                # Check if incoming pending changed
                if len(lgkm_ops) != old_lgkm_in or len(vm_ops) != old_vm_in:
                    changed = True
                    ddg = build_ddg(block, lgkm_pending_in=lgkm_ops, vm_pending_in=vm_ops)
                    ddgs[label] = ddg
            
            if not changed:
                break
    
    return ddgs, waitcnt_deps


# =============================================================================
# JSON Serialization / Deserialization
# =============================================================================

@dataclass
class AnalysisResult:
    """
    Complete analysis result containing CFG, DDGs, and dependencies.
    This is the main data structure that gets serialized to JSON.
    """
    cfg: CFG
    ddgs: Dict[str, DDG]
    inter_block_deps: List[InterBlockDep]
    waitcnt_deps: List[WaitcntInterBlockDep]
    # Register analysis results (optional, computed on demand)
    # Using forward references as these classes are defined later in the file
    register_stats: Optional['RegisterStatistics'] = None
    fgpr_info: Optional['FreeGPRInfo'] = None
    
    def compute_register_analysis(self):
        """Compute register statistics and free GPR info."""
        self.register_stats = compute_register_statistics(self.ddgs)
        self.fgpr_info = compute_fgpr(self.register_stats)
        # Also store in CFG for persistence
        self.cfg.register_stats = self.register_stats.to_dict()
        self.cfg.fgpr = self.fgpr_info.to_dict()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize analysis result to dictionary."""
        result = {
            'version': '1.0',
            'cfg': self.cfg.to_dict(),
            'ddgs': {label: ddg.to_dict() for label, ddg in self.ddgs.items()},
            'inter_block_deps': [dep.to_dict() for dep in self.inter_block_deps],
            'waitcnt_deps': [dep.to_dict() for dep in self.waitcnt_deps],
        }
        # Include register analysis if computed
        if self.register_stats is not None:
            result['register_stats'] = self.register_stats.to_dict()
        if self.fgpr_info is not None:
            result['fgpr'] = self.fgpr_info.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Deserialize analysis result from dictionary."""
        # Restore CFG
        cfg = CFG.from_dict(data['cfg'])
        
        # Restore DDGs (need blocks to reconstruct)
        ddgs = {}
        for label, ddg_data in data.get('ddgs', {}).items():
            block = cfg.blocks.get(label)
            if block:
                ddgs[label] = DDG.from_dict(ddg_data, block)
            else:
                # Create a minimal block for standalone DDG
                minimal_block = BasicBlock(label=label)
                ddgs[label] = DDG.from_dict(ddg_data, minimal_block)
        
        # Restore dependencies
        inter_block_deps = [
            InterBlockDep.from_dict(d) for d in data.get('inter_block_deps', [])
        ]
        waitcnt_deps = [
            WaitcntInterBlockDep.from_dict(d) for d in data.get('waitcnt_deps', [])
        ]
        
        # Restore register analysis if available
        register_stats = None
        fgpr_info = None
        if 'register_stats' in data:
            register_stats = RegisterStatistics.from_dict(data['register_stats'])
        if 'fgpr' in data:
            fgpr_info = FreeGPRInfo.from_dict(data['fgpr'])
        
        return cls(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=inter_block_deps,
            waitcnt_deps=waitcnt_deps,
            register_stats=register_stats,
            fgpr_info=fgpr_info,
        )
    
    def to_amdgcn(self, filepath: str, keep_debug_labels: bool = False):
        """
        Regenerate the .amdgcn assembly file from this analysis result.
        
        This uses the CFG's header_lines, block raw_lines, and footer_lines
        to reconstruct the original file structure. Any modifications made
        to the instructions in the DDGs should be reflected back to the
        block's raw_lines before calling this method.
        
        Args:
            filepath: Output file path
            keep_debug_labels: If False (default), remove .Ltmp* debug labels from output.
                              If True, preserve all labels including .Ltmp* debug labels.
        """
        self.cfg.to_amdgcn(filepath, keep_debug_labels=keep_debug_labels)
    
    def update_block_instructions(self, block_label: str, new_instructions: List[str]):
        """
        Update the raw instruction lines for a specific block.
        
        This replaces all instruction lines in the block while preserving
        directives, comments, and labels.
        
        Args:
            block_label: Label of the block to update
            new_instructions: List of new instruction strings (without line numbers)
        """
        if block_label not in self.cfg.blocks:
            raise ValueError(f"Block '{block_label}' not found in CFG")
        
        block = self.cfg.blocks[block_label]
        
        # Find which lines are instructions vs non-instructions
        instr_line_nums = [instr.address for instr in block.instructions]
        
        if len(new_instructions) != len(instr_line_nums):
            raise ValueError(
                f"Number of new instructions ({len(new_instructions)}) "
                f"doesn't match original ({len(instr_line_nums)})"
            )
        
        # Replace instruction lines with new content
        for line_num, new_instr in zip(instr_line_nums, new_instructions):
            # Preserve original indentation
            old_line = block.raw_lines.get(line_num, "")
            indent = len(old_line) - len(old_line.lstrip())
            indent_str = old_line[:indent] if indent > 0 else "\t"
            
            # Build new line with indentation
            new_line = indent_str + new_instr.strip()
            if not new_line.endswith('\n'):
                new_line += '\n'
            
            block.raw_lines[line_num] = new_line


def save_analysis_to_json(
    cfg: CFG,
    ddgs: Dict[str, DDG],
    inter_block_deps: List[InterBlockDep],
    waitcnt_deps: List[WaitcntInterBlockDep],
    filepath: str,
    indent: int = 2,
    compute_fgpr_analysis: bool = True
):
    """
    Save complete analysis result to JSON file.
    
    Args:
        cfg: The control flow graph
        ddgs: Dictionary of DDGs for each block
        inter_block_deps: Inter-block data dependencies
        waitcnt_deps: Cross-block waitcnt dependencies
        filepath: Output JSON file path
        indent: JSON indentation level (default 2, use None for compact)
        compute_fgpr_analysis: Whether to compute and include register/FGPR analysis
    """
    result = AnalysisResult(
        cfg=cfg,
        ddgs=ddgs,
        inter_block_deps=inter_block_deps,
        waitcnt_deps=waitcnt_deps,
    )
    
    # Compute register analysis (including FGPR) if requested
    if compute_fgpr_analysis:
        result.compute_register_analysis()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=indent, ensure_ascii=False)
    
    print(f"Analysis saved to: {filepath}")


def load_analysis_from_json(filepath: str) -> AnalysisResult:
    """
    Load complete analysis result from JSON file.
    
    Args:
        filepath: Input JSON file path
        
    Returns:
        AnalysisResult containing CFG, DDGs, and dependencies
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = AnalysisResult.from_dict(data)
    print(f"Analysis loaded from: {filepath}")
    print(f"  - CFG: {result.cfg.name} ({len(result.cfg.blocks)} blocks)")
    print(f"  - DDGs: {len(result.ddgs)} blocks")
    print(f"  - Inter-block deps: {len(result.inter_block_deps)}")
    print(f"  - Waitcnt deps: {len(result.waitcnt_deps)}")
    
    return result


def dump_block_instructions(cfg: CFG, output_dir: str) -> None:
    """
    Dump instructions of each basic block to separate text files.
    
    Each instruction is numbered with [idx] prefix to match the instruction
    list index, making it easy to verify scheduling results against log output.
    
    Args:
        cfg: The CFG containing basic blocks
        output_dir: Directory to write the block instruction files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for block_label, block in cfg.blocks.items():
        # Create filename: .LBB0_2 -> LBB0_2.txt
        filename = block_label.lstrip('.') + '.txt'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"; Block: {block_label}\n")
            f.write(f"; Total instructions: {len(block.instructions)}\n")
            f.write(f";\n")
            
            for idx, instr in enumerate(block.instructions):
                # Format: [idx] opcode operands
                if instr.raw_line:
                    # Use raw_line but strip leading whitespace
                    line = instr.raw_line.strip()
                else:
                    # Fallback to opcode + operands
                    line = f"{instr.opcode} {instr.operands}" if instr.operands else instr.opcode
                
                f.write(f"[{idx}] {line}\n")
    
    print(f"Block instructions dumped to: {output_dir}/")


def save_ddg_files(cfg: CFG, ddgs: Dict[str, DDG], output_dir: str, 
                   generate_svg: bool = True,
                   waitcnt_deps: Optional[List[WaitcntInterBlockDep]] = None):
    """
    Save DDG files for each basic block.
    
    Args:
        cfg: The control flow graph
        ddgs: Dictionary of DDGs for each block
        output_dir: Output directory
        generate_svg: Whether to generate SVG files using graphviz
        waitcnt_deps: Cross-block waitcnt dependencies (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save block instructions to text files
    dump_block_instructions(cfg, output_dir)
    
    # Compute inter-block dependencies
    inter_deps = compute_inter_block_deps(cfg, ddgs)
    
    if waitcnt_deps is None:
        waitcnt_deps = []
    
    # Save individual DDGs
    for label, ddg in ddgs.items():
        node_id = label.replace('.', '_').replace('-', '_')
        dot_content = generate_ddg_dot(ddg)
        
        dot_file = os.path.join(output_dir, f"ddg_{node_id}.dot")
        with open(dot_file, 'w') as f:
            f.write(dot_content)
        
        if generate_svg:
            svg_file = os.path.join(output_dir, f"ddg_{node_id}.svg")
            try:
                subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate SVG for {label}: {e}")
            except FileNotFoundError:
                print("Warning: 'dot' command not found. Install graphviz to generate SVG files.")
                generate_svg = False
    
    # Save combined CFG+DDG
    combined_dot = generate_combined_cfg_ddg_dot(cfg, ddgs, inter_deps)
    combined_dot_file = os.path.join(output_dir, "cfg_ddg_combined.dot")
    with open(combined_dot_file, 'w') as f:
        f.write(combined_dot)
    
    if generate_svg:
        combined_svg_file = os.path.join(output_dir, "cfg_ddg_combined.svg")
        try:
            subprocess.run(['dot', '-Tsvg', combined_dot_file, '-o', combined_svg_file],
                         check=True, capture_output=True)
            print(f"Combined CFG+DDG SVG saved to: {combined_svg_file}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate combined SVG: {e}")
    
    # Save summary CFG with links
    summary_dot = generate_summary_cfg_with_ddg_links(cfg, ddgs, output_dir)
    summary_dot_file = os.path.join(output_dir, "cfg_summary.dot")
    with open(summary_dot_file, 'w') as f:
        f.write(summary_dot)
    
    if generate_svg:
        summary_svg_file = os.path.join(output_dir, "cfg_summary.svg")
        try:
            subprocess.run(['dot', '-Tsvg', summary_dot_file, '-o', summary_svg_file],
                         check=True, capture_output=True)
            print(f"Summary CFG SVG saved to: {summary_svg_file}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate summary SVG: {e}")
    
    # Save CFG with inter-block dependencies
    inter_dot = generate_cfg_with_inter_deps_dot(cfg, ddgs, inter_deps, output_dir)
    inter_dot_file = os.path.join(output_dir, "cfg_inter_deps.dot")
    with open(inter_dot_file, 'w') as f:
        f.write(inter_dot)
    
    if generate_svg:
        inter_svg_file = os.path.join(output_dir, "cfg_inter_deps.svg")
        try:
            subprocess.run(['dot', '-Tsvg', inter_dot_file, '-o', inter_svg_file],
                         check=True, capture_output=True)
            print(f"Inter-block dependencies SVG saved to: {inter_svg_file}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate inter-deps SVG: {e}")
    
    # Save CFG with waitcnt dependencies (if available)
    if waitcnt_deps:
        waitcnt_dot = generate_cfg_with_waitcnt_deps_dot(cfg, ddgs, waitcnt_deps)
        waitcnt_dot_file = os.path.join(output_dir, "cfg_waitcnt_deps.dot")
        with open(waitcnt_dot_file, 'w') as f:
            f.write(waitcnt_dot)
        
        if generate_svg:
            waitcnt_svg_file = os.path.join(output_dir, "cfg_waitcnt_deps.svg")
            try:
                subprocess.run(['dot', '-Tsvg', waitcnt_dot_file, '-o', waitcnt_svg_file],
                             check=True, capture_output=True)
                print(f"Waitcnt dependencies SVG saved to: {waitcnt_svg_file}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate waitcnt-deps SVG: {e}")
    
    # Save analysis result to JSON
    json_file = os.path.join(output_dir, "analysis.json")
    save_analysis_to_json(cfg, ddgs, inter_deps, waitcnt_deps, json_file)


def print_ddg_stats(ddgs: Dict[str, DDG]):
    """Print statistics about the DDGs."""
    print("\n" + "=" * 70)
    print("DDG Statistics")
    print("=" * 70)
    
    total_nodes = 0
    total_edges = 0
    total_raw = 0
    total_wait = 0
    total_avail = 0
    total_lgkm_ops = 0
    total_vm_ops = 0
    
    for label, ddg in ddgs.items():
        node_count = len(ddg.nodes)
        edge_count = len(ddg.edges)
        raw_count = sum(1 for _, _, t in ddg.edges if t.startswith("RAW"))
        wait_count = sum(1 for _, _, t in ddg.edges if t.startswith("WAIT"))
        avail_count = sum(1 for _, _, t in ddg.edges if t.startswith("AVAIL"))
        critical_path = ddg.get_critical_path_length()
        lgkm_count = len(ddg.lgkm_ops)
        vm_count = len(ddg.vm_ops)
        
        total_nodes += node_count
        total_edges += edge_count
        total_raw += raw_count
        total_wait += wait_count
        total_avail += avail_count
        total_lgkm_ops += lgkm_count
        total_vm_ops += vm_count
        
        print(f"\n{label}:")
        print(f"  Nodes: {node_count}, Edges: {edge_count}")
        print(f"  RAW deps: {raw_count}, WAIT deps: {wait_count}, AVAIL deps: {avail_count}")
        print(f"  Memory ops: LGKM={lgkm_count}, VM={vm_count}")
        lgkm_in = len(ddg.lgkm_pending_in)
        vm_in = len(ddg.vm_pending_in)
        lgkm_out = len(ddg.lgkm_pending_out)
        vm_out = len(ddg.vm_pending_out)
        if lgkm_in > 0 or vm_in > 0:
            print(f"  Incoming pending: lgkm={lgkm_in}, vm={vm_in}")
        if lgkm_out > 0 or vm_out > 0:
            print(f"  Outgoing pending: lgkm={lgkm_out}, vm={vm_out}")
        print(f"  Critical path length: {critical_path}")
        print(f"  Live-in: {len(ddg.live_in)} regs, Live-out: {len(ddg.live_out)} regs")
        
        # Calculate ILP (Instruction Level Parallelism) estimate
        if critical_path > 0:
            ilp = node_count / critical_path
            print(f"  Estimated ILP: {ilp:.2f}")
    
    print("\n" + "-" * 70)
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")
    print(f"Total RAW dependencies: {total_raw}")
    print(f"Total WAIT dependencies: {total_wait}")
    print(f"Total AVAIL dependencies (from s_waitcnt): {total_avail}")
    print(f"Total LGKM operations (s_load, ds_*): {total_lgkm_ops}")
    print(f"Total VM operations (buffer_load, global_load): {total_vm_ops}")
    
    # Compute and print register statistics
    print("\n" + "-" * 70)
    print("Register Usage Statistics")
    print("-" * 70)
    
    stats = compute_register_statistics(ddgs)
    fgpr_info = compute_fgpr(stats)
    
    # Helper function to format register list
    def format_reg_list(regs: Set[str], prefix: str) -> str:
        if not regs:
            return "none"
        sorted_regs = sorted(regs, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        if len(sorted_regs) <= 20:
            return ", ".join(sorted_regs)
        else:
            return ", ".join(sorted_regs[:10]) + f" ... (+{len(sorted_regs)-10} more)"
    
    # VGPR statistics
    print(f"\nVGPR (Vector GPR):")
    print(f"  Used count: {len(stats.vgpr_used)}")
    print(f"  Max index:  v{stats.vgpr_max_index}" if stats.vgpr_max_index >= 0 else "  Max index:  N/A")
    scatt_v_str = format_reg_list(fgpr_info.scatt_fgpr_v, "v")
    print(f"  Scattered free (within v0-v{stats.vgpr_max_index}): {len(fgpr_info.scatt_fgpr_v)} [{scatt_v_str}]")
    print(f"  Total free (up to v255): {len(fgpr_info.fgpr_v)}")
    
    # AGPR statistics
    print(f"\nAGPR (Accumulator GPR):")
    print(f"  Used count: {len(stats.agpr_used)}")
    print(f"  Max index:  a{stats.agpr_max_index}" if stats.agpr_max_index >= 0 else "  Max index:  N/A")
    scatt_a_str = format_reg_list(fgpr_info.scatt_fgpr_a, "a")
    print(f"  Scattered free (within a0-a{stats.agpr_max_index}): {len(fgpr_info.scatt_fgpr_a)} [{scatt_a_str}]")
    print(f"  Total free (up to a255): {len(fgpr_info.fgpr_a)}")
    
    # SGPR statistics
    print(f"\nSGPR (Scalar GPR):")
    print(f"  Used count: {len(stats.sgpr_used)}")
    print(f"  Max index:  s{stats.sgpr_max_index}" if stats.sgpr_max_index >= 0 else "  Max index:  N/A")
    scatt_s_str = format_reg_list(fgpr_info.scatt_fgpr_s, "s")
    print(f"  Scattered free (within s0-s{stats.sgpr_max_index}): {len(fgpr_info.scatt_fgpr_s)} [{scatt_s_str}]")
    print(f"  Total free (up to s103): {len(fgpr_info.fgpr_s)}")
    
    # Special registers
    print(f"\nSpecial Registers:")
    print(f"  exec: {'Used' if stats.uses_exec else 'Not used'}")
    print(f"  vcc:  {'Used' if stats.uses_vcc else 'Not used'}")
    print(f"  scc:  {'Used' if stats.uses_scc else 'Not used'}")
    print(f"  m0:   {'Used' if stats.uses_m0 else 'Not used'}")
    
    # Summary
    print("\n" + "-" * 70)
    print("Free GPR Summary")
    print("-" * 70)
    print(f"ScattFGPR (scattered free within used range):")
    print(f"  VGPR: {len(fgpr_info.scatt_fgpr_v)}, AGPR: {len(fgpr_info.scatt_fgpr_a)}, SGPR: {len(fgpr_info.scatt_fgpr_s)}")
    print(f"FGPR (total free up to hardware limits):")
    print(f"  VGPR: {len(fgpr_info.fgpr_v)}, AGPR: {len(fgpr_info.fgpr_a)}, SGPR: {len(fgpr_info.fgpr_s)}")
    
    print("=" * 70 + "\n")


def load_hardware_info() -> Dict[str, Any]:
    """Load hardware info from JSON file."""
    hw_info_path = Path(__file__).parent / "gfx942_hardware_info.json"
    if hw_info_path.exists():
        with open(hw_info_path, 'r') as f:
            return json.load(f)
    return {}


@dataclass
class RegisterStatistics:
    """Statistics about register usage across all DDGs."""
    # VGPR statistics
    vgpr_used: Set[str] = field(default_factory=set)
    vgpr_max_index: int = -1
    
    # AGPR statistics
    agpr_used: Set[str] = field(default_factory=set)
    agpr_max_index: int = -1
    
    # SGPR statistics (excludes special registers)
    sgpr_used: Set[str] = field(default_factory=set)
    sgpr_max_index: int = -1
    
    # Special registers
    uses_exec: bool = False
    uses_vcc: bool = False
    uses_scc: bool = False
    uses_m0: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'vgpr': {
                'used_count': len(self.vgpr_used),
                'max_index': self.vgpr_max_index,
                'used_set': sorted(self.vgpr_used, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
            },
            'agpr': {
                'used_count': len(self.agpr_used),
                'max_index': self.agpr_max_index,
                'used_set': sorted(self.agpr_used, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
            },
            'sgpr': {
                'used_count': len(self.sgpr_used),
                'max_index': self.sgpr_max_index,
                'used_set': sorted(self.sgpr_used, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
            },
            'special': {
                'exec': self.uses_exec,
                'vcc': self.uses_vcc,
                'scc': self.uses_scc,
                'm0': self.uses_m0
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegisterStatistics':
        """Deserialize from dictionary."""
        stats = cls()
        if 'vgpr' in data:
            stats.vgpr_used = set(data['vgpr'].get('used_set', []))
            stats.vgpr_max_index = data['vgpr'].get('max_index', -1)
        if 'agpr' in data:
            stats.agpr_used = set(data['agpr'].get('used_set', []))
            stats.agpr_max_index = data['agpr'].get('max_index', -1)
        if 'sgpr' in data:
            stats.sgpr_used = set(data['sgpr'].get('used_set', []))
            stats.sgpr_max_index = data['sgpr'].get('max_index', -1)
        if 'special' in data:
            stats.uses_exec = data['special'].get('exec', False)
            stats.uses_vcc = data['special'].get('vcc', False)
            stats.uses_scc = data['special'].get('scc', False)
            stats.uses_m0 = data['special'].get('m0', False)
        return stats


@dataclass
class FreeGPRInfo:
    """Information about free (unused) GPRs."""
    # Scattered free GPRs (within used range)
    scatt_fgpr_v: Set[str] = field(default_factory=set)
    scatt_fgpr_a: Set[str] = field(default_factory=set)
    scatt_fgpr_s: Set[str] = field(default_factory=set)
    
    # Full free GPRs (scattered + beyond max used)
    fgpr_v: Set[str] = field(default_factory=set)
    fgpr_a: Set[str] = field(default_factory=set)
    fgpr_s: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'scattered_free': {
                'vgpr': sorted(self.scatt_fgpr_v, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
                'agpr': sorted(self.scatt_fgpr_a, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
                'sgpr': sorted(self.scatt_fgpr_s, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
                'vgpr_count': len(self.scatt_fgpr_v),
                'agpr_count': len(self.scatt_fgpr_a),
                'sgpr_count': len(self.scatt_fgpr_s)
            },
            'full_free': {
                'vgpr': sorted(self.fgpr_v, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
                'agpr': sorted(self.fgpr_a, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
                'sgpr': sorted(self.fgpr_s, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0),
                'vgpr_count': len(self.fgpr_v),
                'agpr_count': len(self.fgpr_a),
                'sgpr_count': len(self.fgpr_s)
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FreeGPRInfo':
        """Deserialize from dictionary."""
        info = cls()
        if 'scattered_free' in data:
            info.scatt_fgpr_v = set(data['scattered_free'].get('vgpr', []))
            info.scatt_fgpr_a = set(data['scattered_free'].get('agpr', []))
            info.scatt_fgpr_s = set(data['scattered_free'].get('sgpr', []))
        if 'full_free' in data:
            info.fgpr_v = set(data['full_free'].get('vgpr', []))
            info.fgpr_a = set(data['full_free'].get('agpr', []))
            info.fgpr_s = set(data['full_free'].get('sgpr', []))
        return info


def compute_register_statistics(ddgs: Dict[str, DDG]) -> RegisterStatistics:
    """
    Compute register usage statistics across all DDGs.
    
    Args:
        ddgs: Dictionary of DDGs for each block
        
    Returns:
        RegisterStatistics containing used registers and max indices
    """
    stats = RegisterStatistics()
    
    for label, ddg in ddgs.items():
        for node in ddg.nodes:
            # Collect all registers from defs and uses
            all_regs = node.defs | node.uses | node.available_regs
            
            for reg in all_regs:
                reg_lower = reg.lower()
                
                # Check for special registers first
                if reg_lower == 'exec':
                    stats.uses_exec = True
                elif reg_lower == 'vcc':
                    stats.uses_vcc = True
                elif reg_lower == 'scc':
                    stats.uses_scc = True
                elif reg_lower == 'm0':
                    stats.uses_m0 = True
                # VGPR: v0, v1, ...
                elif reg_lower.startswith('v') and len(reg_lower) > 1:
                    try:
                        idx = int(reg_lower[1:])
                        stats.vgpr_used.add(reg_lower)
                        stats.vgpr_max_index = max(stats.vgpr_max_index, idx)
                    except ValueError:
                        pass  # Not a valid VGPR
                # AGPR: a0, a1, ...
                elif reg_lower.startswith('a') and len(reg_lower) > 1:
                    try:
                        idx = int(reg_lower[1:])
                        stats.agpr_used.add(reg_lower)
                        stats.agpr_max_index = max(stats.agpr_max_index, idx)
                    except ValueError:
                        pass  # Not a valid AGPR
                # SGPR: s0, s1, ...
                elif reg_lower.startswith('s') and len(reg_lower) > 1:
                    try:
                        idx = int(reg_lower[1:])
                        stats.sgpr_used.add(reg_lower)
                        stats.sgpr_max_index = max(stats.sgpr_max_index, idx)
                    except ValueError:
                        pass  # Not a valid SGPR (might be 'scc')
    
    return stats


def compute_fgpr(stats: RegisterStatistics, hw_info: Optional[Dict[str, Any]] = None) -> FreeGPRInfo:
    """
    Compute free (unused) GPRs based on register statistics and hardware limits.
    
    Args:
        stats: Register usage statistics
        hw_info: Hardware info dictionary (loaded from JSON if not provided)
        
    Returns:
        FreeGPRInfo containing scattered free and full free GPR sets
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    fgpr_info = FreeGPRInfo()
    
    # Get hardware limits (defaults if not in JSON)
    reg_limits = hw_info.get('register_limits', {})
    vgpr_max = reg_limits.get('vgpr', {}).get('max_index', 255)
    agpr_max = reg_limits.get('agpr', {}).get('max_index', 255)
    sgpr_max = reg_limits.get('sgpr', {}).get('max_index', 103)
    
    # Compute scattered free VGPRs (within used range)
    if stats.vgpr_max_index >= 0:
        all_vgpr_in_range = {f"v{i}" for i in range(stats.vgpr_max_index + 1)}
        fgpr_info.scatt_fgpr_v = all_vgpr_in_range - stats.vgpr_used
    
    # Compute scattered free AGPRs (within used range)
    if stats.agpr_max_index >= 0:
        all_agpr_in_range = {f"a{i}" for i in range(stats.agpr_max_index + 1)}
        fgpr_info.scatt_fgpr_a = all_agpr_in_range - stats.agpr_used
    
    # Compute scattered free SGPRs (within used range)
    if stats.sgpr_max_index >= 0:
        all_sgpr_in_range = {f"s{i}" for i in range(stats.sgpr_max_index + 1)}
        fgpr_info.scatt_fgpr_s = all_sgpr_in_range - stats.sgpr_used
    
    # Compute full free GPRs (scattered + beyond max used up to hardware limit)
    # Full free VGPRs
    all_vgpr_hw = {f"v{i}" for i in range(vgpr_max + 1)}
    fgpr_info.fgpr_v = all_vgpr_hw - stats.vgpr_used
    
    # Full free AGPRs
    all_agpr_hw = {f"a{i}" for i in range(agpr_max + 1)}
    fgpr_info.fgpr_a = all_agpr_hw - stats.agpr_used
    
    # Full free SGPRs
    all_sgpr_hw = {f"s{i}" for i in range(sgpr_max + 1)}
    fgpr_info.fgpr_s = all_sgpr_hw - stats.sgpr_used
    
    return fgpr_info


def print_inter_block_deps(inter_deps: List[InterBlockDep]):
    """Print inter-block dependencies."""
    print("\n" + "=" * 70)
    print("Inter-Block Data Dependencies")
    print("=" * 70)
    
    for dep in inter_deps:
        print(f"\n{dep.from_block} -> {dep.to_block}:")
        print(f"  Flowing registers ({len(dep.registers)}): ", end="")
        regs = sorted(dep.registers)
        if len(regs) <= 10:
            print(", ".join(regs))
        else:
            print(", ".join(regs[:10]) + f"... (+{len(regs)-10} more)")
    
    print("=" * 70 + "\n")


def print_waitcnt_deps(waitcnt_deps: List[WaitcntInterBlockDep]):
    """Print cross-block waitcnt dependencies."""
    print("\n" + "=" * 70)
    print("Cross-Block Waitcnt Dependencies")
    print("=" * 70)
    
    for dep in waitcnt_deps:
        parts = []
        if dep.lgkm_count > 0:
            parts.append(f"lgkm={dep.lgkm_count}")
        if dep.vm_count > 0:
            parts.append(f"vm={dep.vm_count}")
        
        print(f"\n{dep.from_block} -> {dep.to_block}:")
        print(f"  Memory ops flowing: {', '.join(parts)}")
    
    print("=" * 70 + "\n")




def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Data Dependency Graphs from AMDGCN assembly'
    )
    parser.add_argument('input', nargs='?', help='Input .amdgcn file')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for DOT/SVG files')
    parser.add_argument('--stats', action='store_true',
                       help='Print DDG statistics')
    parser.add_argument('--inter-deps', action='store_true',
                       help='Print inter-block dependencies')
    parser.add_argument('--waitcnt-deps', action='store_true',
                       help='Print cross-block waitcnt dependencies')
    parser.add_argument('--no-svg', action='store_true',
                       help='Do not generate SVG files')
    parser.add_argument('--block', '-b', default=None,
                       help='Generate DDG for specific block only')
    parser.add_argument('--no-cross-block-waitcnt', action='store_true',
                       help='Disable cross-block waitcnt dependency analysis')
    parser.add_argument('--load-json', '-l', default=None,
                       help='Load analysis from JSON file instead of parsing .amdgcn')
    parser.add_argument('--json-only', action='store_true',
                       help='Only save JSON file, skip SVG/DOT generation')
    parser.add_argument('--regenerate', '-r', default=None,
                       help='Regenerate .amdgcn file from loaded JSON to specified path')
    parser.add_argument('--keep-debug-labels', action='store_true',
                       help='Keep .Ltmp* debug labels when regenerating .amdgcn file (default: remove them)')
    parser.add_argument('--move', '-m', nargs=3, metavar=('BLOCK', 'INDEX', 'CYCLES'),
                       help='Move instruction: BLOCK INDEX CYCLES (positive=up, negative=down)')
    parser.add_argument('--distribute', '-d', nargs=3, metavar=('BLOCK', 'OPCODE', 'K'),
                       help='Distribute instructions evenly: BLOCK OPCODE K (e.g., .LBB0_2 global_load_dwordx4 8)')
    
    args = parser.parse_args()
    
    # Check input arguments
    if args.load_json is None and args.input is None:
        parser.error("Either input file or --load-json must be specified")
    
    # Load from JSON or parse from .amdgcn
    if args.load_json:
        # Load from JSON
        result = load_analysis_from_json(args.load_json)
        cfg = result.cfg
        ddgs = result.ddgs
        waitcnt_deps = result.waitcnt_deps
        inter_deps = result.inter_block_deps
        
        # Apply instruction move if requested
        if args.move:
            from amdgcn_passes import move_instruction, MoveResult
            block_label = args.move[0]
            instr_index = int(args.move[1])
            cycles = int(args.move[2])
            
            dir_str = "up" if cycles > 0 else "down"
            print(f"Moving instruction {instr_index} in {block_label} {dir_str} by {abs(cycles)} cycles...")
            move_result = move_instruction(result, block_label, instr_index, cycles, verbose=True)
            
            if move_result.success:
                print(f"Success: {move_result.message}")
                # Update local variables to reflect changes
                cfg = result.cfg
                ddgs = result.ddgs
                waitcnt_deps = result.waitcnt_deps
                inter_deps = result.inter_block_deps
                
                # Dump block instructions after scheduling
                dump_dir = os.path.dirname(args.load_json) or '.'
                dump_block_instructions(cfg, dump_dir)
            else:
                print(f"Failed: {move_result.message}")
                if move_result.blocked_by:
                    print(f"  Blocked by: {move_result.blocked_by}")
                return 1
        
        # Apply instruction distribute if requested
        if args.distribute:
            from amdgcn_passes import distribute_instructions
            block_label = args.distribute[0]
            target_opcode = args.distribute[1]
            distribute_count = int(args.distribute[2])
            
            print(f"Distributing {target_opcode} instructions in {block_label} (K={distribute_count})...")
            success = distribute_instructions(result, block_label, target_opcode, distribute_count, verbose=True)
            
            if success:
                print(f"Distribution completed successfully")
                # Update local variables to reflect changes
                cfg = result.cfg
                ddgs = result.ddgs
                waitcnt_deps = result.waitcnt_deps
                inter_deps = result.inter_block_deps
                
                # Dump block instructions after scheduling
                dump_dir = os.path.dirname(args.load_json) or '.'
                dump_block_instructions(cfg, dump_dir)
            else:
                print(f"Distribution failed or made no changes")
                return 1
        
        # Regenerate .amdgcn file if requested
        if args.regenerate:
            result.to_amdgcn(args.regenerate, keep_debug_labels=args.keep_debug_labels)
            if not args.output_dir and not args.stats and not args.inter_deps and not args.waitcnt_deps:
                print("Done!")
                return 0
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(args.load_json)
            if not output_dir:
                output_dir = '.'
    else:
        # Parse from .amdgcn file
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(args.input)
            if not output_dir:
                output_dir = '.'
            output_dir = os.path.join(output_dir, 'ddg_output')
        
        # Parse the input file
        print(f"Parsing {args.input}...")
        amdgcn_parser = AMDGCNParser()
        cfg = amdgcn_parser.parse_file(args.input)
        
        print(f"Found {len(cfg.blocks)} basic blocks")
        
        # Generate DDGs with cross-block waitcnt analysis
        print("Building Data Dependency Graphs...")
        enable_cross_block = not args.no_cross_block_waitcnt
        ddgs, waitcnt_deps = generate_all_ddgs(cfg, enable_cross_block_waitcnt=enable_cross_block)
        
        if enable_cross_block:
            print(f"Found {len(waitcnt_deps)} cross-block waitcnt dependencies")
        
        # Compute inter-block deps
        inter_deps = compute_inter_block_deps(cfg, ddgs)
    
    # Print stats if requested
    if args.stats:
        print_ddg_stats(ddgs)
    
    # Print inter-block deps if requested
    if args.inter_deps:
        print_inter_block_deps(inter_deps)
    
    # Print waitcnt deps if requested
    if args.waitcnt_deps:
        print_waitcnt_deps(waitcnt_deps)
    
    # Filter to specific block if requested
    if args.block:
        if args.block in ddgs:
            ddgs = {args.block: ddgs[args.block]}
        else:
            print(f"Error: Block '{args.block}' not found")
            print(f"Available blocks: {list(ddgs.keys())}")
            return 1
    
    # Save JSON only mode
    if args.json_only:
        os.makedirs(output_dir, exist_ok=True)
        json_file = os.path.join(output_dir, "analysis.json")
        save_analysis_to_json(cfg, ddgs, inter_deps, waitcnt_deps, json_file)
        print("Done!")
        return 0
    
    # Save files
    print(f"Saving output to {output_dir}...")
    save_ddg_files(cfg, ddgs, output_dir, generate_svg=not args.no_svg, waitcnt_deps=waitcnt_deps)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
