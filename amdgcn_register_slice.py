#!/usr/bin/env python3
"""
AMDGCN Register Slice Analyzer

This module extracts a "slice" of the CFG/DDG based on register dependencies,
finding all instructions that read/write a specific set of registers.

Features:
- Global search across all basic blocks for register-related instructions
- Special handling for s_barrier crossing dependencies
- Includes s_waitcnt instructions in the search
- Multiple output formats: JSON, DOT, SVG, TXT

Usage:
    python amdgcn_register_slice.py input.amdgcn --registers v3,s7,exec,scc --output-dir ./slice_output
"""

import os
import re
import json
import subprocess
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from amdgcn_cfg import (
    AMDGCNParser, CFG, BasicBlock, Instruction,
    escape_dot_string
)
from amdgcn_ddg import (
    DDG, InstructionNode,
    parse_instruction_registers,
    generate_all_ddgs,
    is_lgkm_op, is_vm_op
)
from amdgcn_verify import is_barrier_instruction


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SliceInstruction:
    """
    Instruction in the register slice with metadata.
    
    Attributes:
        address: Original line number in source file
        opcode: Instruction mnemonic
        operands: Operand string
        raw_line: Original line text
        block_label: Label of the containing basic block
        position_in_block: Index within the basic block
        reads: Set of registers read by this instruction
        writes: Set of registers written by this instruction
        is_barrier: True if s_barrier instruction
        is_waitcnt: True if s_waitcnt instruction
        available_regs: For s_waitcnt, registers that become available
    """
    address: int
    opcode: str
    operands: str
    raw_line: str
    block_label: str
    position_in_block: int
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    is_barrier: bool = False
    is_waitcnt: bool = False
    available_regs: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.address)
    
    def __eq__(self, other):
        if isinstance(other, SliceInstruction):
            return self.address == other.address
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'address': self.address,
            'opcode': self.opcode,
            'operands': self.operands,
            'raw_line': self.raw_line,
            'block_label': self.block_label,
            'position_in_block': self.position_in_block,
            'reads': sorted(list(self.reads)),
            'writes': sorted(list(self.writes)),
            'is_barrier': self.is_barrier,
            'is_waitcnt': self.is_waitcnt,
            'available_regs': sorted(list(self.available_regs)),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SliceInstruction':
        """Deserialize from dictionary."""
        return cls(
            address=data['address'],
            opcode=data['opcode'],
            operands=data['operands'],
            raw_line=data['raw_line'],
            block_label=data['block_label'],
            position_in_block=data.get('position_in_block', 0),
            reads=set(data.get('reads', [])),
            writes=set(data.get('writes', [])),
            is_barrier=data.get('is_barrier', False),
            is_waitcnt=data.get('is_waitcnt', False),
            available_regs=set(data.get('available_regs', [])),
        )


@dataclass
class SliceEdge:
    """
    Dependency edge with barrier crossing information.
    
    Attributes:
        from_addr: Source instruction address
        to_addr: Target instruction address
        dep_type: Type of dependency ("RAW", "WAR", "WAIT", "AVAIL")
        registers: Set of registers involved in the dependency
        crosses_barrier: True if edge crosses s_barrier
        barrier_addrs: List of barrier addresses crossed (in order)
    """
    from_addr: int
    to_addr: int
    dep_type: str
    registers: Set[str] = field(default_factory=set)
    crosses_barrier: bool = False
    barrier_addrs: List[int] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.from_addr, self.to_addr, self.dep_type))
    
    def __eq__(self, other):
        if isinstance(other, SliceEdge):
            return (self.from_addr == other.from_addr and 
                    self.to_addr == other.to_addr and
                    self.dep_type == other.dep_type)
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'from_addr': self.from_addr,
            'to_addr': self.to_addr,
            'dep_type': self.dep_type,
            'registers': sorted(list(self.registers)),
            'crosses_barrier': self.crosses_barrier,
            'barrier_addrs': self.barrier_addrs.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SliceEdge':
        """Deserialize from dictionary."""
        return cls(
            from_addr=data['from_addr'],
            to_addr=data['to_addr'],
            dep_type=data['dep_type'],
            registers=set(data.get('registers', [])),
            crosses_barrier=data.get('crosses_barrier', False),
            barrier_addrs=data.get('barrier_addrs', []).copy(),
        )


@dataclass
class RegisterSlice:
    """
    Complete register slice result.
    
    Attributes:
        target_registers: Set of registers that were searched for
        instructions: Dict mapping address to SliceInstruction
        edges: List of dependency edges
        barrier_instructions: Dict of s_barrier instructions in between slice instructions
        cfg_name: Name of the source CFG/function
    """
    target_registers: Set[str] = field(default_factory=set)
    instructions: Dict[int, SliceInstruction] = field(default_factory=dict)
    edges: List[SliceEdge] = field(default_factory=list)
    barrier_instructions: Dict[int, SliceInstruction] = field(default_factory=dict)
    cfg_name: str = ""
    
    def get_instruction_count(self) -> int:
        """Return total number of instructions in slice."""
        return len(self.instructions)
    
    def get_edge_count(self) -> int:
        """Return total number of edges."""
        return len(self.edges)
    
    def get_barrier_crossing_edge_count(self) -> int:
        """Return number of edges that cross barriers."""
        return sum(1 for e in self.edges if e.crosses_barrier)
    
    def get_all_addresses_sorted(self) -> List[int]:
        """Get all instruction addresses in sorted order."""
        return sorted(self.instructions.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'target_registers': sorted(list(self.target_registers)),
            'instructions': {
                str(addr): instr.to_dict() 
                for addr, instr in self.instructions.items()
            },
            'edges': [e.to_dict() for e in self.edges],
            'barrier_instructions': {
                str(addr): instr.to_dict() 
                for addr, instr in self.barrier_instructions.items()
            },
            'cfg_name': self.cfg_name,
            'stats': {
                'instruction_count': self.get_instruction_count(),
                'edge_count': self.get_edge_count(),
                'barrier_crossing_count': self.get_barrier_crossing_edge_count(),
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegisterSlice':
        """Deserialize from dictionary."""
        slice_result = cls(
            target_registers=set(data.get('target_registers', [])),
            cfg_name=data.get('cfg_name', ''),
        )
        
        for addr_str, instr_data in data.get('instructions', {}).items():
            addr = int(addr_str)
            slice_result.instructions[addr] = SliceInstruction.from_dict(instr_data)
        
        for edge_data in data.get('edges', []):
            slice_result.edges.append(SliceEdge.from_dict(edge_data))
        
        for addr_str, instr_data in data.get('barrier_instructions', {}).items():
            addr = int(addr_str)
            slice_result.barrier_instructions[addr] = SliceInstruction.from_dict(instr_data)
        
        return slice_result


# =============================================================================
# Global Position Map
# =============================================================================

@dataclass
class GlobalPosition:
    """
    Global position of an instruction for ordering comparison.
    
    Attributes:
        block_order_idx: Index of block in block_order list
        position_in_block: Index within the block
        address: Original line number
    """
    block_order_idx: int
    position_in_block: int
    address: int
    
    def __lt__(self, other):
        if self.block_order_idx != other.block_order_idx:
            return self.block_order_idx < other.block_order_idx
        return self.position_in_block < other.position_in_block


def build_global_position_map(
    cfg: CFG
) -> Dict[int, GlobalPosition]:
    """
    Build a mapping from instruction addresses to their global positions.
    
    Args:
        cfg: The Control Flow Graph
        
    Returns:
        Dictionary mapping address -> GlobalPosition
    """
    position_map: Dict[int, GlobalPosition] = {}
    block_order = cfg.block_order if cfg.block_order else list(cfg.blocks.keys())
    
    for block_idx, block_label in enumerate(block_order):
        if block_label not in cfg.blocks:
            continue
        block = cfg.blocks[block_label]
        
        for pos, instr in enumerate(block.instructions):
            position_map[instr.address] = GlobalPosition(
                block_order_idx=block_idx,
                position_in_block=pos,
                address=instr.address
            )
    
    return position_map


def is_position_before(
    pos_a: GlobalPosition,
    pos_b: GlobalPosition
) -> bool:
    """Check if position A is before position B in execution order."""
    return pos_a < pos_b


# =============================================================================
# Core Search Functions
# =============================================================================

def find_related_instructions(
    cfg: CFG,
    ddgs: Dict[str, DDG],
    target_registers: Set[str]
) -> Dict[int, SliceInstruction]:
    """
    Find all instructions that read or write any of the target registers.
    
    This performs a global search across all basic blocks.
    
    Args:
        cfg: The Control Flow Graph
        ddgs: Dictionary of DDGs for each block
        target_registers: Set of register names to search for
        
    Returns:
        Dictionary mapping address to SliceInstruction
    """
    result: Dict[int, SliceInstruction] = {}
    block_order = cfg.block_order if cfg.block_order else list(cfg.blocks.keys())
    
    for block_label in block_order:
        if block_label not in cfg.blocks:
            continue
        block = cfg.blocks[block_label]
        ddg = ddgs.get(block_label)
        
        for pos, instr in enumerate(block.instructions):
            # Get defs and uses from DDG if available, otherwise parse
            if ddg and pos < len(ddg.nodes):
                node = ddg.nodes[pos]
                defs = node.defs.copy()
                uses = node.uses.copy()
                available_regs = node.available_regs.copy()
            else:
                defs, uses = parse_instruction_registers(instr)
                available_regs = set()
            
            # Check if instruction touches any target registers
            reads_target = uses & target_registers
            writes_target = defs & target_registers
            avail_target = available_regs & target_registers
            
            # Check if it's an s_waitcnt that makes target registers available
            is_waitcnt = instr.opcode.lower() == 's_waitcnt'
            is_barrier = is_barrier_instruction(
                instr.opcode,
                getattr(instr, 'is_branch', False),
                getattr(instr, 'is_terminator', False)
            )
            
            # Include if reads, writes, or makes available any target register
            if reads_target or writes_target or avail_target:
                slice_instr = SliceInstruction(
                    address=instr.address,
                    opcode=instr.opcode,
                    operands=instr.operands,
                    raw_line=instr.raw_line,
                    block_label=block_label,
                    position_in_block=pos,
                    reads=uses,
                    writes=defs,
                    is_barrier=is_barrier,
                    is_waitcnt=is_waitcnt,
                    available_regs=available_regs,
                )
                result[instr.address] = slice_instr
    
    return result


def find_all_barriers(
    cfg: CFG
) -> Dict[int, SliceInstruction]:
    """
    Find all s_barrier instructions in the CFG.
    
    Args:
        cfg: The Control Flow Graph
        
    Returns:
        Dictionary mapping address to SliceInstruction for barriers
    """
    barriers: Dict[int, SliceInstruction] = {}
    block_order = cfg.block_order if cfg.block_order else list(cfg.blocks.keys())
    
    for block_label in block_order:
        if block_label not in cfg.blocks:
            continue
        block = cfg.blocks[block_label]
        
        for pos, instr in enumerate(block.instructions):
            # Only include s_barrier, not other barrier-like instructions
            if instr.opcode.lower() == 's_barrier':
                defs, uses = parse_instruction_registers(instr)
                slice_instr = SliceInstruction(
                    address=instr.address,
                    opcode=instr.opcode,
                    operands=instr.operands,
                    raw_line=instr.raw_line,
                    block_label=block_label,
                    position_in_block=pos,
                    reads=uses,
                    writes=defs,
                    is_barrier=True,
                    is_waitcnt=False,
                )
                barriers[instr.address] = slice_instr
    
    return barriers


def find_barriers_between(
    from_pos: GlobalPosition,
    to_pos: GlobalPosition,
    barrier_positions: Dict[int, GlobalPosition]
) -> List[int]:
    """
    Find all barrier addresses between two positions.
    
    Args:
        from_pos: Starting position
        to_pos: Ending position
        barrier_positions: Map of barrier addresses to their positions
        
    Returns:
        List of barrier addresses between from_pos and to_pos (sorted)
    """
    barriers_between = []
    
    for barrier_addr, barrier_pos in barrier_positions.items():
        # Check if barrier is strictly between from and to
        if is_position_before(from_pos, barrier_pos) and is_position_before(barrier_pos, to_pos):
            barriers_between.append((barrier_pos, barrier_addr))
    
    # Sort by position and return just the addresses
    barriers_between.sort(key=lambda x: x[0])
    return [addr for _, addr in barriers_between]


# =============================================================================
# Edge Detection Functions
# =============================================================================

def find_dependency_edges(
    instructions: Dict[int, SliceInstruction],
    target_registers: Set[str],
    position_map: Dict[int, GlobalPosition],
    all_barriers: Dict[int, SliceInstruction]
) -> Tuple[List[SliceEdge], Dict[int, SliceInstruction]]:
    """
    Find all dependency edges between slice instructions.
    
    This detects:
    - RAW (Read After Write) dependencies
    - WAR (Write After Read) dependencies
    
    Note: WAW (Write After Write) dependencies are NOT tracked because
    modern GPU hardware (like GFX942) handles concurrent register writes
    correctly without requiring explicit WAW hazard tracking.
    
    Also detects if edges cross s_barrier instructions.
    
    Args:
        instructions: Dictionary of slice instructions
        target_registers: Set of target registers
        position_map: Global position map for all instructions
        all_barriers: All s_barrier instructions in the CFG
        
    Returns:
        Tuple of (edges, barriers_used) where barriers_used contains
        only barriers that are crossed by edges
    """
    edges: List[SliceEdge] = []
    barriers_used: Dict[int, SliceInstruction] = {}
    
    # Build position map for barriers
    barrier_positions: Dict[int, GlobalPosition] = {}
    for barrier_addr, barrier_instr in all_barriers.items():
        if barrier_addr in position_map:
            barrier_positions[barrier_addr] = position_map[barrier_addr]
    
    # Sort instructions by global position
    sorted_addrs = sorted(
        instructions.keys(),
        key=lambda a: position_map.get(a, GlobalPosition(999999, 999999, a))
    )
    
    # Track last writer and readers for each target register
    last_writer: Dict[str, int] = {}  # reg -> address of last writer
    last_readers: Dict[str, Set[int]] = {}  # reg -> set of addresses that read since last write
    
    for addr in sorted_addrs:
        instr = instructions[addr]
        instr_pos = position_map.get(addr)
        if instr_pos is None:
            continue
        
        reads_target = instr.reads & target_registers
        writes_target = instr.writes & target_registers
        
        # Check for RAW dependencies (this reads what someone wrote)
        for reg in reads_target:
            if reg in last_writer:
                writer_addr = last_writer[reg]
                writer_pos = position_map.get(writer_addr)
                
                if writer_pos and is_position_before(writer_pos, instr_pos):
                    # Find barriers between writer and this reader
                    barriers = find_barriers_between(writer_pos, instr_pos, barrier_positions)
                    
                    # Add barrier instructions to barriers_used
                    for b_addr in barriers:
                        if b_addr in all_barriers:
                            barriers_used[b_addr] = all_barriers[b_addr]
                    
                    edge = SliceEdge(
                        from_addr=writer_addr,
                        to_addr=addr,
                        dep_type="RAW",
                        registers={reg},
                        crosses_barrier=len(barriers) > 0,
                        barrier_addrs=barriers,
                    )
                    
                    # Merge with existing edge if same from/to/type
                    merged = False
                    for existing in edges:
                        if (existing.from_addr == edge.from_addr and 
                            existing.to_addr == edge.to_addr and
                            existing.dep_type == edge.dep_type):
                            existing.registers.add(reg)
                            merged = True
                            break
                    if not merged:
                        edges.append(edge)
            
            # Track this as a reader
            if reg not in last_readers:
                last_readers[reg] = set()
            last_readers[reg].add(addr)
        
        # Note: WAW (Write After Write) dependencies are NOT created.
        # Modern GPU hardware (like GFX942) does not require WAW hazard tracking
        # because the register file handles concurrent writes correctly.
        # Only RAW and WAR dependencies are relevant for correctness.
        
        # Check for WAR dependencies (this writes what someone read)
        for reg in writes_target:
            if reg in last_readers:
                for reader_addr in last_readers[reg]:
                    if reader_addr == addr:
                        continue
                    reader_pos = position_map.get(reader_addr)
                    
                    if reader_pos and is_position_before(reader_pos, instr_pos):
                        barriers = find_barriers_between(reader_pos, instr_pos, barrier_positions)
                        
                        for b_addr in barriers:
                            if b_addr in all_barriers:
                                barriers_used[b_addr] = all_barriers[b_addr]
                        
                        edge = SliceEdge(
                            from_addr=reader_addr,
                            to_addr=addr,
                            dep_type="WAR",
                            registers={reg},
                            crosses_barrier=len(barriers) > 0,
                            barrier_addrs=barriers,
                        )
                        
                        merged = False
                        for existing in edges:
                            if (existing.from_addr == edge.from_addr and 
                                existing.to_addr == edge.to_addr and
                                existing.dep_type == edge.dep_type):
                                existing.registers.add(reg)
                                merged = True
                                break
                        if not merged:
                            edges.append(edge)
        
        # Update last writer for written registers
        for reg in writes_target:
            last_writer[reg] = addr
            # Clear readers since we have a new write
            if reg in last_readers:
                last_readers[reg] = set()
    
    return edges, barriers_used


# =============================================================================
# Main Build Function
# =============================================================================

def build_register_slice(
    amdgcn_file: str,
    target_registers: Set[str]
) -> RegisterSlice:
    """
    Build a complete register slice from an AMDGCN file.
    
    This is the main entry point for the register slice analysis.
    
    Args:
        amdgcn_file: Path to the AMDGCN assembly file
        target_registers: Set of register names to search for
        
    Returns:
        RegisterSlice containing all related instructions and dependencies
    """
    # Parse the AMDGCN file
    parser = AMDGCNParser()
    cfg = parser.parse_file(amdgcn_file)
    
    # Generate DDGs for all blocks
    ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=True)
    
    # Build global position map
    position_map = build_global_position_map(cfg)
    
    # Find all instructions related to target registers
    instructions = find_related_instructions(cfg, ddgs, target_registers)
    
    # Find all barriers
    all_barriers = find_all_barriers(cfg)
    
    # Find dependency edges and detect barrier crossings
    edges, barriers_used = find_dependency_edges(
        instructions, target_registers, position_map, all_barriers
    )
    
    # Build the result
    result = RegisterSlice(
        target_registers=target_registers.copy(),
        instructions=instructions,
        edges=edges,
        barrier_instructions=barriers_used,
        cfg_name=cfg.name,
    )
    
    return result


def build_register_slice_from_cfg(
    cfg: CFG,
    ddgs: Dict[str, DDG],
    target_registers: Set[str]
) -> RegisterSlice:
    """
    Build a register slice from pre-parsed CFG and DDGs.
    
    Args:
        cfg: Pre-parsed Control Flow Graph
        ddgs: Pre-built DDGs
        target_registers: Set of register names to search for
        
    Returns:
        RegisterSlice containing all related instructions and dependencies
    """
    # Build global position map
    position_map = build_global_position_map(cfg)
    
    # Find all instructions related to target registers
    instructions = find_related_instructions(cfg, ddgs, target_registers)
    
    # Find all barriers
    all_barriers = find_all_barriers(cfg)
    
    # Find dependency edges and detect barrier crossings
    edges, barriers_used = find_dependency_edges(
        instructions, target_registers, position_map, all_barriers
    )
    
    # Build the result
    result = RegisterSlice(
        target_registers=target_registers.copy(),
        instructions=instructions,
        edges=edges,
        barrier_instructions=barriers_used,
        cfg_name=cfg.name,
    )
    
    return result


# =============================================================================
# Output Generation - DOT Format
# =============================================================================

def generate_slice_dot(
    slice_result: RegisterSlice,
    max_label_len: int = 60
) -> str:
    """
    Generate DOT format representation of the register slice.
    
    Barrier crossings are shown with special dashed orange edges:
    A -> s_barrier -> B (where A actually depends on B through the barrier)
    
    Args:
        slice_result: The register slice to visualize
        max_label_len: Maximum length for instruction labels
        
    Returns:
        DOT format string
    """
    lines = []
    regs_str = "_".join(sorted(slice_result.target_registers)[:5])
    if len(slice_result.target_registers) > 5:
        regs_str += f"_plus{len(slice_result.target_registers)-5}"
    
    lines.append(f'digraph "RegisterSlice_{regs_str}" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, fontname="Courier", fontsize=9];')
    lines.append('    edge [fontname="Courier", fontsize=8];')
    lines.append('    graph [ranksep=0.5, nodesep=0.3];')
    lines.append('')
    
    # Graph title
    target_regs_display = ", ".join(sorted(slice_result.target_registers))
    lines.append(f'    label="Register Slice\\nTarget: {escape_dot_string(target_regs_display)}\\n'
                 f'Instructions: {slice_result.get_instruction_count()}, '
                 f'Edges: {slice_result.get_edge_count()}, '
                 f'Barrier crossings: {slice_result.get_barrier_crossing_edge_count()}";')
    lines.append('    labelloc=t;')
    lines.append('')
    
    # Add instruction nodes
    lines.append('    // Instruction nodes')
    for addr in slice_result.get_all_addresses_sorted():
        instr = slice_result.instructions[addr]
        _add_instruction_node(lines, instr, slice_result.target_registers, max_label_len)
    
    lines.append('')
    
    # Add barrier nodes
    if slice_result.barrier_instructions:
        lines.append('    // Barrier nodes')
        for addr, barrier in slice_result.barrier_instructions.items():
            node_id = f"n{addr}"
            label = f"[{addr}] {barrier.opcode}\\n--- BARRIER ---"
            lines.append(f'    {node_id} [label="{label}", style=filled, '
                        f'fillcolor=lightyellow, shape=diamond];')
        lines.append('')
    
    # Add edges
    lines.append('    // Dependency edges')
    for edge in slice_result.edges:
        _add_edge(lines, edge)
    
    lines.append('}')
    
    return '\n'.join(lines)


def _add_instruction_node(
    lines: List[str],
    instr: SliceInstruction,
    target_registers: Set[str],
    max_label_len: int
) -> None:
    """Add a single instruction node to the DOT output."""
    node_id = f"n{instr.address}"
    
    # Truncate instruction text
    instr_text = f"{instr.opcode} {instr.operands}"
    if len(instr_text) > max_label_len:
        instr_text = instr_text[:max_label_len-3] + "..."
    instr_text = escape_dot_string(instr_text)
    
    # Show which target registers are read/written
    reads_target = instr.reads & target_registers
    writes_target = instr.writes & target_registers
    
    reads_str = ",".join(sorted(reads_target)[:4])
    if len(reads_target) > 4:
        reads_str += "..."
    writes_str = ",".join(sorted(writes_target)[:4])
    if len(writes_target) > 4:
        writes_str += "..."
    
    # Build label
    label = f"[{instr.address}] {instr_text}"
    if writes_target:
        label += f"\\nW: {writes_str}"
    if reads_target:
        label += f"\\nR: {reads_str}"
    label += f"\\n({instr.block_label})"
    
    # Color based on instruction type
    if instr.is_waitcnt:
        color = "lightgray"
    elif instr.is_barrier:
        color = "lightyellow"
    elif is_lgkm_op(instr.opcode.lower()):
        color = "lightskyblue"
    elif is_vm_op(instr.opcode.lower()):
        color = "khaki"
    elif instr.opcode.lower().startswith('v_mfma'):
        color = "lightgreen"
    else:
        color = "white"
    
    lines.append(f'    {node_id} [label="{label}", style=filled, fillcolor={color}];')


def _add_edge(lines: List[str], edge: SliceEdge) -> None:
    """Add a dependency edge to the DOT output."""
    regs_str = ",".join(sorted(edge.registers)[:3])
    if len(edge.registers) > 3:
        regs_str += f"...(+{len(edge.registers)-3})"
    
    # Edge color based on type
    # Note: WAW is not tracked (not needed for modern GPU hardware)
    type_colors = {
        "RAW": "blue",
        "WAR": "purple",
        "WAIT": "green",
        "AVAIL": "forestgreen",
    }
    color = type_colors.get(edge.dep_type, "black")
    
    if edge.crosses_barrier:
        # For barrier crossings, draw: from -> barrier1 -> barrier2 -> ... -> to
        prev_node = f"n{edge.from_addr}"
        
        for barrier_addr in edge.barrier_addrs:
            barrier_node = f"n{barrier_addr}"
            # Dashed orange edge to barrier
            lines.append(f'    {prev_node} -> {barrier_node} '
                        f'[label="{edge.dep_type}:{regs_str}", color=orange, '
                        f'style=dashed, penwidth=2];')
            prev_node = barrier_node
        
        # Final edge to destination
        to_node = f"n{edge.to_addr}"
        lines.append(f'    {prev_node} -> {to_node} '
                    f'[label="{edge.dep_type}:{regs_str}", color=orange, '
                    f'style=dashed, penwidth=2];')
    else:
        # Normal edge
        lines.append(f'    n{edge.from_addr} -> n{edge.to_addr} '
                    f'[label="{edge.dep_type}:{regs_str}", color={color}];')


# =============================================================================
# Output Generation - JSON Format
# =============================================================================

def generate_slice_json(slice_result: RegisterSlice, indent: int = 2) -> str:
    """
    Generate JSON representation of the register slice.
    
    Args:
        slice_result: The register slice to serialize
        indent: JSON indentation level
        
    Returns:
        JSON string
    """
    return json.dumps(slice_result.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# Output Generation - Text Format
# =============================================================================

def generate_slice_text(slice_result: RegisterSlice) -> str:
    """
    Generate plain text representation of the register slice.
    
    This outputs just the instruction lines in address order.
    
    Args:
        slice_result: The register slice
        
    Returns:
        Text string with one instruction per line
    """
    lines = []
    
    # Header
    target_regs = ", ".join(sorted(slice_result.target_registers))
    lines.append(f"# Register Slice for: {target_regs}")
    lines.append(f"# Function: {slice_result.cfg_name}")
    lines.append(f"# Instructions: {slice_result.get_instruction_count()}")
    lines.append(f"# Edges: {slice_result.get_edge_count()}")
    lines.append(f"# Barrier crossings: {slice_result.get_barrier_crossing_edge_count()}")
    lines.append("")
    
    # Instructions in address order
    for addr in slice_result.get_all_addresses_sorted():
        instr = slice_result.instructions[addr]
        reads_target = instr.reads & slice_result.target_registers
        writes_target = instr.writes & slice_result.target_registers
        
        # Show R/W info as comment
        rw_info = []
        if writes_target:
            rw_info.append(f"W:{','.join(sorted(writes_target))}")
        if reads_target:
            rw_info.append(f"R:{','.join(sorted(reads_target))}")
        
        rw_comment = f"  ; {' '.join(rw_info)}" if rw_info else ""
        lines.append(f"{instr.raw_line}{rw_comment}")
    
    # Barrier instructions
    if slice_result.barrier_instructions:
        lines.append("")
        lines.append("# Barriers crossed by dependencies:")
        for addr in sorted(slice_result.barrier_instructions.keys()):
            barrier = slice_result.barrier_instructions[addr]
            lines.append(f"{barrier.raw_line}")
    
    return '\n'.join(lines)


# =============================================================================
# Save Functions
# =============================================================================

def save_slice_outputs(
    slice_result: RegisterSlice,
    output_dir: str,
    generate_svg: bool = True,
    prefix: str = "slice"
) -> Dict[str, str]:
    """
    Save register slice to multiple output formats.
    
    Args:
        slice_result: The register slice to save
        output_dir: Output directory
        generate_svg: Whether to generate SVG using graphviz
        prefix: File name prefix
        
    Returns:
        Dictionary mapping format name to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename suffix from registers
    regs_suffix = "_".join(sorted(slice_result.target_registers)[:5])
    if len(slice_result.target_registers) > 5:
        regs_suffix += f"_plus{len(slice_result.target_registers)-5}"
    
    base_name = f"{prefix}_{regs_suffix}"
    output_files: Dict[str, str] = {}
    
    # Save JSON
    json_file = os.path.join(output_dir, f"{base_name}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(generate_slice_json(slice_result))
    output_files['json'] = json_file
    print(f"JSON saved to: {json_file}")
    
    # Save DOT
    dot_content = generate_slice_dot(slice_result)
    dot_file = os.path.join(output_dir, f"{base_name}.dot")
    with open(dot_file, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    output_files['dot'] = dot_file
    print(f"DOT saved to: {dot_file}")
    
    # Save SVG
    if generate_svg:
        svg_file = os.path.join(output_dir, f"{base_name}.svg")
        try:
            subprocess.run(
                ['dot', '-Tsvg', dot_file, '-o', svg_file],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output_files['svg'] = svg_file
            print(f"SVG saved to: {svg_file}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate SVG: {e}")
        except FileNotFoundError:
            print("Warning: 'dot' command not found. Install graphviz to generate SVG files.")
    
    # Save TXT
    txt_content = generate_slice_text(slice_result)
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    output_files['txt'] = txt_file
    print(f"TXT saved to: {txt_file}")
    
    return output_files


# =============================================================================
# Utility Functions
# =============================================================================

def parse_register_list(register_str: str) -> Set[str]:
    """
    Parse a comma-separated list of register names.
    
    Handles:
    - Single registers: v3, s7, exec, scc
    - Register ranges: v[0:3], s[4:7]
    
    Args:
        register_str: Comma-separated register names
        
    Returns:
        Set of individual register names
    """
    result = set()
    
    for part in register_str.split(','):
        part = part.strip()
        if not part:
            continue
        
        # Check for range notation
        range_match = re.match(r'([sva])\[(\d+):(\d+)\]', part)
        if range_match:
            prefix = range_match.group(1)
            start = int(range_match.group(2))
            end = int(range_match.group(3))
            for i in range(start, end + 1):
                result.add(f"{prefix}{i}")
        else:
            result.add(part)
    
    return result


def print_slice_summary(slice_result: RegisterSlice) -> None:
    """Print a summary of the register slice."""
    print("\n" + "=" * 70)
    print("Register Slice Summary")
    print("=" * 70)
    
    print(f"\nTarget registers: {', '.join(sorted(slice_result.target_registers))}")
    print(f"Function: {slice_result.cfg_name}")
    print(f"\nStatistics:")
    print(f"  Instructions found: {slice_result.get_instruction_count()}")
    print(f"  Dependency edges: {slice_result.get_edge_count()}")
    print(f"  Barrier crossing edges: {slice_result.get_barrier_crossing_edge_count()}")
    print(f"  Barrier instructions: {len(slice_result.barrier_instructions)}")
    
    # Count by dependency type
    type_counts: Dict[str, int] = {}
    for edge in slice_result.edges:
        type_counts[edge.dep_type] = type_counts.get(edge.dep_type, 0) + 1
    
    if type_counts:
        print(f"\nEdge types:")
        for dep_type, count in sorted(type_counts.items()):
            print(f"  {dep_type}: {count}")
    
    # Show blocks involved
    blocks = set(instr.block_label for instr in slice_result.instructions.values())
    print(f"\nBasic blocks involved: {len(blocks)}")
    for block in sorted(blocks):
        count = sum(1 for i in slice_result.instructions.values() if i.block_label == block)
        print(f"  {block}: {count} instructions")
    
    print("=" * 70 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AMDGCN Register Slice Analyzer - Extract instructions related to specific registers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s input.amdgcn --registers v3,s7,exec,scc
  %(prog)s input.amdgcn --registers "v[0:3],s[4:7]" --output-dir ./slice_output
  %(prog)s input.amdgcn --registers vcc,scc --no-svg --quiet
'''
    )
    parser.add_argument(
        'input',
        help='Input .amdgcn assembly file'
    )
    parser.add_argument(
        '--registers', '-r',
        required=True,
        help='Comma-separated list of registers to search for (e.g., v3,s7,exec,scc or v[0:3])'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./slice_output',
        help='Output directory (default: ./slice_output)'
    )
    parser.add_argument(
        '--no-svg',
        action='store_true',
        help='Skip SVG generation (requires graphviz)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress summary output'
    )
    
    args = parser.parse_args()
    
    # Parse register list
    target_registers = parse_register_list(args.registers)
    
    if not target_registers:
        print("Error: No valid registers specified")
        return 1
    
    if not args.quiet:
        print(f"Searching for registers: {', '.join(sorted(target_registers))}")
        print(f"Input file: {args.input}")
    
    # Build the slice
    try:
        slice_result = build_register_slice(args.input, target_registers)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    # Print summary
    if not args.quiet:
        print_slice_summary(slice_result)
    
    # Save outputs
    save_slice_outputs(
        slice_result,
        args.output_dir,
        generate_svg=not args.no_svg
    )
    
    return 0


if __name__ == '__main__':
    exit(main())

