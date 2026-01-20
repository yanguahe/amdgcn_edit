#!/usr/bin/env python3
"""
AMDGCN Latency Module

This module provides hardware latency information and constraint checking
for AMD MI300 (gfx942) MFMA instructions. It loads configuration from a JSON
file and provides functions to:

1. Classify instructions (MFMA type, VALU, memory ops, etc.)
2. Calculate required wait cycles between instruction pairs
3. Detect latency violations in basic blocks
4. Insert s_nop instructions to fix violations

Usage:
    from amdgcn_latency import (
        load_hardware_info,
        get_mfma_pass_count,
        get_required_latency,
        find_latency_violations,
        InsertLatencyNopsPass
    )
    
    hw_info = load_hardware_info()
    latency = get_required_latency(mfma_instr, valu_instr, hw_info)
"""

import json
import os
import re
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from enum import Enum, auto

from amdgcn_cfg import Instruction, BasicBlock, CFG
from amdgcn_ddg import (
    AnalysisResult,
    DDG,
    InstructionNode,
    parse_instruction_registers,
)


# =============================================================================
# Constants and Enums
# =============================================================================

class InstructionType(Enum):
    """Classification of instruction types for latency calculation."""
    XDL = auto()       # XDLOP: Matrix math on I8, F16, BF16, F32
    SGEMM = auto()     # Single-precision GEMM
    DGEMM = auto()     # Double-precision GEMM (V_MFMA*F64)
    VALU = auto()      # Vector ALU operations
    ACCVGPR_READ = auto()   # v_accvgpr_read_b32
    ACCVGPR_WRITE = auto()  # v_accvgpr_write_b32
    VM_LOAD = auto()   # Vector memory loads (global_load, buffer_load)
    VM_STORE = auto()  # Vector memory stores
    LDS = auto()       # Local data share operations (ds_*)
    SMEM = auto()      # Scalar memory operations (s_load, s_store)
    SALU = auto()      # Scalar ALU operations
    SYNC = auto()      # Synchronization (s_waitcnt, s_barrier)
    NOP = auto()       # s_nop
    EXPORT = auto()    # Export operations
    BRANCH = auto()    # Branch/jump instructions
    OTHER = auto()     # Unknown/other


class MFMAReadType(Enum):
    """How an instruction reads MFMA output registers."""
    SRCC_EXACT_SAME = auto()     # SrcC exactly same as MFMA vDst
    SRCC_OVERLAPPED = auto()     # SrcC overlapped with MFMA vDst
    SRCAB = auto()               # SrcA or SrcB
    VALU_RAW = auto()            # VALU read (RAW hazard)
    VM_LDS_FLAT = auto()         # Memory operation read
    NO_DEPENDENCY = auto()       # No register dependency


# =============================================================================
# Hardware Info Data Classes
# =============================================================================

@dataclass
class MFMAInstructionInfo:
    """Information about a specific MFMA instruction."""
    opcode: str
    passes: int
    type: str  # "XDL", "SGEMM", "DGEMM"
    output_size: int


@dataclass
class LatencyRule:
    """A latency rule from the hardware documentation."""
    name: str
    description: str
    latency: Optional[int] = None
    latency_by_passes: Optional[Dict[str, int]] = None
    comment: str = ""


@dataclass
class HardwareInfo:
    """Hardware information loaded from JSON configuration."""
    target: str
    mfma_instructions: Dict[str, MFMAInstructionInfo]
    latency_rules: Dict[str, LatencyRule]
    instruction_cycles: Dict[str, int]
    default_cycles: int
    snop_max_count: int
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HardwareInfo':
        """Create HardwareInfo from parsed JSON dictionary."""
        # Parse MFMA instructions
        mfma_instructions = {}
        for opcode, info in data.get('mfma_instructions', {}).items():
            mfma_instructions[opcode.lower()] = MFMAInstructionInfo(
                opcode=opcode.lower(),
                passes=info['passes'],
                type=info['type'],
                output_size=info.get('output_size', 4)
            )
        
        # Parse latency rules
        latency_rules = {}
        rules_data = data.get('latency_rules', {})
        for name, rule in rules_data.items():
            if name == 'description':
                continue
            latency_rules[name] = LatencyRule(
                name=name,
                description=rule.get('description', ''),
                latency=rule.get('latency'),
                latency_by_passes={str(k): v for k, v in rule.get('latency_by_passes', {}).items()},
                comment=rule.get('comment', '')
            )
        
        # Parse instruction cycles
        cycles_data = data.get('instruction_cycles', {})
        instruction_cycles = cycles_data.get('patterns', {})
        default_cycles = cycles_data.get('default', 4)
        
        # Parse s_nop info
        snop_info = data.get('snop_info', {})
        snop_max_count = snop_info.get('max_count', 15)
        
        return cls(
            target=data.get('target', 'gfx942'),
            mfma_instructions=mfma_instructions,
            latency_rules=latency_rules,
            instruction_cycles=instruction_cycles,
            default_cycles=default_cycles,
            snop_max_count=snop_max_count
        )


# =============================================================================
# Hardware Info Loading
# =============================================================================

_cached_hardware_info: Optional[HardwareInfo] = None


def get_hardware_info_path() -> str:
    """Get the path to the hardware info JSON file."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, 'gfx942_hardware_info.json')


def load_hardware_info(json_path: Optional[str] = None, force_reload: bool = False) -> HardwareInfo:
    """
    Load hardware information from JSON configuration file.
    
    Args:
        json_path: Path to JSON file. If None, uses default path.
        force_reload: If True, bypass cache and reload from file.
        
    Returns:
        HardwareInfo object containing all hardware configuration.
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist.
        json.JSONDecodeError: If JSON is malformed.
    """
    global _cached_hardware_info
    
    if _cached_hardware_info is not None and not force_reload and json_path is None:
        return _cached_hardware_info
    
    if json_path is None:
        json_path = get_hardware_info_path()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    hw_info = HardwareInfo.from_dict(data)
    
    if json_path == get_hardware_info_path():
        _cached_hardware_info = hw_info
    
    return hw_info


def clear_hardware_info_cache():
    """Clear the cached hardware info."""
    global _cached_hardware_info
    _cached_hardware_info = None


# =============================================================================
# Instruction Classification
# =============================================================================

def normalize_mfma_opcode(opcode: str) -> str:
    """
    Normalize an MFMA opcode by removing underscores and lowercasing.
    
    This helps match different naming conventions, e.g.:
    - v_mfma_f32_16x16x16_bf16
    - v_mfma_f32_16x16x16bf16
    """
    return opcode.lower().replace('_', '')


def get_mfma_info(opcode: str, hw_info: HardwareInfo) -> Optional[MFMAInstructionInfo]:
    """
    Get MFMA instruction info for the given opcode.
    
    Args:
        opcode: The MFMA opcode (e.g., 'v_mfma_f32_16x16x16_bf16')
        hw_info: Hardware info object
        
    Returns:
        MFMAInstructionInfo if found, None otherwise.
    """
    opcode_lower = opcode.lower()
    
    # Direct lookup
    if opcode_lower in hw_info.mfma_instructions:
        return hw_info.mfma_instructions[opcode_lower]
    
    # Try normalized lookup (remove underscores)
    normalized = normalize_mfma_opcode(opcode_lower)
    for key, info in hw_info.mfma_instructions.items():
        if normalize_mfma_opcode(key) == normalized:
            return info
    
    # Pattern matching for unknown MFMA variants
    # Parse the opcode to extract dimensions
    match = re.match(r'v_mfma_f(\d+)_(\d+)x(\d+)x(\d+).*', opcode_lower)
    if match:
        # Try to find a similar instruction
        dims = f"{match.group(2)}x{match.group(3)}x{match.group(4)}"
        for key, info in hw_info.mfma_instructions.items():
            if dims in key:
                return info
    
    return None


def get_mfma_pass_count(opcode: str, hw_info: Optional[HardwareInfo] = None) -> int:
    """
    Get the pass count for an MFMA instruction.
    
    Args:
        opcode: The MFMA opcode
        hw_info: Hardware info (will load default if None)
        
    Returns:
        Number of passes (2, 4, 8, or 16), or 4 as default.
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    info = get_mfma_info(opcode, hw_info)
    if info:
        return info.passes
    
    # Default for unknown MFMA
    return 4


def get_mfma_type(opcode: str, hw_info: Optional[HardwareInfo] = None) -> InstructionType:
    """
    Get the MFMA type (XDL, SGEMM, or DGEMM) for an MFMA instruction.
    
    Args:
        opcode: The MFMA opcode
        hw_info: Hardware info (will load default if None)
        
    Returns:
        InstructionType.XDL, SGEMM, or DGEMM
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    info = get_mfma_info(opcode, hw_info)
    if info:
        type_str = info.type.upper()
        if type_str == 'XDL':
            return InstructionType.XDL
        elif type_str == 'SGEMM':
            return InstructionType.SGEMM
        elif type_str == 'DGEMM':
            return InstructionType.DGEMM
    
    # Default: XDL for most modern MFMA
    opcode_lower = opcode.lower()
    if 'f64' in opcode_lower:
        return InstructionType.DGEMM
    elif 'f32' in opcode_lower and ('x1_' in opcode_lower or 'x2_' in opcode_lower):
        return InstructionType.SGEMM
    
    return InstructionType.XDL


def classify_instruction(instr: Instruction, hw_info: Optional[HardwareInfo] = None) -> InstructionType:
    """
    Classify an instruction into one of the InstructionType categories.
    
    Args:
        instr: The instruction to classify
        hw_info: Hardware info (will load default if None)
        
    Returns:
        The InstructionType for this instruction.
    """
    opcode = instr.opcode.lower()
    
    # MFMA instructions
    if opcode.startswith('v_mfma'):
        return get_mfma_type(opcode, hw_info)
    
    # ACCVGPR read/write
    if opcode.startswith('v_accvgpr_read'):
        return InstructionType.ACCVGPR_READ
    if opcode.startswith('v_accvgpr_write'):
        return InstructionType.ACCVGPR_WRITE
    
    # Synchronization
    if opcode in ('s_waitcnt', 's_barrier', 's_wakeup'):
        return InstructionType.SYNC
    if opcode.startswith('s_nop'):
        return InstructionType.NOP
    
    # Memory operations
    if opcode.startswith(('global_load', 'buffer_load', 'flat_load')):
        return InstructionType.VM_LOAD
    if opcode.startswith(('global_store', 'buffer_store', 'flat_store')):
        return InstructionType.VM_STORE
    if opcode.startswith('ds_'):
        return InstructionType.LDS
    if opcode.startswith(('s_load', 's_store', 's_buffer_load')):
        return InstructionType.SMEM
    
    # Export
    if opcode.startswith('exp'):
        return InstructionType.EXPORT
    
    # Branches
    if opcode.startswith(('s_branch', 's_cbranch', 's_call', 's_setpc', 's_getpc')):
        return InstructionType.BRANCH
    if opcode.startswith('s_endpgm'):
        return InstructionType.BRANCH
    
    # Vector ALU
    if opcode.startswith('v_'):
        return InstructionType.VALU
    
    # Scalar ALU
    if opcode.startswith('s_'):
        return InstructionType.SALU
    
    return InstructionType.OTHER


def is_mfma_instruction(instr: Instruction) -> bool:
    """Check if instruction is an MFMA instruction."""
    return instr.opcode.lower().startswith('v_mfma')


def is_accvgpr_read(instr: Instruction) -> bool:
    """Check if instruction is v_accvgpr_read."""
    return instr.opcode.lower().startswith('v_accvgpr_read')


def is_valu_instruction(instr: Instruction) -> bool:
    """Check if instruction is a vector ALU instruction (excluding MFMA)."""
    opcode = instr.opcode.lower()
    return opcode.startswith('v_') and not opcode.startswith('v_mfma')


def is_memory_read_instruction(instr: Instruction) -> bool:
    """Check if instruction reads from memory."""
    opcode = instr.opcode.lower()
    return opcode.startswith(('global_load', 'buffer_load', 'flat_load', 'ds_read', 'ds_load'))


# =============================================================================
# Register Dependency Analysis
# =============================================================================

def parse_agpr_range(operand: str) -> Set[str]:
    """
    Parse AGPR operand to extract individual register names.
    
    Examples:
        'a0' -> {'a0'}
        'a[0:3]' -> {'a0', 'a1', 'a2', 'a3'}
    """
    result = set()
    operand = operand.strip()
    
    # Range format: a[start:end]
    match = re.match(r'a\[(\d+):(\d+)\]', operand)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        for i in range(start, end + 1):
            result.add(f'a{i}')
        return result
    
    # Single register: a0, a1, etc.
    match = re.match(r'a(\d+)', operand)
    if match:
        result.add(operand)
        return result
    
    return result


def get_mfma_dst_registers(instr: Instruction) -> Set[str]:
    """
    Get the destination AGPR registers written by an MFMA instruction.
    
    Args:
        instr: An MFMA instruction
        
    Returns:
        Set of AGPR register names (e.g., {'a0', 'a1', 'a2', 'a3'})
    """
    if not is_mfma_instruction(instr):
        return set()
    
    # MFMA format: v_mfma_... dst, src0, src1, src2
    # First operand is destination
    operands = instr.operands.split(',')
    if operands:
        return parse_agpr_range(operands[0].strip())
    
    return set()


def get_instruction_src_registers(instr: Instruction) -> Set[str]:
    """
    Get source registers read by an instruction.
    """
    _, uses = parse_instruction_registers(instr)
    return uses


def get_instruction_dst_registers(instr: Instruction) -> Set[str]:
    """
    Get destination registers written by an instruction.
    """
    defs, _ = parse_instruction_registers(instr)
    return defs


def check_register_overlap(regs1: Set[str], regs2: Set[str]) -> bool:
    """Check if two register sets have any overlap."""
    return bool(regs1 & regs2)


def check_exact_same_registers(regs1: Set[str], regs2: Set[str]) -> bool:
    """Check if two register sets are exactly the same."""
    return regs1 == regs2


def analyze_mfma_read_type(
    mfma_instr: Instruction,
    reader_instr: Instruction
) -> MFMAReadType:
    """
    Analyze how a reader instruction accesses MFMA output registers.
    
    Args:
        mfma_instr: The MFMA instruction that writes registers
        reader_instr: The instruction that reads registers
        
    Returns:
        MFMAReadType indicating the type of access
    """
    mfma_dst = get_mfma_dst_registers(mfma_instr)
    reader_src = get_instruction_src_registers(reader_instr)
    
    # Check for any dependency
    if not check_register_overlap(mfma_dst, reader_src):
        return MFMAReadType.NO_DEPENDENCY
    
    reader_type = classify_instruction(reader_instr)
    
    # Check if reader is another MFMA using as SrcC
    if is_mfma_instruction(reader_instr):
        # MFMA format: v_mfma dst, srcA, srcB, srcC
        # SrcC is the accumulator (last operand)
        operands = reader_instr.operands.split(',')
        if len(operands) >= 4:
            srcc_regs = parse_agpr_range(operands[3].strip())
            if srcc_regs:
                if check_exact_same_registers(mfma_dst, srcc_regs):
                    return MFMAReadType.SRCC_EXACT_SAME
                elif check_register_overlap(mfma_dst, srcc_regs):
                    return MFMAReadType.SRCC_OVERLAPPED
            
            # Check SrcA and SrcB
            srcab_regs = set()
            for op in operands[1:3]:
                _, uses = parse_instruction_registers(
                    Instruction(0, '', op.strip(), '')
                )
                srcab_regs.update(uses)
            
            if check_register_overlap(mfma_dst, srcab_regs):
                return MFMAReadType.SRCAB
    
    # VALU read (including v_accvgpr_read)
    if reader_type in (InstructionType.VALU, InstructionType.ACCVGPR_READ):
        return MFMAReadType.VALU_RAW
    
    # Memory/LDS/FLAT/Export read
    if reader_type in (InstructionType.VM_LOAD, InstructionType.VM_STORE,
                       InstructionType.LDS, InstructionType.EXPORT):
        return MFMAReadType.VM_LDS_FLAT
    
    return MFMAReadType.VALU_RAW


# =============================================================================
# Latency Calculation
# =============================================================================

def get_required_latency(
    first_instr: Instruction,
    second_instr: Instruction,
    hw_info: Optional[HardwareInfo] = None
) -> int:
    """
    Calculate the required number of independent instructions between two instructions.
    
    This implements the latency rules from AMD MI300 ISA documentation Table 37.
    
    Args:
        first_instr: The first (earlier) instruction
        second_instr: The second (later) instruction
        hw_info: Hardware info (will load default if None)
        
    Returns:
        Number of independent instructions required between them.
        0 means they can be adjacent.
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    first_type = classify_instruction(first_instr, hw_info)
    second_type = classify_instruction(second_instr, hw_info)
    
    # Non-MFMA to MFMA
    if first_type == InstructionType.VALU and second_type in (
        InstructionType.XDL, InstructionType.SGEMM, InstructionType.DGEMM
    ):
        # Check if first writes VGPR that second reads
        first_dst = get_instruction_dst_registers(first_instr)
        second_src = get_instruction_src_registers(second_instr)
        if check_register_overlap(first_dst, second_src):
            return 2  # non_dlops_valu_to_mfma
        return 0
    
    # MFMA to other instructions
    if first_type in (InstructionType.XDL, InstructionType.SGEMM, InstructionType.DGEMM):
        mfma_dst = get_mfma_dst_registers(first_instr)
        second_src = get_instruction_src_registers(second_instr)
        
        # No register dependency
        if not check_register_overlap(mfma_dst, second_src):
            return 0
        
        passes = get_mfma_pass_count(first_instr.opcode, hw_info)
        read_type = analyze_mfma_read_type(first_instr, second_instr)
        
        # Get appropriate latency rule
        if first_type == InstructionType.XDL:
            if read_type == MFMAReadType.SRCC_EXACT_SAME:
                rule = hw_info.latency_rules.get('xdl_to_srcc_exact_same')
            elif read_type == MFMAReadType.SRCC_OVERLAPPED:
                rule = hw_info.latency_rules.get('xdl_to_srcc_overlapped')
            elif read_type == MFMAReadType.SRCAB:
                rule = hw_info.latency_rules.get('xdl_to_mfma_srcab')
            else:  # VALU_RAW or VM_LDS_FLAT
                rule = hw_info.latency_rules.get('xdl_to_valu_vm_lds_flat')
        
        elif first_type == InstructionType.SGEMM:
            if read_type == MFMAReadType.SRCC_EXACT_SAME:
                rule = hw_info.latency_rules.get('sgemm_to_srcc_exact_same')
            elif read_type == MFMAReadType.SRCC_OVERLAPPED:
                rule = hw_info.latency_rules.get('sgemm_to_srcc_overlapped')
            elif read_type == MFMAReadType.SRCAB:
                rule = hw_info.latency_rules.get('sgemm_to_mfma_srcab')
            else:
                rule = hw_info.latency_rules.get('sgemm_to_valu_vm_lds_flat')
        
        elif first_type == InstructionType.DGEMM:
            # DGEMM has special rules based on specific instruction
            opcode_lower = first_instr.opcode.lower()
            if '16x16x4' in opcode_lower:
                if read_type == MFMAReadType.SRCC_EXACT_SAME:
                    rule = hw_info.latency_rules.get('dgemm_16x16x4_to_same_srcc')
                elif read_type == MFMAReadType.SRCAB:
                    rule = hw_info.latency_rules.get('dgemm_16x16x4_to_srcab')
                elif read_type == MFMAReadType.VM_LDS_FLAT:
                    rule = hw_info.latency_rules.get('dgemm_16x16x4_to_vm_lds_flat')
                else:
                    rule = hw_info.latency_rules.get('dgemm_16x16x4_to_valu')
            elif '4x4x4' in opcode_lower:
                if read_type == MFMAReadType.SRCC_EXACT_SAME:
                    rule = hw_info.latency_rules.get('dgemm_4x4x4_to_same_srcc')
                elif read_type == MFMAReadType.SRCAB:
                    rule = hw_info.latency_rules.get('dgemm_4x4x4_to_srcab')
                elif read_type == MFMAReadType.VM_LDS_FLAT:
                    rule = hw_info.latency_rules.get('dgemm_4x4x4_to_vm_lds_flat')
                else:
                    rule = hw_info.latency_rules.get('dgemm_4x4x4_to_valu')
            else:
                rule = None
        else:
            rule = None
        
        if rule:
            if rule.latency is not None:
                return rule.latency
            elif rule.latency_by_passes:
                return rule.latency_by_passes.get(str(passes), 0)
        
        # Fallback: use conservative estimate based on passes
        if first_type in (InstructionType.XDL, InstructionType.SGEMM):
            pass_to_latency = {2: 5, 4: 7, 8: 11, 16: 19}
            return pass_to_latency.get(passes, 7)
    
    return 0


def count_independent_instructions(
    block: BasicBlock,
    start_idx: int,
    end_idx: int,
    first_instr: Instruction,
    hw_info: Optional[HardwareInfo] = None
) -> int:
    """
    Count independent instructions/cycles between start_idx and end_idx.
    
    An instruction is independent if it doesn't have data dependencies
    with the first instruction.
    
    Special handling for s_nop:
    - s_nop N counts as N+1 cycles of delay
    
    Args:
        block: The basic block
        start_idx: Index after the first instruction
        end_idx: Index of the second instruction (exclusive)
        first_instr: The first instruction to check dependencies against
        hw_info: Hardware info
        
    Returns:
        Number of independent instructions/cycles
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    first_dst = get_instruction_dst_registers(first_instr)
    count = 0
    
    for i in range(start_idx, end_idx):
        instr = block.instructions[i]
        instr_type = classify_instruction(instr, hw_info)
        
        # Skip synchronization (but NOT nops)
        if instr_type == InstructionType.SYNC:
            continue
        
        # s_nop N provides N+1 cycles of delay
        if instr_type == InstructionType.NOP:
            # Parse the operand to get N
            try:
                n = int(instr.operands.strip())
                count += n + 1  # s_nop N = N+1 cycles
            except (ValueError, AttributeError):
                count += 1  # Default to 1 cycle if parsing fails
            continue
        
        # Check if instruction uses any of the first instruction's destinations
        instr_src = get_instruction_src_registers(instr)
        if not check_register_overlap(first_dst, instr_src):
            count += 1
    
    return count


# =============================================================================
# Latency Violation Detection
# =============================================================================

@dataclass
class LatencyViolation:
    """Represents a latency constraint violation."""
    first_idx: int
    first_instr: Instruction
    second_idx: int
    second_instr: Instruction
    required_latency: int
    actual_independent: int
    nops_needed: int
    
    def __str__(self) -> str:
        return (
            f"Violation at [{self.first_idx}] -> [{self.second_idx}]: "
            f"need {self.required_latency} independent instrs, have {self.actual_independent}, "
            f"need {self.nops_needed} nops"
        )


def find_latency_violations(
    block: BasicBlock,
    ddg: Optional[DDG] = None,
    hw_info: Optional[HardwareInfo] = None
) -> List[LatencyViolation]:
    """
    Find all latency constraint violations in a basic block.
    
    This scans for MFMA instructions and checks if subsequent instructions
    that depend on their outputs have sufficient distance.
    
    Args:
        block: The basic block to analyze
        ddg: Data dependency graph (optional, for additional info)
        hw_info: Hardware info (will load default if None)
        
    Returns:
        List of LatencyViolation objects describing each violation.
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    violations = []
    
    for i, instr in enumerate(block.instructions):
        if not is_mfma_instruction(instr):
            continue
        
        mfma_dst = get_mfma_dst_registers(instr)
        if not mfma_dst:
            continue
        
        # Check all subsequent instructions
        for j in range(i + 1, len(block.instructions)):
            reader = block.instructions[j]
            reader_src = get_instruction_src_registers(reader)
            
            # Skip if no register dependency
            if not check_register_overlap(mfma_dst, reader_src):
                continue
            
            # Calculate required latency
            required = get_required_latency(instr, reader, hw_info)
            if required == 0:
                continue
            
            # Count actual independent instructions
            actual = count_independent_instructions(block, i + 1, j, instr, hw_info)
            
            if actual < required:
                nops_needed = required - actual
                violations.append(LatencyViolation(
                    first_idx=i,
                    first_instr=instr,
                    second_idx=j,
                    second_instr=reader,
                    required_latency=required,
                    actual_independent=actual,
                    nops_needed=nops_needed
                ))
            
            # Only check the first dependent instruction for each MFMA
            break
    
    return violations


# =============================================================================
# s_nop Instruction Creation
# =============================================================================

def create_snop_instruction(count: int, address: int = 0) -> Instruction:
    """
    Create an s_nop instruction.
    
    Note: s_nop N waits for N+1 cycles. So s_nop 6 = 7 cycle wait.
    
    Args:
        count: The operand for s_nop (0-15)
        address: The address/line number for the instruction
        
    Returns:
        Instruction object for s_nop
    """
    if count < 0:
        count = 0
    elif count > 15:
        count = 15
    
    return Instruction(
        address=address,
        opcode='s_nop',
        operands=str(count),
        raw_line=f'\ts_nop {count}'
    )


def calculate_snop_count(nops_needed: int) -> List[int]:
    """
    Calculate s_nop instruction(s) needed to achieve the required delay.
    
    s_nop N waits for N+1 cycles. For MFMA latency hiding, s_nop N is equivalent
    to N+1 independent instructions. So we should use the minimum number of
    s_nop instructions to achieve the required delay.
    
    Examples:
        - Need 7 cycles: use s_nop 6 (single instruction, 7 cycles)
        - Need 16 cycles: use s_nop 15 (single instruction, 16 cycles)
        - Need 20 cycles: use s_nop 15 + s_nop 3 (two instructions, 16+4=20 cycles)
    
    Args:
        nops_needed: Number of cycles/instructions to wait
        
    Returns:
        List of s_nop operand values (each value N means s_nop N = N+1 cycles)
    """
    if nops_needed <= 0:
        return []
    
    result = []
    remaining = nops_needed
    
    while remaining > 0:
        # s_nop N provides N+1 cycles of delay
        # Maximum operand is 15 (provides 16 cycles)
        if remaining >= 16:
            result.append(15)  # s_nop 15 = 16 cycles
            remaining -= 16
        else:
            # Need 'remaining' cycles, use s_nop (remaining-1)
            result.append(remaining - 1)  # s_nop (remaining-1) = remaining cycles
            remaining = 0
    
    return result


# =============================================================================
# Latency Pass
# =============================================================================

class InsertLatencyNopsPass:
    """
    Pass that inserts s_nop instructions to fix MFMA latency violations.
    
    This pass scans each basic block for latency violations and inserts
    s_nop instructions as needed to ensure correct execution.
    """
    
    @property
    def name(self) -> str:
        return "InsertLatencyNopsPass"
    
    @property
    def description(self) -> str:
        return "Insert s_nop instructions to fix MFMA latency constraint violations"
    
    def __init__(self, hw_info: Optional[HardwareInfo] = None):
        """
        Initialize the pass.
        
        Args:
            hw_info: Hardware info (will load default if None)
        """
        self.hw_info = hw_info if hw_info else load_hardware_info()
        self.inserted_nops: List[Tuple[str, int, int]] = []  # (block_label, index, count)
    
    def run(self, result: AnalysisResult) -> bool:
        """
        Run the pass on the analysis result.
        
        Args:
            result: The AnalysisResult containing CFG and DDGs
            
        Returns:
            True if any nops were inserted, False otherwise
        """
        changed = False
        self.inserted_nops = []
        
        for block_label, block in result.cfg.blocks.items():
            ddg = result.ddgs.get(block_label)
            
            # Find violations
            violations = find_latency_violations(block, ddg, self.hw_info)
            
            if not violations:
                continue
            
            # Consolidate violations by reader instruction index
            # Multiple MFMA instructions may violate latency to the same reader
            # We only need to insert enough nops to satisfy the MAXIMUM requirement
            violations_by_reader: Dict[int, LatencyViolation] = {}
            for v in violations:
                reader_idx = v.second_idx
                if reader_idx not in violations_by_reader:
                    violations_by_reader[reader_idx] = v
                elif v.nops_needed > violations_by_reader[reader_idx].nops_needed:
                    violations_by_reader[reader_idx] = v
            
            # Insert nops (process in reverse order by reader index to maintain indices)
            for reader_idx in sorted(violations_by_reader.keys(), reverse=True):
                violation = violations_by_reader[reader_idx]
                nops = self._insert_nops_for_violation(block, violation)
                if nops > 0:
                    changed = True
                    self.inserted_nops.append((block_label, violation.second_idx, nops))
        
        return changed
    
    def _insert_nops_for_violation(
        self,
        block: BasicBlock,
        violation: LatencyViolation
    ) -> int:
        """
        Insert s_nop instructions to fix a latency violation.
        
        Args:
            block: The basic block
            violation: The violation to fix
            
        Returns:
            Number of nops inserted
        """
        snop_counts = calculate_snop_count(violation.nops_needed)
        
        # Insert before the second instruction
        insert_idx = violation.second_idx
        
        for count in snop_counts:
            snop = create_snop_instruction(count, address=0)
            block.instructions.insert(insert_idx, snop)
            insert_idx += 1
        
        return len(snop_counts)
    
    def get_report(self) -> str:
        """Get a report of all nops inserted."""
        if not self.inserted_nops:
            return "No s_nop instructions inserted."
        
        lines = ["Inserted s_nop instructions:"]
        for block_label, idx, count in self.inserted_nops:
            lines.append(f"  {block_label}[{idx}]: {count} s_nop(s)")
        return "\n".join(lines)


# =============================================================================
# Instruction Cycles Helper
# =============================================================================

def get_instruction_cycles(opcode: str, hw_info: Optional[HardwareInfo] = None) -> int:
    """
    Get the number of cycles required to issue an instruction.
    
    This replaces the hardcoded function in amdgcn_passes.py.
    
    Args:
        opcode: The instruction opcode
        hw_info: Hardware info (will load default if None)
        
    Returns:
        Number of cycles for the instruction
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    opcode_lower = opcode.lower()
    
    for pattern, cycles in hw_info.instruction_cycles.items():
        # Convert glob pattern to regex
        if fnmatch.fnmatch(opcode_lower, pattern):
            return cycles
    
    return hw_info.default_cycles


# =============================================================================
# Utility Functions
# =============================================================================

def validate_block_latency(
    block: BasicBlock,
    hw_info: Optional[HardwareInfo] = None
) -> Tuple[bool, List[LatencyViolation]]:
    """
    Validate that a basic block has no latency violations.
    
    Args:
        block: The basic block to validate
        hw_info: Hardware info
        
    Returns:
        (is_valid, violations) tuple
    """
    violations = find_latency_violations(block, hw_info=hw_info)
    return len(violations) == 0, violations


def check_move_preserves_latency(
    block: BasicBlock,
    from_idx: int,
    to_idx: int,
    hw_info: Optional[HardwareInfo] = None
) -> bool:
    """
    Check if moving an instruction would violate latency constraints.
    
    Args:
        block: The basic block
        from_idx: Current index of instruction
        to_idx: Target index
        hw_info: Hardware info
        
    Returns:
        True if move is safe, False if it would create a violation
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    instr = block.instructions[from_idx]
    
    # Check if instruction is MFMA
    if is_mfma_instruction(instr):
        mfma_dst = get_mfma_dst_registers(instr)
        
        # If moving down, check distance to dependent instructions
        if to_idx > from_idx:
            for j in range(from_idx + 1, len(block.instructions)):
                reader = block.instructions[j]
                reader_src = get_instruction_src_registers(reader)
                
                if check_register_overlap(mfma_dst, reader_src):
                    required = get_required_latency(instr, reader, hw_info)
                    # New distance after move
                    new_distance = j - to_idx - 1 if j > to_idx else j - from_idx - 1
                    if new_distance < required:
                        return False
                    break
    
    # Check if instruction reads MFMA output
    instr_src = get_instruction_src_registers(instr)
    
    # If moving up, check distance from preceding MFMAs
    if to_idx < from_idx:
        for i in range(to_idx - 1, -1, -1):
            prev_instr = block.instructions[i]
            if is_mfma_instruction(prev_instr):
                mfma_dst = get_mfma_dst_registers(prev_instr)
                if check_register_overlap(mfma_dst, instr_src):
                    required = get_required_latency(prev_instr, instr, hw_info)
                    new_distance = to_idx - i - 1
                    if new_distance < required:
                        return False
    
    return True


@dataclass
class LatencyNopsResult:
    """Result of calculating nops needed for a move operation."""
    needs_nops: bool
    nops_count: int
    insert_position: int
    mfma_idx: int = -1
    required_latency: int = 0
    new_distance: int = 0


def calculate_latency_nops_for_move(
    block: BasicBlock,
    from_idx: int,
    to_idx: int,
    hw_info: Optional[HardwareInfo] = None
) -> LatencyNopsResult:
    """
    Calculate how many s_nop instructions are needed to allow a move operation.
    
    This is used when auto_insert_nops is enabled in MoveInstructionPass.
    Instead of blocking the move, we calculate how many nops need to be inserted
    to satisfy the latency constraints.
    
    Args:
        block: The basic block
        from_idx: Current index of instruction to move
        to_idx: Target index
        hw_info: Hardware info
        
    Returns:
        LatencyNopsResult with information about nops needed
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    instr = block.instructions[from_idx]
    
    # Check if instruction is MFMA moving down
    if is_mfma_instruction(instr):
        mfma_dst = get_mfma_dst_registers(instr)
        
        if to_idx > from_idx:
            for j in range(from_idx + 1, len(block.instructions)):
                reader = block.instructions[j]
                reader_src = get_instruction_src_registers(reader)
                
                if check_register_overlap(mfma_dst, reader_src):
                    required = get_required_latency(instr, reader, hw_info)
                    new_distance = j - to_idx - 1 if j > to_idx else j - from_idx - 1
                    if new_distance < required:
                        nops_needed = required - new_distance
                        # Insert nops after the MFMA's new position (to_idx)
                        return LatencyNopsResult(
                            needs_nops=True,
                            nops_count=nops_needed,
                            insert_position=to_idx + 1,
                            mfma_idx=from_idx,
                            required_latency=required,
                            new_distance=new_distance
                        )
                    break
    
    # Check if instruction reads MFMA output and is moving up
    instr_src = get_instruction_src_registers(instr)
    
    if to_idx < from_idx:
        for i in range(to_idx - 1, -1, -1):
            prev_instr = block.instructions[i]
            if is_mfma_instruction(prev_instr):
                mfma_dst = get_mfma_dst_registers(prev_instr)
                if check_register_overlap(mfma_dst, instr_src):
                    required = get_required_latency(prev_instr, instr, hw_info)
                    new_distance = to_idx - i - 1
                    if new_distance < required:
                        nops_needed = required - new_distance
                        # Insert nops before the reader's new position (to_idx)
                        return LatencyNopsResult(
                            needs_nops=True,
                            nops_count=nops_needed,
                            insert_position=to_idx,
                            mfma_idx=i,
                            required_latency=required,
                            new_distance=new_distance
                        )
                    break
    
    return LatencyNopsResult(
        needs_nops=False,
        nops_count=0,
        insert_position=0
    )


def insert_latency_nops(
    block: BasicBlock,
    position: int,
    count: int
) -> int:
    """
    Insert s_nop instructions at the given position to satisfy latency constraints.
    
    Args:
        block: The basic block
        position: Index to insert at
        count: Number of independent cycles/instructions to wait
        
    Returns:
        Number of s_nop instructions inserted
    """
    snop_counts = calculate_snop_count(count)
    
    for i, snop_count in enumerate(snop_counts):
        snop = create_snop_instruction(snop_count)
        block.instructions.insert(position + i, snop)
    
    return len(snop_counts)


def check_move_side_effects_on_latency(
    block: BasicBlock,
    from_idx: int,
    to_idx: int,
    hw_info: Optional[HardwareInfo] = None
) -> bool:
    """
    Check if moving an instruction would cause side effects that violate latency constraints.
    
    When moving instruction A from from_idx to to_idx:
    - If to_idx > from_idx (moving down): Instructions in [from_idx+1, to_idx] shift up by 1
    - If to_idx < from_idx (moving up): Instructions in [to_idx, from_idx-1] shift down by 1
    
    This function checks if any of the shifted instructions would violate MFMA latency.
    
    Args:
        block: The basic block
        from_idx: Current index of instruction being moved
        to_idx: Target index
        hw_info: Hardware info
        
    Returns:
        True if move is safe (no side effects on latency), False if would cause violation
    """
    if hw_info is None:
        hw_info = load_hardware_info()
    
    if to_idx > from_idx:
        # Moving down: instructions in [from_idx+1, to_idx] shift up (index decreases by 1)
        for shifted_idx in range(from_idx + 1, to_idx + 1):
            shifted_instr = block.instructions[shifted_idx]
            shifted_src = get_instruction_src_registers(shifted_instr)
            
            # Check if this shifted instruction reads MFMA output
            # and if shifting it up would violate latency
            for mfma_idx in range(shifted_idx - 1, -1, -1):
                if mfma_idx == from_idx:
                    # Skip the instruction being moved (it won't be there after move)
                    continue
                
                mfma_instr = block.instructions[mfma_idx]
                if is_mfma_instruction(mfma_instr):
                    mfma_dst = get_mfma_dst_registers(mfma_instr)
                    if check_register_overlap(mfma_dst, shifted_src):
                        required = get_required_latency(mfma_instr, shifted_instr, hw_info)
                        if required > 0:
                            # Current distance (before move)
                            current_distance = shifted_idx - mfma_idx - 1
                            # New distance after move (shifted_idx decreases by 1)
                            # Also need to account for mfma_idx potentially shifting
                            new_mfma_idx = mfma_idx if mfma_idx < from_idx else mfma_idx
                            new_shifted_idx = shifted_idx - 1
                            new_distance = new_shifted_idx - new_mfma_idx - 1
                            
                            if new_distance < required:
                                return False
                        break  # Only check first relevant MFMA
    
    elif to_idx < from_idx:
        # Moving up: instructions in [to_idx, from_idx-1] shift down (index increases by 1)
        for shifted_idx in range(to_idx, from_idx):
            shifted_instr = block.instructions[shifted_idx]
            
            # If this is an MFMA, check if any dependent reader above from_idx
            # would now be too close (because MFMA shifted down)
            if is_mfma_instruction(shifted_instr):
                mfma_dst = get_mfma_dst_registers(shifted_instr)
                new_mfma_idx = shifted_idx + 1
                
                # Check readers that are not being shifted (above from_idx)
                for reader_idx in range(from_idx + 1, len(block.instructions)):
                    reader_instr = block.instructions[reader_idx]
                    reader_src = get_instruction_src_registers(reader_instr)
                    
                    if check_register_overlap(mfma_dst, reader_src):
                        required = get_required_latency(shifted_instr, reader_instr, hw_info)
                        if required > 0:
                            # Current distance
                            current_distance = reader_idx - shifted_idx - 1
                            # New distance after MFMA shifts down by 1
                            new_distance = reader_idx - new_mfma_idx - 1
                            
                            if new_distance < required:
                                return False
                        break
    
    return True

