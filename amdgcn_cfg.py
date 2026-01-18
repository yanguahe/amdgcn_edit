#!/usr/bin/env python3
"""
AMDGCN Assembly to Control Flow Graph (CFG) Parser

This module parses AMDGCN assembly files and generates a Control Flow Graph
in DOT format for visualization with Graphviz.

Usage:
    python amdgcn_cfg.py <input.amdgcn> [output.dot]
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Instruction:
    """Represents a single AMDGCN instruction."""
    address: int           # Line number in source (used as pseudo-address)
    opcode: str            # Instruction mnemonic
    operands: str          # Operand string
    raw_line: str          # Original line text
    is_branch: bool = False
    is_conditional: bool = False
    is_terminator: bool = False
    branch_target: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize instruction to dictionary."""
        return {
            'address': self.address,
            'opcode': self.opcode,
            'operands': self.operands,
            'raw_line': self.raw_line,
            'is_branch': self.is_branch,
            'is_conditional': self.is_conditional,
            'is_terminator': self.is_terminator,
            'branch_target': self.branch_target,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Instruction':
        """Deserialize instruction from dictionary."""
        return cls(
            address=data['address'],
            opcode=data['opcode'],
            operands=data['operands'],
            raw_line=data['raw_line'],
            is_branch=data.get('is_branch', False),
            is_conditional=data.get('is_conditional', False),
            is_terminator=data.get('is_terminator', False),
            branch_target=data.get('branch_target'),
        )


@dataclass
class BasicBlock:
    """Represents a basic block in the CFG."""
    label: str
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    # Raw lines for this block (all lines including directives, comments, etc.)
    # Maps line_number -> raw_line_content
    raw_lines: Dict[int, str] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        return len(self.instructions) == 0
    
    def get_terminator(self) -> Optional[Instruction]:
        """Get the terminating instruction of this block."""
        if self.instructions:
            return self.instructions[-1]
        return None
    
    def get_last_branch(self) -> Optional[Instruction]:
        """Get the last branch instruction in the block (may not be at the very end)."""
        for instr in reversed(self.instructions):
            if instr.is_branch or instr.is_terminator:
                return instr
        return None
    
    def get_raw_lines_in_order(self) -> List[str]:
        """Get raw lines in line number order."""
        if not self.raw_lines:
            return []
        return [self.raw_lines[ln] for ln in sorted(self.raw_lines.keys())]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize basic block to dictionary."""
        return {
            'label': self.label,
            'instructions': [instr.to_dict() for instr in self.instructions],
            'successors': self.successors.copy(),
            'predecessors': self.predecessors.copy(),
            'start_line': self.start_line,
            'end_line': self.end_line,
            'raw_lines': {str(k): v for k, v in self.raw_lines.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicBlock':
        """Deserialize basic block from dictionary."""
        block = cls(
            label=data['label'],
            start_line=data.get('start_line', 0),
            end_line=data.get('end_line', 0),
        )
        block.instructions = [Instruction.from_dict(instr) for instr in data.get('instructions', [])]
        block.successors = data.get('successors', []).copy()
        block.predecessors = data.get('predecessors', []).copy()
        block.raw_lines = {int(k): v for k, v in data.get('raw_lines', {}).items()}
        return block


@dataclass
class CFG:
    """Control Flow Graph representation."""
    name: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry_block: Optional[str] = None
    # Preserve original file structure for regeneration
    header_lines: List[str] = field(default_factory=list)  # Lines before function body
    footer_lines: List[str] = field(default_factory=list)  # Lines after function body
    block_order: List[str] = field(default_factory=list)   # Order of blocks in file
    
    def add_block(self, block: BasicBlock):
        self.blocks[block.label] = block
        if self.entry_block is None:
            self.entry_block = block.label
        if block.label not in self.block_order:
            self.block_order.append(block.label)
    
    def add_edge(self, from_block: str, to_block: str):
        if from_block in self.blocks and to_block in self.blocks:
            if to_block not in self.blocks[from_block].successors:
                self.blocks[from_block].successors.append(to_block)
            if from_block not in self.blocks[to_block].predecessors:
                self.blocks[to_block].predecessors.append(from_block)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize CFG to dictionary."""
        return {
            'name': self.name,
            'entry_block': self.entry_block,
            'blocks': {label: block.to_dict() for label, block in self.blocks.items()},
            'header_lines': self.header_lines,
            'footer_lines': self.footer_lines,
            'block_order': self.block_order,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CFG':
        """Deserialize CFG from dictionary."""
        cfg = cls(name=data['name'])
        cfg.entry_block = data.get('entry_block')
        cfg.header_lines = data.get('header_lines', [])
        cfg.footer_lines = data.get('footer_lines', [])
        cfg.block_order = data.get('block_order', [])
        for label, block_data in data.get('blocks', {}).items():
            cfg.blocks[label] = BasicBlock.from_dict(block_data)
        return cfg
    
    def to_json(self, filepath: str, indent: int = 2):
        """Save CFG to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'CFG':
        """Load CFG from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_amdgcn(self, filepath: str, keep_debug_labels: bool = False):
        """
        Regenerate the .amdgcn assembly file from this CFG.
        
        This reconstructs the file using:
        - header_lines: preserved from original file
        - blocks: in block_order, using raw_lines from each block
        - footer_lines: preserved from original file
        
        Args:
            filepath: Output file path
            keep_debug_labels: If False (default), remove .Ltmp* debug labels from output
                              AND strip debug sections to avoid undefined symbol errors.
                              If True, preserve all labels including .Ltmp* debug labels
                              and keep all debug sections intact.
        
        Note:
            When .Ltmp* labels are removed from the code section, the debug sections
            (.debug_abbrev, .debug_info, .debug_ranges, .debug_str, .debug_line) still
            reference these labels, causing assembler errors like:
            "symbol '.Ltmp2' can not be undefined in a subtraction expression"
            
            To avoid this, when keep_debug_labels=False, we also strip all debug sections
            while preserving the .amdgpu_metadata section (required for kernel execution).
        """
        import re
        # Pattern to match .Ltmp* debug labels (lines that are just the label)
        ltmp_pattern = re.compile(r'^\s*\.Ltmp\d+:\s*$')
        
        lines = []
        
        # Add header
        lines.extend(self.header_lines)
        
        # Add blocks in order
        for label in self.block_order:
            if label in self.blocks:
                block = self.blocks[label]
                block_lines = block.get_raw_lines_in_order()
                if keep_debug_labels:
                    lines.extend(block_lines)
                else:
                    # Filter out .Ltmp* debug labels
                    for line in block_lines:
                        if not ltmp_pattern.match(line):
                            lines.extend([line])
        
        # Add footer
        lines.extend(self.footer_lines)
        
        # Convert to string for processing - strip trailing newlines from each line first
        # to avoid double newlines when joining
        stripped_lines = [line.rstrip('\n\r') for line in lines]
        content = '\n'.join(stripped_lines)
        
        # If not keeping debug labels, strip debug sections to avoid undefined symbol errors
        if not keep_debug_labels:
            content = self._strip_debug_sections(content)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            if not content.endswith('\n'):
                f.write('\n')
        
        print(f"AMDGCN file saved to: {filepath}")
    
    @staticmethod
    def _strip_debug_sections(asm_src: str) -> str:
        """
        Strip debug sections from assembly source to avoid undefined .Ltmp* symbol errors.
        
        When modifying assembly code, the .Ltmp* temporary labels used for DWARF debug info
        may be removed or relocated. However, the debug sections (.debug_abbrev, .debug_info,
        .debug_ranges, .debug_str, .debug_line) still reference these labels, causing
        assembler errors like "symbol '.Ltmp2' can not be undefined in a subtraction expression".
        
        This function removes all debug sections while preserving:
        - The .text code section
        - The .rodata kernel descriptor section (.amdhsa_kernel)
        - The .AMDGPU.csdata and .AMDGPU.gpr_maximums sections
        - The .amdgpu_metadata section (required for kernel execution)
        
        The debug sections are only used for debugging tools (GDB, profilers) and are not
        required for kernel execution.
        
        Args:
            asm_src: The assembly source code as a string
            
        Returns:
            The assembly source with debug sections removed
        """
        # Find where debug sections start
        debug_start_marker = '\t.section\t.debug_abbrev'
        debug_start_idx = asm_src.find(debug_start_marker)
        
        if debug_start_idx == -1:
            # No debug sections found, return original
            return asm_src
        
        # Keep everything before debug sections
        pre_debug = asm_src[:debug_start_idx]
        
        # Extract .amdgpu_metadata section (required for kernel execution)
        metadata_start_marker = '\t.amdgpu_metadata'
        metadata_end_marker = '\t.end_amdgpu_metadata'
        
        metadata_start = asm_src.find(metadata_start_marker)
        metadata_end = asm_src.find(metadata_end_marker)
        
        if metadata_start != -1 and metadata_end != -1:
            # Include the end marker and its newline
            metadata_end = asm_src.find('\n', metadata_end) + 1
            metadata_section = asm_src[metadata_start:metadata_end]
            # Combine: code + kernel descriptor + amdgpu_metadata
            return pre_debug + metadata_section
        else:
            # No metadata section found, just return pre-debug content
            return pre_debug.rstrip() + '\n'


# =============================================================================
# AMDGCN Instruction Classification
# =============================================================================

# Unconditional branch instructions
UNCONDITIONAL_BRANCHES = {
    's_branch',
    's_setpc_b64',
}

# Conditional branch instructions
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

# Program termination instructions
TERMINATOR_INSTRUCTIONS = {
    's_endpgm',
    's_endpgm_saved',
    's_trap',
    's_rfe_b64',
}

# Call instructions (also affect control flow)
CALL_INSTRUCTIONS = {
    's_call_b64',
    's_swappc_b64',
}


def classify_instruction(opcode: str) -> Tuple[bool, bool, bool]:
    """
    Classify an instruction by its control flow properties.
    
    Returns:
        (is_branch, is_conditional, is_terminator)
    """
    opcode_lower = opcode.lower()
    
    if opcode_lower in UNCONDITIONAL_BRANCHES:
        return (True, False, False)
    elif opcode_lower in CONDITIONAL_BRANCHES:
        return (True, True, False)
    elif opcode_lower in TERMINATOR_INSTRUCTIONS:
        return (False, False, True)
    elif opcode_lower in CALL_INSTRUCTIONS:
        return (True, False, False)
    
    return (False, False, False)


def extract_branch_target(operands: str) -> Optional[str]:
    """Extract the branch target label from operands."""
    # Match labels like .LBB0_0, .Ltmp0, etc.
    match = re.search(r'(\.L\w+)', operands)
    if match:
        return match.group(1)
    return None


# =============================================================================
# Parser
# =============================================================================

class AMDGCNParser:
    """Parser for AMDGCN assembly files."""
    
    # Pattern to match labels (e.g., .LBB0_0:, paged_attention_decode_v2_gluon_dot_kernel:)
    LABEL_PATTERN = re.compile(r'^(\.[A-Za-z_]\w*|\w+):')
    
    # Pattern to match basic block labels specifically (exclude debug labels)
    # Basic block labels: .LBB<func_num>_<block_num> or .Lfunc_begin<num>
    BB_LABEL_PATTERN = re.compile(r'^\.L(BB\d+_\d+|func_begin\d+):')
    
    # Pattern to match instructions
    # Format: [optional_label:] opcode [operands] [; comment]
    INSTRUCTION_PATTERN = re.compile(
        r'^\s*([a-z_][a-z0-9_]*)\s*(.*?)(?:\s*;.*)?$',
        re.IGNORECASE
    )
    
    # Pattern for directives (lines starting with .)
    DIRECTIVE_PATTERN = re.compile(r'^\s*\.[a-z_]', re.IGNORECASE)
    
    def __init__(self):
        self.blocks: Dict[str, BasicBlock] = {}
        self.block_order: List[str] = []  # Maintain order of blocks
        self.current_block: Optional[BasicBlock] = None
        self.function_name: str = "unknown"
        self.in_function: bool = False
        
    def is_skip_line(self, line: str) -> bool:
        """Check if a line should be skipped (comments, empty, most directives)."""
        stripped = line.strip()
        
        # Don't skip label lines
        if self.LABEL_PATTERN.match(stripped):
            return False
        
        # Skip comments
        if stripped.startswith(';'):
            return True
        
        # Skip empty lines
        if not stripped:
            return True
        
        # Skip directives
        if stripped.startswith('.'):
            return True
        
        return False
    
    def is_bb_label(self, line: str) -> Optional[str]:
        """
        Check if line is a basic block label definition. 
        Returns label name or None.
        Only returns true for actual basic block labels (.LBB*), 
        not debug labels (.Ltmp*).
        """
        stripped = line.strip()
        match = self.BB_LABEL_PATTERN.match(stripped)
        if match:
            # Return the full label including the .L prefix
            return '.' + match.group(0).rstrip(':').lstrip('.')
        return None
    
    def is_any_label(self, line: str) -> Optional[str]:
        """Check if line is any label definition. Returns label name or None."""
        stripped = line.strip()
        match = self.LABEL_PATTERN.match(stripped)
        if match:
            return match.group(1)
        return None
    
    def parse_instruction(self, line: str, line_num: int) -> Optional[Instruction]:
        """Parse a single instruction line."""
        stripped = line.strip()
        
        # Skip lines that are just labels
        if stripped.endswith(':') and ' ' not in stripped:
            return None
        
        # Skip lines that start with a directive
        if stripped.startswith('.'):
            return None
        
        # Skip comment lines
        if stripped.startswith(';'):
            return None
        
        # Try to match instruction pattern
        match = self.INSTRUCTION_PATTERN.match(stripped)
        if not match:
            return None
        
        opcode = match.group(1)
        operands = match.group(2).strip() if match.group(2) else ""
        
        # Remove trailing comments from operands
        if ';' in operands:
            operands = operands.split(';')[0].strip()
        
        # Classify the instruction
        is_branch, is_conditional, is_terminator = classify_instruction(opcode)
        
        # Extract branch target if applicable
        branch_target = None
        if is_branch:
            branch_target = extract_branch_target(operands)
        
        return Instruction(
            address=line_num,
            opcode=opcode,
            operands=operands,
            raw_line=line.rstrip(),
            is_branch=is_branch,
            is_conditional=is_conditional,
            is_terminator=is_terminator,
            branch_target=branch_target
        )
    
    def start_new_block(self, label: str, line_num: int):
        """Start a new basic block."""
        if self.current_block and not self.current_block.is_empty():
            self.current_block.end_line = line_num - 1
        
        self.current_block = BasicBlock(label=label, start_line=line_num)
        self.blocks[label] = self.current_block
        self.block_order.append(label)
    
    def parse_file(self, filepath: str) -> CFG:
        """Parse an AMDGCN assembly file and build the CFG."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Track file structure for regeneration
        header_lines: List[str] = []
        footer_lines: List[str] = []
        func_start_line = -1
        func_end_line = -1
        
        # First pass: identify function boundaries and extract function name
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Look for function definition
            if stripped.startswith('.type') and '@function' in stripped:
                # Extract function name
                match = re.search(r'\.type\s+(\w+)', stripped)
                if match:
                    self.function_name = match.group(1)
        
        # Find function start (the function label line)
        for i, line in enumerate(lines, 1):
            any_label = self.is_any_label(line.strip())
            if any_label == self.function_name:
                func_start_line = i
                break
        
        # Find function end (.Lfunc_end* label)
        for i, line in enumerate(lines, 1):
            any_label = self.is_any_label(line.strip())
            if any_label and any_label.startswith('.Lfunc_end'):
                func_end_line = i
                break
        
        # Capture header (everything before function label, inclusive)
        if func_start_line > 0:
            header_lines = [lines[i] for i in range(func_start_line)]  # Include func label line
        
        # Capture footer (from .Lfunc_end* onwards)
        if func_end_line > 0:
            footer_lines = [lines[i] for i in range(func_end_line - 1, len(lines))]
        
        # Second pass: identify all basic block labels and their line numbers
        bb_labels = {}  # label -> line number
        for i, line in enumerate(lines, 1):
            label = self.is_bb_label(line)
            if label:
                bb_labels[label] = i
        
        # Also collect branch targets - these become basic blocks too
        branch_targets = set()
        for i, line in enumerate(lines, 1):
            instr = self.parse_instruction(line, i)
            if instr and instr.branch_target:
                branch_targets.add(instr.branch_target)
        
        # Add branch targets that are LBB labels to our bb_labels
        for target in branch_targets:
            if target.startswith('.LBB') and target not in bb_labels:
                # Find the line number for this label
                for i, line in enumerate(lines, 1):
                    label = self.is_any_label(line)
                    if label == target:
                        bb_labels[target] = i
                        break
        
        # Sort bb_labels by line number to determine block boundaries
        sorted_bb_labels = sorted(bb_labels.items(), key=lambda x: x[1])
        
        # Build a mapping: line_num -> which block it belongs to
        # Lines between func_start and first bb_label belong to the first block
        # Lines between bb_labels belong to the earlier block
        line_to_block: Dict[int, str] = {}
        if sorted_bb_labels:
            # Lines from func_start+1 to first bb_label-1 belong to first bb
            first_bb_label, first_bb_line = sorted_bb_labels[0]
            for ln in range(func_start_line + 1, first_bb_line):
                line_to_block[ln] = first_bb_label
            
            # Assign lines for each basic block
            for idx, (label, start_ln) in enumerate(sorted_bb_labels):
                if idx + 1 < len(sorted_bb_labels):
                    end_ln = sorted_bb_labels[idx + 1][1]
                else:
                    end_ln = func_end_line if func_end_line > 0 else len(lines) + 1
                
                for ln in range(start_ln, end_ln):
                    line_to_block[ln] = label
        
        # Third pass: parse instructions and build blocks with raw lines
        self.in_function = False
        entry_block_created = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for function label to start parsing
            any_label = self.is_any_label(stripped)
            if any_label:
                if any_label == self.function_name:
                    self.in_function = True
                    continue
                elif any_label.startswith('.Lfunc_end'):
                    # End of function
                    self.in_function = False
                    if self.current_block:
                        self.current_block.end_line = i - 1
                    continue
                elif self.in_function and any_label in bb_labels:
                    # Start a new basic block
                    self.start_new_block(any_label, i)
                    entry_block_created = True
                    # Add label line to raw_lines
                    self.current_block.raw_lines[i] = line
                    continue
                elif self.in_function:
                    # Other labels (like .Ltmp*) - still store raw line
                    if i in line_to_block:
                        block_label = line_to_block[i]
                        if block_label in self.blocks:
                            self.blocks[block_label].raw_lines[i] = line
                    elif self.current_block:
                        self.current_block.raw_lines[i] = line
                    continue
            
            if not self.in_function:
                continue
            
            # Store raw line for current block (all lines, not just instructions)
            # Use line_to_block mapping if available
            if i in line_to_block:
                block_label = line_to_block[i]
                if block_label in self.blocks:
                    self.blocks[block_label].raw_lines[i] = line
            elif self.current_block:
                self.current_block.raw_lines[i] = line
            
            # Skip non-instruction lines for instruction parsing
            if self.is_skip_line(line):
                continue
            
            # Create entry block if we haven't yet
            if not entry_block_created:
                self.start_new_block('.Lentry', i)
                entry_block_created = True
            
            # Parse instruction
            instr = self.parse_instruction(line, i)
            if instr and self.current_block:
                self.current_block.instructions.append(instr)
        
        # Set end line for last block
        if self.current_block:
            self.current_block.end_line = func_end_line - 1 if func_end_line > 0 else len(lines)
        
        # Build CFG
        cfg = CFG(name=self.function_name)
        cfg.header_lines = header_lines
        cfg.footer_lines = footer_lines
        cfg.block_order = self.block_order.copy()
        
        for label in self.block_order:
            block = self.blocks[label]
            cfg.add_block(block)
        
        # Add edges based on control flow
        self._build_edges(cfg)
        
        return cfg
    
    def _build_edges(self, cfg: CFG):
        """Build CFG edges based on branch instructions."""
        block_labels = self.block_order
        
        for i, label in enumerate(block_labels):
            block = cfg.blocks[label]
            
            # Find ALL branch instructions in the block
            # This handles cases where there are multiple branches
            # (e.g., conditional branch followed by unconditional branch)
            branch_instrs = [instr for instr in block.instructions 
                           if instr.is_branch or instr.is_terminator]
            
            if not branch_instrs:
                # No branch/terminator, fall through to next block
                if i + 1 < len(block_labels):
                    cfg.add_edge(label, block_labels[i + 1])
                continue
            
            has_terminator = any(instr.is_terminator for instr in branch_instrs)
            has_unconditional = any(instr.is_branch and not instr.is_conditional 
                                   for instr in branch_instrs)
            
            if has_terminator:
                # s_endpgm or similar - no successors
                continue
            
            # Add edges for all branch targets
            for branch_instr in branch_instrs:
                if branch_instr.is_branch:
                    target = branch_instr.branch_target
                    if target:
                        # Resolve target - might need to map .Ltmp to .LBB
                        if target in cfg.blocks:
                            cfg.add_edge(label, target)
                        else:
                            # Try to find the containing basic block for this label
                            resolved = self._resolve_label_to_bb(target, block_labels)
                            if resolved and resolved in cfg.blocks:
                                cfg.add_edge(label, resolved)
            
            # Add fall-through edge only if no unconditional branch ends the block
            # This handles the case where there's a conditional branch at the end
            # that could fall through to the next block
            last_instr = branch_instrs[-1] if branch_instrs else None
            if last_instr and last_instr.is_conditional and not has_unconditional:
                if i + 1 < len(block_labels):
                    cfg.add_edge(label, block_labels[i + 1])
    
    def _resolve_label_to_bb(self, label: str, bb_labels: List[str]) -> Optional[str]:
        """
        If the target label is not a basic block, find which basic block contains it.
        This handles the case where a branch targets a debug label inside a basic block.
        """
        # For now, just return None - branch targets should be basic block labels
        # This could be enhanced to look up the line number of the target label
        # and find the containing basic block
        return None


# =============================================================================
# DOT Output Generation
# =============================================================================

def escape_dot_string(s: str) -> str:
    """Escape special characters for DOT format."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('<', '\\<')
    s = s.replace('>', '\\>')
    s = s.replace('{', '\\{')
    s = s.replace('}', '\\}')
    s = s.replace('|', '\\|')
    return s


def truncate_instruction(instr: Instruction, max_len: int = 60) -> str:
    """Truncate instruction for display."""
    text = f"{instr.opcode}"
    if instr.operands:
        text += f" {instr.operands}"
    if len(text) > max_len:
        text = text[:max_len-3] + "..."
    return text


def generate_dot(cfg: CFG, show_instructions: bool = True, max_instructions: int = 15) -> str:
    """
    Generate DOT format representation of the CFG.
    
    Args:
        cfg: The control flow graph
        show_instructions: Whether to show instructions in blocks
        max_instructions: Maximum number of instructions to show per block
    
    Returns:
        DOT format string
    """
    lines = []
    lines.append(f'digraph "{escape_dot_string(cfg.name)}" {{')
    lines.append('    // Graph attributes')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=record, fontname="Courier", fontsize=10];')
    lines.append('    edge [fontname="Courier", fontsize=9];')
    lines.append('')
    
    # Generate nodes
    lines.append('    // Basic blocks')
    for label, block in cfg.blocks.items():
        node_id = label.replace('.', '_').replace('-', '_')
        
        # Build label content
        header = f"{label}"
        if block.start_line > 0:
            header += f" (L{block.start_line}-{block.end_line})"
        
        if show_instructions and block.instructions:
            # Show instructions (truncated if too many)
            instr_lines = []
            display_count = min(len(block.instructions), max_instructions)
            
            for instr in block.instructions[:display_count]:
                instr_text = truncate_instruction(instr)
                instr_text = escape_dot_string(instr_text)
                instr_lines.append(instr_text)
            
            if len(block.instructions) > max_instructions:
                remaining = len(block.instructions) - max_instructions
                instr_lines.append(f"... ({remaining} more instructions)")
            
            # Format as record label
            instr_content = "\\l".join(instr_lines) + "\\l"
            label_content = f"{{{escape_dot_string(header)}|{instr_content}}}"
        else:
            # Simple label with instruction count
            instr_count = len(block.instructions)
            label_content = f"{{{escape_dot_string(header)}|{instr_count} instructions}}"
        
        # Color code special blocks
        style = ""
        if label == cfg.entry_block:
            style = ', style=filled, fillcolor=lightgreen'
        elif block.get_terminator() and block.get_terminator().is_terminator:
            style = ', style=filled, fillcolor=lightcoral'
        elif not block.successors:
            # Exit block (no successors but not because of terminator)
            style = ', style=filled, fillcolor=lightyellow'
        
        lines.append(f'    {node_id} [label="{label_content}"{style}];')
    
    lines.append('')
    
    # Generate edges
    lines.append('    // Control flow edges')
    for label, block in cfg.blocks.items():
        from_id = label.replace('.', '_').replace('-', '_')
        
        # Find the control flow instruction
        control_instr = None
        for instr in reversed(block.instructions):
            if instr.is_branch or instr.is_terminator:
                control_instr = instr
                break
        
        for succ in block.successors:
            to_id = succ.replace('.', '_').replace('-', '_')
            
            # Determine edge style based on type
            edge_style = ""
            edge_label = ""
            
            if control_instr and control_instr.is_conditional:
                # For conditional branches, distinguish taken vs fall-through
                if control_instr.branch_target and succ.endswith(control_instr.branch_target.lstrip('.')):
                    edge_style = 'color=blue'
                    edge_label = 'label="taken"'
                elif control_instr.branch_target == succ:
                    edge_style = 'color=blue'
                    edge_label = 'label="taken"'
                else:
                    edge_style = 'color=red, style=dashed'
                    edge_label = 'label="fall"'
            
            attrs = ", ".join(filter(None, [edge_style, edge_label]))
            if attrs:
                lines.append(f'    {from_id} -> {to_id} [{attrs}];')
            else:
                lines.append(f'    {from_id} -> {to_id};')
    
    lines.append('}')
    
    return '\n'.join(lines)


def generate_simple_dot(cfg: CFG) -> str:
    """
    Generate a simplified DOT representation (just blocks and edges).
    """
    lines = []
    lines.append(f'digraph "{escape_dot_string(cfg.name)}" {{')
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, fontname="Courier"];')
    lines.append('')
    
    for label, block in cfg.blocks.items():
        node_id = label.replace('.', '_').replace('-', '_')
        instr_count = len(block.instructions)
        
        # Color coding
        color = ""
        if label == cfg.entry_block:
            color = ', style=filled, fillcolor=lightgreen'
        elif block.get_terminator() and block.get_terminator().is_terminator:
            color = ', style=filled, fillcolor=lightcoral'
        
        lines.append(f'    {node_id} [label="{label}\\n({instr_count} instr)"{color}];')
    
    lines.append('')
    
    for label, block in cfg.blocks.items():
        from_id = label.replace('.', '_').replace('-', '_')
        for succ in block.successors:
            to_id = succ.replace('.', '_').replace('-', '_')
            lines.append(f'    {from_id} -> {to_id};')
    
    lines.append('}')
    
    return '\n'.join(lines)


# =============================================================================
# Statistics and Analysis
# =============================================================================

def print_cfg_stats(cfg: CFG):
    """Print statistics about the CFG."""
    print(f"\n{'='*60}")
    print(f"CFG Statistics for: {cfg.name}")
    print(f"{'='*60}")
    
    total_blocks = len(cfg.blocks)
    total_instructions = sum(len(b.instructions) for b in cfg.blocks.values())
    total_edges = sum(len(b.successors) for b in cfg.blocks.values())
    
    print(f"Total basic blocks: {total_blocks}")
    print(f"Total instructions: {total_instructions}")
    print(f"Total edges: {total_edges}")
    
    # Find entry and exit blocks
    entry_blocks = [l for l, b in cfg.blocks.items() if not b.predecessors]
    exit_blocks = [l for l, b in cfg.blocks.items() if not b.successors]
    
    print(f"\nEntry blocks: {entry_blocks}")
    print(f"Exit blocks: {exit_blocks}")
    
    # Block size distribution
    sizes = [len(b.instructions) for b in cfg.blocks.values()]
    if sizes:
        print(f"\nBlock size statistics:")
        print(f"  Min: {min(sizes)} instructions")
        print(f"  Max: {max(sizes)} instructions")
        print(f"  Avg: {sum(sizes)/len(sizes):.1f} instructions")
    
    # Branch statistics
    branch_count = 0
    cond_branch_count = 0
    for block in cfg.blocks.values():
        for instr in block.instructions:
            if instr.is_branch:
                branch_count += 1
                if instr.is_conditional:
                    cond_branch_count += 1
    
    print(f"\nBranch instructions: {branch_count}")
    print(f"  Conditional: {cond_branch_count}")
    print(f"  Unconditional: {branch_count - cond_branch_count}")
    
    print(f"{'='*60}\n")


def list_basic_blocks(cfg: CFG):
    """List all basic blocks with their instruction counts."""
    print(f"\nBasic Blocks in {cfg.name}:")
    print("-" * 50)
    
    for label, block in cfg.blocks.items():
        # Find control flow instruction
        control_instr = None
        for instr in reversed(block.instructions):
            if instr.is_branch or instr.is_terminator:
                control_instr = instr
                break
        
        term_info = ""
        if control_instr:
            if control_instr.is_terminator:
                term_info = f" [EXIT: {control_instr.opcode}]"
            elif control_instr.is_branch:
                if control_instr.is_conditional:
                    term_info = f" [COND: {control_instr.opcode} -> {control_instr.branch_target}]"
                else:
                    term_info = f" [JUMP: {control_instr.opcode} -> {control_instr.branch_target}]"
        
        pred_str = ", ".join(block.predecessors) if block.predecessors else "none"
        succ_str = ", ".join(block.successors) if block.successors else "none"
        
        print(f"\n{label}:{term_info}")
        print(f"  Instructions: {len(block.instructions)}")
        print(f"  Predecessors: {pred_str}")
        print(f"  Successors: {succ_str}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse AMDGCN assembly and generate Control Flow Graph (DOT format)'
    )
    parser.add_argument('input', help='Input .amdgcn file')
    parser.add_argument('output', nargs='?', help='Output .dot file (default: stdout)')
    parser.add_argument('--simple', action='store_true', 
                       help='Generate simplified DOT (no instructions)')
    parser.add_argument('--stats', action='store_true',
                       help='Print CFG statistics')
    parser.add_argument('--list-blocks', action='store_true',
                       help='List all basic blocks')
    parser.add_argument('--max-instr', type=int, default=15,
                       help='Max instructions to show per block (default: 15)')
    
    args = parser.parse_args()
    
    # Parse the input file
    amdgcn_parser = AMDGCNParser()
    cfg = amdgcn_parser.parse_file(args.input)
    
    # Print statistics if requested
    if args.stats:
        print_cfg_stats(cfg)
    
    # List blocks if requested
    if args.list_blocks:
        list_basic_blocks(cfg)
    
    # Generate DOT output
    if args.simple:
        dot_content = generate_simple_dot(cfg)
    else:
        dot_content = generate_dot(cfg, show_instructions=True, 
                                   max_instructions=args.max_instr)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(dot_content)
        print(f"DOT file written to: {args.output}")
    else:
        # Only print DOT if no stats/list options
        if not args.stats and not args.list_blocks:
            print(dot_content)


if __name__ == '__main__':
    main()
