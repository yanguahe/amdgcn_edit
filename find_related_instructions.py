#!/usr/bin/env python3
"""
Find Related Instructions Tool

This tool finds all instructions in a CFG that use the same registers
as a specified instruction type in a given basic block.

Usage:
    python find_related_instructions.py <input.amdgcn> <block_label> <opcode>
    
Example:
    python find_related_instructions.py kernel.amdgcn .LBB0_2 global_load_dwordx4

Output:
    Lists all instructions that use registers (as src or dst) from the
    specified instruction type, grouped by basic block and sorted by
    address within each block.
    
    The output is organized into three sections:
    1. Instructions using Target Defs (registers written by target instructions)
    2. Instructions using Target Uses (registers read by target instructions)
    3. Combined view (all related instructions)
"""

import argparse
import sys
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from amdgcn_cfg import AMDGCNParser, CFG, BasicBlock, Instruction
from amdgcn_ddg import parse_instruction_registers


@dataclass
class RelatedInstruction:
    """Represents an instruction related to the target registers."""
    address: int
    raw_line: str
    opcode: str
    operands: str
    defs: Set[str]
    uses: Set[str]
    matching_defs: Set[str]  # registers that match target regs (in defs)
    matching_uses: Set[str]  # registers that match target regs (in uses)
    # Separate tracking for Target Defs vs Target Uses matching
    matching_target_defs_in_defs: Set[str] = field(default_factory=set)  # Target Defs regs found in instr defs
    matching_target_defs_in_uses: Set[str] = field(default_factory=set)  # Target Defs regs found in instr uses
    matching_target_uses_in_defs: Set[str] = field(default_factory=set)  # Target Uses regs found in instr defs
    matching_target_uses_in_uses: Set[str] = field(default_factory=set)  # Target Uses regs found in instr uses


@dataclass
class BlockInstructions:
    """All related instructions in a basic block."""
    block_label: str
    instructions: List[RelatedInstruction] = field(default_factory=list)


def find_instructions_by_opcode(
    block: BasicBlock,
    target_opcode: str
) -> List[Instruction]:
    """
    Find all instructions in a block that match the target opcode.
    
    Args:
        block: The basic block to search
        target_opcode: The opcode to match (case-insensitive, supports partial match)
        
    Returns:
        List of matching instructions
    """
    target_lower = target_opcode.lower()
    matching = []
    
    for instr in block.instructions:
        opcode_lower = instr.opcode.lower()
        # Support exact match or prefix match
        if opcode_lower == target_lower or opcode_lower.startswith(target_lower):
            matching.append(instr)
    
    return matching


def collect_target_registers(
    instructions: List[Instruction]
) -> Tuple[Set[str], Set[str]]:
    """
    Collect all registers (defs and uses) from a list of instructions.
    
    Args:
        instructions: List of instructions to analyze
        
    Returns:
        (all_defs, all_uses) - sets of all defined and used registers
    """
    all_defs = set()
    all_uses = set()
    
    for instr in instructions:
        defs, uses = parse_instruction_registers(instr)
        all_defs.update(defs)
        all_uses.update(uses)
    
    return all_defs, all_uses


def find_related_instructions_in_cfg(
    cfg: CFG,
    target_regs: Set[str],
    target_defs: Set[str] = None,
    target_uses: Set[str] = None
) -> Dict[str, BlockInstructions]:
    """
    Find all instructions in the CFG that use any of the target registers.
    
    Args:
        cfg: The control flow graph
        target_regs: Set of registers to search for (combined)
        target_defs: Set of registers defined by target instructions (optional)
        target_uses: Set of registers used by target instructions (optional)
        
    Returns:
        Dictionary mapping block label to BlockInstructions
    """
    results = {}
    target_defs = target_defs or set()
    target_uses = target_uses or set()
    
    for block_label in cfg.block_order:
        block = cfg.blocks.get(block_label)
        if not block:
            continue
        
        block_instrs = BlockInstructions(block_label=block_label)
        
        for instr in block.instructions:
            defs, uses = parse_instruction_registers(instr)
            
            # Check if this instruction uses any target registers
            matching_defs = defs & target_regs
            matching_uses = uses & target_regs
            
            if matching_defs or matching_uses:
                related = RelatedInstruction(
                    address=instr.address,
                    raw_line=instr.raw_line.strip() if instr.raw_line else f"{instr.opcode} {instr.operands}",
                    opcode=instr.opcode,
                    operands=instr.operands,
                    defs=defs,
                    uses=uses,
                    matching_defs=matching_defs,
                    matching_uses=matching_uses,
                    # Separate tracking: which Target Defs/Uses regs appear in this instr's defs/uses
                    matching_target_defs_in_defs=defs & target_defs,
                    matching_target_defs_in_uses=uses & target_defs,
                    matching_target_uses_in_defs=defs & target_uses,
                    matching_target_uses_in_uses=uses & target_uses,
                )
                block_instrs.instructions.append(related)
        
        # Only add blocks that have related instructions
        if block_instrs.instructions:
            # Sort by address
            block_instrs.instructions.sort(key=lambda x: x.address)
            results[block_label] = block_instrs
    
    return results


def format_register_set(regs: Set[str]) -> str:
    """Format a set of registers for display."""
    if not regs:
        return ""
    # Sort registers for consistent display
    sorted_regs = sorted(regs)
    return ", ".join(sorted_regs)


def print_section_header(title: str, char: str = "="):
    """Print a section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def print_instructions_for_section(
    results: Dict[str, BlockInstructions],
    cfg: CFG,
    filter_func,
    match_info_func
) -> int:
    """
    Print instructions that match the filter function.
    
    Args:
        results: The search results
        cfg: The CFG (for block order)
        filter_func: Function(instr) -> bool to filter instructions
        match_info_func: Function(instr) -> str to generate match info string
        
    Returns:
        Total number of instructions printed
    """
    total = 0
    blocks_with_output = 0
    
    for block_label in cfg.block_order:
        if block_label not in results:
            continue
        
        block_instrs = results[block_label]
        filtered = [i for i in block_instrs.instructions if filter_func(i)]
        
        if not filtered:
            continue
        
        blocks_with_output += 1
        print(f"\n; Block: {block_label} ({len(filtered)} instructions)")
        print("-" * 70)
        
        for instr in filtered:
            total += 1
            match_str = match_info_func(instr)
            print(f"[{instr.address:4d}] {instr.raw_line}{match_str}")
    
    return total, blocks_with_output


def print_results(
    target_block: str,
    target_opcode: str,
    target_defs: Set[str],
    target_uses: Set[str],
    results: Dict[str, BlockInstructions],
    cfg: CFG,
    verbose: bool = True,
    separate_sections: bool = True
):
    """
    Print the results in a formatted way.
    
    Args:
        target_block: The source block label
        target_opcode: The target opcode
        target_defs: Registers defined by target instructions
        target_uses: Registers used by target instructions
        results: The search results
        cfg: The CFG (for block order)
        verbose: Whether to show detailed register information
        separate_sections: Whether to show separate sections for Target Defs and Target Uses
    """
    all_target_regs = target_defs | target_uses
    
    print("=" * 80)
    print(f"Find Related Instructions")
    print("=" * 80)
    print(f"Source Block: {target_block}")
    print(f"Target Opcode: {target_opcode}")
    print(f"Target Defs ({len(target_defs)} regs): {format_register_set(target_defs)}")
    print(f"Target Uses ({len(target_uses)} regs): {format_register_set(target_uses)}")
    print(f"All Target Registers ({len(all_target_regs)} regs): {format_register_set(all_target_regs)}")
    
    if separate_sections and target_defs and target_uses:
        # ========== Section 1: Instructions using Target Defs ==========
        print_section_header(
            f"SECTION 1: Instructions using Target Defs (dst regs of {target_opcode})"
        )
        print(f"Target Defs: {format_register_set(target_defs)}")
        
        def filter_target_defs(instr):
            return bool(instr.matching_target_defs_in_defs or instr.matching_target_defs_in_uses)
        
        def match_info_target_defs(instr):
            info = []
            if instr.matching_target_defs_in_defs:
                info.append(f"def:{format_register_set(instr.matching_target_defs_in_defs)}")
            if instr.matching_target_defs_in_uses:
                info.append(f"use:{format_register_set(instr.matching_target_defs_in_uses)}")
            return f"  ; {{{', '.join(info)}}}" if info else ""
        
        count1, blocks1 = print_instructions_for_section(
            results, cfg, filter_target_defs, match_info_target_defs
        )
        print(f"\n[Section 1 Total: {count1} instructions in {blocks1} blocks]")
        
        # ========== Section 2: Instructions using Target Uses ==========
        print_section_header(
            f"SECTION 2: Instructions using Target Uses (src regs of {target_opcode})"
        )
        print(f"Target Uses: {format_register_set(target_uses)}")
        
        def filter_target_uses(instr):
            return bool(instr.matching_target_uses_in_defs or instr.matching_target_uses_in_uses)
        
        def match_info_target_uses(instr):
            info = []
            if instr.matching_target_uses_in_defs:
                info.append(f"def:{format_register_set(instr.matching_target_uses_in_defs)}")
            if instr.matching_target_uses_in_uses:
                info.append(f"use:{format_register_set(instr.matching_target_uses_in_uses)}")
            return f"  ; {{{', '.join(info)}}}" if info else ""
        
        count2, blocks2 = print_instructions_for_section(
            results, cfg, filter_target_uses, match_info_target_uses
        )
        print(f"\n[Section 2 Total: {count2} instructions in {blocks2} blocks]")
        
        # ========== Section 3: Combined View ==========
        print_section_header(
            f"SECTION 3: Combined View (all instructions using any target register)"
        )
    
    # Print combined view (or only view if not using separate sections)
    total_instructions = 0
    
    # Output in block order
    for block_label in cfg.block_order:
        if block_label not in results:
            continue
        
        block_instrs = results[block_label]
        
        print(f"\n; Block: {block_label} ({len(block_instrs.instructions)} instructions)")
        print("-" * 70)
        
        for instr in block_instrs.instructions:
            total_instructions += 1
            
            # Format: [address] instruction_text
            # Show which registers match with detailed breakdown
            match_info = []
            if instr.matching_defs:
                match_info.append(f"def:{format_register_set(instr.matching_defs)}")
            if instr.matching_uses:
                match_info.append(f"use:{format_register_set(instr.matching_uses)}")
            
            match_str = f"  ; {{{', '.join(match_info)}}}" if match_info else ""
            
            print(f"[{instr.address:4d}] {instr.raw_line}{match_str}")
    
    print("\n" + "=" * 80)
    print(f"Total: {total_instructions} instructions in {len(results)} blocks")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Find all instructions related to a specific instruction type in a basic block',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Find all instructions using registers from global_load_dwordx4 in .LBB0_2
    python find_related_instructions.py kernel.amdgcn .LBB0_2 global_load_dwordx4
    
    # Find all instructions related to v_mfma instructions in .LBB0_10
    python find_related_instructions.py kernel.amdgcn .LBB0_10 v_mfma
    
    # Save output to file
    python find_related_instructions.py kernel.amdgcn .LBB0_2 global_load -o output.txt
    
    # Show only combined view (no separate sections)
    python find_related_instructions.py kernel.amdgcn .LBB0_2 global_load --combined-only
"""
    )
    
    parser.add_argument('input', help='Input .amdgcn assembly file')
    parser.add_argument('block_label', help='Basic block label (e.g., .LBB0_2)')
    parser.add_argument('opcode', help='Target instruction opcode (e.g., global_load_dwordx4)')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--defs-only', action='store_true',
                       help='Only search for registers in target instruction defs')
    parser.add_argument('--uses-only', action='store_true',
                       help='Only search for registers in target instruction uses')
    parser.add_argument('--combined-only', action='store_true',
                       help='Only show combined view, skip separate Target Defs/Uses sections')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show verbose output with all register details')
    
    args = parser.parse_args()
    
    # Redirect output if specified
    output_file = None
    if args.output:
        output_file = open(args.output, 'w')
        sys.stdout = output_file
    
    try:
        # Parse the AMDGCN file
        print(f"Parsing {args.input}...")
        amdgcn_parser = AMDGCNParser()
        cfg = amdgcn_parser.parse_file(args.input)
        print(f"Parsed CFG: {cfg.name} ({len(cfg.blocks)} blocks)")
        
        # Normalize block label (add . prefix if missing)
        block_label = args.block_label
        if not block_label.startswith('.'):
            block_label = '.' + block_label
        
        # Check if block exists
        if block_label not in cfg.blocks:
            print(f"\nError: Block '{block_label}' not found in CFG.")
            print(f"Available blocks: {', '.join(cfg.block_order)}")
            sys.exit(1)
        
        target_block = cfg.blocks[block_label]
        
        # Find instructions with target opcode
        matching_instrs = find_instructions_by_opcode(target_block, args.opcode)
        
        if not matching_instrs:
            print(f"\nError: No instructions with opcode '{args.opcode}' found in {block_label}.")
            print(f"Available opcodes in {block_label}:")
            opcodes = set(instr.opcode for instr in target_block.instructions)
            for op in sorted(opcodes):
                print(f"  - {op}")
            sys.exit(1)
        
        print(f"Found {len(matching_instrs)} '{args.opcode}' instructions in {block_label}")
        
        # Collect target registers
        target_defs, target_uses = collect_target_registers(matching_instrs)
        
        # Determine which registers to search for
        if args.defs_only:
            search_defs = target_defs
            search_uses = set()
            target_regs = target_defs
        elif args.uses_only:
            search_defs = set()
            search_uses = target_uses
            target_regs = target_uses
        else:
            search_defs = target_defs
            search_uses = target_uses
            target_regs = target_defs | target_uses
        
        if not target_regs:
            print(f"\nWarning: No registers found in target instructions.")
            sys.exit(0)
        
        # Find all related instructions
        results = find_related_instructions_in_cfg(
            cfg, target_regs, 
            target_defs=search_defs, 
            target_uses=search_uses
        )
        
        # Determine whether to show separate sections
        # Show separate sections only if both target_defs and target_uses are non-empty
        # and --combined-only is not specified
        separate_sections = (
            not args.combined_only and 
            not args.defs_only and 
            not args.uses_only and
            bool(target_defs) and 
            bool(target_uses)
        )
        
        # Print results
        print_results(
            block_label,
            args.opcode,
            search_defs,
            search_uses,
            results,
            cfg,
            verbose=args.verbose,
            separate_sections=separate_sections
        )
        
    finally:
        if output_file:
            output_file.close()
            sys.stdout = sys.__stdout__
            print(f"Output written to: {args.output}")


if __name__ == '__main__':
    main()
