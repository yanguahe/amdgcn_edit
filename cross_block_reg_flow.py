#!/usr/bin/env python3
"""
Cross-Block Register Flow Analyzer

This tool analyzes the data flow between two basic blocks by tracking
which producer instructions' destination registers are used by consumer
instructions in another block.

Usage:
    python cross_block_reg_flow.py <input.amdgcn> \\
        --bb0 .LBB0_0 --opcode-a global_load_dwordx4 \\
        --bb1 .LBB0_2 --opcode-b v_mfma_f32_16x16x16_bf16

    # Or using JSON analysis file:
    python cross_block_reg_flow.py <analysis.json> \\
        --bb0 .LBB0_0 --opcode-a global_load_dwordx4 \\
        --bb1 .LBB0_2 --opcode-b v_mfma_f32_16x16x16_bf16 \\
        --json

Example Output:
    A[0] @ line 45: global_load_dwordx4 v[0:3], ...
      dst regs: v0, v1, v2, v3
      -> B[2] @ line 120 uses: v0, v1
      -> B[5] @ line 135 uses: v2, v3

    A[1] @ line 48: global_load_dwordx4 v[4:7], ...
      dst regs: v4, v5, v6, v7
      -> B[3] @ line 125 uses: v4, v5
"""

import argparse
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

# Import from existing modules
from amdgcn_cfg import AMDGCNParser, CFG, BasicBlock, Instruction
from amdgcn_ddg import parse_instruction_registers, AnalysisResult


@dataclass
class InstructionInfo:
    """Information about an instruction for analysis."""
    index: int              # Assigned index (0, 1, 2, ...)
    address: int            # Original line number in source
    opcode: str
    operands: str
    raw_line: str
    defs: Set[str]          # Destination registers
    uses: Set[str]          # Source registers


@dataclass  
class RegFlowMatch:
    """A match between producer and consumer instructions."""
    consumer_index: int     # B instruction index
    consumer_address: int   # B instruction line number
    matched_regs: Set[str]  # Registers that flow from A to B


@dataclass
class CrossBlockFlowResult:
    """Result of cross-block register flow analysis."""
    bb0_label: str
    bb1_label: str
    opcode_a: str
    opcode_b: str
    producers: List[InstructionInfo]       # A instructions in BB0
    consumers: List[InstructionInfo]       # B instructions in BB1
    # Mapping: producer_index -> list of RegFlowMatch
    flow_map: Dict[int, List[RegFlowMatch]] = field(default_factory=dict)


def expand_register_range(reg_str: str) -> Set[str]:
    """
    Expand a register range notation to individual registers.
    
    Examples:
        "v[0:3]" -> {"v0", "v1", "v2", "v3"}
        "s[4:5]" -> {"s4", "s5"}
        "a[0:15]" -> {"a0", "a1", ..., "a15"}
        "v0" -> {"v0"}
    """
    result = set()
    
    # Match range pattern: prefix[start:end]
    range_match = re.match(r'([vsaVSA])\[(\d+):(\d+)\]', reg_str)
    if range_match:
        prefix = range_match.group(1).lower()
        start = int(range_match.group(2))
        end = int(range_match.group(3))
        for i in range(start, end + 1):
            result.add(f"{prefix}{i}")
        return result
    
    # Match single register: prefix + number
    single_match = re.match(r'([vsaVSA])(\d+)', reg_str)
    if single_match:
        prefix = single_match.group(1).lower()
        num = single_match.group(2)
        result.add(f"{prefix}{num}")
        return result
    
    # Return as-is for special registers like vcc, scc, exec
    if reg_str.lower() in ('vcc', 'scc', 'exec'):
        result.add(reg_str.lower())
    
    return result


def normalize_register_set(regs: Set[str]) -> Set[str]:
    """
    Normalize a set of registers by expanding all range notations.
    
    Input: {"v[0:3]", "s4"}
    Output: {"v0", "v1", "v2", "v3", "s4"}
    """
    normalized = set()
    for reg in regs:
        expanded = expand_register_range(reg)
        if expanded:
            normalized.update(expanded)
        else:
            normalized.add(reg)
    return normalized


def find_instructions_by_opcode(
    block: BasicBlock,
    target_opcode: str,
    exact_match: bool = False
) -> List[InstructionInfo]:
    """
    Find all instructions in a block that match the target opcode.
    
    Args:
        block: The basic block to search
        target_opcode: The opcode to match (e.g., "global_load_dwordx4")
        exact_match: If True, require exact opcode match. 
                     If False, allow prefix match (e.g., "global_load" matches "global_load_dwordx4")
    
    Returns:
        List of InstructionInfo sorted by address, indexed from 0
    """
    matches = []
    target_lower = target_opcode.lower()
    
    for instr in block.instructions:
        opcode_lower = instr.opcode.lower()
        
        if exact_match:
            matched = opcode_lower == target_lower
        else:
            # Allow prefix match
            matched = opcode_lower == target_lower or opcode_lower.startswith(target_lower + '_') or target_lower.startswith(opcode_lower)
            # Also try exact match first
            if opcode_lower == target_lower:
                matched = True
            # Check if target is a prefix of the actual opcode
            elif opcode_lower.startswith(target_lower):
                matched = True
        
        if matched:
            defs, uses = parse_instruction_registers(instr)
            # Normalize registers to individual form
            defs_normalized = normalize_register_set(defs)
            uses_normalized = normalize_register_set(uses)
            
            matches.append(InstructionInfo(
                index=-1,  # Will be assigned later
                address=instr.address,
                opcode=instr.opcode,
                operands=instr.operands,
                raw_line=instr.raw_line,
                defs=defs_normalized,
                uses=uses_normalized
            ))
    
    # Sort by address and assign indices
    matches.sort(key=lambda x: x.address)
    for i, info in enumerate(matches):
        info.index = i
    
    return matches


def analyze_cross_block_flow(
    cfg: CFG,
    bb0_label: str,
    opcode_a: str,
    bb1_label: str,
    opcode_b: str,
    exact_match: bool = False
) -> CrossBlockFlowResult:
    """
    Analyze the register flow from BB0's A instructions to BB1's B instructions.
    
    For each A instruction in BB0:
    - Find its destination registers
    - Find all B instructions in BB1 that use any of these registers
    
    Args:
        cfg: The control flow graph
        bb0_label: Label of the producer block (e.g., ".LBB0_0")
        opcode_a: Opcode of producer instructions (e.g., "global_load_dwordx4")
        bb1_label: Label of the consumer block (e.g., ".LBB0_2")
        opcode_b: Opcode of consumer instructions (e.g., "v_mfma_f32_16x16x16_bf16")
        exact_match: If True, require exact opcode match
    
    Returns:
        CrossBlockFlowResult with the analysis
    """
    # Validate blocks exist
    if bb0_label not in cfg.blocks:
        raise ValueError(f"Block '{bb0_label}' not found in CFG. Available: {list(cfg.blocks.keys())}")
    if bb1_label not in cfg.blocks:
        raise ValueError(f"Block '{bb1_label}' not found in CFG. Available: {list(cfg.blocks.keys())}")
    
    block0 = cfg.blocks[bb0_label]
    block1 = cfg.blocks[bb1_label]
    
    # Find A instructions in BB0 (producers)
    producers = find_instructions_by_opcode(block0, opcode_a, exact_match)
    
    # Find B instructions in BB1 (consumers)
    consumers = find_instructions_by_opcode(block1, opcode_b, exact_match)
    
    # Build the flow map
    flow_map: Dict[int, List[RegFlowMatch]] = {}
    
    for producer in producers:
        matches = []
        producer_defs = producer.defs
        
        for consumer in consumers:
            consumer_uses = consumer.uses
            
            # Find registers that flow from producer to consumer
            common_regs = producer_defs & consumer_uses
            
            if common_regs:
                matches.append(RegFlowMatch(
                    consumer_index=consumer.index,
                    consumer_address=consumer.address,
                    matched_regs=common_regs
                ))
        
        flow_map[producer.index] = matches
    
    return CrossBlockFlowResult(
        bb0_label=bb0_label,
        bb1_label=bb1_label,
        opcode_a=opcode_a,
        opcode_b=opcode_b,
        producers=producers,
        consumers=consumers,
        flow_map=flow_map
    )


def format_register_set(regs: Set[str], max_display: int = 20) -> str:
    """Format a set of registers for display."""
    # Sort registers by type and number
    def reg_sort_key(r):
        match = re.match(r'([a-z]+)(\d+)?', r.lower())
        if match:
            prefix = match.group(1)
            num = int(match.group(2)) if match.group(2) else 0
            return (prefix, num)
        return (r, 0)
    
    sorted_regs = sorted(regs, key=reg_sort_key)
    
    if len(sorted_regs) > max_display:
        displayed = sorted_regs[:max_display]
        return ", ".join(displayed) + f", ... (+{len(sorted_regs) - max_display} more)"
    else:
        return ", ".join(sorted_regs)


def print_flow_result(result: CrossBlockFlowResult, verbose: bool = False):
    """Print the analysis result in a readable format."""
    print(f"\n{'='*70}")
    print(f"Cross-Block Register Flow Analysis")
    print(f"{'='*70}")
    print(f"Producer Block: {result.bb0_label}")
    print(f"Producer Opcode: {result.opcode_a}")
    print(f"Consumer Block: {result.bb1_label}")
    print(f"Consumer Opcode: {result.opcode_b}")
    print(f"{'='*70}")
    
    print(f"\nFound {len(result.producers)} A instructions ({result.opcode_a}) in {result.bb0_label}")
    print(f"Found {len(result.consumers)} B instructions ({result.opcode_b}) in {result.bb1_label}")
    
    if verbose:
        print(f"\n--- All A Instructions ---")
        for p in result.producers:
            print(f"  A[{p.index}] @ line {p.address}: {p.opcode} {p.operands}")
            print(f"         defs: {format_register_set(p.defs)}")
        
        print(f"\n--- All B Instructions ---")
        for c in result.consumers:
            print(f"  B[{c.index}] @ line {c.address}: {c.opcode} {c.operands}")
            print(f"         uses: {format_register_set(c.uses)}")
    
    print(f"\n{'='*70}")
    print(f"Register Flow: B <- A")
    print(f"{'='*70}\n")
    
    # Build reverse mapping: consumer_index -> list of (producer_index, matched_regs)
    reverse_map: Dict[int, List[Tuple[int, Set[str]]]] = {}
    for producer_idx, matches in result.flow_map.items():
        for match in matches:
            if match.consumer_index not in reverse_map:
                reverse_map[match.consumer_index] = []
            reverse_map[match.consumer_index].append((producer_idx, match.matched_regs))
    
    # Output by B instruction order
    for consumer in result.consumers:
        # Truncate operands if too long
        operands_display = consumer.operands
        if len(operands_display) > 50:
            operands_display = operands_display[:47] + "..."
        
        print(f"B[{consumer.index}] @ line {consumer.address}: {consumer.opcode} {operands_display}")
        print(f"  src regs: {format_register_set(consumer.uses)}")
        
        producer_matches = reverse_map.get(consumer.index, [])
        if producer_matches:
            # Sort by producer index
            for producer_idx, matched_regs in sorted(producer_matches, key=lambda x: x[0]):
                producer = result.producers[producer_idx]
                print(f"  <- A[{producer_idx}] @ line {producer.address} defs: {format_register_set(matched_regs)}")
                # Show full A instruction
                print(f"     {producer.opcode} {producer.operands}")
        else:
            print(f"  <- (no matching A instructions)")
        
        print()
    
    # Summary statistics
    print(f"{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    
    total_matches = sum(len(matches) for matches in result.flow_map.values())
    producers_with_matches = sum(1 for matches in result.flow_map.values() if matches)
    
    print(f"Total A->B register flow edges: {total_matches}")
    print(f"A instructions with B consumers: {producers_with_matches}/{len(result.producers)}")
    
    # Find B instructions that have A producers
    consumers_with_producers = set()
    for matches in result.flow_map.values():
        for m in matches:
            consumers_with_producers.add(m.consumer_index)
    
    print(f"B instructions with A producers: {len(consumers_with_producers)}/{len(result.consumers)}")


def export_to_json(result: CrossBlockFlowResult, filepath: str):
    """Export the analysis result to JSON format."""
    data = {
        "bb0_label": result.bb0_label,
        "bb1_label": result.bb1_label,
        "opcode_a": result.opcode_a,
        "opcode_b": result.opcode_b,
        "producers": [
            {
                "index": p.index,
                "address": p.address,
                "opcode": p.opcode,
                "operands": p.operands,
                "defs": sorted(list(p.defs))
            }
            for p in result.producers
        ],
        "consumers": [
            {
                "index": c.index,
                "address": c.address,
                "opcode": c.opcode,
                "operands": c.operands,
                "uses": sorted(list(c.uses))
            }
            for c in result.consumers
        ],
        "flow_map": {
            str(k): [
                {
                    "consumer_index": m.consumer_index,
                    "consumer_address": m.consumer_address,
                    "matched_regs": sorted(list(m.matched_regs))
                }
                for m in v
            ]
            for k, v in result.flow_map.items()
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nExported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze cross-block register flow between instruction types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from .amdgcn file
  python cross_block_reg_flow.py kernel.amdgcn \\
      --bb0 .LBB0_0 --opcode-a global_load_dwordx4 \\
      --bb1 .LBB0_2 --opcode-b v_mfma_f32_16x16x16_bf16

  # Analyze from JSON analysis file
  python cross_block_reg_flow.py analysis.json --json \\
      --bb0 .LBB0_0 --opcode-a global_load_dwordx4 \\
      --bb1 .LBB0_2 --opcode-b v_mfma_f32_16x16x16_bf16

  # Export results to JSON
  python cross_block_reg_flow.py kernel.amdgcn \\
      --bb0 .LBB0_0 --opcode-a global_load \\
      --bb1 .LBB0_2 --opcode-b v_mfma \\
      --output flow_result.json
        """
    )
    
    parser.add_argument('input', help='Input .amdgcn or .json analysis file')
    parser.add_argument('--json', action='store_true',
                        help='Input is a JSON analysis file (from amdgcn_ddg.py)')
    parser.add_argument('--bb0', required=True,
                        help='Producer block label (e.g., .LBB0_0)')
    parser.add_argument('--opcode-a', required=True,
                        help='Producer instruction opcode (e.g., global_load_dwordx4)')
    parser.add_argument('--bb1', required=True,
                        help='Consumer block label (e.g., .LBB0_2)')
    parser.add_argument('--opcode-b', required=True,
                        help='Consumer instruction opcode (e.g., v_mfma_f32_16x16x16_bf16)')
    parser.add_argument('--exact', action='store_true',
                        help='Require exact opcode match (no prefix matching)')
    parser.add_argument('--output', '-o',
                        help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output including all instruction details')
    
    args = parser.parse_args()
    
    # Load the CFG
    if args.json:
        # Load from JSON analysis file
        result = AnalysisResult.from_dict(
            json.load(open(args.input, 'r', encoding='utf-8'))
        )
        cfg = result.cfg
        print(f"Loaded analysis from: {args.input}")
    else:
        # Parse .amdgcn file
        amdgcn_parser = AMDGCNParser()
        cfg = amdgcn_parser.parse_file(args.input)
        print(f"Parsed: {args.input}")
    
    print(f"Function: {cfg.name}")
    print(f"Total blocks: {len(cfg.blocks)}")
    
    # Run the analysis
    try:
        result = analyze_cross_block_flow(
            cfg=cfg,
            bb0_label=args.bb0,
            opcode_a=args.opcode_a,
            bb1_label=args.bb1,
            opcode_b=args.opcode_b,
            exact_match=args.exact
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Print results
    print_flow_result(result, verbose=args.verbose)
    
    # Export if requested
    if args.output:
        export_to_json(result, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())
