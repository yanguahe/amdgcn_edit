"""
AMDGCN Assembly Editing Tools

This package provides tools for parsing and analyzing AMDGCN assembly files,
including control flow graph (CFG) generation, data dependency graph (DDG)
analysis, and visualization.

Modules:
    amdgcn_cfg: Parse AMDGCN assembly files and generate CFG in DOT format
    amdgcn_ddg: Generate Data Dependency Graphs for basic blocks
"""

from .amdgcn_cfg import (
    AMDGCNParser,
    CFG,
    BasicBlock,
    Instruction,
    generate_dot,
    generate_simple_dot,
    print_cfg_stats,
    list_basic_blocks,
)

from .amdgcn_ddg import (
    DDG,
    InstructionNode,
    AnalysisResult,
    build_ddg,
    generate_all_ddgs,
    generate_ddg_dot,
    generate_combined_cfg_ddg_dot,
    save_ddg_files,
    print_ddg_stats,
    parse_instruction_registers,
    save_analysis_to_json,
    load_analysis_from_json,
)

from .amdgcn_passes import (
    Pass,
    PassManager,
    MoveInstructionPass,
    DistributeInstructionPass,
    MoveResult,
    apply_passes,
    move_instruction,
    distribute_instructions,
    get_instruction_cycles,
)

__all__ = [
    # CFG
    'AMDGCNParser',
    'CFG',
    'BasicBlock',
    'Instruction',
    'generate_dot',
    'generate_simple_dot',
    'print_cfg_stats',
    'list_basic_blocks',
    # DDG
    'DDG',
    'InstructionNode',
    'AnalysisResult',
    'build_ddg',
    'generate_all_ddgs',
    'generate_ddg_dot',
    'generate_combined_cfg_ddg_dot',
    'save_ddg_files',
    'print_ddg_stats',
    'parse_instruction_registers',
    'save_analysis_to_json',
    'load_analysis_from_json',
    # Passes
    'Pass',
    'PassManager',
    'MoveInstructionPass',
    'DistributeInstructionPass',
    'MoveResult',
    'apply_passes',
    'move_instruction',
    'distribute_instructions',
    'get_instruction_cycles',
]

