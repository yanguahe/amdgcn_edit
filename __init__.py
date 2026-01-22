"""
AMDGCN Assembly Editing Tools

This package provides tools for parsing and analyzing AMDGCN assembly files,
including control flow graph (CFG) generation, data dependency graph (DDG)
analysis, register slice analysis, and visualization.

Modules:
    amdgcn_cfg: Parse AMDGCN assembly files and generate CFG in DOT format
    amdgcn_ddg: Generate Data Dependency Graphs for basic blocks
    amdgcn_register_slice: Analyze register dependencies and generate slice views
    amdgcn_passes: Instruction scheduling passes
    amdgcn_latency: Hardware latency analysis
    amdgcn_verify: Scheduling verification
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
    RegisterStatistics,
    FreeGPRInfo,
    RegisterMetadata,
    build_ddg,
    generate_all_ddgs,
    generate_ddg_dot,
    generate_combined_cfg_ddg_dot,
    save_ddg_files,
    print_ddg_stats,
    parse_instruction_registers,
    save_analysis_to_json,
    load_analysis_from_json,
    compute_register_statistics,
    compute_fgpr,
    compute_register_metadata,
    load_hardware_info,
)

from .amdgcn_passes import (
    Pass,
    PassManager,
    MoveInstructionPass,
    DistributeInstructionPass,
    RegisterReplacePass,
    RegisterSegment,
    MoveResult,
    apply_passes,
    move_instruction,
    distribute_instructions,
    replace_registers,
    get_instruction_cycles,
    parse_register_segment,
    find_aligned_free_registers,
)

from .amdgcn_register_slice import (
    SliceInstruction,
    SliceEdge,
    RegisterSlice,
    GlobalPosition,
    build_register_slice,
    build_global_position_map,
    find_related_instructions,
    find_dependency_edges,
    find_all_barriers,
    generate_slice_dot,
    generate_slice_json,
    generate_slice_text,
    save_slice_outputs,
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
    'RegisterStatistics',
    'FreeGPRInfo',
    'RegisterMetadata',
    'build_ddg',
    'generate_all_ddgs',
    'generate_ddg_dot',
    'generate_combined_cfg_ddg_dot',
    'save_ddg_files',
    'print_ddg_stats',
    'parse_instruction_registers',
    'save_analysis_to_json',
    'load_analysis_from_json',
    'compute_register_statistics',
    'compute_fgpr',
    'compute_register_metadata',
    'load_hardware_info',
    # Passes
    'Pass',
    'PassManager',
    'MoveInstructionPass',
    'DistributeInstructionPass',
    'RegisterReplacePass',
    'RegisterSegment',
    'MoveResult',
    'apply_passes',
    'move_instruction',
    'distribute_instructions',
    'replace_registers',
    'get_instruction_cycles',
    'parse_register_segment',
    'find_aligned_free_registers',
    # Register Slice
    'SliceInstruction',
    'SliceEdge',
    'RegisterSlice',
    'GlobalPosition',
    'build_register_slice',
    'build_global_position_map',
    'find_related_instructions',
    'find_dependency_edges',
    'find_all_barriers',
    'generate_slice_dot',
    'generate_slice_json',
    'generate_slice_text',
    'save_slice_outputs',
]

