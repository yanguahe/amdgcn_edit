# AMDGCN Assembly Editing Tools

A comprehensive toolkit for parsing, analyzing, and optimizing AMDGCN assembly files. This package provides tools for Control Flow Graph (CFG) generation, Data Dependency Graph (DDG) analysis, instruction scheduling optimization, and register management.

## Features

- **CFG Generation**: Parse AMDGCN assembly files and generate Control Flow Graphs
- **DDG Analysis**: Build Data Dependency Graphs with RAW/WAR dependency tracking
- **Instruction Scheduling**: Move and distribute instructions with dependency verification
- **Register Replacement**: Replace registers within instruction ranges with alignment support
- **Register Slice Analysis**: Extract instruction subgraphs related to specific registers
- **Visualization**: Generate DOT/SVG visualizations of CFG and DDG structures

## Installation

Ensure you have Python 3.6+ and optionally Graphviz for SVG generation:

```bash
# Install graphviz for visualization (optional)
apt-get install graphviz  # Debian/Ubuntu
yum install graphviz      # RHEL/CentOS
```

## Quick Start

### 1. Parse Assembly and Generate DDG

```bash
# Parse .amdgcn file and generate DDG analysis
python3 amdgcn_ddg.py kernel.amdgcn --output-dir ./ddg_output

# Generate JSON analysis only (skip SVG)
python3 amdgcn_ddg.py kernel.amdgcn --output-dir ./ddg_output --json-only

# Print statistics
python3 amdgcn_ddg.py kernel.amdgcn --stats
```

### 2. Load Analysis from JSON

```bash
# Load previously saved analysis
python3 amdgcn_ddg.py --load-json ./ddg_output/analysis.json --stats
```

### 3. Regenerate Assembly File

```bash
# Regenerate .amdgcn from JSON (useful after transformations)
python3 amdgcn_ddg.py --load-json ./ddg_output/analysis.json --regenerate output.amdgcn
```

---

## Transformation Passes

### Move Instruction Pass

Move an instruction up or down within a basic block while respecting data dependencies.

```bash
# Move instruction at index 10 in .LBB0_2 up by 5 cycles
python3 amdgcn_ddg.py --load-json analysis.json --move .LBB0_2 10 5

# Move instruction down by 3 cycles (use negative value)
python3 amdgcn_ddg.py --load-json analysis.json --move .LBB0_2 15 -3
```

### Distribute Instruction Pass

Evenly distribute instructions with a specific opcode throughout a basic block.

```bash
# Distribute global_load_dwordx4 instructions in .LBB0_2 with K=8
python3 amdgcn_ddg.py --load-json analysis.json --distribute .LBB0_2 global_load_dwordx4 8

# Distribute v_mfma instructions
python3 amdgcn_ddg.py --load-json analysis.json --distribute .LBB0_2 v_mfma_f32_32x32x16_fp8_fp8 16
```

### Register Replacement Pass

Replace registers within a specified instruction range with new registers from the free register pool.

```bash
# Replace v[40:45] in instructions 267-1000 with alignment 2
python3 amdgcn_ddg.py --load-json analysis.json --replace-regs 267 1000 v[40:45] 2

# Replace multiple register segments
python3 amdgcn_ddg.py --load-json analysis.json --replace-regs 267 1000 v[40:45] 2 s[37:40] 1
```

### Transform Pass Pipeline (JSON)

Define multiple passes in a JSON file for batch execution:

```json
[
  {
    "type": "replace_registers",
    "range_start": 267,
    "range_end": 1000,
    "registers": ["v[40:45]", "s[37:40]"],
    "alignments": [2, 1]
  },
  {
    "type": "distribute",
    "block": ".LBB0_2",
    "opcode": "global_load_dwordx4",
    "k": 16
  },
  {
    "type": "move",
    "block": ".LBB0_2",
    "index": 10,
    "cycles": 5
  }
]
```

Execute the pipeline:

```bash
python3 amdgcn_ddg.py --load-json analysis.json --transform-json transform_passes.json
```

---

## Register Slice Analysis

Extract all instructions related to specific registers and visualize their dependencies.

```bash
# Analyze instructions using v40-v45
python3 amdgcn_register_slice.py kernel.amdgcn --registers v40,v41,v42,v43,v44,v45 --output-dir ./slice_output

# Use register range notation
python3 amdgcn_register_slice.py kernel.amdgcn --registers "v[40:45]" --output-dir ./slice_output

# Load from JSON instead of parsing .amdgcn
python3 amdgcn_register_slice.py --load-json analysis.json --registers v40,v41 --output-dir ./slice_output

# Analyze special registers
python3 amdgcn_register_slice.py kernel.amdgcn --registers exec,vcc,scc --output-dir ./slice_output
```

Output files:
- `slice.json`: Structured analysis data
- `slice.dot`: GraphViz DOT format
- `slice.svg`: Visual dependency graph
- `slice.txt`: Human-readable summary

---

## Python API

### Basic Usage

```python
from amdgcn_cfg import AMDGCNParser, CFG
from amdgcn_ddg import (
    generate_all_ddgs,
    compute_inter_block_deps,
    save_analysis_to_json,
    load_analysis_from_json,
    AnalysisResult,
)

# Parse assembly file
parser = AMDGCNParser()
cfg = parser.parse_file("kernel.amdgcn")

# Generate DDGs
ddgs, waitcnt_deps = generate_all_ddgs(cfg)
inter_deps = compute_inter_block_deps(cfg, ddgs)

# Create analysis result
result = AnalysisResult(
    cfg=cfg,
    ddgs=ddgs,
    inter_block_deps=inter_deps,
    waitcnt_deps=waitcnt_deps
)

# Save to JSON
save_analysis_to_json(cfg, ddgs, inter_deps, waitcnt_deps, "analysis.json")

# Load from JSON
result = load_analysis_from_json("analysis.json")
```

### Using Transformation Passes

```python
from amdgcn_passes import (
    PassManager,
    MoveInstructionPass,
    DistributeInstructionPass,
    RegisterReplacePass,
    move_instruction,
    distribute_instructions,
    replace_registers,
)

# Method 1: Use convenience functions
move_result = move_instruction(result, ".LBB0_2", 10, 5, verbose=True)
if move_result.success:
    print(f"Moved: {move_result.message}")

# Method 2: Use PassManager
pm = PassManager()
pm.add_pass(RegisterReplacePass(
    range_start=267,
    range_end=1000,
    registers_to_replace=["v[40:45]"],
    alignments=[2],
    verbose=True
))
pm.add_pass(DistributeInstructionPass(
    block_label=".LBB0_2",
    target_opcode="global_load_dwordx4",
    k=16,
    verbose=True
))
pm.run(result)

# Regenerate assembly with updated metadata
result.to_amdgcn("output.amdgcn", update_metadata=True)
```

### Register Statistics and Free GPR Analysis

```python
from amdgcn_ddg import (
    compute_register_statistics,
    compute_fgpr,
    compute_register_metadata,
    RegisterMetadata,
)

# Compute register usage
stats = compute_register_statistics(ddgs)
print(f"VGPR max index: {stats.vgpr_max_index}")
print(f"AGPR max index: {stats.agpr_max_index}")
print(f"SGPR max index: {stats.sgpr_max_index}")
print(f"Uses VCC: {stats.uses_vcc}")

# Compute free GPRs
fgpr = compute_fgpr(stats)
print(f"Free VGPRs: {len(fgpr.fgpr_v)}")
print(f"Free SGPRs: {len(fgpr.fgpr_s)}")

# Compute metadata for .amdgcn regeneration
metadata = compute_register_metadata(stats)
print(f"next_free_vgpr: {metadata.next_free_vgpr}")
print(f"accum_offset: {metadata.accum_offset}")
print(f"total_num_sgprs: {metadata.total_num_sgprs}")
```

### Register Slice Analysis

```python
from amdgcn_register_slice import (
    build_register_slice,
    generate_slice_dot,
    generate_slice_json,
    save_slice_outputs,
)

# Build slice for specific registers
target_registers = {"v40", "v41", "v42"}
slice_result = build_register_slice(cfg, ddgs, target_registers)

# Generate outputs
save_slice_outputs(slice_result, "./slice_output", generate_svg=True)
```

---

## File Structure

```
amdgcn_edit/
├── amdgcn_cfg.py           # CFG parser and data structures
├── amdgcn_ddg.py           # DDG generation and main CLI
├── amdgcn_passes.py        # Instruction scheduling passes
├── amdgcn_register_slice.py # Register slice analyzer
├── amdgcn_latency.py       # Hardware latency analysis
├── amdgcn_verify.py        # Scheduling verification
├── gfx942_hardware_info.json # GFX942/GFX950 hardware info
├── __init__.py             # Package exports
└── test_*.py               # Unit tests
```

---

## Hardware Support

Currently supports AMD GFX942/GFX950 (MI300) architecture with:
- VGPR: v0-v255 (256 registers)
- AGPR: a0-a255 (256 accumulator registers)
- SGPR: s0-s103 (104 scalar registers)
- Special registers: exec, vcc, scc, m0

Hardware information is stored in `gfx942_hardware_info.json` including:
- MFMA instruction latencies
- Register limits
- Instruction cycle costs

---

## Running Tests

```bash
# Run all tests
python3 -m pytest test_*.py -v

# Run specific test file
python3 test_amdgcn_ddg.py
python3 test_amdgcn_passes.py
python3 test_amdgcn_register_slice.py
```
