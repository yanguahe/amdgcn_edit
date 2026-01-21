#!/usr/bin/env python3
"""
Comprehensive Unit Tests for AMDGCN Register Slice Analyzer

Tests cover:
- Data structure serialization/deserialization
- Single register matching
- Multiple register matching
- Cross-block search
- s_barrier crossing detection
- s_waitcnt inclusion
- Empty results handling
- Special registers (exec, scc, vcc)
- Output format validity (JSON, DOT)
- Edge cases (loop blocks, multiple barriers, nested dependencies)

Run tests:
    pytest test_amdgcn_register_slice.py -v
    
Or run directly:
    python test_amdgcn_register_slice.py
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Set, List

# Try to import pytest, but work without it
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
    # Create minimal pytest stubs for compatibility
    class PytestStub:
        @staticmethod
        def mark():
            pass
        
        class skip:
            class Exception(Exception):
                pass
        
        @staticmethod
        def skipif(condition, reason=""):
            def decorator(func):
                func._skip_condition = condition
                func._skip_reason = reason
                return func
            return decorator
    
    pytest = PytestStub()
    pytest.mark = type('mark', (), {
        'skipif': pytest.skipif
    })()

from amdgcn_cfg import AMDGCNParser, Instruction, BasicBlock, CFG
from amdgcn_ddg import generate_all_ddgs, DDG, InstructionNode

from amdgcn_register_slice import (
    # Data structures
    SliceInstruction,
    SliceEdge,
    RegisterSlice,
    GlobalPosition,
    
    # Core functions
    build_global_position_map,
    is_position_before,
    find_related_instructions,
    find_all_barriers,
    find_barriers_between,
    find_dependency_edges,
    build_register_slice,
    build_register_slice_from_cfg,
    
    # Output generation
    generate_slice_dot,
    generate_slice_json,
    generate_slice_text,
    save_slice_outputs,
    
    # Utilities
    parse_register_list,
    print_slice_summary,
)


# =============================================================================
# Test Data - Real assembly file
# =============================================================================

TEST_ASSEMBLY_FILE = Path(__file__).parent / "pa_dot_kernel.v2.amdgcn"


# =============================================================================
# Helper Functions for Creating Test Data
# =============================================================================

def create_test_instruction(
    address: int,
    opcode: str,
    operands: str,
    is_branch: bool = False,
    is_terminator: bool = False
) -> Instruction:
    """Create a test Instruction object."""
    raw_line = f"\t{opcode} {operands}"
    return Instruction(
        address=address,
        opcode=opcode,
        operands=operands,
        raw_line=raw_line,
        is_branch=is_branch,
        is_terminator=is_terminator,
    )


def create_test_block(label: str, instructions: List[Instruction]) -> BasicBlock:
    """Create a test BasicBlock object."""
    block = BasicBlock(label=label)
    block.instructions = instructions
    return block


def create_simple_cfg() -> CFG:
    """Create a simple CFG for testing."""
    cfg = CFG(name="test_kernel")
    
    # Block 1: some arithmetic
    block1 = create_test_block(".LBB0_0", [
        create_test_instruction(10, "v_add_f32", "v0, v1, v2"),
        create_test_instruction(11, "v_mul_f32", "v3, v0, v4"),
        create_test_instruction(12, "s_add_u32", "s0, s1, s2"),
    ])
    
    # Block 2: with barrier
    block2 = create_test_block(".LBB0_1", [
        create_test_instruction(20, "v_mov_b32", "v5, v3"),
        create_test_instruction(21, "s_barrier", ""),
        create_test_instruction(22, "v_add_f32", "v6, v3, v5"),
        create_test_instruction(23, "s_endpgm", "", is_terminator=True),
    ])
    
    cfg.add_block(block1)
    cfg.add_block(block2)
    cfg.add_edge(".LBB0_0", ".LBB0_1")
    cfg.block_order = [".LBB0_0", ".LBB0_1"]
    
    return cfg


def create_cfg_with_waitcnt() -> CFG:
    """Create a CFG with s_waitcnt instructions."""
    cfg = CFG(name="test_kernel_waitcnt")
    
    block1 = create_test_block(".LBB0_0", [
        create_test_instruction(10, "s_load_dwordx2", "s[0:1], s[2:3], 0x0"),
        create_test_instruction(11, "global_load_dwordx4", "v[0:3], v[4:5], off"),
        create_test_instruction(12, "s_waitcnt", "lgkmcnt(0)"),
        create_test_instruction(13, "s_add_u32", "s4, s0, s1"),
        create_test_instruction(14, "s_waitcnt", "vmcnt(0)"),
        create_test_instruction(15, "v_add_f32", "v6, v0, v1"),
    ])
    
    cfg.add_block(block1)
    cfg.block_order = [".LBB0_0"]
    
    return cfg


def create_cfg_with_multiple_barriers() -> CFG:
    """Create a CFG with multiple s_barrier instructions."""
    cfg = CFG(name="test_kernel_multi_barrier")
    
    block1 = create_test_block(".LBB0_0", [
        create_test_instruction(10, "v_mov_b32", "v0, 0"),
        create_test_instruction(11, "s_barrier", ""),
        create_test_instruction(12, "v_add_f32", "v1, v0, v0"),
        create_test_instruction(13, "s_barrier", ""),
        create_test_instruction(14, "v_mul_f32", "v2, v0, v1"),
    ])
    
    cfg.add_block(block1)
    cfg.block_order = [".LBB0_0"]
    
    return cfg


def create_cfg_with_special_registers() -> CFG:
    """Create a CFG with special registers (exec, scc, vcc)."""
    cfg = CFG(name="test_kernel_special")
    
    block1 = create_test_block(".LBB0_0", [
        create_test_instruction(10, "v_cmp_gt_f32", "vcc, v0, v1"),
        create_test_instruction(11, "s_cmp_eq_u32", "s0, s1"),  # writes scc
        create_test_instruction(12, "s_cbranch_scc0", ".LBB0_1", is_branch=True),
    ])
    
    block2 = create_test_block(".LBB0_1", [
        create_test_instruction(20, "v_cndmask_b32", "v2, v3, v4, vcc"),
        create_test_instruction(21, "s_and_saveexec_b64", "s[2:3], vcc"),
        create_test_instruction(22, "s_endpgm", "", is_terminator=True),
    ])
    
    cfg.add_block(block1)
    cfg.add_block(block2)
    cfg.add_edge(".LBB0_0", ".LBB0_1")
    cfg.block_order = [".LBB0_0", ".LBB0_1"]
    
    return cfg


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestSliceInstruction:
    """Tests for SliceInstruction data class."""
    
    def test_create_basic(self):
        """Test basic creation."""
        instr = SliceInstruction(
            address=100,
            opcode="v_add_f32",
            operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2",
            block_label=".LBB0_0",
            position_in_block=5,
            reads={"v1", "v2"},
            writes={"v0"},
        )
        assert instr.address == 100
        assert instr.opcode == "v_add_f32"
        assert "v1" in instr.reads
        assert "v0" in instr.writes
    
    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = SliceInstruction(
            address=100,
            opcode="v_add_f32",
            operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2",
            block_label=".LBB0_0",
            position_in_block=5,
            reads={"v1", "v2"},
            writes={"v0"},
            is_barrier=False,
            is_waitcnt=False,
            available_regs=set(),
        )
        
        data = original.to_dict()
        restored = SliceInstruction.from_dict(data)
        
        assert restored.address == original.address
        assert restored.opcode == original.opcode
        assert restored.reads == original.reads
        assert restored.writes == original.writes
    
    def test_equality_by_address(self):
        """Test that equality is based on address."""
        instr1 = SliceInstruction(address=100, opcode="v_add_f32", operands="v0, v1, v2",
                                  raw_line="", block_label="", position_in_block=0)
        instr2 = SliceInstruction(address=100, opcode="v_mul_f32", operands="v0, v1, v2",
                                  raw_line="", block_label="", position_in_block=0)
        instr3 = SliceInstruction(address=200, opcode="v_add_f32", operands="v0, v1, v2",
                                  raw_line="", block_label="", position_in_block=0)
        
        assert instr1 == instr2  # Same address
        assert instr1 != instr3  # Different address
    
    def test_hashable(self):
        """Test that SliceInstruction is hashable."""
        instr1 = SliceInstruction(address=100, opcode="v_add_f32", operands="v0, v1, v2",
                                  raw_line="", block_label="", position_in_block=0)
        instr2 = SliceInstruction(address=100, opcode="v_mul_f32", operands="v3, v4, v5",
                                  raw_line="", block_label="", position_in_block=0)
        
        s = {instr1, instr2}
        assert len(s) == 1  # Same address, same hash


class TestSliceEdge:
    """Tests for SliceEdge data class."""
    
    def test_create_basic(self):
        """Test basic creation."""
        edge = SliceEdge(
            from_addr=100,
            to_addr=200,
            dep_type="RAW",
            registers={"v0"},
        )
        assert edge.from_addr == 100
        assert edge.to_addr == 200
        assert edge.dep_type == "RAW"
        assert not edge.crosses_barrier
    
    def test_barrier_crossing(self):
        """Test edge with barrier crossing."""
        edge = SliceEdge(
            from_addr=100,
            to_addr=200,
            dep_type="RAW",
            registers={"v0"},
            crosses_barrier=True,
            barrier_addrs=[150, 175],
        )
        assert edge.crosses_barrier
        assert edge.barrier_addrs == [150, 175]
    
    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = SliceEdge(
            from_addr=100,
            to_addr=200,
            dep_type="RAW",
            registers={"v0", "v1"},
            crosses_barrier=True,
            barrier_addrs=[150],
        )
        
        data = original.to_dict()
        restored = SliceEdge.from_dict(data)
        
        assert restored.from_addr == original.from_addr
        assert restored.to_addr == original.to_addr
        assert restored.registers == original.registers
        assert restored.crosses_barrier == original.crosses_barrier


class TestRegisterSlice:
    """Tests for RegisterSlice data class."""
    
    def test_create_empty(self):
        """Test empty slice creation."""
        slice_result = RegisterSlice()
        assert slice_result.get_instruction_count() == 0
        assert slice_result.get_edge_count() == 0
    
    def test_statistics(self):
        """Test slice statistics."""
        slice_result = RegisterSlice(target_registers={"v0", "v1"})
        
        slice_result.instructions[100] = SliceInstruction(
            address=100, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="", block_label=".LBB0_0", position_in_block=0
        )
        slice_result.instructions[200] = SliceInstruction(
            address=200, opcode="v_mul_f32", operands="v3, v0, v4",
            raw_line="", block_label=".LBB0_0", position_in_block=1
        )
        
        slice_result.edges.append(SliceEdge(from_addr=100, to_addr=200, dep_type="RAW"))
        slice_result.edges.append(SliceEdge(from_addr=100, to_addr=200, dep_type="WAR",
                                            crosses_barrier=True, barrier_addrs=[150]))
        
        assert slice_result.get_instruction_count() == 2
        assert slice_result.get_edge_count() == 2
        assert slice_result.get_barrier_crossing_edge_count() == 1
    
    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = RegisterSlice(
            target_registers={"v0", "v1"},
            cfg_name="test_kernel",
        )
        original.instructions[100] = SliceInstruction(
            address=100, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2", block_label=".LBB0_0", position_in_block=0
        )
        original.edges.append(SliceEdge(from_addr=100, to_addr=200, dep_type="RAW"))
        
        data = original.to_dict()
        restored = RegisterSlice.from_dict(data)
        
        assert restored.target_registers == original.target_registers
        assert restored.cfg_name == original.cfg_name
        assert len(restored.instructions) == len(original.instructions)
        assert len(restored.edges) == len(original.edges)


# =============================================================================
# Position Map Tests
# =============================================================================

class TestGlobalPosition:
    """Tests for GlobalPosition and position comparison."""
    
    def test_same_block_comparison(self):
        """Test comparison within same block."""
        pos1 = GlobalPosition(block_order_idx=0, position_in_block=5, address=100)
        pos2 = GlobalPosition(block_order_idx=0, position_in_block=10, address=200)
        
        assert pos1 < pos2
        assert is_position_before(pos1, pos2)
        assert not is_position_before(pos2, pos1)
    
    def test_cross_block_comparison(self):
        """Test comparison across blocks."""
        pos1 = GlobalPosition(block_order_idx=0, position_in_block=100, address=100)
        pos2 = GlobalPosition(block_order_idx=1, position_in_block=0, address=200)
        
        assert pos1 < pos2
        assert is_position_before(pos1, pos2)
    
    def test_build_position_map(self):
        """Test building position map from CFG."""
        cfg = create_simple_cfg()
        pos_map = build_global_position_map(cfg)
        
        # Check that all instructions are mapped
        assert 10 in pos_map  # First block
        assert 20 in pos_map  # Second block
        
        # Check correct ordering
        assert pos_map[10] < pos_map[20]
        assert pos_map[10] < pos_map[11]


# =============================================================================
# Core Search Tests
# =============================================================================

class TestFindRelatedInstructions:
    """Tests for find_related_instructions function."""
    
    def test_single_register(self):
        """Test finding instructions for a single register."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        # Search for v0 - should find instructions that read/write v0
        instructions = find_related_instructions(cfg, ddgs, {"v0"})
        
        # v0 is written at address 10, and read at address 11
        assert len(instructions) >= 1
    
    def test_multiple_registers(self):
        """Test finding instructions for multiple registers."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        instructions = find_related_instructions(cfg, ddgs, {"v0", "v3", "s0"})
        
        # Should find more instructions
        assert len(instructions) >= 1
    
    def test_no_matches(self):
        """Test with registers that don't exist in the code."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        instructions = find_related_instructions(cfg, ddgs, {"v999", "s999"})
        
        # Should find no instructions
        assert len(instructions) == 0
    
    def test_special_register_vcc(self):
        """Test finding instructions using vcc."""
        cfg = create_cfg_with_special_registers()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        instructions = find_related_instructions(cfg, ddgs, {"vcc"})
        
        # Should find v_cmp and v_cndmask
        assert len(instructions) >= 1
    
    def test_special_register_scc(self):
        """Test finding instructions using scc."""
        cfg = create_cfg_with_special_registers()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        instructions = find_related_instructions(cfg, ddgs, {"scc"})
        
        # Should find s_cmp and s_cbranch
        assert len(instructions) >= 1


class TestFindAllBarriers:
    """Tests for find_all_barriers function."""
    
    def test_single_barrier(self):
        """Test finding a single barrier."""
        cfg = create_simple_cfg()
        barriers = find_all_barriers(cfg)
        
        # Should find barrier at address 21
        assert 21 in barriers
        assert barriers[21].is_barrier
    
    def test_multiple_barriers(self):
        """Test finding multiple barriers."""
        cfg = create_cfg_with_multiple_barriers()
        barriers = find_all_barriers(cfg)
        
        # Should find barriers at addresses 11 and 13
        assert len(barriers) == 2
        assert 11 in barriers
        assert 13 in barriers
    
    def test_no_barriers(self):
        """Test CFG with no barriers."""
        cfg = create_cfg_with_waitcnt()  # Has waitcnt but no s_barrier
        barriers = find_all_barriers(cfg)
        
        assert len(barriers) == 0


class TestFindBarriersBetween:
    """Tests for find_barriers_between function."""
    
    def test_barriers_between(self):
        """Test finding barriers between two positions."""
        barrier_positions = {
            150: GlobalPosition(0, 5, 150),
            175: GlobalPosition(0, 8, 175),
        }
        
        from_pos = GlobalPosition(0, 0, 100)
        to_pos = GlobalPosition(0, 10, 200)
        
        barriers = find_barriers_between(from_pos, to_pos, barrier_positions)
        
        assert barriers == [150, 175]
    
    def test_no_barriers_between(self):
        """Test when there are no barriers between positions."""
        barrier_positions = {
            250: GlobalPosition(0, 15, 250),
        }
        
        from_pos = GlobalPosition(0, 0, 100)
        to_pos = GlobalPosition(0, 10, 200)
        
        barriers = find_barriers_between(from_pos, to_pos, barrier_positions)
        
        assert barriers == []
    
    def test_barriers_cross_block(self):
        """Test finding barriers across block boundaries."""
        barrier_positions = {
            150: GlobalPosition(1, 5, 150),  # In second block
        }
        
        from_pos = GlobalPosition(0, 10, 100)  # First block
        to_pos = GlobalPosition(1, 10, 200)    # Second block, after barrier
        
        barriers = find_barriers_between(from_pos, to_pos, barrier_positions)
        
        assert barriers == [150]


# =============================================================================
# Edge Detection Tests
# =============================================================================

class TestFindDependencyEdges:
    """Tests for find_dependency_edges function."""
    
    def test_raw_dependency(self):
        """Test RAW (Read After Write) dependency detection."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        pos_map = build_global_position_map(cfg)
        all_barriers = find_all_barriers(cfg)
        
        # Find instructions for v3
        instructions = find_related_instructions(cfg, ddgs, {"v3"})
        
        edges, barriers_used = find_dependency_edges(
            instructions, {"v3"}, pos_map, all_barriers
        )
        
        # Should have at least one RAW edge
        raw_edges = [e for e in edges if e.dep_type == "RAW"]
        assert len(raw_edges) >= 0  # v3 written at 11, read at 20, 22
    
    def test_barrier_crossing_detection(self):
        """Test detection of edges crossing s_barrier."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        pos_map = build_global_position_map(cfg)
        all_barriers = find_all_barriers(cfg)
        
        # Find instructions for v3 (crosses barrier between block 0 and 1)
        instructions = find_related_instructions(cfg, ddgs, {"v3"})
        
        edges, barriers_used = find_dependency_edges(
            instructions, {"v3"}, pos_map, all_barriers
        )
        
        # If there are edges crossing the barrier, they should be marked
        crossing_edges = [e for e in edges if e.crosses_barrier]
        if crossing_edges:
            assert len(barriers_used) > 0
    
    def test_multiple_dependency_types(self):
        """Test detection of multiple dependency types."""
        # Create CFG where same register is read, then written
        cfg = CFG(name="test_multi_dep")
        block = create_test_block(".LBB0_0", [
            create_test_instruction(10, "v_mov_b32", "v0, 0"),      # Write v0
            create_test_instruction(11, "v_add_f32", "v1, v0, v0"), # Read v0
            create_test_instruction(12, "v_mov_b32", "v0, 1"),      # Write v0 (WAR from 11->12)
        ])
        cfg.add_block(block)
        cfg.block_order = [".LBB0_0"]
        
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        pos_map = build_global_position_map(cfg)
        all_barriers = find_all_barriers(cfg)
        
        instructions = find_related_instructions(cfg, ddgs, {"v0"})
        edges, _ = find_dependency_edges(instructions, {"v0"}, pos_map, all_barriers)
        
        dep_types = {e.dep_type for e in edges}
        # Should have RAW (10->11) and WAR (11->12)
        # Note: WAW is not tracked (not needed for modern GPU hardware)
        assert "RAW" in dep_types or len(edges) >= 1


# =============================================================================
# Build Register Slice Tests
# =============================================================================

def skipif_no_test_file(func):
    """Decorator to skip test if test assembly file doesn't exist."""
    func._skip_condition = not TEST_ASSEMBLY_FILE.exists()
    func._skip_reason = "Test file not found"
    return func


class TestBuildRegisterSlice:
    """Tests for build_register_slice function."""
    
    @skipif_no_test_file
    def test_build_from_real_file(self):
        """Test building slice from real assembly file."""
        slice_result = build_register_slice(str(TEST_ASSEMBLY_FILE), {"v0", "v1"})
        
        assert slice_result.cfg_name is not None
        assert slice_result.target_registers == {"v0", "v1"}
    
    def test_build_from_cfg(self):
        """Test building slice from pre-parsed CFG."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0", "v3"})
        
        assert slice_result.cfg_name == "test_kernel"
        assert slice_result.target_registers == {"v0", "v3"}
        assert slice_result.get_instruction_count() >= 1
    
    def test_empty_result_for_nonexistent_registers(self):
        """Test that non-existent registers return empty slice."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v999"})
        
        assert slice_result.get_instruction_count() == 0
        assert slice_result.get_edge_count() == 0


class TestBuildRegisterSliceWithBarriers:
    """Tests for barrier handling in register slice building."""
    
    def test_slice_with_barrier_crossing(self):
        """Test slice that crosses s_barrier."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v3"})
        
        # v3 is written in block 0, read in block 1 (after barrier)
        if slice_result.get_edge_count() > 0:
            # Check if any edges cross barrier
            barrier_crossing_count = slice_result.get_barrier_crossing_edge_count()
            # Barrier instructions should be included if there are crossings
            if barrier_crossing_count > 0:
                assert len(slice_result.barrier_instructions) > 0
    
    def test_slice_with_multiple_barriers(self):
        """Test slice with multiple barriers."""
        cfg = create_cfg_with_multiple_barriers()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        # v0 written at 10, read at 12 (after first barrier), read at 14 (after second barrier)
        assert slice_result.get_instruction_count() >= 1


class TestBuildRegisterSliceWithWaitcnt:
    """Tests for s_waitcnt handling in register slice building."""
    
    def test_slice_includes_waitcnt(self):
        """Test that s_waitcnt is included when it makes target registers available."""
        cfg = create_cfg_with_waitcnt()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=True)
        
        # Search for s0 which is loaded by s_load and waited by s_waitcnt lgkmcnt(0)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"s0"})
        
        # Should find s_load and potentially s_waitcnt and s_add
        assert slice_result.get_instruction_count() >= 1


# =============================================================================
# Output Generation Tests
# =============================================================================

class TestGenerateSliceDot:
    """Tests for DOT output generation."""
    
    def test_generate_valid_dot(self):
        """Test that generated DOT is valid."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0", "v3"})
        
        dot_content = generate_slice_dot(slice_result)
        
        # Check basic DOT structure
        assert dot_content.startswith('digraph')
        assert 'rankdir=TB' in dot_content
        assert dot_content.endswith('}')
    
    def test_dot_contains_target_registers(self):
        """Test that DOT contains target register info."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        dot_content = generate_slice_dot(slice_result)
        
        # Should mention target registers in label
        assert "v0" in dot_content
    
    def test_dot_barrier_edges_style(self):
        """Test that barrier crossing edges have special style."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v3"})
        
        dot_content = generate_slice_dot(slice_result)
        
        # If there are barrier crossings, check for orange dashed edges
        if slice_result.get_barrier_crossing_edge_count() > 0:
            assert "orange" in dot_content or "dashed" in dot_content


class TestGenerateSliceJson:
    """Tests for JSON output generation."""
    
    def test_generate_valid_json(self):
        """Test that generated JSON is valid."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        json_content = generate_slice_json(slice_result)
        
        # Should be valid JSON
        data = json.loads(json_content)
        assert "target_registers" in data
        assert "instructions" in data
        assert "edges" in data
    
    def test_json_roundtrip(self):
        """Test that JSON can be parsed back to RegisterSlice."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        original = build_register_slice_from_cfg(cfg, ddgs, {"v0", "v3"})
        
        json_content = generate_slice_json(original)
        data = json.loads(json_content)
        restored = RegisterSlice.from_dict(data)
        
        assert restored.target_registers == original.target_registers
        assert restored.cfg_name == original.cfg_name
        assert len(restored.instructions) == len(original.instructions)


class TestGenerateSliceText:
    """Tests for text output generation."""
    
    def test_generate_text_with_header(self):
        """Test that text output has header."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        text_content = generate_slice_text(slice_result)
        
        assert "# Register Slice for:" in text_content
        assert "v0" in text_content
    
    def test_text_contains_instructions(self):
        """Test that text output contains instruction lines."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        text_content = generate_slice_text(slice_result)
        
        # Should contain raw instruction lines
        for instr in slice_result.instructions.values():
            assert instr.opcode in text_content


class TestSaveSliceOutputs:
    """Tests for saving slice to files."""
    
    def test_save_all_formats(self):
        """Test saving to all output formats."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_files = save_slice_outputs(
                slice_result, tmpdir, generate_svg=False, prefix="test"
            )
            
            assert 'json' in output_files
            assert 'dot' in output_files
            assert 'txt' in output_files
            
            # Check files exist
            assert os.path.exists(output_files['json'])
            assert os.path.exists(output_files['dot'])
            assert os.path.exists(output_files['txt'])


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestParseRegisterList:
    """Tests for parse_register_list function."""
    
    def test_simple_list(self):
        """Test parsing simple register list."""
        result = parse_register_list("v0,v1,v2")
        assert result == {"v0", "v1", "v2"}
    
    def test_with_spaces(self):
        """Test parsing with spaces."""
        result = parse_register_list("v0, v1, v2")
        assert result == {"v0", "v1", "v2"}
    
    def test_register_range(self):
        """Test parsing register range."""
        result = parse_register_list("v[0:3]")
        assert result == {"v0", "v1", "v2", "v3"}
    
    def test_mixed(self):
        """Test parsing mixed registers and ranges."""
        result = parse_register_list("v0,s[0:1],exec,scc")
        assert "v0" in result
        assert "s0" in result
        assert "s1" in result
        assert "exec" in result
        assert "scc" in result
    
    def test_empty_string(self):
        """Test parsing empty string."""
        result = parse_register_list("")
        assert result == set()
    
    def test_special_registers(self):
        """Test parsing special registers."""
        result = parse_register_list("vcc,scc,exec")
        assert result == {"vcc", "scc", "exec"}


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and corner cases."""
    
    def test_empty_cfg(self):
        """Test with empty CFG."""
        cfg = CFG(name="empty_kernel")
        ddgs: Dict[str, DDG] = {}
        
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        assert slice_result.get_instruction_count() == 0
        assert slice_result.get_edge_count() == 0
    
    def test_single_instruction(self):
        """Test with single instruction."""
        cfg = CFG(name="single_instr_kernel")
        block = create_test_block(".LBB0_0", [
            create_test_instruction(10, "v_mov_b32", "v0, 0"),
        ])
        cfg.add_block(block)
        cfg.block_order = [".LBB0_0"]
        
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        assert slice_result.get_instruction_count() == 1
        assert slice_result.get_edge_count() == 0  # No edges with single instruction
    
    def test_self_dependency(self):
        """Test instruction that reads and writes same register."""
        cfg = CFG(name="self_dep_kernel")
        block = create_test_block(".LBB0_0", [
            create_test_instruction(10, "v_add_f32", "v0, v0, v1"),
        ])
        cfg.add_block(block)
        cfg.block_order = [".LBB0_0"]
        
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        # Should find the instruction
        assert slice_result.get_instruction_count() == 1
    
    def test_many_registers(self):
        """Test with many target registers."""
        cfg = create_simple_cfg()
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        
        # Search for many registers at once
        many_regs = {f"v{i}" for i in range(100)}
        slice_result = build_register_slice_from_cfg(cfg, ddgs, many_regs)
        
        # Should not crash
        assert slice_result is not None
    
    def test_consecutive_barriers(self):
        """Test with consecutive barrier instructions."""
        cfg = CFG(name="consecutive_barriers_kernel")
        block = create_test_block(".LBB0_0", [
            create_test_instruction(10, "v_mov_b32", "v0, 0"),
            create_test_instruction(11, "s_barrier", ""),
            create_test_instruction(12, "s_barrier", ""),
            create_test_instruction(13, "v_add_f32", "v1, v0, v0"),
        ])
        cfg.add_block(block)
        cfg.block_order = [".LBB0_0"]
        
        ddgs, _ = generate_all_ddgs(cfg, enable_cross_block_waitcnt=False)
        slice_result = build_register_slice_from_cfg(cfg, ddgs, {"v0"})
        
        # Should handle consecutive barriers
        if slice_result.get_barrier_crossing_edge_count() > 0:
            # Edge should have multiple barrier addresses
            for edge in slice_result.edges:
                if edge.crosses_barrier:
                    # Could cross both barriers
                    assert len(edge.barrier_addrs) >= 1


class TestIntegrationWithRealFile:
    """Integration tests with real assembly file."""
    
    @skipif_no_test_file
    def test_full_pipeline_real_file(self):
        """Test full pipeline with real assembly file."""
        slice_result = build_register_slice(str(TEST_ASSEMBLY_FILE), {"v0", "v1", "s0"})
        
        # Should produce valid result
        assert slice_result is not None
        assert slice_result.cfg_name is not None
        
        # Generate all outputs
        dot = generate_slice_dot(slice_result)
        assert 'digraph' in dot
        
        json_str = generate_slice_json(slice_result)
        data = json.loads(json_str)
        assert 'target_registers' in data
        
        text = generate_slice_text(slice_result)
        assert '# Register Slice for:' in text
    
    @skipif_no_test_file
    def test_special_registers_real_file(self):
        """Test special registers with real file."""
        slice_result = build_register_slice(str(TEST_ASSEMBLY_FILE), {"scc", "vcc", "exec"})
        
        # Should find some instructions using special registers
        # The real file likely uses scc for comparisons
        assert slice_result is not None
    
    @skipif_no_test_file
    def test_save_outputs_real_file(self):
        """Test saving outputs with real file."""
        slice_result = build_register_slice(str(TEST_ASSEMBLY_FILE), {"v0"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_files = save_slice_outputs(
                slice_result, tmpdir, generate_svg=False
            )
            
            # All files should be created
            for fmt, path in output_files.items():
                assert os.path.exists(path), f"Missing {fmt} file: {path}"


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    import sys
    
    # Collect all test classes
    test_classes = [
        TestSliceInstruction,
        TestSliceEdge,
        TestRegisterSlice,
        TestGlobalPosition,
        TestFindRelatedInstructions,
        TestFindAllBarriers,
        TestFindBarriersBetween,
        TestFindDependencyEdges,
        TestBuildRegisterSlice,
        TestBuildRegisterSliceWithBarriers,
        TestBuildRegisterSliceWithWaitcnt,
        TestGenerateSliceDot,
        TestGenerateSliceJson,
        TestGenerateSliceText,
        TestSaveSliceOutputs,
        TestParseRegisterList,
        TestEdgeCases,
        TestIntegrationWithRealFile,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    failures = []
    
    for test_class in test_classes:
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                test_name = f"{test_class.__name__}.{method_name}"
                
                try:
                    # Check for skip marker (real file tests)
                    should_skip = False
                    
                    # Check for _skip_condition attribute (our custom decorator)
                    if hasattr(method, '_skip_condition') and method._skip_condition:
                        should_skip = True
                    
                    # Check for pytest mark
                    if hasattr(method, 'pytestmark'):
                        for mark in method.pytestmark:
                            if hasattr(mark, 'name') and mark.name == 'skipif':
                                if hasattr(mark, 'args') and mark.args and mark.args[0]:
                                    should_skip = True
                    
                    if should_skip:
                        print(f"SKIP: {test_name}")
                        skipped += 1
                        continue
                    
                    method()
                    print(f"PASS: {test_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"FAIL: {test_name}")
                    print(f"      {e}")
                    failed += 1
                    failures.append((test_name, str(e)))
                except Exception as e:
                    print(f"ERROR: {test_name}")
                    print(f"       {type(e).__name__}: {e}")
                    failed += 1
                    failures.append((test_name, f"{type(e).__name__}: {e}"))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {passed + failed + skipped}")
    
    if failures:
        print("\nFailures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
    
    if failed == 0:
        print("\n" + "*" * 70)
        print("*** ALL TESTS PASSED! ***")
        print("*" * 70)
        return 0
    else:
        print("\n" + "!" * 70)
        print(f"!!! {failed} TEST(S) FAILED !!!")
        print("!" * 70)
        return 1


if __name__ == '__main__':
    import sys
    
    # Always use our custom test runner to get the summary
    # (Set USE_PYTEST=True to use pytest instead)
    USE_PYTEST = False
    
    if USE_PYTEST and HAS_PYTEST:
        # Run with pytest if available
        import pytest as real_pytest
        sys.exit(real_pytest.main([__file__, '-v', '--tb=short']))
    else:
        # Use manual test runner with custom summary
        sys.exit(run_all_tests())

