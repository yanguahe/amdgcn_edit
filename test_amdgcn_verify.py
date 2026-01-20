#!/usr/bin/env python3
"""
Unit tests for AMDGCN Verification module.

Tests cover:
- Barrier instruction detection
- Global DDG construction
- Position mapping and comparison
- RAW dependency verification
- Barrier constraint verification
- AVAIL dependency verification
- End-to-end optimization verification
"""

import pytest
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

from amdgcn_cfg import AMDGCNParser, Instruction, BasicBlock, CFG
from amdgcn_ddg import DDG, InstructionNode, build_ddg, generate_all_ddgs
from amdgcn_verify import (
    # Exceptions
    SchedulingVerificationError,
    
    # Data structures
    InstructionInfo,
    BarrierRegion,
    AvailDependency,
    GlobalDependencyGraph,
    VerificationResult,
    
    # Barrier detection
    is_barrier_instruction,
    CONDITIONAL_BRANCHES,
    UNCONDITIONAL_BRANCHES,
    TERMINATOR_INSTRUCTIONS,
    
    # DDG construction
    build_global_ddg,
    
    # Position comparison
    build_position_map,
    get_block_order_index,
    is_before,
    
    # Verification
    verify_optimization,
    verify_and_report,
    compare_before_after,
)


# =============================================================================
# Test Data - Path to real assembly file
# =============================================================================

TEST_ASSEMBLY_FILE = Path(__file__).parent / "pa_dot_kernel.v2.amdgcn"


# =============================================================================
# Helper Functions
# =============================================================================

def create_instruction(opcode: str, operands: str, address: int) -> Instruction:
    """Helper to create an instruction."""
    return Instruction(
        address=address,
        opcode=opcode,
        operands=operands,
        raw_line=f"\t{opcode} {operands}"
    )


def create_block_with_instructions(label: str, instructions: List[Instruction]) -> BasicBlock:
    """Helper to create a block with instructions."""
    block = BasicBlock(label=label)
    for i, instr in enumerate(instructions):
        block.instructions.append(instr)
        block.raw_lines[instr.address] = instr.raw_line + "\n"
    return block


def create_simple_cfg(blocks: List[BasicBlock]) -> CFG:
    """Create a simple CFG from blocks."""
    cfg = CFG(name="test")
    for block in blocks:
        cfg.add_block(block)
        # Note: add_block already appends to block_order if not present
    return cfg


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestInstructionInfo:
    """Tests for InstructionInfo dataclass."""

    def test_creation(self):
        """Test creating InstructionInfo."""
        info = InstructionInfo(
            address=100,
            block_label=".LBB0_0",
            position=5,
            opcode="v_add_f32",
            operands="v0, v1, v2"
        )
        assert info.address == 100
        assert info.block_label == ".LBB0_0"
        assert info.position == 5
        assert info.opcode == "v_add_f32"

    def test_hash(self):
        """Test that InstructionInfo is hashable by address."""
        info1 = InstructionInfo(address=100, block_label=".LBB0_0", position=0, opcode="v_add_f32", operands="v0, v1, v2")
        info2 = InstructionInfo(address=100, block_label=".LBB0_1", position=5, opcode="v_mul_f32", operands="v3, v4, v5")
        assert hash(info1) == hash(info2)

    def test_equality(self):
        """Test equality based on address."""
        info1 = InstructionInfo(address=100, block_label=".LBB0_0", position=0, opcode="v_add_f32", operands="v0, v1, v2")
        info2 = InstructionInfo(address=100, block_label=".LBB0_1", position=5, opcode="v_mul_f32", operands="v3, v4, v5")
        info3 = InstructionInfo(address=200, block_label=".LBB0_0", position=0, opcode="v_add_f32", operands="v0, v1, v2")
        assert info1 == info2  # Same address
        assert info1 != info3  # Different address

    def test_default_values(self):
        """Test default values for optional fields."""
        info = InstructionInfo(address=1, block_label=".LBB0_0", position=0, opcode="nop", operands="")
        assert info.defs == set()
        assert info.uses == set()
        assert info.is_barrier is False
        assert info.raw_line == ""


class TestBarrierRegion:
    """Tests for BarrierRegion dataclass."""

    def test_creation(self):
        """Test creating BarrierRegion."""
        region = BarrierRegion(
            barrier_addr=50,
            barrier_opcode="s_barrier"
        )
        assert region.barrier_addr == 50
        assert region.barrier_opcode == "s_barrier"
        assert region.before_addrs == set()
        assert region.after_addrs == set()

    def test_with_addresses(self):
        """Test BarrierRegion with instruction addresses."""
        region = BarrierRegion(
            barrier_addr=50,
            barrier_opcode="s_barrier",
            before_addrs={10, 20, 30},
            after_addrs={60, 70}
        )
        assert len(region.before_addrs) == 3
        assert len(region.after_addrs) == 2
        assert 10 in region.before_addrs
        assert 60 in region.after_addrs


class TestAvailDependency:
    """Tests for AvailDependency dataclass."""

    def test_creation(self):
        """Test creating AvailDependency."""
        dep = AvailDependency(
            waitcnt_addr=100,
            user_addr=150,
            needed_regs={"v0", "v1"}
        )
        assert dep.waitcnt_addr == 100
        assert dep.user_addr == 150
        assert "v0" in dep.needed_regs


class TestGlobalDependencyGraph:
    """Tests for GlobalDependencyGraph dataclass."""

    def test_creation(self):
        """Test creating GlobalDependencyGraph."""
        gdg = GlobalDependencyGraph()
        assert len(gdg.instructions) == 0
        assert len(gdg.raw_edges) == 0
        assert len(gdg.barrier_regions) == 0
        assert len(gdg.avail_deps) == 0

    def test_counts(self):
        """Test count methods."""
        gdg = GlobalDependencyGraph()
        
        # Add some data
        gdg.instructions[1] = InstructionInfo(1, ".LBB0_0", 0, "nop", "")
        gdg.instructions[2] = InstructionInfo(2, ".LBB0_0", 1, "nop", "")
        gdg.raw_edges.add((1, 2, "v0"))
        gdg.barrier_regions.append(BarrierRegion(barrier_addr=10, barrier_opcode="s_barrier"))
        gdg.avail_deps.append(AvailDependency(waitcnt_addr=5, user_addr=10))
        
        assert gdg.get_instruction_count() == 2
        assert gdg.get_raw_edge_count() == 1
        assert gdg.get_barrier_count() == 1
        assert gdg.get_avail_dep_count() == 1


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_success_default(self):
        """Test default success status."""
        result = VerificationResult()
        assert result.success is True
        assert len(result.raw_violations) == 0
        assert len(result.barrier_violations) == 0

    def test_get_all_errors(self):
        """Test getting all errors."""
        result = VerificationResult(
            success=False,
            raw_violations=["RAW error 1", "RAW error 2"],
            barrier_violations=["Barrier error 1"],
            avail_violations=["AVAIL error 1"]
        )
        errors = result.get_all_errors()
        assert len(errors) == 4
        assert "RAW error 1" in errors
        assert "Barrier error 1" in errors


# =============================================================================
# Barrier Detection Tests
# =============================================================================

class TestBarrierDetection:
    """Tests for barrier instruction detection."""

    def test_s_barrier(self):
        """Test s_barrier detection."""
        assert is_barrier_instruction("s_barrier") is True
        assert is_barrier_instruction("S_BARRIER") is True

    def test_conditional_branches(self):
        """Test conditional branch detection."""
        for branch in CONDITIONAL_BRANCHES:
            assert is_barrier_instruction(branch) is True
        
        # Test specific ones
        assert is_barrier_instruction("s_cbranch_scc0") is True
        assert is_barrier_instruction("s_cbranch_vccz") is True
        assert is_barrier_instruction("s_cbranch_execz") is True

    def test_unconditional_branches(self):
        """Test unconditional branch detection."""
        for branch in UNCONDITIONAL_BRANCHES:
            assert is_barrier_instruction(branch) is True
        
        assert is_barrier_instruction("s_branch") is True
        assert is_barrier_instruction("s_setpc_b64") is True

    def test_terminators(self):
        """Test terminator detection."""
        for term in TERMINATOR_INSTRUCTIONS:
            assert is_barrier_instruction(term) is True
        
        assert is_barrier_instruction("s_endpgm") is True

    def test_non_barriers(self):
        """Test that non-barrier instructions return False."""
        assert is_barrier_instruction("v_add_f32") is False
        assert is_barrier_instruction("s_mov_b32") is False
        assert is_barrier_instruction("global_load_dwordx4") is False
        assert is_barrier_instruction("v_mfma_f32_16x16x16_bf16") is False

    def test_with_flags(self):
        """Test using flags for branch/terminator detection."""
        assert is_barrier_instruction("custom_branch", is_branch=True) is True
        assert is_barrier_instruction("custom_term", is_terminator=True) is True
        assert is_barrier_instruction("custom_op", is_branch=False, is_terminator=False) is False


# =============================================================================
# Position Comparison Tests
# =============================================================================

class TestPositionComparison:
    """Tests for position mapping and comparison."""

    def test_build_position_map(self):
        """Test building position map from CFG."""
        block1 = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 10),
            create_instruction("v_add_f32", "v1, v0, v2", 11),
        ])
        block2 = create_block_with_instructions(".LBB0_1", [
            create_instruction("v_mul_f32", "v3, v1, v4", 20),
        ])
        cfg = create_simple_cfg([block1, block2])
        
        pos_map = build_position_map(cfg)
        
        assert pos_map[10] == (".LBB0_0", 0)
        assert pos_map[11] == (".LBB0_0", 1)
        assert pos_map[20] == (".LBB0_1", 0)

    def test_get_block_order_index(self):
        """Test getting block order index."""
        block_order = [".LBB0_0", ".LBB0_1", ".LBB0_2"]
        
        assert get_block_order_index(".LBB0_0", block_order) == 0
        assert get_block_order_index(".LBB0_1", block_order) == 1
        assert get_block_order_index(".LBB0_2", block_order) == 2
        assert get_block_order_index(".LBB0_999", block_order) == -1

    def test_is_before_same_block(self):
        """Test is_before in same block."""
        block_order = [".LBB0_0"]
        
        pos_a = (".LBB0_0", 0)
        pos_b = (".LBB0_0", 5)
        
        assert is_before(pos_a, pos_b, block_order) is True
        assert is_before(pos_b, pos_a, block_order) is False
        assert is_before(pos_a, pos_a, block_order) is False  # Same position

    def test_is_before_cross_block(self):
        """Test is_before across blocks."""
        block_order = [".LBB0_0", ".LBB0_1", ".LBB0_2"]
        
        pos_a = (".LBB0_0", 5)
        pos_b = (".LBB0_1", 0)
        pos_c = (".LBB0_2", 10)
        
        assert is_before(pos_a, pos_b, block_order) is True
        assert is_before(pos_b, pos_c, block_order) is True
        assert is_before(pos_a, pos_c, block_order) is True
        assert is_before(pos_c, pos_a, block_order) is False


# =============================================================================
# Global DDG Construction Tests
# =============================================================================

class TestBuildGlobalDDG:
    """Tests for building global dependency graph."""

    def test_simple_cfg(self):
        """Test building DDG from simple CFG."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("v_add_f32", "v1, v0, v2", 2),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        
        gdg = build_global_ddg(cfg, ddgs)
        
        assert len(gdg.instructions) == 2
        assert 1 in gdg.instructions
        assert 2 in gdg.instructions
        assert gdg.instructions[1].opcode == "v_mov_b32"

    def test_with_barrier(self):
        """Test that barriers are detected."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("s_barrier", "", 2),
            create_instruction("v_add_f32", "v1, v0, v2", 3),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        
        gdg = build_global_ddg(cfg, ddgs)
        
        assert len(gdg.barrier_regions) == 1
        barrier = gdg.barrier_regions[0]
        assert barrier.barrier_addr == 2
        assert barrier.barrier_opcode == "s_barrier"
        assert 1 in barrier.before_addrs
        assert 3 in barrier.after_addrs

    def test_block_order(self):
        """Test that block order is preserved."""
        block1 = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
        ])
        block2 = create_block_with_instructions(".LBB0_1", [
            create_instruction("v_mov_b32", "v1, 2.0", 10),
        ])
        cfg = create_simple_cfg([block1, block2])
        ddgs, _ = generate_all_ddgs(cfg)
        
        gdg = build_global_ddg(cfg, ddgs)
        
        assert gdg.block_order == [".LBB0_0", ".LBB0_1"]


# =============================================================================
# Verification Tests
# =============================================================================

class TestVerifyOptimization:
    """Tests for optimization verification."""

    def test_unchanged_code_passes(self):
        """Test that unchanged code passes verification."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("v_add_f32", "v1, v0, v2", 2),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Verify same CFG - should pass
        verify_optimization(gdg, cfg)  # No exception = pass

    def test_raw_violation_detected(self):
        """Test that RAW violation is detected."""
        # Original: writer at line 1, reader at line 2
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("v_add_f32", "v1, v0, v2", 2),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Optimized: swap order (reader before writer - violates RAW)
        swapped_block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_add_f32", "v1, v0, v2", 2),  # Reader first
            create_instruction("v_mov_b32", "v0, 1.0", 1),  # Writer second
        ])
        optimized_cfg = create_simple_cfg([swapped_block])
        
        with pytest.raises(SchedulingVerificationError) as exc_info:
            verify_optimization(gdg, optimized_cfg)
        
        assert len(exc_info.value.errors) > 0
        assert "RAW" in exc_info.value.errors[0]

    def test_barrier_violation_detected(self):
        """Test that barrier crossing is detected."""
        # Original: instr at 1, barrier at 2, instr at 3
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("s_barrier", "", 2),
            create_instruction("v_mov_b32", "v1, 2.0", 3),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Optimized: move instruction from before barrier to after
        # (instr 3, barrier 2, instr 1) - instr 1 crossed after barrier
        reordered_block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v1, 2.0", 3),
            create_instruction("s_barrier", "", 2),
            create_instruction("v_mov_b32", "v0, 1.0", 1),  # Was before barrier
        ])
        optimized_cfg = create_simple_cfg([reordered_block])
        
        with pytest.raises(SchedulingVerificationError) as exc_info:
            verify_optimization(gdg, optimized_cfg)
        
        errors = exc_info.value.errors
        # Should have barrier violation
        assert any("Barrier" in e or "barrier" in e for e in errors)

    def test_valid_reordering_passes(self):
        """Test that valid reordering (independent instructions) passes."""
        # Two independent instructions
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("v_mov_b32", "v1, 2.0", 2),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Swap them - should be valid since no dependency
        swapped_block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v1, 2.0", 2),
            create_instruction("v_mov_b32", "v0, 1.0", 1),
        ])
        optimized_cfg = create_simple_cfg([swapped_block])
        
        # Should not raise
        verify_optimization(gdg, optimized_cfg)


class TestVerifyAndReport:
    """Tests for verify_and_report function."""

    def test_returns_result(self):
        """Test that verify_and_report returns VerificationResult."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        result = verify_and_report(gdg, cfg, verbose=False)
        
        assert isinstance(result, VerificationResult)
        assert result.success is True

    def test_collects_errors(self):
        """Test that errors are collected in result."""
        # Create scenario with RAW violation
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("v_add_f32", "v1, v0, v2", 2),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Swap to create violation
        swapped_block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_add_f32", "v1, v0, v2", 2),
            create_instruction("v_mov_b32", "v0, 1.0", 1),
        ])
        optimized_cfg = create_simple_cfg([swapped_block])
        
        result = verify_and_report(gdg, optimized_cfg, verbose=False)
        
        assert result.success is False
        assert len(result.raw_violations) > 0


# =============================================================================
# Scheduling Verification Error Tests
# =============================================================================

class TestSchedulingVerificationError:
    """Tests for SchedulingVerificationError exception."""

    def test_error_creation(self):
        """Test creating exception with errors."""
        errors = ["Error 1", "Error 2"]
        exc = SchedulingVerificationError(errors)
        
        assert exc.errors == errors
        assert "Error 1" in str(exc)
        assert "Error 2" in str(exc)

    def test_error_message_format(self):
        """Test error message format."""
        errors = ["RAW violation: v0"]
        exc = SchedulingVerificationError(errors)
        
        assert "Scheduling verification failed" in str(exc)


# =============================================================================
# Integration Tests with Real Assembly
# =============================================================================

class TestVerifyIntegration:
    """Integration tests using real assembly file."""

    @pytest.fixture
    def cfg_and_gdg(self):
        """Parse the test assembly file and build DDG."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test assembly file not found: {TEST_ASSEMBLY_FILE}")
        
        parser = AMDGCNParser()
        cfg = parser.parse_file(str(TEST_ASSEMBLY_FILE))
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        return cfg, gdg

    def test_real_file_self_verification(self, cfg_and_gdg):
        """Test that real file verifies against itself."""
        cfg, gdg = cfg_and_gdg
        
        # Should pass - no changes
        verify_optimization(gdg, cfg)

    def test_gdg_has_instructions(self, cfg_and_gdg):
        """Test that GDG has instructions from real file."""
        cfg, gdg = cfg_and_gdg
        
        assert gdg.get_instruction_count() > 0

    def test_gdg_has_raw_edges(self, cfg_and_gdg):
        """Test that GDG has RAW edges from real file."""
        cfg, gdg = cfg_and_gdg
        
        # Real code should have many dependencies
        assert gdg.get_raw_edge_count() > 0

    def test_gdg_may_have_barriers(self, cfg_and_gdg):
        """Test that GDG detects barriers from real file."""
        cfg, gdg = cfg_and_gdg
        
        # File may or may not have barriers - just verify it works
        assert isinstance(gdg.get_barrier_count(), int)


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestVerifyEdgeCases:
    """Tests for edge cases in verification."""

    def test_empty_cfg(self):
        """Test verification with empty CFG."""
        cfg = CFG(name="empty")
        gdg = GlobalDependencyGraph()
        
        # Should not raise
        verify_optimization(gdg, cfg)

    def test_single_instruction(self):
        """Test verification with single instruction."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_endpgm", "", 1),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        verify_optimization(gdg, cfg)

    def test_multiple_blocks_same_register(self):
        """Test cross-block register usage."""
        block1 = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
        ])
        block2 = create_block_with_instructions(".LBB0_1", [
            create_instruction("v_add_f32", "v1, v0, v2", 10),
        ])
        cfg = create_simple_cfg([block1, block2])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Should work correctly
        assert len(gdg.instructions) == 2
        verify_optimization(gdg, cfg)

    def test_branch_at_block_end(self):
        """Test blocks ending with branches."""
        block1 = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("s_cbranch_scc0", ".LBB0_1", 2),
        ])
        block2 = create_block_with_instructions(".LBB0_1", [
            create_instruction("v_mov_b32", "v1, 2.0", 10),
        ])
        cfg = create_simple_cfg([block1, block2])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Branch should be detected as barrier
        assert len(gdg.barrier_regions) >= 1
        barrier_ops = [r.barrier_opcode for r in gdg.barrier_regions]
        assert "s_cbranch_scc0" in barrier_ops

    def test_multiple_barriers(self):
        """Test multiple barriers in sequence."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("s_barrier", "", 2),
            create_instruction("v_mov_b32", "v1, 2.0", 3),
            create_instruction("s_barrier", "", 4),
            create_instruction("v_mov_b32", "v2, 3.0", 5),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        assert len(gdg.barrier_regions) == 2


# =============================================================================
# Corner Cases Tests
# =============================================================================

class TestVerifyCornerCases:
    """Tests for corner cases in verification."""

    def test_register_ranges(self):
        """Test handling of register ranges."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("global_load_dwordx4", "v[0:3], v[4:5], off", 1),
            create_instruction("s_waitcnt", "vmcnt(0)", 2),
            create_instruction("v_add_f32", "v10, v0, v1", 3),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Should handle register ranges correctly
        assert len(gdg.instructions) == 3

    def test_mfma_instructions(self):
        """Test handling of MFMA instructions."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0", 1),
            create_instruction("v_mfma_f32_16x16x16_bf16", "a[0:3], v[0:1], v[2:3], 0", 2),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        assert len(gdg.instructions) == 2
        verify_optimization(gdg, cfg)

    def test_waitcnt_variations(self):
        """Test different s_waitcnt formats."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("global_load_dwordx4", "v[0:3], v[4:5], off", 1),
            create_instruction("s_waitcnt", "vmcnt(0) lgkmcnt(0)", 2),
            create_instruction("v_add_f32", "v10, v0, v1", 3),
        ])
        cfg = create_simple_cfg([block])
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        assert len(gdg.instructions) == 3

    def test_all_branch_types(self):
        """Test all conditional branch types are detected."""
        blocks = []
        addr = 1
        
        for i, branch in enumerate(list(CONDITIONAL_BRANCHES)[:3]):  # Test first 3
            block = create_block_with_instructions(f".LBB0_{i}", [
                create_instruction("v_mov_b32", "v0, 1.0", addr),
                create_instruction(branch, f".LBB0_{i+10}", addr + 1),
            ])
            blocks.append(block)
            addr += 10
        
        cfg = create_simple_cfg(blocks)
        ddgs, _ = generate_all_ddgs(cfg)
        gdg = build_global_ddg(cfg, ddgs)
        
        # Should detect barriers
        assert gdg.get_barrier_count() >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

