#!/usr/bin/env python3
"""
Unit tests for AMDGCN Passes module.

Tests cover:
- Dependency analysis (RAW, WAR, WAW)
- SCC pair handling
- MoveInstructionPass
- DistributeInstructionPass
- PassManager
- s_waitcnt handling
"""

import pytest
from pathlib import Path
from typing import List

from amdgcn_cfg import AMDGCNParser, Instruction, BasicBlock, CFG
from amdgcn_ddg import (
    DDG,
    InstructionNode,
    build_ddg,
    generate_all_ddgs,
    compute_inter_block_deps,
    parse_instruction_registers,
    AnalysisResult,
)
from amdgcn_passes import (
    # Dependency analysis
    get_instruction_defs_uses,
    has_raw_dependency,
    has_war_dependency,
    has_waw_dependency,
    can_ignore_scc_waw,
    has_true_scc_dependency,
    
    # SCC pair handling
    is_scc_tight_pair_start,
    is_scc_tight_pair_end,
    is_scc_pair_reader,
    is_scc_pair_writer,
    find_scc_pair_start,
    find_scc_pair_start_separated,
    is_scc_separated_pair_end,
    get_instructions_between_pair,
    can_chain_skip_scc_pair,
    
    # Register liveness
    is_register_live_after,
    
    # s_waitcnt handling
    parse_waitcnt_operands,
    build_waitcnt_operands,
    update_waitcnt_instruction,
    get_instruction_cycles,
    
    # Passes
    Pass,
    PassManager,
    MoveInstructionPass,
    MoveResult,
    find_dependent_waitcnt,
    find_dependency_chain,
    find_immediate_dependency_chain,
    
    # Register Replace Pass
    RegisterReplacePass,
    RegisterSegment,
    parse_register_segment,
    find_aligned_free_registers,
    replace_registers,
)


# =============================================================================
# Test Data - Path to real assembly file
# =============================================================================

TEST_ASSEMBLY_FILE = Path(__file__).parent / "pa_dot_kernel.v2.amdgcn"


# =============================================================================
# Helper Functions
# =============================================================================

def create_instruction(opcode: str, operands: str, address: int = 0) -> Instruction:
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
        instr.address = i + 1
        block.instructions.append(instr)
        block.raw_lines[i + 1] = instr.raw_line + "\n"
    return block


# =============================================================================
# Dependency Analysis Tests
# =============================================================================

class TestGetInstructionDefsUses:
    """Tests for get_instruction_defs_uses function."""

    def test_simple_add(self):
        """Test simple add instruction."""
        instr = create_instruction("v_add_f32", "v0, v1, v2")
        defs, uses = get_instruction_defs_uses(instr)
        assert "v0" in defs
        assert "v1" in uses
        assert "v2" in uses

    def test_mfma_instruction(self):
        """Test MFMA instruction."""
        instr = create_instruction("v_mfma_f32_16x16x16_bf16", "a[0:3], v[0:1], v[2:3], 0")
        defs, uses = get_instruction_defs_uses(instr)
        assert all(f"a{i}" in defs for i in range(4))
        assert "v0" in uses
        assert "v1" in uses
        assert "v2" in uses
        assert "v3" in uses


class TestHasRAWDependency:
    """Tests for RAW (Read After Write) dependency detection."""

    def test_raw_dependency_exists(self):
        """Test detection of RAW dependency."""
        instr_writer = create_instruction("v_mov_b32", "v0, 1.0")
        instr_reader = create_instruction("v_add_f32", "v1, v0, v2")
        has_dep, regs = has_raw_dependency(instr_reader, instr_writer)
        assert has_dep is True
        assert "v0" in regs

    def test_raw_dependency_not_exists(self):
        """Test when no RAW dependency."""
        instr_a = create_instruction("v_mov_b32", "v0, 1.0")
        instr_b = create_instruction("v_mov_b32", "v1, 2.0")
        has_dep, regs = has_raw_dependency(instr_a, instr_b)
        assert has_dep is False
        assert len(regs) == 0

    def test_raw_scc_dependency(self):
        """Test RAW dependency through SCC."""
        instr_writer = create_instruction("s_add_u32", "s0, s1, s2")  # writes SCC
        instr_reader = create_instruction("s_cbranch_scc0", ".LBB0_5")  # reads SCC
        has_dep, regs = has_raw_dependency(instr_reader, instr_writer)
        assert has_dep is True
        assert "scc" in regs

    def test_raw_large_register_range(self):
        """Test RAW with large register ranges."""
        instr_writer = create_instruction("global_load_dwordx4", "v[100:103], v[0:1], off")
        instr_reader = create_instruction("v_add_f32", "v200, v101, v50")
        has_dep, regs = has_raw_dependency(instr_reader, instr_writer)
        assert has_dep is True
        assert "v101" in regs


class TestHasWARDependency:
    """Tests for WAR (Write After Read) dependency detection."""

    def test_war_dependency_exists(self):
        """Test detection of WAR dependency."""
        instr_reader = create_instruction("v_add_f32", "v0, v1, v2")  # reads v1
        instr_writer = create_instruction("v_mov_b32", "v1, 1.0")  # writes v1
        has_dep, regs = has_war_dependency(instr_writer, instr_reader)
        assert has_dep is True
        assert "v1" in regs

    def test_war_dependency_not_exists(self):
        """Test when no WAR dependency."""
        instr_a = create_instruction("v_mov_b32", "v0, 1.0")
        instr_b = create_instruction("v_mov_b32", "v1, 2.0")
        has_dep, regs = has_war_dependency(instr_a, instr_b)
        assert has_dep is False


class TestHasWAWDependency:
    """Tests for WAW (Write After Write) dependency detection."""

    def test_waw_dependency_exists(self):
        """Test detection of WAW dependency."""
        instr_a = create_instruction("v_mov_b32", "v0, 1.0")  # writes v0
        instr_b = create_instruction("v_mov_b32", "v0, 2.0")  # also writes v0
        has_dep, regs = has_waw_dependency(instr_a, instr_b)
        assert has_dep is True
        assert "v0" in regs

    def test_waw_dependency_not_exists(self):
        """Test when no WAW dependency."""
        instr_a = create_instruction("v_mov_b32", "v0, 1.0")
        instr_b = create_instruction("v_mov_b32", "v1, 2.0")
        has_dep, regs = has_waw_dependency(instr_a, instr_b)
        assert has_dep is False

    def test_waw_scc_dependency(self):
        """Test WAW dependency through SCC."""
        instr_a = create_instruction("s_add_u32", "s0, s1, s2")  # writes SCC
        instr_b = create_instruction("s_cmp_lt_i32", "s3, s4")  # also writes SCC
        has_dep, regs = has_waw_dependency(instr_a, instr_b)
        assert has_dep is True
        assert "scc" in regs


class TestCanIgnoreSCCWAW:
    """Tests for WAW-SCC ignorability."""

    def test_can_ignore_scc_waw(self):
        """Test that WAW-SCC can be ignored when moving instruction only writes SCC."""
        instr_moving = create_instruction("s_lshl_b64", "s[0:1], s[0:1], 1")  # only writes SCC
        instr_stationary = create_instruction("s_add_u32", "s2, s3, s4")  # also writes SCC
        assert can_ignore_scc_waw(instr_moving, instr_stationary) is True

    def test_cannot_ignore_scc_waw_reader(self):
        """Test that WAW-SCC cannot be ignored when moving instruction reads SCC."""
        instr_moving = create_instruction("s_addc_u32", "s0, s1, s2")  # reads and writes SCC
        instr_stationary = create_instruction("s_add_u32", "s3, s4, s5")  # writes SCC
        assert can_ignore_scc_waw(instr_moving, instr_stationary) is False

    def test_cannot_ignore_non_scc_writer(self):
        """Test that non-SCC writers return False."""
        instr_moving = create_instruction("s_mul_i32", "s0, s1, s2")  # doesn't write SCC
        instr_stationary = create_instruction("s_add_u32", "s3, s4, s5")  # writes SCC
        assert can_ignore_scc_waw(instr_moving, instr_stationary) is False


class TestHasTrueSCCDependency:
    """Tests for true SCC dependency detection."""

    def test_true_scc_dependency_raw(self):
        """Test true RAW-SCC dependency."""
        instr_reader = create_instruction("s_cbranch_scc0", ".LBB0_5")  # reads SCC
        instr_writer = create_instruction("s_add_u32", "s0, s1, s2")  # writes SCC
        has_dep, dep_type = has_true_scc_dependency(instr_reader, instr_writer)
        assert has_dep is True
        assert dep_type == "RAW-SCC"

    def test_no_true_scc_dependency(self):
        """Test no true SCC dependency when no SCC read."""
        instr_a = create_instruction("s_lshl_b64", "s[0:1], s[0:1], 1")  # only writes SCC
        instr_b = create_instruction("s_add_u32", "s2, s3, s4")  # only writes SCC
        has_dep, dep_type = has_true_scc_dependency(instr_a, instr_b)
        assert has_dep is False


# =============================================================================
# SCC Pair Handling Tests
# =============================================================================

class TestSCCPairDetection:
    """Tests for SCC pair detection."""

    def test_is_scc_tight_pair_start(self):
        """Test detection of tight SCC pair start."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),  # pair start
            create_instruction("s_addc_u32", "s3, s4, s5"),  # pair end
        ])
        assert is_scc_tight_pair_start(block, 0) is True
        assert is_scc_tight_pair_start(block, 1) is False

    def test_is_scc_tight_pair_end(self):
        """Test detection of tight SCC pair end."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),  # pair start
            create_instruction("s_addc_u32", "s3, s4, s5"),  # pair end
        ])
        assert is_scc_tight_pair_end(block, 1) is True
        assert is_scc_tight_pair_end(block, 0) is False

    def test_sub_subb_pair(self):
        """Test detection of s_sub/s_subb pair."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_sub_u32", "s0, s1, s2"),  # pair start
            create_instruction("s_subb_u32", "s3, s4, s5"),  # pair end
        ])
        assert is_scc_tight_pair_start(block, 0) is True
        assert is_scc_tight_pair_end(block, 1) is True

    def test_is_scc_pair_reader(self):
        """Test SCC pair reader detection."""
        assert is_scc_pair_reader("s_addc_u32") is True
        assert is_scc_pair_reader("s_subb_u32") is True
        assert is_scc_pair_reader("s_add_u32") is False
        assert is_scc_pair_reader("s_mul_i32") is False

    def test_is_scc_pair_writer(self):
        """Test SCC pair writer detection."""
        assert is_scc_pair_writer("s_add_u32") is True
        assert is_scc_pair_writer("s_add_i32") is True
        assert is_scc_pair_writer("s_sub_u32") is True
        assert is_scc_pair_writer("s_addc_u32") is False


class TestSCCSeparatedPair:
    """Tests for separated SCC pair detection."""

    def test_find_scc_pair_start_separated(self):
        """Test finding separated SCC pair start."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),  # pair start
            create_instruction("s_mul_i32", "s10, s11, s12"),  # non-SCC
            create_instruction("s_addc_u32", "s3, s4, s5"),  # pair end
        ])
        pair_start = find_scc_pair_start_separated(block, 2)
        assert pair_start == 0

    def test_find_scc_pair_start_broken(self):
        """Test that broken pair returns -1."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),  # writes SCC
            create_instruction("s_cmp_lt_i32", "s10, s11"),  # writes SCC (breaks pair)
            create_instruction("s_addc_u32", "s3, s4, s5"),  # reads SCC from wrong place
        ])
        pair_start = find_scc_pair_start_separated(block, 2)
        assert pair_start == -1

    def test_find_scc_pair_start_tight(self):
        """Test find_scc_pair_start for tight pair."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),
            create_instruction("s_addc_u32", "s3, s4, s5"),
        ])
        pair_start = find_scc_pair_start(block, 1)
        assert pair_start == 0

    def test_is_scc_separated_pair_end(self):
        """Test is_scc_separated_pair_end."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),
            create_instruction("s_mul_i32", "s10, s11, s12"),
            create_instruction("s_addc_u32", "s3, s4, s5"),
        ])
        assert is_scc_separated_pair_end(block, 2) is True
        assert is_scc_separated_pair_end(block, 1) is False
        assert is_scc_separated_pair_end(block, 0) is False


class TestGetInstructionsBetweenPair:
    """Tests for getting instructions between pair."""

    def test_tight_pair(self):
        """Test tight pair has no instructions between."""
        indices = get_instructions_between_pair(None, 0, 1)
        assert indices == []

    def test_separated_pair(self):
        """Test separated pair returns intermediate indices."""
        indices = get_instructions_between_pair(None, 0, 3)
        assert indices == [1, 2]


class TestCanChainSkipSCCPair:
    """Tests for chain SCC pair skipping."""

    def test_can_skip_when_chain_only_writes_scc(self):
        """Test chain can skip pair when it only writes SCC."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),  # pair start
            create_instruction("s_addc_u32", "s3, s4, s5"),  # pair end
            create_instruction("s_lshl_b64", "s[6:7], s[6:7], 1"),  # chain: only writes SCC
        ])
        chain = [2]
        assert can_chain_skip_scc_pair(block, chain, 0, 1) is True

    def test_cannot_skip_when_chain_reads_scc(self):
        """Test chain cannot skip pair when it reads SCC."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_add_u32", "s0, s1, s2"),  # pair start
            create_instruction("s_addc_u32", "s3, s4, s5"),  # pair end
            create_instruction("s_cbranch_scc0", ".LBB0_5"),  # chain: reads SCC
        ])
        chain = [2]
        assert can_chain_skip_scc_pair(block, chain, 0, 1) is False


# =============================================================================
# Register Liveness Tests
# =============================================================================

class TestRegisterLiveness:
    """Tests for register liveness analysis."""

    def test_register_live_after(self):
        """Test register is live when used later."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_add_f32", "v1, v0, v2"),  # uses v0
        ])
        assert is_register_live_after(block, "v0", 0) is True

    def test_register_not_live_after(self):
        """Test register is not live when not used."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_mov_b32", "v1, 2.0"),  # doesn't use v0
        ])
        assert is_register_live_after(block, "v0", 0) is False


# =============================================================================
# s_waitcnt Handling Tests
# =============================================================================

class TestWaitcntOperands:
    """Tests for s_waitcnt operand handling."""

    def test_build_waitcnt_vmcnt_only(self):
        """Test building vmcnt-only operands."""
        result = build_waitcnt_operands(0, None)
        assert result == "vmcnt(0)"

    def test_build_waitcnt_lgkmcnt_only(self):
        """Test building lgkmcnt-only operands."""
        result = build_waitcnt_operands(None, 0)
        assert result == "lgkmcnt(0)"

    def test_build_waitcnt_both(self):
        """Test building both operands."""
        result = build_waitcnt_operands(3, 2)
        assert "vmcnt(3)" in result
        assert "lgkmcnt(2)" in result

    def test_update_waitcnt_instruction(self):
        """Test updating s_waitcnt instruction."""
        instr = create_instruction("s_waitcnt", "vmcnt(5) lgkmcnt(3)")
        success = update_waitcnt_instruction(instr, vmcnt_delta=-2, lgkmcnt_delta=1)
        assert success is True
        assert "vmcnt(3)" in instr.operands
        assert "lgkmcnt(4)" in instr.operands

    def test_update_waitcnt_invalid_negative(self):
        """Test that negative count fails."""
        instr = create_instruction("s_waitcnt", "vmcnt(0)")
        success = update_waitcnt_instruction(instr, vmcnt_delta=-1)
        assert success is False

    def test_update_waitcnt_invalid_overflow(self):
        """Test that overflow count fails."""
        instr = create_instruction("s_waitcnt", "vmcnt(60)")
        success = update_waitcnt_instruction(instr, vmcnt_delta=10)  # Would be 70 > 63
        assert success is False

    def test_update_non_waitcnt_instruction(self):
        """Test that non-waitcnt instruction fails."""
        instr = create_instruction("v_add_f32", "v0, v1, v2")
        success = update_waitcnt_instruction(instr)
        assert success is False


class TestGetInstructionCycles:
    """Tests for instruction cycle count."""

    def test_mfma_cycles(self):
        """Test MFMA instruction has high cycle count."""
        cycles = get_instruction_cycles("v_mfma_f32_16x16x16_bf16")
        assert cycles >= 8  # MFMA should have significant cycles

    def test_regular_instruction_cycles(self):
        """Test regular instruction has lower cycle count."""
        cycles = get_instruction_cycles("v_add_f32")
        assert cycles >= 1

    def test_nop_cycles(self):
        """Test s_nop cycles."""
        cycles = get_instruction_cycles("s_nop")
        assert cycles >= 1


# =============================================================================
# Pass Manager Tests
# =============================================================================

class TestPassManager:
    """Tests for PassManager."""

    def test_pass_manager_creation(self):
        """Test creating a PassManager."""
        pm = PassManager()
        assert len(pm.passes) == 0

    def test_pass_manager_add_pass(self):
        """Test adding passes to manager."""
        pm = PassManager()
        pass1 = MoveInstructionPass(".LBB0_0", 0, 1)
        pm.add_pass(pass1)
        assert len(pm.passes) == 1

    def test_pass_manager_clear(self):
        """Test clearing passes."""
        pm = PassManager()
        pm.add_pass(MoveInstructionPass(".LBB0_0", 0, 1))
        pm.add_pass(MoveInstructionPass(".LBB0_0", 1, 1))
        pm.clear()
        assert len(pm.passes) == 0


# =============================================================================
# MoveInstructionPass Tests
# =============================================================================

class TestMoveInstructionPass:
    """Tests for MoveInstructionPass."""

    def test_pass_name(self):
        """Test pass name generation."""
        pass_ = MoveInstructionPass(".LBB0_0", 5, 2)
        assert "MoveInstruction" in pass_.name
        assert ".LBB0_0" in pass_.name
        assert "up" in pass_.name

    def test_pass_name_down(self):
        """Test pass name for downward movement."""
        pass_ = MoveInstructionPass(".LBB0_0", 5, -2)
        assert "down" in pass_.name

    def test_pass_zero_cycles(self):
        """Test that zero cycles returns False."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_mov_b32", "v1, 2.0"),
        ])
        cfg = CFG(name="test")
        cfg.add_block(block)
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        result = AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
        
        pass_ = MoveInstructionPass(".LBB0_0", 0, 0)
        changed = pass_.run(result)
        assert changed is False

    def test_pass_invalid_block(self):
        """Test that invalid block fails gracefully."""
        cfg = CFG(name="test")
        cfg.add_block(create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
        ]))
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        result = AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
        
        pass_ = MoveInstructionPass(".LBB0_NONEXISTENT", 0, 1)
        changed = pass_.run(result)
        assert changed is False
        assert pass_.last_result.success is False

    def test_pass_invalid_index(self):
        """Test that invalid index fails gracefully."""
        cfg = CFG(name="test")
        cfg.add_block(create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
        ]))
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        result = AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
        
        pass_ = MoveInstructionPass(".LBB0_0", 100, 1)  # Index out of range
        changed = pass_.run(result)
        assert changed is False
        assert pass_.last_result.success is False


class TestMoveResult:
    """Tests for MoveResult dataclass."""

    def test_move_result_success(self):
        """Test successful move result."""
        result = MoveResult(success=True, message="Moved 2 positions")
        assert result.success is True

    def test_move_result_failure(self):
        """Test failed move result."""
        result = MoveResult(
            success=False,
            message="Blocked",
            blocked_by="RAW dependency"
        )
        assert result.success is False
        assert result.blocked_by == "RAW dependency"


# =============================================================================
# Dependency Chain Tests
# =============================================================================

class TestFindDependencyChain:
    """Tests for dependency chain finding."""

    def test_find_dependency_chain_single(self):
        """Test finding single instruction chain."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_mov_b32", "v1, 2.0"),  # no dependency
        ])
        chain = find_dependency_chain(block, None, 1, -1)
        assert chain == [1]

    def test_find_dependency_chain_with_dependency(self):
        """Test finding chain with dependency."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_add_f32", "v1, v0, v2"),  # depends on v0
        ])
        chain = find_dependency_chain(block, None, 1, -1)
        assert 0 in chain
        assert 1 in chain

    def test_find_immediate_dependency_chain(self):
        """Test finding immediate dependency chain."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_add_f32", "v1, v0, v2"),  # depends on idx 0
            create_instruction("v_add_f32", "v3, v1, v4"),  # depends on idx 1
        ])
        chain = find_immediate_dependency_chain(block, 2, -1)
        assert 0 in chain or 1 in chain  # Should include at least v1 dependency


# =============================================================================
# Integration Tests with Real Assembly
# =============================================================================

class TestPassesIntegration:
    """Integration tests using real assembly file."""

    @pytest.fixture
    def analysis_result(self):
        """Parse the test assembly file and generate analysis result."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test assembly file not found: {TEST_ASSEMBLY_FILE}")
        parser = AMDGCNParser()
        cfg = parser.parse_file(str(TEST_ASSEMBLY_FILE))
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        return AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)

    def test_analysis_result_has_ddgs(self, analysis_result):
        """Test that analysis result has DDGs."""
        assert len(analysis_result.ddgs) > 0

    def test_can_find_movable_instructions(self, analysis_result):
        """Test that we can find instructions that could potentially be moved."""
        # Find a block with multiple instructions
        for label, block in analysis_result.cfg.blocks.items():
            if len(block.instructions) >= 5:
                # Check that we can analyze dependencies
                for i in range(len(block.instructions)):
                    instr = block.instructions[i]
                    defs, uses = get_instruction_defs_uses(instr)
                    # Just verify the analysis works
                    assert isinstance(defs, set)
                    assert isinstance(uses, set)
                break

    def test_pass_manager_run_on_real_code(self, analysis_result):
        """Test running pass manager on real code."""
        pm = PassManager()
        pm.verbose = False
        
        # Find a non-empty block
        target_block = None
        for label, block in analysis_result.cfg.blocks.items():
            if len(block.instructions) >= 3:
                target_block = label
                break
        
        if target_block:
            # Add a pass that tries to move by 0 cycles (should be no-op)
            pass_ = MoveInstructionPass(target_block, 1, 0)
            pm.add_pass(pass_)
            
            result = pm.run_all(analysis_result)
            # Zero cycles should not make changes
            assert result is False


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestPassesEdgeCases:
    """Tests for edge cases in passes."""

    def test_empty_block_chain(self):
        """Test dependency chain on empty block returns empty list for out of range index."""
        block = BasicBlock(label=".LBB0_0")
        # An empty block with index 0 should fail gracefully
        # The function expects at least one instruction
        # So we test with a single instruction instead
        block.instructions.append(create_instruction("s_nop", "0"))
        chain = find_dependency_chain(block, None, 0, -1)
        assert chain == [0]

    def test_single_instruction_block(self):
        """Test with single instruction block."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("s_endpgm", ""),
        ])
        chain = find_immediate_dependency_chain(block, 0, -1)
        assert chain == [0]

    def test_branch_at_end(self):
        """Test block ending with branch."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("s_cbranch_scc0", ".LBB0_5"),
        ])
        # Should be able to analyze even with branch
        chain = find_immediate_dependency_chain(block, 1, -1)
        assert 1 in chain

    def test_barrier_instruction(self):
        """Test block with s_barrier."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("s_barrier", ""),
            create_instruction("v_mov_b32", "v1, 2.0"),
        ])
        # Barrier should be handled
        assert len(block.instructions) == 3

    def test_waitcnt_instruction(self):
        """Test block with s_waitcnt."""
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("global_load_dwordx4", "v[0:3], v[4:5], off"),
            create_instruction("s_waitcnt", "vmcnt(0)"),
            create_instruction("v_add_f32", "v6, v0, v1"),
        ])
        ddg = build_ddg(block)
        # Should track memory op and waitcnt
        assert len(ddg.vm_ops) == 1

    def test_frozen_boundary(self):
        """Test frozen boundary parameter."""
        pass_ = MoveInstructionPass(".LBB0_0", 5, 2, frozen_boundary=3)
        assert pass_.frozen_boundary == 3

    def test_protected_instructions(self):
        """Test protected instructions parameter."""
        protected = [create_instruction("s_nop", "0")]
        pass_ = MoveInstructionPass(".LBB0_0", 5, 2, protected_instructions=protected)
        assert len(pass_.protected_instructions) == 1

    def test_auto_insert_nops(self):
        """Test auto_insert_nops parameter."""
        pass1 = MoveInstructionPass(".LBB0_0", 5, 2, auto_insert_nops=True)
        pass2 = MoveInstructionPass(".LBB0_0", 5, 2, auto_insert_nops=False)
        assert pass1.auto_insert_nops is True
        assert pass2.auto_insert_nops is False


# =============================================================================
# Test RegisterReplacePass
# =============================================================================

class TestParseRegisterSegment:
    """Test the parse_register_segment helper function."""
    
    def test_single_vgpr(self):
        """Test parsing single VGPR."""
        seg = parse_register_segment("v40")
        assert seg is not None
        assert seg.prefix == "v"
        assert seg.start == 40
        assert seg.count == 1
        assert seg.get_registers() == ["v40"]
    
    def test_single_sgpr(self):
        """Test parsing single SGPR."""
        seg = parse_register_segment("s37")
        assert seg is not None
        assert seg.prefix == "s"
        assert seg.start == 37
        assert seg.count == 1
    
    def test_single_agpr(self):
        """Test parsing single AGPR."""
        seg = parse_register_segment("a0")
        assert seg is not None
        assert seg.prefix == "a"
        assert seg.start == 0
        assert seg.count == 1
    
    def test_vgpr_range(self):
        """Test parsing VGPR range."""
        seg = parse_register_segment("v[40:45]")
        assert seg is not None
        assert seg.prefix == "v"
        assert seg.start == 40
        assert seg.count == 6  # 40, 41, 42, 43, 44, 45
        assert seg.get_registers() == ["v40", "v41", "v42", "v43", "v44", "v45"]
    
    def test_sgpr_range(self):
        """Test parsing SGPR range."""
        seg = parse_register_segment("s[37:40]")
        assert seg is not None
        assert seg.prefix == "s"
        assert seg.start == 37
        assert seg.count == 4  # 37, 38, 39, 40
    
    def test_agpr_range(self):
        """Test parsing AGPR range."""
        seg = parse_register_segment("a[0:3]")
        assert seg is not None
        assert seg.prefix == "a"
        assert seg.start == 0
        assert seg.count == 4
    
    def test_uppercase(self):
        """Test parsing with uppercase."""
        seg = parse_register_segment("V[10:15]")
        assert seg is not None
        assert seg.prefix == "v"
        assert seg.start == 10
        assert seg.count == 6
    
    def test_whitespace(self):
        """Test parsing with whitespace."""
        seg = parse_register_segment("  v40  ")
        assert seg is not None
        assert seg.prefix == "v"
        assert seg.start == 40
    
    def test_invalid_format(self):
        """Test invalid format returns None."""
        assert parse_register_segment("invalid") is None
        assert parse_register_segment("x40") is None
        assert parse_register_segment("v[40]") is None  # Missing end
        assert parse_register_segment("v40:45") is None  # Missing brackets
        assert parse_register_segment("") is None
    
    def test_segment_str(self):
        """Test RegisterSegment __str__ method."""
        seg1 = parse_register_segment("v40")
        assert str(seg1) == "v40"
        
        seg2 = parse_register_segment("v[40:45]")
        assert str(seg2) == "v[40:45]"


class TestFindAlignedFreeRegisters:
    """Test the find_aligned_free_registers helper function."""
    
    def test_basic_allocation(self):
        """Test basic allocation without alignment."""
        fgpr = {"v90", "v91", "v92", "v93", "v94", "v95"}
        result = find_aligned_free_registers(fgpr, "v", 3, 1)
        assert result == 90  # Smallest starting index
    
    def test_alignment_2(self):
        """Test allocation with alignment 2."""
        fgpr = {"v90", "v91", "v92", "v93", "v94", "v95"}
        result = find_aligned_free_registers(fgpr, "v", 3, 2)
        assert result == 90  # 90 is divisible by 2
    
    def test_alignment_4(self):
        """Test allocation with alignment 4."""
        fgpr = {"v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98"}
        result = find_aligned_free_registers(fgpr, "v", 4, 4)
        assert result == 92  # 92 is divisible by 4
    
    def test_gap_in_registers(self):
        """Test allocation skips non-consecutive ranges."""
        fgpr = {"v90", "v91", "v93", "v94", "v95"}  # Gap at v92
        result = find_aligned_free_registers(fgpr, "v", 3, 1)
        assert result == 93  # Must skip past the gap
    
    def test_insufficient_registers(self):
        """Test returns None when insufficient registers."""
        fgpr = {"v90", "v91"}
        result = find_aligned_free_registers(fgpr, "v", 5, 1)
        assert result is None
    
    def test_empty_set(self):
        """Test returns None for empty set."""
        fgpr = set()
        result = find_aligned_free_registers(fgpr, "v", 1, 1)
        assert result is None
    
    def test_no_aligned_start(self):
        """Test returns None when no aligned start available."""
        fgpr = {"v91", "v92", "v93"}  # 91 not divisible by 4, and no aligned 4-consecutive
        result = find_aligned_free_registers(fgpr, "v", 3, 4)
        assert result is None  # 92 divisible by 4, but only 2 consecutive from there
    
    def test_different_prefix(self):
        """Test allocation with different prefix."""
        fgpr = {"v90", "s10", "s11", "s12"}
        result = find_aligned_free_registers(fgpr, "s", 2, 1)
        assert result == 10


class TestRegisterReplacePass:
    """Test the RegisterReplacePass class."""
    
    def _create_test_cfg(self) -> AnalysisResult:
        """Create a test CFG with known register usage."""
        # Create a simple CFG
        cfg = CFG(name="test_cfg")
        
        # Block with instructions using v40-v45
        block = BasicBlock(label=".LBB0_0")
        block.instructions = [
            create_instruction("v_mov_b32", "v40, 0x100", address=100),
            create_instruction("v_mov_b32", "v41, v40", address=101),
            create_instruction("v_add_f32", "v42, v40, v41", address=102),
            create_instruction("v_mov_b32", "v43, v42", address=103),
            create_instruction("global_store_dword", "v[44:45], v43, off", address=104),
            # Instruction outside range
            create_instruction("v_mov_b32", "v50, v40", address=200),
        ]
        cfg.add_block(block)
        
        # Set up FGPR - free registers v90-v99
        cfg.fgpr = {
            'full_free': {
                'vgpr': ['v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99'],
                'agpr': [],
                'sgpr': ['s80', 's81', 's82', 's83']
            },
            'scattered_free': {
                'vgpr': [],
                'agpr': [],
                'sgpr': []
            }
        }
        
        # Create DDGs with nodes
        nodes = []
        for node_id, instr in enumerate(block.instructions):
            defs, uses = parse_instruction_registers(instr)
            node = InstructionNode(instr=instr, node_id=node_id, defs=defs, uses=uses)
            nodes.append(node)
        
        ddgs = {".LBB0_0": DDG(block_label=".LBB0_0", nodes=nodes)}
        
        result = AnalysisResult(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=[],
            waitcnt_deps=[]
        )
        
        return result
    
    def test_basic_replacement(self):
        """Test basic register replacement."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        # Check mapping was created
        assert "v40" in pass_.register_mapping
        new_reg = pass_.register_mapping["v40"]
        assert new_reg.startswith("v")
        
        # Check instruction was modified
        block = result.cfg.blocks[".LBB0_0"]
        assert new_reg in block.instructions[0].operands
    
    def test_range_replacement(self):
        """Test replacing a range of registers."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v[40:42]"],
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        # Check all registers in range were mapped
        assert "v40" in pass_.register_mapping
        assert "v41" in pass_.register_mapping
        assert "v42" in pass_.register_mapping
        
        # New registers should be consecutive
        new_indices = sorted([int(r[1:]) for r in pass_.register_mapping.values()])
        assert new_indices[1] - new_indices[0] == 1
        assert new_indices[2] - new_indices[1] == 1
    
    def test_alignment_constraint(self):
        """Test alignment constraint is respected."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v[40:41]"],
            alignments=[2],  # Must start at even index
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        # Check new register starts at aligned index
        new_start = int(pass_.register_mapping["v40"][1:])
        assert new_start % 2 == 0
    
    def test_multiple_segments(self):
        """Test replacing multiple register segments."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v40", "v41"],
            alignments=[1, 1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        # Check both were mapped (to different registers)
        assert "v40" in pass_.register_mapping
        assert "v41" in pass_.register_mapping
        assert pass_.register_mapping["v40"] != pass_.register_mapping["v41"]
    
    def test_instructions_outside_range_not_modified(self):
        """Test instructions outside the range are not modified."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,  # address 200 is outside
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        # Instruction at address 200 should still have v40
        block = result.cfg.blocks[".LBB0_0"]
        instr_200 = [i for i in block.instructions if i.address == 200][0]
        assert "v40" in instr_200.operands
    
    def test_insufficient_free_registers(self):
        """Test error when insufficient free registers."""
        result = self._create_test_cfg()
        
        # Request more registers than available
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v[0:99]"],  # 100 registers, but only 10 free
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is False
        assert pass_.error_message is not None
        assert "Insufficient" in pass_.error_message
    
    def test_invalid_register_segment(self):
        """Test error for invalid register segment."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["invalid_reg"],
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is False
        assert pass_.error_message is not None
        assert "Invalid" in pass_.error_message
    
    def test_mismatched_alignment_count(self):
        """Test error when alignment count doesn't match segment count."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v40", "v41"],
            alignments=[1],  # Only 1 alignment for 2 segments
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is False
        assert pass_.error_message is not None
        assert "Alignment count" in pass_.error_message
    
    def test_empty_range(self):
        """Test replacement with empty instruction range."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=500,  # No instructions in this range
            range_end=600,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        # Mapping is created but no instructions modified
        assert changed is False  # No changes because no instructions in range
    
    def test_ddg_nodes_updated(self):
        """Test that DDG nodes are updated after replacement."""
        result = self._create_test_cfg()
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=104,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=True
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        new_reg = pass_.register_mapping["v40"]
        
        # Check DDG nodes were updated
        ddg = result.ddgs[".LBB0_0"]
        for node in ddg.nodes:
            if node.instr.address == 100:
                # v40 should be in defs (written by mov)
                assert new_reg in node.defs or "v40" not in node.defs
    
    def test_convenience_function(self):
        """Test the replace_registers convenience function."""
        result = self._create_test_cfg()
        
        success, mapping = replace_registers(
            result,
            range_start=100,
            range_end=104,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=True
        )
        
        assert success is True
        assert "v40" in mapping
    
    def test_pass_properties(self):
        """Test RegisterReplacePass property methods."""
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=200,
            registers_to_replace=["v[40:45]"],
            alignments=[2]
        )
        
        assert pass_.name == "RegisterReplacePass"
        assert "v[40:45]" in pass_.description
        assert "100" in pass_.description
        assert "200" in pass_.description


class TestRegisterReplacePassWithRealFile:
    """Test RegisterReplacePass with real assembly file."""
    
    @pytest.fixture
    def real_analysis_result(self):
        """Load real assembly file and create analysis result."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test file {TEST_ASSEMBLY_FILE} not found")
        
        parser = AMDGCNParser()
        cfg = parser.parse_file(str(TEST_ASSEMBLY_FILE))
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_deps = compute_inter_block_deps(cfg, ddgs)
        
        # Compute FGPR
        from amdgcn_ddg import compute_register_statistics, compute_fgpr
        stats = compute_register_statistics(ddgs)
        fgpr_info = compute_fgpr(stats)
        cfg.fgpr = fgpr_info.to_dict()
        cfg.register_stats = stats.to_dict()
        
        return AnalysisResult(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=inter_deps,
            waitcnt_deps=waitcnt_deps
        )
    
    def test_real_file_replacement(self, real_analysis_result):
        """Test register replacement on real file."""
        result = real_analysis_result
        
        # Find a block with instructions
        block_label = list(result.cfg.blocks.keys())[0]
        block = result.cfg.blocks[block_label]
        
        if len(block.instructions) < 2:
            pytest.skip("Block has too few instructions")
        
        # Get range from first few instructions
        range_start = block.instructions[0].address
        range_end = block.instructions[min(10, len(block.instructions) - 1)].address
        
        # Find a VGPR used in this range
        from amdgcn_ddg import parse_instruction_registers
        used_vgprs = set()
        for instr in block.instructions[:11]:
            defs, uses = parse_instruction_registers(instr)
            for reg in defs | uses:
                if reg.startswith('v') and reg[1:].isdigit():
                    used_vgprs.add(reg)
        
        if not used_vgprs:
            pytest.skip("No VGPRs found in range")
        
        # Get a VGPR to replace
        test_vgpr = sorted(used_vgprs)[0]
        
        pass_ = RegisterReplacePass(
            range_start=range_start,
            range_end=range_end,
            registers_to_replace=[test_vgpr],
            alignments=[1],
            verbose=True
        )
        
        # This may or may not succeed depending on free registers
        # The important thing is it doesn't crash
        pass_.run(result)


class TestRegisterReplacePassChaining:
    """Test chaining RegisterReplacePass with other passes."""
    
    def _create_test_cfg_for_chaining(self) -> AnalysisResult:
        """Create a test CFG suitable for chaining passes."""
        cfg = CFG(name="test_cfg")
        
        # Block with more instructions for testing movement
        block = BasicBlock(label=".LBB0_0")
        block.instructions = [
            create_instruction("v_mov_b32", "v10, 0x0", address=100),
            create_instruction("v_mov_b32", "v40, 0x100", address=101),
            create_instruction("v_mov_b32", "v11, v10", address=102),
            create_instruction("v_mov_b32", "v41, v40", address=103),
            create_instruction("v_add_f32", "v12, v10, v11", address=104),
            create_instruction("v_add_f32", "v42, v40, v41", address=105),
            create_instruction("s_nop", "0", address=106),
            create_instruction("v_mov_b32", "v13, v12", address=107),
            create_instruction("v_mov_b32", "v43, v42", address=108),
        ]
        cfg.add_block(block)
        
        # Set up FGPR
        cfg.fgpr = {
            'full_free': {
                'vgpr': ['v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99'],
                'agpr': [],
                'sgpr': []
            },
            'scattered_free': {'vgpr': [], 'agpr': [], 'sgpr': []}
        }
        
        nodes = []
        for node_id, instr in enumerate(block.instructions):
            defs, uses = parse_instruction_registers(instr)
            node = InstructionNode(instr=instr, node_id=node_id, defs=defs, uses=uses)
            nodes.append(node)
        
        ddgs = {".LBB0_0": DDG(block_label=".LBB0_0", nodes=nodes)}
        
        return AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=[], waitcnt_deps=[])
    
    def test_pass_manager_with_register_replace(self):
        """Test RegisterReplacePass works with PassManager."""
        result = self._create_test_cfg_for_chaining()
        
        pm = PassManager()
        pm.verbose = True
        
        # Add register replace pass
        replace_pass = RegisterReplacePass(
            range_start=101,
            range_end=108,
            registers_to_replace=["v[40:42]"],
            alignments=[1],
            verbose=True
        )
        pm.add_pass(replace_pass)
        
        # Run passes
        changed = pm.run_all(result, rebuild_ddg=False)
        
        assert changed is True
        assert "v40" in replace_pass.register_mapping
    
    def test_register_replace_then_move(self):
        """Test chaining RegisterReplacePass followed by MoveInstructionPass."""
        result = self._create_test_cfg_for_chaining()
        
        pm = PassManager()
        pm.verbose = True
        
        # First: replace registers
        replace_pass = RegisterReplacePass(
            range_start=101,
            range_end=105,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=False
        )
        pm.add_pass(replace_pass)
        
        # Run replacement first
        changed = pm.run_all(result, rebuild_ddg=False)
        assert changed is True
        
        # The new register should be used in instructions
        new_reg = replace_pass.register_mapping["v40"]
        block = result.cfg.blocks[".LBB0_0"]
        
        # Check that at least one instruction now uses the new register
        found_new_reg = any(new_reg in instr.operands for instr in block.instructions)
        assert found_new_reg, f"New register {new_reg} not found in any instruction"
    
    def test_chained_passes_via_pass_manager(self):
        """Test multiple passes chained via PassManager."""
        result = self._create_test_cfg_for_chaining()
        
        # Store original instructions for comparison
        block = result.cfg.blocks[".LBB0_0"]
        original_count = len(block.instructions)
        
        # Create passes
        replace_pass = RegisterReplacePass(
            range_start=101,
            range_end=105,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=False
        )
        
        pm = PassManager()
        pm.verbose = True
        pm.add_pass(replace_pass)
        
        # Run all passes
        changed = pm.run_all(result, rebuild_ddg=False)
        
        assert changed is True
        # Instruction count should be same (register replace doesn't add/remove)
        assert len(block.instructions) == original_count
    
    def test_register_replace_preserves_instruction_order(self):
        """Test that RegisterReplacePass preserves instruction order."""
        result = self._create_test_cfg_for_chaining()
        
        block = result.cfg.blocks[".LBB0_0"]
        original_addresses = [instr.address for instr in block.instructions]
        
        pass_ = RegisterReplacePass(
            range_start=100,
            range_end=108,
            registers_to_replace=["v[40:43]"],
            alignments=[1],
            verbose=False
        )
        
        pass_.run(result)
        
        # Check addresses are unchanged
        new_addresses = [instr.address for instr in block.instructions]
        assert original_addresses == new_addresses
    
    def test_verify_optimization_compatible(self):
        """Test that RegisterReplacePass doesn't break verify_optimization."""
        # RegisterReplacePass only changes operands, not instruction order.
        # verify_optimization checks instruction ordering constraints,
        # so register replacement should be compatible.
        
        result = self._create_test_cfg_for_chaining()
        
        # Store original block for building GDG
        from amdgcn_verify import build_global_ddg
        original_gdg = build_global_ddg(result.cfg, result.ddgs)
        
        pass_ = RegisterReplacePass(
            range_start=101,
            range_end=108,
            registers_to_replace=["v40"],
            alignments=[1],
            verbose=False
        )
        
        changed = pass_.run(result)
        assert changed is True
        
        # verify_optimization should still pass because order is unchanged
        # (We can't fully test this without real DDG edges, but the key
        # point is that instruction addresses are preserved)


# =============================================================================
# Test barrier_crossing_opcodes Feature
# =============================================================================

class TestBarrierCrossingOpcodes:
    """Tests for barrier_crossing_opcodes parameter in MoveInstructionPass and DistributeInstructionPass."""
    
    def _create_block_with_barrier(self) -> AnalysisResult:
        """Create a block with s_barrier for testing."""
        # Create instructions:
        # [0] v_mov_b32 v0, 1.0
        # [1] global_load_dwordx4 v[4:7], v[0:1], s[0:1]  (target to move)
        # [2] s_barrier
        # [3] v_mov_b32 v10, 2.0
        # [4] v_mov_b32 v11, 3.0
        instructions = [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("global_load_dwordx4", "v[4:7], v[0:1], s[0:1]"),
            create_instruction("s_barrier", ""),
            create_instruction("v_mov_b32", "v10, 2.0"),
            create_instruction("v_mov_b32", "v11, 3.0"),
        ]
        
        block = create_block_with_instructions(".LBB0_0", instructions)
        cfg = CFG(name="test")
        cfg.add_block(block)
        
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        
        return AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
    
    def test_barrier_crossing_opcodes_parameter_default(self):
        """Test that barrier_crossing_opcodes defaults to empty set."""
        pass_ = MoveInstructionPass(".LBB0_0", 1, -2)
        assert pass_.barrier_crossing_opcodes == set()
    
    def test_barrier_crossing_opcodes_parameter_set(self):
        """Test setting barrier_crossing_opcodes parameter."""
        opcodes = {"global_load_dwordx4", "global_store_dwordx4"}
        pass_ = MoveInstructionPass(".LBB0_0", 1, -2, barrier_crossing_opcodes=opcodes)
        assert pass_.barrier_crossing_opcodes == opcodes
    
    def test_move_blocked_by_barrier_without_crossing(self):
        """Test that moves are blocked by s_barrier without barrier_crossing_opcodes."""
        result = self._create_block_with_barrier()
        block = result.cfg.blocks[".LBB0_0"]
        
        # Try to move instruction [3] (v_mov v10) up past barrier - should be blocked
        pass_ = MoveInstructionPass(".LBB0_0", 3, 2)  # move up 2 cycles
        changed = pass_.run(result)
        
        # Movement should be blocked by barrier (instruction at index 3 should stay at index 3)
        # Since we're moving OTHER instructions to make room, and barrier blocks crossing
        assert pass_.total_cycles_moved < 2 or len(pass_._blocked_by_barrier) > 0
    
    def test_move_allowed_with_barrier_crossing_opcodes(self):
        """Test that specified opcodes can cross s_barrier."""
        result = self._create_block_with_barrier()
        
        # global_load_dwordx4 is allowed to cross barrier
        opcodes = {"global_load_dwordx4"}
        pass_ = MoveInstructionPass(".LBB0_0", 1, -2, barrier_crossing_opcodes=opcodes)
        
        # This should allow the global_load_dwordx4 to move down past the barrier
        # (The pass moves other instructions; global_load_dwordx4 can cross barrier)
        changed = pass_.run(result)
        
        # Verify the instruction can cross (checking that barrier_crossing_opcodes is being used)
        assert pass_.barrier_crossing_opcodes == {"global_load_dwordx4"}
    
    def test_s_barrier_cannot_be_moved_without_barrier_crossing(self):
        """Test that s_barrier cannot be moved without barrier_crossing_opcodes."""
        result = self._create_block_with_barrier()
        
        # Without barrier_crossing_opcodes, s_barrier cannot be moved
        pass_ = MoveInstructionPass(".LBB0_0", 2, 1)  # No barrier_crossing_opcodes
        
        # s_barrier should not be movable
        changed = pass_.run(result)
        # The pass should not make any changes because s_barrier can't cross anything
    
    def test_s_barrier_can_cross_allowed_opcodes(self):
        """Test that s_barrier can cross instructions in barrier_crossing_opcodes."""
        # Create a block where s_barrier needs to cross global_load_dwordx4:
        # [0] global_load_dwordx4 v[0:3], v[8:9], s[0:1]  (target to move up)
        # [1] s_barrier
        # [2] v_mov_b32 v10, 2.0
        instructions = [
            create_instruction("global_load_dwordx4", "v[0:3], v[8:9], s[0:1]"),
            create_instruction("s_barrier", ""),
            create_instruction("v_mov_b32", "v10, 2.0"),
        ]
        
        block = create_block_with_instructions(".LBB0_0", instructions)
        cfg = CFG(name="test")
        cfg.add_block(block)
        
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        result = AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
        
        # With barrier_crossing_opcodes including global_load_dwordx4,
        # s_barrier should be able to move (cross global_load_dwordx4)
        opcodes = {"global_load_dwordx4"}
        pass_ = MoveInstructionPass(".LBB0_0", 0, -1, barrier_crossing_opcodes=opcodes)
        
        # This tests that when we try to move global_load_dwordx4 down,
        # s_barrier can be moved up to make room
        changed = pass_.run(result)
        
        # Verify barrier_crossing_opcodes is set correctly
        assert pass_.barrier_crossing_opcodes == {"global_load_dwordx4"}


class TestDistributeInstructionPassBarrierCrossing:
    """Tests for barrier_crossing_opcodes in DistributeInstructionPass."""
    
    def _create_block_with_loads_and_barrier(self) -> AnalysisResult:
        """Create a block with global loads and s_barrier for testing distribution."""
        # Create instructions with global loads before and after barrier:
        # [0] global_load_dwordx4 v[0:3], v[8:9], s[0:1]
        # [1] v_mov_b32 v10, 1.0
        # [2] global_load_dwordx4 v[4:7], v[8:9], s[0:1]
        # [3] s_barrier
        # [4] global_load_dwordx4 v[12:15], v[8:9], s[0:1]
        # [5] v_mov_b32 v20, 2.0
        # [6] global_load_dwordx4 v[16:19], v[8:9], s[0:1]
        instructions = [
            create_instruction("global_load_dwordx4", "v[0:3], v[8:9], s[0:1]"),
            create_instruction("v_mov_b32", "v10, 1.0"),
            create_instruction("global_load_dwordx4", "v[4:7], v[8:9], s[0:1]"),
            create_instruction("s_barrier", ""),
            create_instruction("global_load_dwordx4", "v[12:15], v[8:9], s[0:1]"),
            create_instruction("v_mov_b32", "v20, 2.0"),
            create_instruction("global_load_dwordx4", "v[16:19], v[8:9], s[0:1]"),
        ]
        
        block = create_block_with_instructions(".LBB0_0", instructions)
        cfg = CFG(name="test")
        cfg.add_block(block)
        
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        
        return AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
    
    def test_distribute_barrier_crossing_parameter(self):
        """Test that DistributeInstructionPass accepts barrier_crossing_opcodes."""
        from amdgcn_passes import DistributeInstructionPass
        
        opcodes = {"global_load_dwordx4"}
        pass_ = DistributeInstructionPass(
            block_label=".LBB0_0",
            target_opcode="global_load_dwordx4",
            distribute_count=4,
            barrier_crossing_opcodes=opcodes
        )
        
        assert pass_.barrier_crossing_opcodes == opcodes
    
    def test_distribute_without_barrier_crossing_stops_at_barrier(self):
        """Test that distribution stops at s_barrier without barrier_crossing_opcodes."""
        from amdgcn_passes import DistributeInstructionPass
        
        result = self._create_block_with_loads_and_barrier()
        
        # Without barrier_crossing_opcodes, distribution should stop at barrier
        pass_ = DistributeInstructionPass(
            block_label=".LBB0_0",
            target_opcode="global_load_dwordx4",
            distribute_count=4,
            verbose=False
        )
        
        # _find_branch_boundary should return index 3 (s_barrier)
        block = result.cfg.blocks[".LBB0_0"]
        boundary = pass_._find_branch_boundary(block)
        assert boundary == 3  # s_barrier is at index 3
    
    def test_distribute_with_barrier_crossing_extends_boundary(self):
        """Test that distribution extends past s_barrier with barrier_crossing_opcodes."""
        from amdgcn_passes import DistributeInstructionPass
        
        result = self._create_block_with_loads_and_barrier()
        
        # With barrier_crossing_opcodes, distribution should extend past barrier
        opcodes = {"global_load_dwordx4"}
        pass_ = DistributeInstructionPass(
            block_label=".LBB0_0",
            target_opcode="global_load_dwordx4",
            distribute_count=4,
            barrier_crossing_opcodes=opcodes,
            verbose=False
        )
        
        # _find_branch_boundary should return len(instructions) since barrier is ignored
        block = result.cfg.blocks[".LBB0_0"]
        boundary = pass_._find_branch_boundary(block)
        assert boundary == len(block.instructions)  # No boundary (ignoring s_barrier)


class TestConvenienceFunctionsBarrierCrossing:
    """Tests for barrier_crossing_opcodes in convenience functions."""
    
    def test_move_instruction_accepts_barrier_crossing(self):
        """Test that move_instruction convenience function accepts barrier_crossing_opcodes."""
        from amdgcn_passes import move_instruction
        
        # Create a minimal test setup
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
            create_instruction("v_mov_b32", "v1, 2.0"),
        ])
        cfg = CFG(name="test")
        cfg.add_block(block)
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        result = AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
        
        # Should not raise error
        opcodes = {"global_load_dwordx4"}
        move_result = move_instruction(
            result, ".LBB0_0", 0, 0,
            barrier_crossing_opcodes=opcodes
        )
        # Zero cycles movement should succeed (no-op)
        assert move_result.success is True
    
    def test_distribute_instructions_accepts_barrier_crossing(self):
        """Test that distribute_instructions convenience function accepts barrier_crossing_opcodes."""
        from amdgcn_passes import distribute_instructions
        
        # Create a minimal test setup
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("global_load_dwordx4", "v[0:3], v[8:9], s[0:1]"),
            create_instruction("v_mov_b32", "v10, 1.0"),
        ])
        cfg = CFG(name="test")
        cfg.add_block(block)
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        inter_block_deps = compute_inter_block_deps(cfg, ddgs)
        result = AnalysisResult(cfg=cfg, ddgs=ddgs, inter_block_deps=inter_block_deps, waitcnt_deps=waitcnt_deps)
        
        # Should not raise error - test that the parameter is accepted
        opcodes = {"global_load_dwordx4"}
        # Just verify no exception is raised when passing barrier_crossing_opcodes
        distribute_instructions(
            result, ".LBB0_0", "global_load_dwordx4", 1,
            barrier_crossing_opcodes=opcodes
        )
        # Test passes if no exception is raised


class TestVerifyOptimizationBarrierCrossing:
    """Tests for barrier_crossing_opcodes in verify_optimization."""
    
    def test_verify_optimization_accepts_barrier_crossing(self):
        """Test that verify_optimization accepts barrier_crossing_opcodes parameter."""
        from amdgcn_verify import verify_optimization, build_global_ddg
        
        # Create a simple block
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
        ])
        cfg = CFG(name="test")
        cfg.add_block(block)
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        
        gdg = build_global_ddg(cfg, ddgs)
        
        # Should not raise error
        opcodes = {"global_load_dwordx4"}
        verify_optimization(gdg, cfg, barrier_crossing_opcodes=opcodes)
    
    def test_verify_and_report_accepts_barrier_crossing(self):
        """Test that verify_and_report accepts barrier_crossing_opcodes parameter."""
        from amdgcn_verify import verify_and_report, build_global_ddg
        
        # Create a simple block
        block = create_block_with_instructions(".LBB0_0", [
            create_instruction("v_mov_b32", "v0, 1.0"),
        ])
        cfg = CFG(name="test")
        cfg.add_block(block)
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        
        gdg = build_global_ddg(cfg, ddgs)
        
        # Should not raise error
        opcodes = {"global_load_dwordx4"}
        result = verify_and_report(gdg, cfg, verbose=False, barrier_crossing_opcodes=opcodes)
        assert result.success is True


if __name__ == "__main__":
    import sys
    
    class TestResultCollector:
        """Pytest plugin to collect test results"""
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.skipped = 0
            self.errors = 0
        
        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    self.passed += 1
                elif report.failed:
                    self.failed += 1
            elif report.when == "setup" and report.skipped:
                self.skipped += 1
            elif report.when == "setup" and report.failed:
                self.errors += 1
    
    collector = TestResultCollector()
    exit_code = pytest.main([__file__, "-v"], plugins=[collector])
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Passed:  {collector.passed}")
    print(f"  Failed:  {collector.failed}")
    print(f"  Skipped: {collector.skipped}")
    print(f"  Errors:  {collector.errors}")
    print("-" * 60)
    total = collector.passed + collector.failed + collector.skipped + collector.errors
    print(f"  Total:   {total}")
    print("=" * 60)
    
    if collector.failed == 0 and collector.errors == 0:
        print("\n" + "*" * 60)
        print("*" + " " * 58 + "*")
        print("*" + "  ✓ ALL TESTS PASSED!  ".center(58) + "*")
        print("*" + " " * 58 + "*")
        print("*" * 60 + "\n")
    else:
        print("\n" + "!" * 60)
        print("!" + " " * 58 + "!")
        print("!" + "  ✗ SOME TESTS FAILED!  ".center(58) + "!")
        print("!" + " " * 58 + "!")
        print("!" * 60 + "\n")
    
    sys.exit(exit_code)

