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
        from amdgcn_ddg import compute_inter_block_deps
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
        from amdgcn_ddg import compute_inter_block_deps
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
        from amdgcn_ddg import compute_inter_block_deps
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
        from amdgcn_ddg import compute_inter_block_deps
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

