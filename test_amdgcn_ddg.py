#!/usr/bin/env python3
"""
Unit tests for AMDGCN DDG (Data Dependency Graph) module.

Tests cover:
- Register parsing (SGPR, VGPR, AGPR, VCC, EXEC, SCC)
- SCC handling (writers, readers, dead SCC detection)
- Memory operation classification (VM, LGKM)
- DDG construction and dependency edges
- Live-in/Live-out analysis
- s_waitcnt parsing
"""

import pytest
from pathlib import Path

from amdgcn_cfg import AMDGCNParser, Instruction, BasicBlock, CFG
from amdgcn_ddg import (
    # Register parsing
    expand_register_range,
    parse_operand_registers,
    parse_instruction_registers,
    
    # SCC handling
    SCC_WRITERS,
    SCC_READERS,
    SCC_ONLY_WRITERS,
    SCC_READ_WRITE,
    SCC_NON_WRITERS,
    is_scc_only_writer,
    is_scc_reader,
    is_scc_writer,
    is_dead_scc_write,
    
    # VCC/EXEC handling
    VCC_WRITERS,
    VCC_READERS,
    EXEC_WRITERS,
    EXEC_READERS,
    
    # Memory operation classification
    is_lgkm_op,
    is_vm_op,
    parse_waitcnt_operands,
    
    # DDG structures
    DDG,
    InstructionNode,
    PendingMemOp,
    InterBlockDep,
    
    # DDG building
    build_ddg,
    generate_all_ddgs,
    compute_inter_block_deps,
    AnalysisResult,
)


# =============================================================================
# Test Data - Path to real assembly file
# =============================================================================

TEST_ASSEMBLY_FILE = Path(__file__).parent / "pa_dot_kernel.v2.amdgcn"


# =============================================================================
# Register Expansion Tests
# =============================================================================

class TestExpandRegisterRange:
    """Tests for register range expansion."""

    def test_single_sgpr(self):
        """Test single SGPR."""
        result = expand_register_range("s0")
        assert result == ["s0"]

    def test_single_vgpr(self):
        """Test single VGPR."""
        result = expand_register_range("v42")
        assert result == ["v42"]

    def test_single_agpr(self):
        """Test single AGPR."""
        result = expand_register_range("a7")
        assert result == ["a7"]

    def test_sgpr_range_small(self):
        """Test small SGPR range."""
        result = expand_register_range("s[0:1]")
        assert result == ["s0", "s1"]

    def test_sgpr_range_medium(self):
        """Test medium SGPR range."""
        result = expand_register_range("s[2:5]")
        assert result == ["s2", "s3", "s4", "s5"]

    def test_vgpr_range(self):
        """Test VGPR range."""
        result = expand_register_range("v[120:123]")
        assert result == ["v120", "v121", "v122", "v123"]

    def test_agpr_range_large(self):
        """Test large AGPR range."""
        result = expand_register_range("a[0:3]")
        assert result == ["a0", "a1", "a2", "a3"]

    def test_agpr_range_32(self):
        """Test large 32-element AGPR range."""
        result = expand_register_range("a[0:31]")
        assert len(result) == 32
        assert result[0] == "a0"
        assert result[31] == "a31"

    def test_non_register_passthrough(self):
        """Test that non-register strings pass through."""
        result = expand_register_range("0x100")
        assert result == ["0x100"]

    def test_special_keyword_passthrough(self):
        """Test that special keywords pass through."""
        result = expand_register_range("off")
        assert result == ["off"]


# =============================================================================
# Operand Register Parsing Tests
# =============================================================================

class TestParseOperandRegisters:
    """Tests for parsing registers from operands."""

    def test_single_sgpr(self):
        """Test parsing single SGPR."""
        regs, reg_type = parse_operand_registers("s0")
        assert "s0" in regs
        assert reg_type == "sgpr"

    def test_sgpr_range(self):
        """Test parsing SGPR range."""
        regs, reg_type = parse_operand_registers("s[2:3]")
        assert "s2" in regs
        assert "s3" in regs
        assert reg_type == "sgpr"

    def test_single_vgpr(self):
        """Test parsing single VGPR."""
        regs, reg_type = parse_operand_registers("v0")
        assert "v0" in regs
        assert reg_type == "vgpr"

    def test_vgpr_range(self):
        """Test parsing VGPR range."""
        regs, reg_type = parse_operand_registers("v[120:123]")
        assert "v120" in regs
        assert "v121" in regs
        assert "v122" in regs
        assert "v123" in regs
        assert reg_type == "vgpr"

    def test_single_agpr(self):
        """Test parsing single AGPR."""
        regs, reg_type = parse_operand_registers("a0")
        assert "a0" in regs
        assert reg_type == "agpr"

    def test_agpr_range(self):
        """Test parsing AGPR range."""
        regs, reg_type = parse_operand_registers("a[0:3]")
        assert "a0" in regs
        assert "a3" in regs
        assert reg_type == "agpr"

    def test_vcc(self):
        """Test parsing VCC."""
        regs, reg_type = parse_operand_registers("vcc")
        assert "vcc" in regs
        assert reg_type == "vcc"

    def test_exec(self):
        """Test parsing EXEC."""
        regs, reg_type = parse_operand_registers("exec")
        assert "exec" in regs
        assert reg_type == "exec"

    def test_immediate_value(self):
        """Test that immediate values return empty."""
        regs, reg_type = parse_operand_registers("0x100")
        assert len(regs) == 0

    def test_off_keyword(self):
        """Test that 'off' keyword returns empty."""
        regs, reg_type = parse_operand_registers("off")
        assert len(regs) == 0

    def test_offen_keyword(self):
        """Test that 'offen' keyword returns empty."""
        regs, reg_type = parse_operand_registers("offen")
        assert len(regs) == 0

    def test_floating_point_literal(self):
        """Test that floating point literals return empty."""
        regs, reg_type = parse_operand_registers("1.0")
        assert len(regs) == 0

    def test_hex_constant(self):
        """Test that hex constants return empty."""
        regs, reg_type = parse_operand_registers("0x3fb8aa3b")
        assert len(regs) == 0


# =============================================================================
# Instruction Register Parsing Tests
# =============================================================================

class TestParseInstructionRegisters:
    """Tests for parsing defs and uses from instructions."""

    def test_simple_add(self):
        """Test simple v_add instruction."""
        instr = Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v0" in defs
        assert "v1" in uses
        assert "v2" in uses
        assert "v0" not in uses

    def test_mfma_instruction(self):
        """Test MFMA instruction parsing."""
        instr = Instruction(
            address=10, opcode="v_mfma_f32_16x16x16_bf16",
            operands="a[0:3], v[120:121], v[56:57], 0",
            raw_line="\tv_mfma_f32_16x16x16_bf16 a[0:3], v[120:121], v[56:57], 0"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "a0" in defs
        assert "a1" in defs
        assert "a2" in defs
        assert "a3" in defs
        assert "v120" in uses
        assert "v121" in uses
        assert "v56" in uses
        assert "v57" in uses

    def test_global_load(self):
        """Test global_load instruction parsing."""
        instr = Instruction(
            address=10, opcode="global_load_dwordx4",
            operands="v[120:123], v[26:27], off",
            raw_line="\tglobal_load_dwordx4 v[120:123], v[26:27], off"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v120" in defs
        assert "v123" in defs
        assert "v26" in uses
        assert "v27" in uses

    def test_global_store(self):
        """Test global_store instruction parsing (all sources)."""
        instr = Instruction(
            address=10, opcode="global_store_dwordx4",
            operands="v[36:37], v[88:91], off",
            raw_line="\tglobal_store_dwordx4 v[36:37], v[88:91], off"
        )
        defs, uses = parse_instruction_registers(instr)
        assert len(defs) == 0  # Stores have no defs
        assert "v36" in uses
        assert "v37" in uses
        assert "v88" in uses
        assert "v91" in uses

    def test_s_load(self):
        """Test s_load instruction parsing."""
        instr = Instruction(
            address=10, opcode="s_load_dwordx2",
            operands="s[2:3], s[0:1], 0x0",
            raw_line="\ts_load_dwordx2 s[2:3], s[0:1], 0x0"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "s2" in defs
        assert "s3" in defs
        assert "s0" in uses
        assert "s1" in uses

    def test_ds_read(self):
        """Test ds_read instruction parsing."""
        instr = Instruction(
            address=10, opcode="ds_read_b128",
            operands="v[56:59], v14",
            raw_line="\tds_read_b128 v[56:59], v14"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v56" in defs
        assert "v59" in defs
        assert "v14" in uses

    def test_ds_write(self):
        """Test ds_write instruction parsing (all sources)."""
        instr = Instruction(
            address=10, opcode="ds_write_b128",
            operands="v6, v[2:5]",
            raw_line="\tds_write_b128 v6, v[2:5]"
        )
        defs, uses = parse_instruction_registers(instr)
        assert len(defs) == 0  # Writes have no defs
        assert "v6" in uses
        assert "v2" in uses
        assert "v5" in uses

    def test_accvgpr_read(self):
        """Test v_accvgpr_read_b32 instruction parsing."""
        instr = Instruction(
            address=10, opcode="v_accvgpr_read_b32",
            operands="v36, a0",
            raw_line="\tv_accvgpr_read_b32 v36, a0"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v36" in defs
        assert "a0" in uses

    def test_accvgpr_write(self):
        """Test v_accvgpr_write_b32 instruction parsing."""
        instr = Instruction(
            address=10, opcode="v_accvgpr_write_b32",
            operands="a0, v36",
            raw_line="\tv_accvgpr_write_b32 a0, v36"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "a0" in defs
        assert "v36" in uses

    def test_s_waitcnt(self):
        """Test s_waitcnt has no defs/uses."""
        instr = Instruction(
            address=10, opcode="s_waitcnt",
            operands="vmcnt(0) lgkmcnt(0)",
            raw_line="\ts_waitcnt vmcnt(0) lgkmcnt(0)"
        )
        defs, uses = parse_instruction_registers(instr)
        assert len(defs) == 0
        assert len(uses) == 0

    def test_s_barrier(self):
        """Test s_barrier has no defs/uses."""
        instr = Instruction(
            address=10, opcode="s_barrier",
            operands="",
            raw_line="\ts_barrier"
        )
        defs, uses = parse_instruction_registers(instr)
        assert len(defs) == 0
        assert len(uses) == 0

    def test_s_branch_scc_reader(self):
        """Test s_cbranch_scc0 reads SCC."""
        instr = Instruction(
            address=10, opcode="s_cbranch_scc0",
            operands=".LBB0_5",
            raw_line="\ts_cbranch_scc0 .LBB0_5",
            is_branch=True, is_conditional=True
        )
        defs, uses = parse_instruction_registers(instr)
        assert "scc" in uses

    def test_s_add_u32_writes_scc(self):
        """Test s_add_u32 writes SCC."""
        instr = Instruction(
            address=10, opcode="s_add_u32",
            operands="s0, s1, s2",
            raw_line="\ts_add_u32 s0, s1, s2"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "scc" in defs
        assert "s0" in defs
        assert "s1" in uses
        assert "s2" in uses

    def test_s_addc_u32_reads_writes_scc(self):
        """Test s_addc_u32 reads and writes SCC."""
        instr = Instruction(
            address=10, opcode="s_addc_u32",
            operands="s3, s4, s5",
            raw_line="\ts_addc_u32 s3, s4, s5"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "scc" in defs
        assert "scc" in uses
        assert "s3" in defs
        assert "s4" in uses
        assert "s5" in uses

    def test_s_mul_i32_no_scc(self):
        """Test s_mul_i32 does NOT write SCC."""
        instr = Instruction(
            address=10, opcode="s_mul_i32",
            operands="s0, s1, s2",
            raw_line="\ts_mul_i32 s0, s1, s2"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "scc" not in defs
        assert "s0" in defs

    def test_v_cmp_e64(self):
        """Test v_cmp with explicit destination (e64 encoding)."""
        instr = Instruction(
            address=10, opcode="v_cmp_gt_i32_e64",
            operands="s[24:25], s17, v46",
            raw_line="\tv_cmp_gt_i32_e64 s[24:25], s17, v46"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "s24" in defs
        assert "s25" in defs
        assert "s17" in uses
        assert "v46" in uses

    def test_v_cmp_e32_implicit_vcc(self):
        """Test v_cmp with implicit VCC destination (e32 encoding)."""
        instr = Instruction(
            address=10, opcode="v_cmp_eq_u32_e32",
            operands="vcc, 0, v1",
            raw_line="\tv_cmp_eq_u32_e32 vcc, 0, v1"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "vcc" in defs
        assert "v1" in uses

    def test_v_cndmask_reads_vcc(self):
        """Test v_cndmask_b32_e32 reads VCC."""
        instr = Instruction(
            address=10, opcode="v_cndmask_b32_e32",
            operands="v4, v14, v9",
            raw_line="\tv_cndmask_b32_e32 v4, v14, v9"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v4" in defs
        assert "vcc" in uses  # Implicit VCC use

    def test_readfirstlane(self):
        """Test v_readfirstlane_b32 parsing."""
        instr = Instruction(
            address=10, opcode="v_readfirstlane_b32",
            operands="s8, v200",
            raw_line="\tv_readfirstlane_b32 s8, v200"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "s8" in defs
        assert "v200" in uses

    def test_pk_fma(self):
        """Test v_pk_fma_f32 parsing with op_sel_hi modifier."""
        # This test covers a regression bug where op_sel modifiers caused
        # the last operand's registers to be skipped entirely
        instr = Instruction(
            address=10, opcode="v_pk_fma_f32",
            operands="v[16:17], v[16:17], v[22:23], v[38:39] op_sel_hi:[1,0,1]",
            raw_line="\tv_pk_fma_f32 v[16:17], v[16:17], v[22:23], v[38:39] op_sel_hi:[1,0,1]"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v16" in defs
        assert "v17" in defs
        assert "v16" in uses  # Same register can be src and dst
        assert "v22" in uses
        # Regression test: v38, v39 must be parsed even with op_sel modifier
        assert "v38" in uses, "v38 should be parsed even with op_sel_hi modifier"
        assert "v39" in uses, "v39 should be parsed even with op_sel_hi modifier"


# =============================================================================
# SCC Handling Tests
# =============================================================================

class TestSCCHandling:
    """Tests for SCC-related functions."""

    def test_is_scc_writer(self):
        """Test is_scc_writer function."""
        assert is_scc_writer("s_add_u32") is True
        assert is_scc_writer("s_addc_u32") is True
        assert is_scc_writer("s_cmp_lt_i32") is True
        assert is_scc_writer("s_and_b64") is True
        assert is_scc_writer("s_mul_i32") is False
        assert is_scc_writer("s_mov_b32") is False

    def test_is_scc_reader(self):
        """Test is_scc_reader function."""
        assert is_scc_reader("s_cbranch_scc0") is True
        assert is_scc_reader("s_cbranch_scc1") is True
        assert is_scc_reader("s_addc_u32") is True
        assert is_scc_reader("s_cselect_b32") is True
        assert is_scc_reader("s_add_u32") is False

    def test_is_scc_only_writer(self):
        """Test is_scc_only_writer function."""
        assert is_scc_only_writer("s_add_u32") is True  # Only writes SCC
        assert is_scc_only_writer("s_cmp_lt_i32") is True  # Only writes SCC
        assert is_scc_only_writer("s_addc_u32") is False  # Also reads SCC
        assert is_scc_only_writer("s_mul_i32") is False  # Doesn't write SCC

    def test_scc_writer_sets(self):
        """Test that SCC writer sets are consistent."""
        # All SCC readers that also write should be in both sets
        for op in SCC_READ_WRITE:
            assert op in SCC_WRITERS, f"{op} should be in SCC_WRITERS"
            assert op in SCC_READERS, f"{op} should be in SCC_READERS"

    def test_scc_only_writers_not_readers(self):
        """Test that SCC_ONLY_WRITERS don't read SCC."""
        for op in SCC_ONLY_WRITERS:
            assert op not in SCC_READERS, f"{op} should not be in SCC_READERS"

    def test_scc_non_writers(self):
        """Test that SCC_NON_WRITERS don't write SCC."""
        for op in SCC_NON_WRITERS:
            assert op not in SCC_WRITERS, f"{op} should not be in SCC_WRITERS"


class TestDeadSCCWrite:
    """Tests for dead SCC write detection."""

    def test_dead_scc_write_simple(self):
        """Test detection of dead SCC write."""
        instructions = [
            Instruction(address=1, opcode="s_lshl_b64", operands="s[8:9], s[8:9], 1",
                       raw_line="\ts_lshl_b64 s[8:9], s[8:9], 1"),  # writes SCC
            Instruction(address=2, opcode="s_mul_i32", operands="s0, s1, s2",
                       raw_line="\ts_mul_i32 s0, s1, s2"),  # no SCC
            Instruction(address=3, opcode="s_lshl_b64", operands="s[0:1], s[0:1], 1",
                       raw_line="\ts_lshl_b64 s[0:1], s[0:1], 1"),  # writes SCC (overwrites)
        ]
        assert is_dead_scc_write(instructions, 0) is True  # First SCC write is dead

    def test_not_dead_scc_write_read(self):
        """Test that SCC write is not dead when read by next SCC instruction."""
        instructions = [
            Instruction(address=1, opcode="s_add_u32", operands="s0, s1, s2",
                       raw_line="\ts_add_u32 s0, s1, s2"),  # writes SCC
            Instruction(address=2, opcode="s_addc_u32", operands="s3, s4, s5",
                       raw_line="\ts_addc_u32 s3, s4, s5"),  # reads SCC
        ]
        assert is_dead_scc_write(instructions, 0) is False  # SCC is read

    def test_not_dead_scc_no_subsequent(self):
        """Test that SCC write is not considered dead if no subsequent SCC instruction."""
        instructions = [
            Instruction(address=1, opcode="s_add_u32", operands="s0, s1, s2",
                       raw_line="\ts_add_u32 s0, s1, s2"),  # writes SCC
            Instruction(address=2, opcode="v_add_f32", operands="v0, v1, v2",
                       raw_line="\tv_add_f32 v0, v1, v2"),  # no SCC
        ]
        assert is_dead_scc_write(instructions, 0) is False  # SCC might be live-out

    def test_non_scc_writer(self):
        """Test that non-SCC writer returns False."""
        instructions = [
            Instruction(address=1, opcode="v_add_f32", operands="v0, v1, v2",
                       raw_line="\tv_add_f32 v0, v1, v2"),
        ]
        assert is_dead_scc_write(instructions, 0) is False


# =============================================================================
# Memory Operation Classification Tests
# =============================================================================

class TestMemoryOperationClassification:
    """Tests for memory operation classification."""

    def test_is_lgkm_op_s_load(self):
        """Test s_load is LGKM operation."""
        assert is_lgkm_op("s_load_dwordx2") is True
        assert is_lgkm_op("s_load_dwordx4") is True
        assert is_lgkm_op("s_load_dword") is True

    def test_is_lgkm_op_s_store(self):
        """Test s_store is LGKM operation."""
        assert is_lgkm_op("s_store_dword") is True
        assert is_lgkm_op("s_store_dwordx2") is True

    def test_is_lgkm_op_ds(self):
        """Test ds_* are LGKM operations."""
        assert is_lgkm_op("ds_read_b32") is True
        assert is_lgkm_op("ds_read_b128") is True
        assert is_lgkm_op("ds_write_b32") is True
        assert is_lgkm_op("ds_write_b128") is True
        assert is_lgkm_op("ds_bpermute_b32") is True
        assert is_lgkm_op("ds_swizzle_b32") is True

    def test_is_lgkm_op_not_vm(self):
        """Test that VM operations are not LGKM."""
        assert is_lgkm_op("global_load_dwordx4") is False
        assert is_lgkm_op("buffer_load_dword") is False

    def test_is_vm_op_global_load(self):
        """Test global_load is VM operation."""
        assert is_vm_op("global_load_dword") is True
        assert is_vm_op("global_load_dwordx2") is True
        assert is_vm_op("global_load_dwordx4") is True

    def test_is_vm_op_global_store(self):
        """Test global_store is VM operation."""
        assert is_vm_op("global_store_dword") is True
        assert is_vm_op("global_store_dwordx4") is True

    def test_is_vm_op_buffer_load(self):
        """Test buffer_load is VM operation."""
        assert is_vm_op("buffer_load_dword") is True
        assert is_vm_op("buffer_load_dwordx4") is True

    def test_is_vm_op_buffer_store(self):
        """Test buffer_store is VM operation."""
        assert is_vm_op("buffer_store_dword") is True
        assert is_vm_op("buffer_store_dwordx2") is True

    def test_is_vm_op_not_lgkm(self):
        """Test that LGKM operations are not VM."""
        assert is_vm_op("s_load_dwordx2") is False
        assert is_vm_op("ds_read_b128") is False


# =============================================================================
# s_waitcnt Parsing Tests
# =============================================================================

class TestParseWaitcntOperands:
    """Tests for s_waitcnt operand parsing."""

    def test_vmcnt_only(self):
        """Test parsing vmcnt only."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("vmcnt(0)")
        assert vmcnt == 0
        assert lgkmcnt is None

    def test_lgkmcnt_only(self):
        """Test parsing lgkmcnt only."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("lgkmcnt(0)")
        assert vmcnt is None
        assert lgkmcnt == 0

    def test_both_counts(self):
        """Test parsing both vmcnt and lgkmcnt."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("vmcnt(3) lgkmcnt(2)")
        assert vmcnt == 3
        assert lgkmcnt == 2

    def test_both_counts_reversed(self):
        """Test parsing both counts in reverse order."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("lgkmcnt(1) vmcnt(5)")
        assert vmcnt == 5
        assert lgkmcnt == 1

    def test_wait_all(self):
        """Test parsing wait all (just 0)."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("0")
        assert vmcnt == 0
        assert lgkmcnt == 0

    def test_non_zero_counts(self):
        """Test parsing non-zero counts."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("vmcnt(15) lgkmcnt(7)")
        assert vmcnt == 15
        assert lgkmcnt == 7

    def test_spaces_in_operands(self):
        """Test parsing with extra spaces."""
        vmcnt, lgkmcnt = parse_waitcnt_operands("vmcnt( 3 ) lgkmcnt( 2 )")
        assert vmcnt == 3
        assert lgkmcnt == 2


# =============================================================================
# DDG Data Structure Tests
# =============================================================================

class TestDDGStructures:
    """Tests for DDG data structures."""

    def test_instruction_node_creation(self):
        """Test creating InstructionNode."""
        instr = Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        )
        node = InstructionNode(
            instr=instr,
            node_id=0,
            defs={"v0"},
            uses={"v1", "v2"}
        )
        assert node.node_id == 0
        assert "v0" in node.defs
        assert "v1" in node.uses
        assert "v2" in node.uses

    def test_instruction_node_hash(self):
        """Test InstructionNode hashing."""
        instr = Instruction(address=10, opcode="s_nop", operands="0", raw_line="\ts_nop 0")
        node1 = InstructionNode(instr=instr, node_id=5)
        node2 = InstructionNode(instr=instr, node_id=5)
        node3 = InstructionNode(instr=instr, node_id=6)
        
        assert hash(node1) == hash(node2)  # Same node_id
        assert hash(node1) != hash(node3)  # Different node_id

    def test_ddg_creation(self):
        """Test creating DDG."""
        ddg = DDG(block_label=".LBB0_0")
        assert ddg.block_label == ".LBB0_0"
        assert ddg.nodes == []
        assert ddg.edges == []

    def test_ddg_critical_path_empty(self):
        """Test critical path length on empty DDG."""
        ddg = DDG(block_label=".LBB0_0")
        assert ddg.get_critical_path_length() == 0

    def test_pending_mem_op_creation(self):
        """Test creating PendingMemOp."""
        op = PendingMemOp(
            regs={"v0", "v1", "v2", "v3"},
            block_label=".LBB0_0",
            node_id=5,
            op_type="vm",
            instr_text="global_load_dwordx4 v[0:3], v[4:5], off"
        )
        assert "v0" in op.regs
        assert op.block_label == ".LBB0_0"
        assert op.op_type == "vm"

    def test_inter_block_dep_creation(self):
        """Test creating InterBlockDep."""
        dep = InterBlockDep(
            from_block=".LBB0_0",
            to_block=".LBB0_1",
            registers={"v0", "s0"}
        )
        assert dep.from_block == ".LBB0_0"
        assert dep.to_block == ".LBB0_1"
        assert "v0" in dep.registers


# =============================================================================
# DDG Serialization Tests
# =============================================================================

class TestDDGSerialization:
    """Tests for DDG serialization."""

    def test_instruction_node_to_dict(self):
        """Test InstructionNode serialization."""
        instr = Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        )
        node = InstructionNode(
            instr=instr,
            node_id=0,
            defs={"v0"},
            uses={"v1", "v2"}
        )
        d = node.to_dict()
        assert d['node_id'] == 0
        assert 'v0' in d['defs']
        assert 'v1' in d['uses']

    def test_pending_mem_op_to_dict(self):
        """Test PendingMemOp serialization."""
        op = PendingMemOp(
            regs={"v0", "v1"},
            block_label=".LBB0_0",
            node_id=3,
            op_type="lgkm"
        )
        d = op.to_dict()
        assert d['block_label'] == ".LBB0_0"
        assert d['op_type'] == "lgkm"

    def test_pending_mem_op_from_dict(self):
        """Test PendingMemOp deserialization."""
        d = {
            'regs': ['v0', 'v1'],
            'block_label': '.LBB0_1',
            'node_id': 5,
            'op_type': 'vm',
            'instr_text': 'test'
        }
        op = PendingMemOp.from_dict(d)
        assert op.block_label == '.LBB0_1'
        assert 'v0' in op.regs

    def test_inter_block_dep_to_dict(self):
        """Test InterBlockDep serialization."""
        dep = InterBlockDep(
            from_block=".LBB0_0",
            to_block=".LBB0_1",
            registers={"v0", "s0"}
        )
        d = dep.to_dict()
        assert d['from_block'] == ".LBB0_0"
        assert 'v0' in d['registers']

    def test_inter_block_dep_from_dict(self):
        """Test InterBlockDep deserialization."""
        d = {
            'from_block': '.LBB0_2',
            'to_block': '.LBB0_3',
            'registers': ['s0', 's1']
        }
        dep = InterBlockDep.from_dict(d)
        assert dep.from_block == '.LBB0_2'
        assert 's0' in dep.registers

    def test_ddg_to_dict(self):
        """Test DDG serialization."""
        ddg = DDG(block_label=".LBB0_0")
        ddg.live_in = {"v0", "s0"}
        ddg.live_out = {"v1", "s1"}
        d = ddg.to_dict()
        assert d['block_label'] == ".LBB0_0"
        assert 'v0' in d['live_in']
        assert 's1' in d['live_out']


# =============================================================================
# DDG Building Tests
# =============================================================================

class TestBuildDDG:
    """Tests for DDG building."""

    def test_build_ddg_empty_block(self):
        """Test building DDG for empty block."""
        block = BasicBlock(label=".LBB0_0")
        ddg = build_ddg(block)
        assert ddg.block_label == ".LBB0_0"
        assert len(ddg.nodes) == 0
        assert len(ddg.edges) == 0

    def test_build_ddg_single_instruction(self):
        """Test building DDG for single instruction."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        ))
        ddg = build_ddg(block)
        assert len(ddg.nodes) == 1
        assert "v1" in ddg.live_in
        assert "v2" in ddg.live_in

    def test_build_ddg_raw_dependency(self):
        """Test DDG detects RAW dependency."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="v_mov_b32", operands="v0, 1.0",
            raw_line="\tv_mov_b32 v0, 1.0"
        ))
        block.instructions.append(Instruction(
            address=11, opcode="v_add_f32", operands="v1, v0, v2",
            raw_line="\tv_add_f32 v1, v0, v2"
        ))
        ddg = build_ddg(block)
        assert len(ddg.nodes) == 2
        # Check for RAW edge (0 -> 1 on v0)
        raw_edges = [e for e in ddg.edges if e[2].startswith("RAW:")]
        assert len(raw_edges) > 0

    def test_build_ddg_live_in(self):
        """Test DDG calculates live-in correctly."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        ))
        ddg = build_ddg(block)
        assert "v1" in ddg.live_in  # Used before defined
        assert "v2" in ddg.live_in
        assert "v0" not in ddg.live_in  # Defined first

    def test_build_ddg_memory_tracking(self):
        """Test DDG tracks memory operations."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="global_load_dwordx4", operands="v[0:3], v[4:5], off",
            raw_line="\tglobal_load_dwordx4 v[0:3], v[4:5], off"
        ))
        block.instructions.append(Instruction(
            address=11, opcode="ds_read_b128", operands="v[6:9], v10",
            raw_line="\tds_read_b128 v[6:9], v10"
        ))
        ddg = build_ddg(block)
        assert len(ddg.vm_ops) == 1  # global_load
        assert len(ddg.lgkm_ops) == 1  # ds_read


# =============================================================================
# Integration Tests with Real Assembly
# =============================================================================

class TestDDGIntegration:
    """Integration tests using real assembly file."""

    @pytest.fixture
    def parsed_data(self):
        """Parse the test assembly file and generate DDGs."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test assembly file not found: {TEST_ASSEMBLY_FILE}")
        parser = AMDGCNParser()
        cfg = parser.parse_file(str(TEST_ASSEMBLY_FILE))
        ddgs, waitcnt_deps = generate_all_ddgs(cfg)
        return cfg, ddgs, waitcnt_deps

    def test_generate_ddgs_for_all_blocks(self, parsed_data):
        """Test that DDGs are generated for all blocks."""
        cfg, ddgs, _ = parsed_data
        assert len(ddgs) > 0
        # All blocks should have DDGs
        for label in cfg.block_order:
            if label in cfg.blocks and cfg.blocks[label].instructions:
                assert label in ddgs, f"Missing DDG for {label}"

    def test_ddg_nodes_match_instructions(self, parsed_data):
        """Test that DDG node count matches instruction count."""
        cfg, ddgs, _ = parsed_data
        for label, ddg in ddgs.items():
            if label in cfg.blocks:
                block = cfg.blocks[label]
                assert len(ddg.nodes) == len(block.instructions)

    def test_ddg_has_mfma_instructions(self, parsed_data):
        """Test that MFMA instructions are tracked."""
        _, ddgs, _ = parsed_data
        has_mfma = False
        for ddg in ddgs.values():
            for node in ddg.nodes:
                if 'mfma' in node.instr.opcode.lower():
                    has_mfma = True
                    # MFMA should have AGPR defs
                    assert any(r.startswith('a') for r in node.defs)
                    break
        assert has_mfma

    def test_ddg_has_memory_ops(self, parsed_data):
        """Test that memory operations are tracked."""
        _, ddgs, _ = parsed_data
        total_vm_ops = sum(len(ddg.vm_ops) for ddg in ddgs.values())
        total_lgkm_ops = sum(len(ddg.lgkm_ops) for ddg in ddgs.values())
        assert total_vm_ops > 0
        assert total_lgkm_ops > 0

    def test_ddg_critical_path(self, parsed_data):
        """Test that critical paths are computed."""
        _, ddgs, _ = parsed_data
        for ddg in ddgs.values():
            if len(ddg.nodes) > 0:
                crit_path = ddg.get_critical_path_length()
                assert crit_path >= 0


class TestComputeInterBlockDeps:
    """Tests for inter-block dependency computation."""

    @pytest.fixture
    def parsed_data(self):
        """Parse the test assembly file and generate DDGs."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test assembly file not found: {TEST_ASSEMBLY_FILE}")
        parser = AMDGCNParser()
        cfg = parser.parse_file(str(TEST_ASSEMBLY_FILE))
        ddgs, _ = generate_all_ddgs(cfg)
        return cfg, ddgs

    def test_compute_inter_block_deps(self, parsed_data):
        """Test computing inter-block dependencies."""
        cfg, ddgs = parsed_data
        deps = compute_inter_block_deps(cfg, ddgs)
        # Should have some dependencies for non-trivial CFGs
        assert deps is not None


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestDDGEdgeCases:
    """Tests for edge cases in DDG building."""

    def test_block_with_only_nops(self):
        """Test block containing only s_nop instructions."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="s_nop", operands="0",
            raw_line="\ts_nop 0"
        ))
        ddg = build_ddg(block)
        assert len(ddg.nodes) == 1
        assert len(ddg.nodes[0].defs) == 0
        assert len(ddg.nodes[0].uses) == 0

    def test_block_with_only_waitcnt(self):
        """Test block containing only s_waitcnt."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="s_waitcnt", operands="vmcnt(0) lgkmcnt(0)",
            raw_line="\ts_waitcnt vmcnt(0) lgkmcnt(0)"
        ))
        ddg = build_ddg(block)
        assert len(ddg.nodes) == 1

    def test_block_with_barrier(self):
        """Test block containing s_barrier."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="s_barrier", operands="",
            raw_line="\ts_barrier"
        ))
        ddg = build_ddg(block)
        assert len(ddg.nodes) == 1

    def test_large_register_range(self):
        """Test instruction with large register range."""
        instr = Instruction(
            address=10, opcode="global_load_dwordx4",
            operands="v[200:203], v[204:205], off",
            raw_line="\tglobal_load_dwordx4 v[200:203], v[204:205], off"
        )
        defs, uses = parse_instruction_registers(instr)
        assert "v200" in defs
        assert "v203" in defs
        assert len([r for r in defs if r.startswith('v')]) == 4

    def test_chain_of_dependencies(self):
        """Test chain of RAW dependencies."""
        block = BasicBlock(label=".LBB0_0")
        # v0 <- defined
        block.instructions.append(Instruction(
            address=10, opcode="v_mov_b32", operands="v0, 1.0",
            raw_line="\tv_mov_b32 v0, 1.0"
        ))
        # v1 <- v0 (depends on v0)
        block.instructions.append(Instruction(
            address=11, opcode="v_mov_b32", operands="v1, v0",
            raw_line="\tv_mov_b32 v1, v0"
        ))
        # v2 <- v1 (depends on v1)
        block.instructions.append(Instruction(
            address=12, opcode="v_mov_b32", operands="v2, v1",
            raw_line="\tv_mov_b32 v2, v1"
        ))
        ddg = build_ddg(block)
        assert len(ddg.nodes) == 3
        raw_edges = [e for e in ddg.edges if e[2].startswith("RAW:")]
        assert len(raw_edges) >= 2  # v0->v1, v1->v2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

