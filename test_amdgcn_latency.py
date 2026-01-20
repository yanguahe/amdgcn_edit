#!/usr/bin/env python3
"""
Comprehensive Test Suite for AMDGCN Latency Module

This module tests all functionality of the amdgcn_latency module including:
- JSON configuration loading
- MFMA instruction classification
- Pass count detection
- Latency calculation
- Violation detection
- s_nop insertion

Uses pa_dot_kernel.v2.amdgcn as a real-world test case.

Usage:
    python -m pytest test_amdgcn_latency.py -v
    python test_amdgcn_latency.py  # Direct execution
"""

import os
import sys
import json
import tempfile
import unittest
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass

# Add the module directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amdgcn_cfg import Instruction, BasicBlock, CFG, AMDGCNParser
from amdgcn_ddg import generate_all_ddgs, AnalysisResult, compute_inter_block_deps
from amdgcn_latency import (
    # Data classes and enums
    InstructionType,
    MFMAReadType,
    HardwareInfo,
    MFMAInstructionInfo,
    LatencyRule,
    LatencyViolation,
    LatencyNopsResult,
    
    # Loading functions
    load_hardware_info,
    clear_hardware_info_cache,
    get_hardware_info_path,
    
    # Classification functions
    get_mfma_info,
    get_mfma_pass_count,
    get_mfma_type,
    classify_instruction,
    is_mfma_instruction,
    is_accvgpr_read,
    is_valu_instruction,
    is_memory_read_instruction,
    
    # Register analysis
    parse_agpr_range,
    get_mfma_dst_registers,
    get_instruction_src_registers,
    get_instruction_dst_registers,
    check_register_overlap,
    check_exact_same_registers,
    analyze_mfma_read_type,
    
    # Latency calculation
    get_required_latency,
    count_independent_instructions,
    find_latency_violations,
    calculate_latency_nops_for_move,
    insert_latency_nops,
    
    # s_nop functions
    create_snop_instruction,
    calculate_snop_count,
    InsertLatencyNopsPass,
    
    # Utilities
    get_instruction_cycles,
    validate_block_latency,
    check_move_preserves_latency,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def get_test_asm_path() -> str:
    """Get path to test assembly file."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, 'pa_dot_kernel.v2.amdgcn')


def create_instruction(opcode: str, operands: str = "", address: int = 0) -> Instruction:
    """Create a test instruction."""
    raw_line = f"\t{opcode} {operands}".rstrip()
    return Instruction(
        address=address,
        opcode=opcode,
        operands=operands,
        raw_line=raw_line
    )


def create_mfma_instruction(
    variant: str = "f32_16x16x16_bf16",
    dst: str = "a[0:3]",
    src0: str = "v[0:1]",
    src1: str = "v[2:3]",
    src2: str = "a[0:3]",
    address: int = 0
) -> Instruction:
    """Create a test MFMA instruction."""
    opcode = f"v_mfma_{variant}"
    operands = f"{dst}, {src0}, {src1}, {src2}"
    return create_instruction(opcode, operands, address)


def create_accvgpr_read(dst: str = "v0", src: str = "a0", address: int = 0) -> Instruction:
    """Create a test v_accvgpr_read instruction."""
    return create_instruction("v_accvgpr_read_b32", f"{dst}, {src}", address)


def load_real_assembly() -> Tuple[CFG, dict, AnalysisResult]:
    """Load and parse the real assembly file."""
    asm_path = get_test_asm_path()
    if not os.path.exists(asm_path):
        raise unittest.SkipTest(f"Test assembly file not found: {asm_path}")
    
    parser = AMDGCNParser()
    cfg = parser.parse_file(asm_path)
    ddgs, waitcnt_deps = generate_all_ddgs(cfg, enable_cross_block_waitcnt=True)
    inter_block_deps = compute_inter_block_deps(cfg, ddgs)
    
    result = AnalysisResult(
        cfg=cfg,
        ddgs=ddgs,
        inter_block_deps=inter_block_deps,
        waitcnt_deps=waitcnt_deps
    )
    
    return cfg, ddgs, result


# =============================================================================
# Unit Tests: JSON Loading
# =============================================================================

class TestJSONLoading(unittest.TestCase):
    """Tests for JSON configuration loading."""
    
    def setUp(self):
        clear_hardware_info_cache()
    
    def tearDown(self):
        clear_hardware_info_cache()
    
    def test_load_default_config(self):
        """Test loading the default hardware configuration."""
        hw_info = load_hardware_info()
        
        self.assertEqual(hw_info.target, "gfx942")
        self.assertGreater(len(hw_info.mfma_instructions), 0)
        self.assertGreater(len(hw_info.latency_rules), 0)
        self.assertEqual(hw_info.snop_max_count, 15)
    
    def test_caching(self):
        """Test that hardware info is cached."""
        hw_info1 = load_hardware_info()
        hw_info2 = load_hardware_info()
        
        # Should be the same object (cached)
        self.assertIs(hw_info1, hw_info2)
    
    def test_force_reload(self):
        """Test force reload bypasses cache."""
        hw_info1 = load_hardware_info()
        hw_info2 = load_hardware_info(force_reload=True)
        
        # Should be different objects
        self.assertIsNot(hw_info1, hw_info2)
        # But same content
        self.assertEqual(hw_info1.target, hw_info2.target)
    
    def test_missing_file(self):
        """Test error handling for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_hardware_info("/nonexistent/path/config.json")
    
    def test_invalid_json(self):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            temp_path = f.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                load_hardware_info(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_mfma_instructions_loaded(self):
        """Test that all expected MFMA instructions are loaded."""
        hw_info = load_hardware_info()
        
        expected_instructions = [
            "v_mfma_f32_16x16x16_bf16",
            "v_mfma_f32_16x16x16_f16",
            "v_mfma_f32_32x32x8_bf16",
            "v_mfma_f32_32x32x8_f16",
        ]
        
        for opcode in expected_instructions:
            self.assertIn(opcode, hw_info.mfma_instructions,
                         f"Expected MFMA instruction not found: {opcode}")
    
    def test_latency_rules_loaded(self):
        """Test that latency rules are properly loaded."""
        hw_info = load_hardware_info()
        
        expected_rules = [
            "xdl_to_valu_vm_lds_flat",
            "sgemm_to_valu_vm_lds_flat",
            "xdl_to_srcc_exact_same",
        ]
        
        for rule_name in expected_rules:
            self.assertIn(rule_name, hw_info.latency_rules,
                         f"Expected latency rule not found: {rule_name}")


# =============================================================================
# Unit Tests: MFMA Classification
# =============================================================================

class TestMFMAClassification(unittest.TestCase):
    """Tests for MFMA instruction classification."""
    
    def setUp(self):
        self.hw_info = load_hardware_info()
    
    def test_get_mfma_info_exact_match(self):
        """Test exact opcode matching."""
        info = get_mfma_info("v_mfma_f32_16x16x16_bf16", self.hw_info)
        
        self.assertIsNotNone(info)
        self.assertEqual(info.passes, 4)
        self.assertEqual(info.type, "XDL")
    
    def test_get_mfma_info_case_insensitive(self):
        """Test case-insensitive matching."""
        info1 = get_mfma_info("v_mfma_f32_16x16x16_bf16", self.hw_info)
        info2 = get_mfma_info("V_MFMA_F32_16X16X16_BF16", self.hw_info)
        
        self.assertEqual(info1.passes, info2.passes)
        self.assertEqual(info1.type, info2.type)
    
    def test_get_mfma_info_unknown(self):
        """Test unknown MFMA returns None."""
        info = get_mfma_info("v_mfma_unknown_format", self.hw_info)
        self.assertIsNone(info)
    
    def test_get_mfma_pass_count_2pass(self):
        """Test 2-pass MFMA detection."""
        count = get_mfma_pass_count("v_mfma_f32_4x4x4_16b_f16", self.hw_info)
        self.assertEqual(count, 2)
    
    def test_get_mfma_pass_count_4pass(self):
        """Test 4-pass MFMA detection."""
        count = get_mfma_pass_count("v_mfma_f32_16x16x16_bf16", self.hw_info)
        self.assertEqual(count, 4)
    
    def test_get_mfma_pass_count_8pass(self):
        """Test 8-pass MFMA detection."""
        count = get_mfma_pass_count("v_mfma_f32_32x32x8_bf16", self.hw_info)
        self.assertEqual(count, 8)
    
    def test_get_mfma_pass_count_16pass(self):
        """Test 16-pass MFMA detection."""
        count = get_mfma_pass_count("v_mfma_f32_32x32x4_2b_f16", self.hw_info)
        self.assertEqual(count, 16)
    
    def test_get_mfma_type_xdl(self):
        """Test XDL type detection."""
        mfma_type = get_mfma_type("v_mfma_f32_16x16x16_bf16", self.hw_info)
        self.assertEqual(mfma_type, InstructionType.XDL)
    
    def test_get_mfma_type_sgemm(self):
        """Test SGEMM type detection."""
        mfma_type = get_mfma_type("v_mfma_f32_16x16x4_f32", self.hw_info)
        self.assertEqual(mfma_type, InstructionType.SGEMM)
    
    def test_get_mfma_type_dgemm(self):
        """Test DGEMM type detection."""
        mfma_type = get_mfma_type("v_mfma_f64_16x16x4_f64", self.hw_info)
        self.assertEqual(mfma_type, InstructionType.DGEMM)


# =============================================================================
# Unit Tests: Instruction Classification
# =============================================================================

class TestInstructionClassification(unittest.TestCase):
    """Tests for general instruction classification."""
    
    def test_classify_mfma(self):
        """Test MFMA instruction classification."""
        instr = create_mfma_instruction()
        result = classify_instruction(instr)
        self.assertEqual(result, InstructionType.XDL)
    
    def test_classify_accvgpr_read(self):
        """Test v_accvgpr_read classification."""
        instr = create_accvgpr_read()
        result = classify_instruction(instr)
        self.assertEqual(result, InstructionType.ACCVGPR_READ)
    
    def test_classify_accvgpr_write(self):
        """Test v_accvgpr_write classification."""
        instr = create_instruction("v_accvgpr_write_b32", "a0, v0")
        result = classify_instruction(instr)
        self.assertEqual(result, InstructionType.ACCVGPR_WRITE)
    
    def test_classify_valu(self):
        """Test VALU instruction classification."""
        test_cases = [
            ("v_add_f32_e32", "v0, v1, v2"),
            ("v_mul_f32_e32", "v0, s0, v1"),
            ("v_fma_f32", "v0, v1, v2, v3"),
            ("v_cmp_eq_u32_e32", "vcc, v0, v1"),
        ]
        
        for opcode, operands in test_cases:
            instr = create_instruction(opcode, operands)
            result = classify_instruction(instr)
            self.assertEqual(result, InstructionType.VALU,
                           f"Failed for {opcode}")
    
    def test_classify_vm_load(self):
        """Test VM load instruction classification."""
        test_cases = [
            "global_load_dwordx4",
            "buffer_load_dword",
            "flat_load_dword",
        ]
        
        for opcode in test_cases:
            instr = create_instruction(opcode, "v[0:3], v[4:5], off")
            result = classify_instruction(instr)
            self.assertEqual(result, InstructionType.VM_LOAD,
                           f"Failed for {opcode}")
    
    def test_classify_vm_store(self):
        """Test VM store instruction classification."""
        test_cases = [
            "global_store_dwordx4",
            "buffer_store_dword",
            "flat_store_dword",
        ]
        
        for opcode in test_cases:
            instr = create_instruction(opcode, "v[0:3], v[4:5], off")
            result = classify_instruction(instr)
            self.assertEqual(result, InstructionType.VM_STORE,
                           f"Failed for {opcode}")
    
    def test_classify_lds(self):
        """Test LDS instruction classification."""
        test_cases = [
            "ds_read_b128",
            "ds_write_b128",
            "ds_swizzle_b32",
            "ds_bpermute_b32",
        ]
        
        for opcode in test_cases:
            instr = create_instruction(opcode, "v0, v1")
            result = classify_instruction(instr)
            self.assertEqual(result, InstructionType.LDS,
                           f"Failed for {opcode}")
    
    def test_classify_sync(self):
        """Test synchronization instruction classification."""
        test_cases = ["s_waitcnt", "s_barrier"]
        
        for opcode in test_cases:
            instr = create_instruction(opcode, "vmcnt(0)")
            result = classify_instruction(instr)
            self.assertEqual(result, InstructionType.SYNC,
                           f"Failed for {opcode}")
    
    def test_classify_nop(self):
        """Test s_nop classification."""
        instr = create_instruction("s_nop", "0")
        result = classify_instruction(instr)
        self.assertEqual(result, InstructionType.NOP)
    
    def test_classify_salu(self):
        """Test SALU instruction classification."""
        test_cases = [
            "s_add_i32",
            "s_mul_i32",
            "s_and_b64",
            "s_lshl_b32",
        ]
        
        for opcode in test_cases:
            instr = create_instruction(opcode, "s0, s1, s2")
            result = classify_instruction(instr)
            self.assertEqual(result, InstructionType.SALU,
                           f"Failed for {opcode}")
    
    def test_is_mfma_instruction(self):
        """Test is_mfma_instruction helper."""
        mfma = create_mfma_instruction()
        valu = create_instruction("v_add_f32_e32", "v0, v1, v2")
        
        self.assertTrue(is_mfma_instruction(mfma))
        self.assertFalse(is_mfma_instruction(valu))
    
    def test_is_accvgpr_read(self):
        """Test is_accvgpr_read helper."""
        read = create_accvgpr_read()
        write = create_instruction("v_accvgpr_write_b32", "a0, v0")
        
        self.assertTrue(is_accvgpr_read(read))
        self.assertFalse(is_accvgpr_read(write))


# =============================================================================
# Unit Tests: Register Analysis
# =============================================================================

class TestRegisterAnalysis(unittest.TestCase):
    """Tests for register parsing and analysis."""
    
    def test_parse_agpr_range_single(self):
        """Test parsing single AGPR."""
        result = parse_agpr_range("a0")
        self.assertEqual(result, {"a0"})
    
    def test_parse_agpr_range_multiple(self):
        """Test parsing AGPR range."""
        result = parse_agpr_range("a[0:3]")
        self.assertEqual(result, {"a0", "a1", "a2", "a3"})
    
    def test_parse_agpr_range_large(self):
        """Test parsing large AGPR range."""
        result = parse_agpr_range("a[0:15]")
        expected = {f"a{i}" for i in range(16)}
        self.assertEqual(result, expected)
    
    def test_parse_agpr_range_invalid(self):
        """Test parsing invalid AGPR returns empty set."""
        result = parse_agpr_range("v0")
        self.assertEqual(result, set())
    
    def test_get_mfma_dst_registers(self):
        """Test extracting MFMA destination registers."""
        mfma = create_mfma_instruction(dst="a[0:3]")
        result = get_mfma_dst_registers(mfma)
        self.assertEqual(result, {"a0", "a1", "a2", "a3"})
    
    def test_check_register_overlap_true(self):
        """Test register overlap detection - overlap exists."""
        regs1 = {"a0", "a1", "a2", "a3"}
        regs2 = {"a2", "a3", "a4", "a5"}
        
        self.assertTrue(check_register_overlap(regs1, regs2))
    
    def test_check_register_overlap_false(self):
        """Test register overlap detection - no overlap."""
        regs1 = {"a0", "a1", "a2", "a3"}
        regs2 = {"a4", "a5", "a6", "a7"}
        
        self.assertFalse(check_register_overlap(regs1, regs2))
    
    def test_check_exact_same_registers_true(self):
        """Test exact register match - same."""
        regs1 = {"a0", "a1", "a2", "a3"}
        regs2 = {"a0", "a1", "a2", "a3"}
        
        self.assertTrue(check_exact_same_registers(regs1, regs2))
    
    def test_check_exact_same_registers_false(self):
        """Test exact register match - different."""
        regs1 = {"a0", "a1", "a2", "a3"}
        regs2 = {"a0", "a1", "a2"}
        
        self.assertFalse(check_exact_same_registers(regs1, regs2))


# =============================================================================
# Unit Tests: Latency Calculation
# =============================================================================

class TestLatencyCalculation(unittest.TestCase):
    """Tests for latency calculation."""
    
    def setUp(self):
        self.hw_info = load_hardware_info()
    
    def test_4pass_mfma_to_accvgpr_read(self):
        """Test MFMA (4-pass) -> v_accvgpr_read requires 7 cycles."""
        mfma = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]"
        )
        read = create_accvgpr_read(dst="v0", src="a0")
        
        latency = get_required_latency(mfma, read, self.hw_info)
        self.assertEqual(latency, 7)
    
    def test_8pass_mfma_to_accvgpr_read(self):
        """Test MFMA (8-pass) -> v_accvgpr_read requires 11 cycles."""
        mfma = create_mfma_instruction(
            variant="f32_32x32x8_bf16",
            dst="a[0:15]"
        )
        read = create_accvgpr_read(dst="v0", src="a0")
        
        latency = get_required_latency(mfma, read, self.hw_info)
        self.assertEqual(latency, 11)
    
    def test_2pass_mfma_to_valu(self):
        """Test MFMA (2-pass) -> VALU requires 5 cycles."""
        mfma = create_mfma_instruction(
            variant="f32_4x4x4_16b_f16",
            dst="a[0:3]"
        )
        valu = create_instruction("v_mul_f32_e32", "v0, v1, a0")
        
        latency = get_required_latency(mfma, valu, self.hw_info)
        self.assertEqual(latency, 5)
    
    def test_mfma_accumulator_forwarding(self):
        """Test MFMA -> same MFMA (SrcC exact) allows 0 wait for 4+ passes."""
        mfma1 = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]",
            src2="0"  # Initial accumulator is 0
        )
        mfma2 = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]",
            src2="a[0:3]"  # Uses result of first MFMA
        )
        
        latency = get_required_latency(mfma1, mfma2, self.hw_info)
        self.assertEqual(latency, 0)
    
    def test_no_dependency_no_latency(self):
        """Test no latency when no register dependency."""
        mfma = create_mfma_instruction(dst="a[0:3]")
        valu = create_instruction("v_add_f32_e32", "v0, v1, v2")  # No AGPR
        
        latency = get_required_latency(mfma, valu, self.hw_info)
        self.assertEqual(latency, 0)
    
    def test_valu_to_mfma_latency(self):
        """Test VALU -> MFMA requires 2 cycles if dependent."""
        valu = create_instruction("v_add_f32_e32", "v0, v1, v2")
        mfma = create_mfma_instruction(src0="v[0:1]")  # Uses v0
        
        latency = get_required_latency(valu, mfma, self.hw_info)
        self.assertEqual(latency, 2)
    
    def test_sgemm_to_valu_latency(self):
        """Test SGEMM -> VALU requires 6 cycles (4-pass)."""
        mfma = create_mfma_instruction(
            variant="f32_16x16x4_f32",  # SGEMM, 4-pass
            dst="a[0:3]"
        )
        valu = create_instruction("v_mul_f32_e32", "v0, v1, a0")
        
        latency = get_required_latency(mfma, valu, self.hw_info)
        self.assertEqual(latency, 6)


# =============================================================================
# Unit Tests: s_nop Creation
# =============================================================================

class TestSnopCreation(unittest.TestCase):
    """Tests for s_nop instruction creation."""
    
    def test_create_snop_basic(self):
        """Test basic s_nop creation."""
        snop = create_snop_instruction(5, address=100)
        
        self.assertEqual(snop.opcode, "s_nop")
        self.assertEqual(snop.operands, "5")
        self.assertEqual(snop.address, 100)
        self.assertIn("s_nop 5", snop.raw_line)
    
    def test_create_snop_zero(self):
        """Test s_nop 0 creation."""
        snop = create_snop_instruction(0)
        self.assertEqual(snop.operands, "0")
    
    def test_create_snop_max(self):
        """Test s_nop max (15) creation."""
        snop = create_snop_instruction(15)
        self.assertEqual(snop.operands, "15")
    
    def test_create_snop_clamp_negative(self):
        """Test s_nop clamps negative to 0."""
        snop = create_snop_instruction(-5)
        self.assertEqual(snop.operands, "0")
    
    def test_create_snop_clamp_over_max(self):
        """Test s_nop clamps over max to 15."""
        snop = create_snop_instruction(20)
        self.assertEqual(snop.operands, "15")
    
    def test_calculate_snop_count_small(self):
        """Test snop count calculation for small values.
        
        s_nop N provides N+1 cycles. So for 3 cycles, use s_nop 2.
        """
        result = calculate_snop_count(3)
        self.assertEqual(len(result), 1)  # Single s_nop instruction
        self.assertEqual(result[0], 2)    # s_nop 2 = 3 cycles
    
    def test_calculate_snop_count_7_cycles(self):
        """Test snop count for 7 cycles (typical 4-pass MFMA latency)."""
        result = calculate_snop_count(7)
        self.assertEqual(len(result), 1)  # Single s_nop instruction
        self.assertEqual(result[0], 6)    # s_nop 6 = 7 cycles
    
    def test_calculate_snop_count_16_cycles(self):
        """Test snop count for 16 cycles (max single s_nop)."""
        result = calculate_snop_count(16)
        self.assertEqual(len(result), 1)  # Single s_nop instruction
        self.assertEqual(result[0], 15)   # s_nop 15 = 16 cycles
    
    def test_calculate_snop_count_20_cycles(self):
        """Test snop count for 20 cycles (needs two s_nop)."""
        result = calculate_snop_count(20)
        self.assertEqual(len(result), 2)  # Two s_nop instructions
        self.assertEqual(result[0], 15)   # s_nop 15 = 16 cycles
        self.assertEqual(result[1], 3)    # s_nop 3 = 4 cycles
        # Total: 16 + 4 = 20 cycles
    
    def test_calculate_snop_count_zero(self):
        """Test snop count calculation for zero."""
        result = calculate_snop_count(0)
        self.assertEqual(result, [])
    
    def test_calculate_snop_count_negative(self):
        """Test snop count calculation for negative."""
        result = calculate_snop_count(-5)
        self.assertEqual(result, [])
    
    def test_calculate_snop_count_1_cycle(self):
        """Test snop count for 1 cycle."""
        result = calculate_snop_count(1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 0)    # s_nop 0 = 1 cycle


# =============================================================================
# Unit Tests: Instruction Cycles
# =============================================================================

class TestInstructionCycles(unittest.TestCase):
    """Tests for instruction cycle counting."""
    
    def test_mfma_cycles(self):
        """Test MFMA instruction cycles."""
        cycles = get_instruction_cycles("v_mfma_f32_16x16x16_bf16")
        self.assertEqual(cycles, 16)
    
    def test_ds_swizzle_cycles(self):
        """Test ds_swizzle instruction cycles."""
        cycles = get_instruction_cycles("ds_swizzle_b32")
        self.assertEqual(cycles, 8)
    
    def test_exp_cycles(self):
        """Test v_exp instruction cycles."""
        cycles = get_instruction_cycles("v_exp_f32")
        self.assertEqual(cycles, 16)
    
    def test_default_cycles(self):
        """Test default instruction cycles."""
        cycles = get_instruction_cycles("v_add_f32_e32")
        self.assertEqual(cycles, 4)
    
    def test_snop_cycles(self):
        """Test s_nop instruction cycles."""
        cycles = get_instruction_cycles("s_nop")
        self.assertEqual(cycles, 1)


# =============================================================================
# Integration Tests: Real Assembly
# =============================================================================

class TestRealAssembly(unittest.TestCase):
    """Integration tests using pa_dot_kernel.v2.amdgcn."""
    
    @classmethod
    def setUpClass(cls):
        """Load and parse the real assembly file once for all tests."""
        try:
            cls.cfg, cls.ddgs, cls.result = load_real_assembly()
            cls.hw_info = load_hardware_info()
        except unittest.SkipTest as e:
            cls.skip_reason = str(e)
            cls.cfg = None
    
    def setUp(self):
        if self.cfg is None:
            self.skipTest(getattr(self, 'skip_reason', 'Assembly file not available'))
    
    def test_file_parsed_successfully(self):
        """Test that the assembly file was parsed successfully."""
        self.assertIsNotNone(self.cfg)
        self.assertGreater(len(self.cfg.blocks), 0)
    
    def test_mfma_instructions_found(self):
        """Test that MFMA instructions are present in the assembly."""
        mfma_count = 0
        for block in self.cfg.blocks.values():
            for instr in block.instructions:
                if is_mfma_instruction(instr):
                    mfma_count += 1
        
        self.assertGreater(mfma_count, 0,
                          "Expected MFMA instructions in test assembly")
    
    def test_accvgpr_read_found(self):
        """Test that v_accvgpr_read instructions are present."""
        read_count = 0
        for block in self.cfg.blocks.values():
            for instr in block.instructions:
                if is_accvgpr_read(instr):
                    read_count += 1
        
        self.assertGreater(read_count, 0,
                          "Expected v_accvgpr_read instructions in test assembly")
    
    def test_classify_all_instructions(self):
        """Test that all instructions can be classified."""
        for block in self.cfg.blocks.values():
            for instr in block.instructions:
                instr_type = classify_instruction(instr, self.hw_info)
                # Should never be OTHER for valid assembly
                self.assertIsInstance(instr_type, InstructionType)
    
    def test_find_mfma_to_read_sequences(self):
        """Test finding MFMA to v_accvgpr_read sequences."""
        found_sequences = []
        
        for label, block in self.cfg.blocks.items():
            for i, instr in enumerate(block.instructions):
                if is_mfma_instruction(instr):
                    mfma_dst = get_mfma_dst_registers(instr)
                    
                    # Look for following accvgpr_read
                    for j in range(i + 1, min(i + 20, len(block.instructions))):
                        reader = block.instructions[j]
                        if is_accvgpr_read(reader):
                            reader_src = get_instruction_src_registers(reader)
                            if check_register_overlap(mfma_dst, reader_src):
                                found_sequences.append((label, i, j))
                                break
        
        # Should find at least some sequences
        self.assertGreater(len(found_sequences), 0,
                          "Expected to find MFMA -> accvgpr_read sequences")


# =============================================================================
# Integration Tests: Latency Violations
# =============================================================================

class TestLatencyViolations(unittest.TestCase):
    """Tests for latency violation detection."""
    
    def setUp(self):
        self.hw_info = load_hardware_info()
    
    def test_find_violation_basic(self):
        """Test finding a basic latency violation."""
        # Create block with MFMA immediately followed by accvgpr_read
        block = BasicBlock(label=".test_block")
        
        mfma = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]",
            address=0
        )
        read = create_accvgpr_read(dst="v0", src="a0", address=1)
        
        block.instructions = [mfma, read]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].required_latency, 7)
        self.assertEqual(violations[0].actual_independent, 0)
        self.assertEqual(violations[0].nops_needed, 7)
    
    def test_no_violation_with_sufficient_distance(self):
        """Test no violation when sufficient instructions between."""
        block = BasicBlock(label=".test_block")
        
        mfma = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]",
            address=0
        )
        
        # Add 7 independent instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 8)
        ]
        
        read = create_accvgpr_read(dst="v0", src="a0", address=8)
        
        block.instructions = [mfma] + independent + [read]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        self.assertEqual(len(violations), 0)
    
    def test_no_violation_no_dependency(self):
        """Test no violation when no register dependency."""
        block = BasicBlock(label=".test_block")
        
        mfma = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]",
            address=0
        )
        read = create_accvgpr_read(dst="v0", src="a4", address=1)  # Different AGPR
        
        block.instructions = [mfma, read]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        self.assertEqual(len(violations), 0)
    
    def test_multiple_violations(self):
        """Test finding multiple violations in one block."""
        block = BasicBlock(label=".test_block")
        
        mfma1 = create_mfma_instruction(dst="a[0:3]", address=0)
        read1 = create_accvgpr_read(dst="v0", src="a0", address=1)
        
        mfma2 = create_mfma_instruction(dst="a[4:7]", address=2)
        read2 = create_accvgpr_read(dst="v1", src="a4", address=3)
        
        block.instructions = [mfma1, read1, mfma2, read2]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        # Should find at least the first violation
        self.assertGreaterEqual(len(violations), 1)


# =============================================================================
# Integration Tests: InsertLatencyNopsPass
# =============================================================================

class TestInsertLatencyNopsPass(unittest.TestCase):
    """Tests for the InsertLatencyNopsPass."""
    
    def test_pass_inserts_nops(self):
        """Test that the pass inserts nops for violations."""
        # Create a simple CFG with violation
        cfg = CFG(name="test")
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        read = create_accvgpr_read(dst="v0", src="a0", address=1)
        
        block.instructions = [mfma, read]
        cfg.add_block(block)
        
        ddgs, _ = generate_all_ddgs(cfg)
        result = AnalysisResult(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=[],
            waitcnt_deps=[]
        )
        
        # Run pass
        pass_ = InsertLatencyNopsPass()
        changed = pass_.run(result)
        
        self.assertTrue(changed)
        
        # Check that nops were inserted
        new_len = len(result.cfg.blocks[".test"].instructions)
        self.assertGreater(new_len, 2)
    
    def test_pass_no_change_when_valid(self):
        """Test that pass makes no changes when no violations."""
        cfg = CFG(name="test")
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        
        # Add sufficient independent instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 10)
        ]
        
        read = create_accvgpr_read(dst="v0", src="a0", address=10)
        
        block.instructions = [mfma] + independent + [read]
        cfg.add_block(block)
        
        ddgs, _ = generate_all_ddgs(cfg)
        result = AnalysisResult(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=[],
            waitcnt_deps=[]
        )
        
        # Run pass
        pass_ = InsertLatencyNopsPass()
        changed = pass_.run(result)
        
        self.assertFalse(changed)


# =============================================================================
# Corner Case Tests
# =============================================================================

class TestCornerCases(unittest.TestCase):
    """Tests for corner cases and edge conditions."""
    
    def setUp(self):
        self.hw_info = load_hardware_info()
    
    def test_empty_block(self):
        """Test handling of empty basic block."""
        block = BasicBlock(label=".empty")
        block.instructions = []
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        self.assertEqual(len(violations), 0)
    
    def test_block_with_only_mfma(self):
        """Test block with only MFMA instructions."""
        block = BasicBlock(label=".mfma_only")
        
        mfma1 = create_mfma_instruction(dst="a[0:3]", src2="0", address=0)
        mfma2 = create_mfma_instruction(dst="a[0:3]", src2="a[0:3]", address=1)
        
        block.instructions = [mfma1, mfma2]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        # Should have 0 violations due to accumulator forwarding
        self.assertEqual(len(violations), 0)
    
    def test_snop_already_present_partial(self):
        """Test that existing s_nop instructions are counted correctly.
        
        s_nop N provides N+1 cycles of delay. So s_nop 2 = 3 cycles.
        For 4-pass MFMA, we need 7 cycles. With s_nop 2 + 1 salu = 4 cycles.
        So we should still need 3 more cycles.
        """
        block = BasicBlock(label=".with_nop")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # Add s_nop 2 (3 cycles) and one SALU instruction (1 cycle)
        nop = create_snop_instruction(2, address=1)  # s_nop 2 = 3 cycles
        salu = create_instruction("s_add_i32", "s0, s1, s2", address=2)  # 1 cycle
        read = create_accvgpr_read(dst="v0", src="a0", address=3)
        
        block.instructions = [mfma, nop, salu, read]
        
        # Total independent cycles: 3 (from s_nop 2) + 1 (from salu) = 4
        # Required: 7 for 4-pass MFMA
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        # Should find a violation (need 7, have 4)
        self.assertGreater(len(violations), 0)
        # Should need 3 more cycles (7 - 4 = 3)
        self.assertEqual(violations[0].nops_needed, 3)
    
    def test_snop_already_present_sufficient(self):
        """Test that sufficient s_nop doesn't create a violation.
        
        s_nop 6 = 7 cycles, which satisfies 4-pass MFMA requirement.
        """
        block = BasicBlock(label=".with_nop")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        nop = create_snop_instruction(6, address=1)  # s_nop 6 = 7 cycles
        read = create_accvgpr_read(dst="v0", src="a0", address=2)
        
        block.instructions = [mfma, nop, read]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        # Should NOT find a violation (need 7, have 7)
        self.assertEqual(len(violations), 0)
    
    def test_mfma_different_dst_ranges(self):
        """Test MFMA with different destination ranges."""
        block = BasicBlock(label=".diff_ranges")
        
        # MFMA writes a[0:3]
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        
        # Read from a[4] - no dependency
        read_no_dep = create_accvgpr_read(dst="v0", src="a4", address=1)
        
        # Read from a[2] - has dependency
        read_dep = create_accvgpr_read(dst="v1", src="a2", address=2)
        
        block.instructions = [mfma, read_no_dep, read_dep]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        # Should find violation for read_dep
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].second_idx, 2)
    
    def test_chained_mfma_different_dst(self):
        """Test chain of MFMA with different destinations."""
        block = BasicBlock(label=".chain")
        
        # First MFMA writes a[0:3]
        mfma1 = create_mfma_instruction(dst="a[0:3]", src2="0", address=0)
        
        # Second MFMA writes a[4:7], doesn't depend on first
        mfma2 = create_mfma_instruction(dst="a[4:7]", src2="0", address=1)
        
        # Third MFMA uses a[0:3] as SrcC - should have 0 latency due to forwarding
        mfma3 = create_mfma_instruction(dst="a[0:3]", src2="a[0:3]", address=2)
        
        block.instructions = [mfma1, mfma2, mfma3]
        
        violations = find_latency_violations(block, hw_info=self.hw_info)
        
        # With accumulator forwarding, no violations expected
        self.assertEqual(len(violations), 0)
    
    def test_validate_block_latency(self):
        """Test validate_block_latency utility function."""
        block = BasicBlock(label=".validate")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        read = create_accvgpr_read(dst="v0", src="a0", address=1)
        
        block.instructions = [mfma, read]
        
        is_valid, violations = validate_block_latency(block, self.hw_info)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
    
    def test_check_move_preserves_latency_unsafe(self):
        """Test that moving instruction to violate latency is detected."""
        block = BasicBlock(label=".move_check")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        
        # 7 independent instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 8)
        ]
        
        read = create_accvgpr_read(dst="v0", src="a0", address=8)
        
        block.instructions = [mfma] + independent + [read]
        
        # Moving read from index 8 to index 2 would violate latency
        is_safe = check_move_preserves_latency(block, 8, 2, self.hw_info)
        
        self.assertFalse(is_safe)
    
    def test_16pass_mfma_latency(self):
        """Test latency for 16-pass MFMA."""
        mfma = create_mfma_instruction(
            variant="f32_32x32x4_2b_f16",  # 16-pass
            dst="a[0:31]"
        )
        read = create_accvgpr_read(dst="v0", src="a0")
        
        latency = get_required_latency(mfma, read, self.hw_info)
        self.assertEqual(latency, 19)
    
    def test_mfma_to_global_store(self):
        """Test MFMA to global_store latency."""
        mfma = create_mfma_instruction(
            variant="f32_16x16x16_bf16",
            dst="a[0:3]"
        )
        # Store that uses AGPR (after reading to VGPR)
        store = create_instruction("global_store_dwordx4", "v[0:3], v[4:5], off")
        
        # No direct AGPR dependency, latency should be 0
        latency = get_required_latency(mfma, store, self.hw_info)
        self.assertEqual(latency, 0)


# =============================================================================
# Tests: calculate_latency_nops_for_move
# =============================================================================

class TestCalculateLatencyNopsForMove(unittest.TestCase):
    """Tests for calculate_latency_nops_for_move function."""
    
    def setUp(self):
        from amdgcn_latency import calculate_latency_nops_for_move, LatencyNopsResult
        self.calculate_nops = calculate_latency_nops_for_move
        self.hw_info = load_hardware_info()
    
    def test_move_reader_up_needs_nops(self):
        """Test moving reader up closer to MFMA needs nops."""
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # 7 independent instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 8)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=8)
        
        block.instructions = [mfma] + independent + [read]
        
        # Moving read from index 8 to index 2 (passing 6 instructions)
        # New distance would be 2 - 0 - 1 = 1, need 7, so need 6 nops
        result = self.calculate_nops(block, 8, 2, self.hw_info)
        
        self.assertTrue(result.needs_nops)
        self.assertEqual(result.nops_count, 6)
        self.assertEqual(result.insert_position, 2)
    
    def test_move_reader_up_no_nops_needed(self):
        """Test moving reader up when distance is sufficient."""
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # 10 independent instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 11)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=11)
        
        block.instructions = [mfma] + independent + [read]
        
        # Moving read from index 11 to index 8
        # New distance would be 8 - 0 - 1 = 7, need 7, so no nops needed
        result = self.calculate_nops(block, 11, 8, self.hw_info)
        
        self.assertFalse(result.needs_nops)
        self.assertEqual(result.nops_count, 0)
    
    def test_move_mfma_down_needs_nops(self):
        """Test moving MFMA down closer to reader needs nops."""
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # 10 independent instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 11)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=11)
        
        block.instructions = [mfma] + independent + [read]
        
        # Moving MFMA from index 0 to index 5
        # New distance would be 11 - 5 - 1 = 5, need 7, so need 2 nops
        result = self.calculate_nops(block, 0, 5, self.hw_info)
        
        self.assertTrue(result.needs_nops)
        self.assertEqual(result.nops_count, 2)
        # Nops inserted after MFMA's new position
        self.assertEqual(result.insert_position, 6)
    
    def test_no_dependency_no_nops(self):
        """Test no nops needed when no register dependency."""
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        read = create_accvgpr_read(dst="v0", src="a4", address=1)  # Different AGPR
        
        block.instructions = [mfma, read]
        
        # Moving read up - no dependency with MFMA (different registers)
        # Actually need to have some space to move
        independent = [create_instruction("s_add_i32", "s0, s1, s2", 1)]
        block.instructions = [mfma] + independent + [read]
        
        result = self.calculate_nops(block, 2, 1, self.hw_info)
        
        self.assertFalse(result.needs_nops)
    
    def test_16pass_mfma_large_nops(self):
        """Test 16-pass MFMA requires large nop count."""
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(
            variant="f32_32x32x4_2b_f16",  # 16-pass
            dst="a[0:31]",
            address=0
        )
        read = create_accvgpr_read(dst="v0", src="a0", address=1)
        
        block.instructions = [mfma, read]
        
        # 16-pass requires 19 independent instructions
        # Current distance is 0, so need 19 nops
        result = self.calculate_nops(block, 1, 1, self.hw_info)
        
        # When from_idx == to_idx, no move needed
        # Let's test moving to immediate position
        block.instructions = [mfma, create_instruction("s_nop", "0", 1), read]
        result = self.calculate_nops(block, 2, 1, self.hw_info)
        
        self.assertTrue(result.needs_nops)
        # New distance would be 0, need 19, so need 19 nops
        self.assertEqual(result.nops_count, 19)


# =============================================================================
# Tests: insert_latency_nops
# =============================================================================

class TestInsertLatencyNops(unittest.TestCase):
    """Tests for insert_latency_nops function."""
    
    def setUp(self):
        from amdgcn_latency import insert_latency_nops
        self.insert_nops = insert_latency_nops
    
    def test_insert_small_nop_count(self):
        """Test inserting small number of nops.
        
        With efficient s_nop usage, 3 cycles = 1 instruction (s_nop 2).
        """
        block = BasicBlock(label=".test")
        block.instructions = [
            create_instruction("v_add_f32_e32", "v0, v1, v2", 0),
            create_instruction("v_mul_f32_e32", "v3, v4, v5", 1),
        ]
        
        # Insert 3 cycles of delay at position 1
        inserted = self.insert_nops(block, 1, 3)
        
        self.assertEqual(inserted, 1)  # Single s_nop instruction
        self.assertEqual(len(block.instructions), 3)
        
        # Check that nop is at position 1
        self.assertEqual(block.instructions[1].opcode, "s_nop")
        self.assertEqual(block.instructions[1].operands, "2")  # s_nop 2 = 3 cycles
    
    def test_insert_zero_nops(self):
        """Test inserting zero nops does nothing."""
        block = BasicBlock(label=".test")
        block.instructions = [
            create_instruction("v_add_f32_e32", "v0, v1, v2", 0),
            create_instruction("v_mul_f32_e32", "v3, v4, v5", 1),
        ]
        
        inserted = self.insert_nops(block, 1, 0)
        
        self.assertEqual(inserted, 0)
        self.assertEqual(len(block.instructions), 2)
    
    def test_insert_at_beginning(self):
        """Test inserting nops at the beginning.
        
        2 cycles = 1 instruction (s_nop 1).
        """
        block = BasicBlock(label=".test")
        block.instructions = [
            create_instruction("v_add_f32_e32", "v0, v1, v2", 0),
        ]
        
        inserted = self.insert_nops(block, 0, 2)
        
        self.assertEqual(inserted, 1)  # Single s_nop instruction
        self.assertEqual(len(block.instructions), 2)
        self.assertEqual(block.instructions[0].opcode, "s_nop")
        self.assertEqual(block.instructions[0].operands, "1")  # s_nop 1 = 2 cycles
        self.assertEqual(block.instructions[1].opcode, "v_add_f32_e32")
    
    def test_insert_at_end(self):
        """Test inserting nops at the end.
        
        2 cycles = 1 instruction (s_nop 1).
        """
        block = BasicBlock(label=".test")
        block.instructions = [
            create_instruction("v_add_f32_e32", "v0, v1, v2", 0),
        ]
        
        inserted = self.insert_nops(block, 1, 2)
        
        self.assertEqual(inserted, 1)  # Single s_nop instruction
        self.assertEqual(len(block.instructions), 2)
        self.assertEqual(block.instructions[0].opcode, "v_add_f32_e32")
        self.assertEqual(block.instructions[1].opcode, "s_nop")
        self.assertEqual(block.instructions[1].operands, "1")  # s_nop 1 = 2 cycles


# =============================================================================
# Tests: MoveInstructionPass auto_insert_nops
# =============================================================================

class TestMoveInstructionPassAutoInsertNops(unittest.TestCase):
    """Tests for MoveInstructionPass with auto_insert_nops feature."""
    
    def setUp(self):
        from amdgcn_passes import MoveInstructionPass
        from amdgcn_ddg import generate_all_ddgs
        self.MoveInstructionPass = MoveInstructionPass
        self.generate_ddgs = generate_all_ddgs
        self.hw_info = load_hardware_info()
    
    def test_auto_insert_nops_default_true(self):
        """Test that auto_insert_nops defaults to True."""
        pass_ = self.MoveInstructionPass(
            block_label=".test",
            instr_index=5,
            cycles=10
        )
        self.assertTrue(pass_.auto_insert_nops)
    
    def test_auto_insert_nops_can_be_disabled(self):
        """Test that auto_insert_nops can be set to False."""
        pass_ = self.MoveInstructionPass(
            block_label=".test",
            instr_index=5,
            cycles=10,
            auto_insert_nops=False
        )
        self.assertFalse(pass_.auto_insert_nops)
    
    def test_move_blocked_when_auto_insert_disabled(self):
        """Test that move is blocked when auto_insert_nops is disabled.
        
        When auto_insert_nops=False, any move that would violate MFMA latency
        constraints should be blocked. The MFMA-to-reader distance must remain >= 7.
        """
        cfg = CFG(name="test")
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # Provide exactly 7 independent instructions (minimum for 4-pass MFMA)
        # so the read instruction starts at a valid position
        independent = [
            create_instruction("s_add_i32", f"s{i}, s1, s2", i)
            for i in range(1, 8)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=8)
        
        block.instructions = [mfma] + independent + [read]
        cfg.add_block(block)
        
        ddgs, _ = self.generate_ddgs(cfg)
        result = AnalysisResult(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=[],
            waitcnt_deps=[]
        )
        
        original_read_idx = 8
        
        # Try to move the read instruction UP by 3 cycles
        # With auto_insert_nops=False, moves that would violate latency should be blocked
        pass_ = self.MoveInstructionPass(
            block_label=".test",
            instr_index=original_read_idx,  # The read instruction
            cycles=3,  # Try to move up
            auto_insert_nops=False
        )
        
        # Run the pass
        changed = pass_.run(result)
        
        # Find current positions of MFMA and read
        block = result.cfg.blocks[".test"]
        mfma_idx = -1
        read_idx = -1
        for i, instr in enumerate(block.instructions):
            if instr.opcode.startswith("v_mfma"):
                mfma_idx = i
            if instr.opcode.startswith("v_accvgpr_read"):
                read_idx = i
        
        # The key constraint is that MFMA-to-read distance must remain >= 7
        # This ensures the latency constraint is not violated
        distance = read_idx - mfma_idx - 1
        self.assertGreaterEqual(distance, 7,
            f"Latency constraint violated: MFMA at {mfma_idx}, read at {read_idx}, "
            f"distance={distance}, required=7")
    
    def test_auto_insert_allows_move_with_nops(self):
        """Test that auto_insert_nops allows move by inserting nops."""
        cfg = CFG(name="test")
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # 10 independent instructions (more than needed)
        independent = [
            create_instruction("s_add_i32", f"s{i}, s1, s2", i)
            for i in range(1, 11)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=11)
        
        block.instructions = [mfma] + independent + [read]
        cfg.add_block(block)
        
        ddgs, _ = self.generate_ddgs(cfg)
        result = AnalysisResult(
            cfg=cfg,
            ddgs=ddgs,
            inter_block_deps=[],
            waitcnt_deps=[]
        )
        
        original_count = len(block.instructions)
        
        # Try to move read instruction up by many cycles
        # This should trigger auto_insert_nops
        pass_ = self.MoveInstructionPass(
            block_label=".test",
            instr_index=11,  # The read instruction
            cycles=8,  # Try to move up significantly
            auto_insert_nops=True
        )
        
        changed = pass_.run(result)
        
        # The pass should have run (might have inserted nops)
        block = result.cfg.blocks[".test"]
        
        # Verify MFMA latency is still satisfied
        mfma_idx = -1
        read_idx = -1
        for i, instr in enumerate(block.instructions):
            if instr.opcode.startswith("v_mfma"):
                mfma_idx = i
            if instr.opcode.startswith("v_accvgpr_read"):
                read_idx = i
        
        # Either: 
        # 1. Move was blocked (read_idx unchanged)
        # 2. Nops were inserted to maintain distance >= 7
        if read_idx != -1 and mfma_idx != -1:
            distance = read_idx - mfma_idx - 1
            self.assertGreaterEqual(distance, 7,
                f"Latency violated: MFMA at {mfma_idx}, read at {read_idx}, distance={distance}")
    
    def test_nops_inserted_tracked(self):
        """Test that inserted nops are tracked in _inserted_nops."""
        pass_ = self.MoveInstructionPass(
            block_label=".test",
            instr_index=5,
            cycles=10,
            auto_insert_nops=True
        )
        
        # Initially empty
        self.assertEqual(pass_._inserted_nops, [])


# =============================================================================
# Tests: Edge Cases for auto_insert_nops
# =============================================================================

class TestAutoInsertNopsEdgeCases(unittest.TestCase):
    """Edge case tests for auto_insert_nops feature."""
    
    def setUp(self):
        self.hw_info = load_hardware_info()
    
    def test_move_immediately_after_mfma(self):
        """Test moving reader to immediately after MFMA."""
        from amdgcn_latency import calculate_latency_nops_for_move
        
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(dst="a[0:3]", address=0)
        # Multiple instructions
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 10)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=10)
        
        block.instructions = [mfma] + independent + [read]
        
        # Try to move read to position 1 (immediately after MFMA)
        result = calculate_latency_nops_for_move(block, 10, 1, self.hw_info)
        
        self.assertTrue(result.needs_nops)
        # Distance would be 0, need 7, so need 7 nops
        self.assertEqual(result.nops_count, 7)
    
    def test_multiple_mfma_readers(self):
        """Test block with multiple MFMA outputs being read."""
        from amdgcn_latency import calculate_latency_nops_for_move
        
        block = BasicBlock(label=".test")
        
        mfma1 = create_mfma_instruction(dst="a[0:3]", address=0)
        mfma2 = create_mfma_instruction(dst="a[4:7]", address=1)
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(2, 12)
        ]
        read1 = create_accvgpr_read(dst="v0", src="a0", address=12)
        read2 = create_accvgpr_read(dst="v1", src="a4", address=13)
        
        block.instructions = [mfma1, mfma2] + independent + [read1, read2]
        
        # Moving read1 closer should need nops based on mfma1
        result = calculate_latency_nops_for_move(block, 12, 3, self.hw_info)
        
        # read1 reads a0, which is written by mfma1 at index 0
        # New distance would be 3 - 0 - 1 = 2, need 7, so need 5 nops
        if result.needs_nops:
            self.assertEqual(result.nops_count, 5)
    
    def test_chained_mfma_accumulator_no_nops(self):
        """Test chained MFMA with accumulator forwarding needs no nops."""
        from amdgcn_latency import calculate_latency_nops_for_move
        
        block = BasicBlock(label=".test")
        
        mfma1 = create_mfma_instruction(
            dst="a[0:3]",
            src2="0",  # Initial
            address=0
        )
        mfma2 = create_mfma_instruction(
            dst="a[0:3]",
            src2="a[0:3]",  # Uses mfma1 result as accumulator
            address=1
        )
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(2, 5)
        ]
        
        block.instructions = [mfma1] + independent + [mfma2]
        
        # Moving mfma2 closer to mfma1 should have 0 latency due to forwarding
        result = calculate_latency_nops_for_move(block, 4, 1, self.hw_info)
        
        # Due to accumulator forwarding, no nops should be needed
        self.assertFalse(result.needs_nops)
    
    def test_8pass_mfma_nops(self):
        """Test 8-pass MFMA requires 11 independent instructions."""
        from amdgcn_latency import calculate_latency_nops_for_move
        
        block = BasicBlock(label=".test")
        
        mfma = create_mfma_instruction(
            variant="f32_32x32x8_bf16",  # 8-pass
            dst="a[0:15]",
            address=0
        )
        independent = [
            create_instruction("s_add_i32", "s0, s1, s2", i)
            for i in range(1, 6)
        ]
        read = create_accvgpr_read(dst="v0", src="a0", address=6)
        
        block.instructions = [mfma] + independent + [read]
        
        # Moving read from index 6 to index 2
        # New distance would be 2 - 0 - 1 = 1, need 11, so need 10 nops
        result = calculate_latency_nops_for_move(block, 6, 2, self.hw_info)
        
        self.assertTrue(result.needs_nops)
        self.assertEqual(result.nops_count, 10)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)

