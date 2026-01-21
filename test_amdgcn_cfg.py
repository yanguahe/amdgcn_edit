#!/usr/bin/env python3
"""
Unit tests for AMDGCN CFG (Control Flow Graph) module.

Tests cover:
- Instruction parsing and classification
- Basic block detection
- CFG construction and edge building
- Serialization/deserialization
- Integration with real assembly files
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from amdgcn_cfg import (
    Instruction,
    BasicBlock,
    CFG,
    AMDGCNParser,
    classify_instruction,
    extract_branch_target,
    UNCONDITIONAL_BRANCHES,
    CONDITIONAL_BRANCHES,
    TERMINATOR_INSTRUCTIONS,
    escape_dot_string,
    truncate_instruction,
    generate_dot,
    generate_simple_dot,
)


# =============================================================================
# Test Data - Path to real assembly file
# =============================================================================

TEST_ASSEMBLY_FILE = Path(__file__).parent / "pa_dot_kernel.v2.amdgcn"


# =============================================================================
# Instruction Parsing Tests
# =============================================================================

class TestClassifyInstruction:
    """Tests for instruction classification."""

    def test_unconditional_branch(self):
        """Test classification of unconditional branch instructions."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_branch")
        assert is_branch is True
        assert is_conditional is False
        assert is_terminator is False

    def test_unconditional_branch_setpc(self):
        """Test classification of s_setpc_b64."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_setpc_b64")
        assert is_branch is True
        assert is_conditional is False
        assert is_terminator is False

    def test_conditional_branch_scc0(self):
        """Test classification of s_cbranch_scc0."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_cbranch_scc0")
        assert is_branch is True
        assert is_conditional is True
        assert is_terminator is False

    def test_conditional_branch_scc1(self):
        """Test classification of s_cbranch_scc1."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_cbranch_scc1")
        assert is_branch is True
        assert is_conditional is True
        assert is_terminator is False

    def test_conditional_branch_vccz(self):
        """Test classification of s_cbranch_vccz."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_cbranch_vccz")
        assert is_branch is True
        assert is_conditional is True
        assert is_terminator is False

    def test_conditional_branch_vccnz(self):
        """Test classification of s_cbranch_vccnz."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_cbranch_vccnz")
        assert is_branch is True
        assert is_conditional is True
        assert is_terminator is False

    def test_conditional_branch_execz(self):
        """Test classification of s_cbranch_execz."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_cbranch_execz")
        assert is_branch is True
        assert is_conditional is True
        assert is_terminator is False

    def test_conditional_branch_execnz(self):
        """Test classification of s_cbranch_execnz."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_cbranch_execnz")
        assert is_branch is True
        assert is_conditional is True
        assert is_terminator is False

    def test_terminator_endpgm(self):
        """Test classification of s_endpgm."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_endpgm")
        assert is_branch is False
        assert is_conditional is False
        assert is_terminator is True

    def test_terminator_trap(self):
        """Test classification of s_trap."""
        is_branch, is_conditional, is_terminator = classify_instruction("s_trap")
        assert is_branch is False
        assert is_conditional is False
        assert is_terminator is True

    def test_regular_instruction(self):
        """Test classification of regular (non-control-flow) instruction."""
        is_branch, is_conditional, is_terminator = classify_instruction("v_add_f32")
        assert is_branch is False
        assert is_conditional is False
        assert is_terminator is False

    def test_memory_instruction(self):
        """Test classification of memory instruction."""
        is_branch, is_conditional, is_terminator = classify_instruction("global_load_dwordx4")
        assert is_branch is False
        assert is_conditional is False
        assert is_terminator is False

    def test_mfma_instruction(self):
        """Test classification of MFMA instruction."""
        is_branch, is_conditional, is_terminator = classify_instruction("v_mfma_f32_16x16x16_bf16")
        assert is_branch is False
        assert is_conditional is False
        assert is_terminator is False

    def test_case_insensitive(self):
        """Test that classification is case-insensitive."""
        is_branch1, is_conditional1, is_terminator1 = classify_instruction("S_BRANCH")
        is_branch2, is_conditional2, is_terminator2 = classify_instruction("s_branch")
        assert is_branch1 == is_branch2
        assert is_conditional1 == is_conditional2
        assert is_terminator1 == is_terminator2

    def test_all_unconditional_branches(self):
        """Test all unconditional branch instructions are classified correctly."""
        for opcode in UNCONDITIONAL_BRANCHES:
            is_branch, is_conditional, is_terminator = classify_instruction(opcode)
            assert is_branch is True, f"{opcode} should be branch"
            assert is_conditional is False, f"{opcode} should not be conditional"
            assert is_terminator is False, f"{opcode} should not be terminator"

    def test_all_conditional_branches(self):
        """Test all conditional branch instructions are classified correctly."""
        for opcode in CONDITIONAL_BRANCHES:
            is_branch, is_conditional, is_terminator = classify_instruction(opcode)
            assert is_branch is True, f"{opcode} should be branch"
            assert is_conditional is True, f"{opcode} should be conditional"
            assert is_terminator is False, f"{opcode} should not be terminator"

    def test_all_terminators(self):
        """Test all terminator instructions are classified correctly."""
        for opcode in TERMINATOR_INSTRUCTIONS:
            is_branch, is_conditional, is_terminator = classify_instruction(opcode)
            assert is_branch is False, f"{opcode} should not be branch"
            assert is_terminator is True, f"{opcode} should be terminator"


class TestExtractBranchTarget:
    """Tests for branch target extraction."""

    def test_simple_lbb_target(self):
        """Test extraction of simple .LBB target."""
        target = extract_branch_target(".LBB0_0")
        assert target == ".LBB0_0"

    def test_lbb_target_in_operands(self):
        """Test extraction of .LBB target from full operand string."""
        target = extract_branch_target(".LBB0_17")
        assert target == ".LBB0_17"

    def test_ltmp_target(self):
        """Test extraction of .Ltmp target."""
        target = extract_branch_target(".Ltmp5")
        assert target == ".Ltmp5"

    def test_lfunc_target(self):
        """Test extraction of .Lfunc target."""
        target = extract_branch_target(".Lfunc_begin0")
        assert target == ".Lfunc_begin0"

    def test_no_target(self):
        """Test extraction when no label present."""
        target = extract_branch_target("s0, s1, s2")
        assert target is None

    def test_target_with_offset(self):
        """Test extraction of target with offset."""
        target = extract_branch_target(".LBB0_2 offset:16")
        assert target == ".LBB0_2"

    def test_multiple_labels(self):
        """Test extraction returns first label found."""
        target = extract_branch_target(".LBB0_0, .LBB0_1")
        assert target == ".LBB0_0"


# =============================================================================
# Instruction Data Structure Tests
# =============================================================================

class TestInstruction:
    """Tests for Instruction data structure."""

    def test_instruction_creation(self):
        """Test creating an Instruction object."""
        instr = Instruction(
            address=10,
            opcode="v_add_f32",
            operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        )
        assert instr.address == 10
        assert instr.opcode == "v_add_f32"
        assert instr.operands == "v0, v1, v2"
        assert instr.is_branch is False
        assert instr.is_conditional is False
        assert instr.is_terminator is False
        assert instr.branch_target is None

    def test_instruction_branch_flags(self):
        """Test creating an Instruction with branch flags."""
        instr = Instruction(
            address=20,
            opcode="s_cbranch_scc0",
            operands=".LBB0_5",
            raw_line="\ts_cbranch_scc0 .LBB0_5",
            is_branch=True,
            is_conditional=True,
            branch_target=".LBB0_5"
        )
        assert instr.is_branch is True
        assert instr.is_conditional is True
        assert instr.branch_target == ".LBB0_5"

    def test_instruction_to_dict(self):
        """Test serialization of Instruction to dictionary."""
        instr = Instruction(
            address=10,
            opcode="v_add_f32",
            operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2",
            is_branch=False,
            is_conditional=False,
            is_terminator=False,
            branch_target=None
        )
        d = instr.to_dict()
        assert d['address'] == 10
        assert d['opcode'] == "v_add_f32"
        assert d['operands'] == "v0, v1, v2"
        assert d['is_branch'] is False

    def test_instruction_from_dict(self):
        """Test deserialization of Instruction from dictionary."""
        d = {
            'address': 15,
            'opcode': 's_branch',
            'operands': '.LBB0_3',
            'raw_line': '\ts_branch .LBB0_3',
            'is_branch': True,
            'is_conditional': False,
            'is_terminator': False,
            'branch_target': '.LBB0_3'
        }
        instr = Instruction.from_dict(d)
        assert instr.address == 15
        assert instr.opcode == 's_branch'
        assert instr.is_branch is True
        assert instr.branch_target == '.LBB0_3'

    def test_instruction_roundtrip(self):
        """Test instruction serialization roundtrip."""
        original = Instruction(
            address=100,
            opcode="v_mfma_f32_16x16x16_bf16",
            operands="a[0:3], v[0:1], v[2:3], a[0:3]",
            raw_line="\tv_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3]",
            is_branch=False,
            is_conditional=False,
            is_terminator=False
        )
        d = original.to_dict()
        restored = Instruction.from_dict(d)
        assert restored.address == original.address
        assert restored.opcode == original.opcode
        assert restored.operands == original.operands


# =============================================================================
# BasicBlock Tests
# =============================================================================

class TestBasicBlock:
    """Tests for BasicBlock data structure."""

    def test_basic_block_creation(self):
        """Test creating a BasicBlock."""
        block = BasicBlock(label=".LBB0_0")
        assert block.label == ".LBB0_0"
        assert block.instructions == []
        assert block.successors == []
        assert block.predecessors == []

    def test_basic_block_is_empty(self):
        """Test is_empty() on empty block."""
        block = BasicBlock(label=".LBB0_0")
        assert block.is_empty() is True

    def test_basic_block_is_not_empty(self):
        """Test is_empty() on non-empty block."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="s_nop", operands="0", raw_line="\ts_nop 0"
        ))
        assert block.is_empty() is False

    def test_basic_block_get_terminator(self):
        """Test get_terminator() returns last instruction."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        ))
        block.instructions.append(Instruction(
            address=20, opcode="s_endpgm", operands="",
            raw_line="\ts_endpgm", is_terminator=True
        ))
        term = block.get_terminator()
        assert term is not None
        assert term.opcode == "s_endpgm"

    def test_basic_block_get_terminator_empty(self):
        """Test get_terminator() on empty block returns None."""
        block = BasicBlock(label=".LBB0_0")
        assert block.get_terminator() is None

    def test_basic_block_get_last_branch(self):
        """Test get_last_branch() finds branch instruction."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        ))
        block.instructions.append(Instruction(
            address=20, opcode="s_branch", operands=".LBB0_5",
            raw_line="\ts_branch .LBB0_5", is_branch=True
        ))
        branch = block.get_last_branch()
        assert branch is not None
        assert branch.opcode == "s_branch"

    def test_basic_block_raw_lines(self):
        """Test raw_lines storage and retrieval."""
        block = BasicBlock(label=".LBB0_0")
        block.raw_lines = {
            10: ".LBB0_0:\n",
            11: "\tv_add_f32 v0, v1, v2\n",
            12: "\ts_branch .LBB0_5\n"
        }
        lines = block.get_raw_lines_in_order()
        assert len(lines) == 3
        assert ".LBB0_0:" in lines[0]

    def test_basic_block_to_dict(self):
        """Test serialization of BasicBlock."""
        block = BasicBlock(label=".LBB0_1", start_line=10, end_line=20)
        block.successors = [".LBB0_2"]
        block.predecessors = [".LBB0_0"]
        d = block.to_dict()
        assert d['label'] == ".LBB0_1"
        assert d['successors'] == [".LBB0_2"]
        assert d['predecessors'] == [".LBB0_0"]

    def test_basic_block_from_dict(self):
        """Test deserialization of BasicBlock."""
        d = {
            'label': '.LBB0_3',
            'instructions': [],
            'successors': ['.LBB0_4'],
            'predecessors': ['.LBB0_2'],
            'start_line': 50,
            'end_line': 70,
            'raw_lines': {'50': 'line1\n', '51': 'line2\n'}
        }
        block = BasicBlock.from_dict(d)
        assert block.label == '.LBB0_3'
        assert block.successors == ['.LBB0_4']
        assert block.start_line == 50
        assert 50 in block.raw_lines


# =============================================================================
# CFG Tests
# =============================================================================

class TestCFG:
    """Tests for CFG data structure."""

    def test_cfg_creation(self):
        """Test creating a CFG."""
        cfg = CFG(name="test_kernel")
        assert cfg.name == "test_kernel"
        assert cfg.blocks == {}
        assert cfg.entry_block is None

    def test_cfg_add_block(self):
        """Test adding blocks to CFG."""
        cfg = CFG(name="test_kernel")
        block1 = BasicBlock(label=".LBB0_0")
        cfg.add_block(block1)
        assert ".LBB0_0" in cfg.blocks
        assert cfg.entry_block == ".LBB0_0"
        assert cfg.block_order == [".LBB0_0"]

    def test_cfg_add_multiple_blocks(self):
        """Test adding multiple blocks to CFG."""
        cfg = CFG(name="test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_block(BasicBlock(label=".LBB0_1"))
        cfg.add_block(BasicBlock(label=".LBB0_2"))
        assert len(cfg.blocks) == 3
        assert cfg.entry_block == ".LBB0_0"  # First block is entry
        assert cfg.block_order == [".LBB0_0", ".LBB0_1", ".LBB0_2"]

    def test_cfg_add_edge(self):
        """Test adding edges between blocks."""
        cfg = CFG(name="test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_block(BasicBlock(label=".LBB0_1"))
        cfg.add_edge(".LBB0_0", ".LBB0_1")
        assert ".LBB0_1" in cfg.blocks[".LBB0_0"].successors
        assert ".LBB0_0" in cfg.blocks[".LBB0_1"].predecessors

    def test_cfg_add_edge_duplicate(self):
        """Test that duplicate edges are not added."""
        cfg = CFG(name="test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_block(BasicBlock(label=".LBB0_1"))
        cfg.add_edge(".LBB0_0", ".LBB0_1")
        cfg.add_edge(".LBB0_0", ".LBB0_1")
        assert cfg.blocks[".LBB0_0"].successors.count(".LBB0_1") == 1

    def test_cfg_add_edge_nonexistent_block(self):
        """Test adding edge with nonexistent block."""
        cfg = CFG(name="test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_edge(".LBB0_0", ".LBB0_1")  # .LBB0_1 doesn't exist
        assert ".LBB0_1" not in cfg.blocks[".LBB0_0"].successors

    def test_cfg_to_dict(self):
        """Test CFG serialization to dictionary."""
        cfg = CFG(name="test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_block(BasicBlock(label=".LBB0_1"))
        cfg.add_edge(".LBB0_0", ".LBB0_1")
        cfg.header_lines = ["header line\n"]
        cfg.footer_lines = ["footer line\n"]
        
        d = cfg.to_dict()
        assert d['name'] == "test_kernel"
        assert d['entry_block'] == ".LBB0_0"
        assert ".LBB0_0" in d['blocks']
        assert d['header_lines'] == ["header line\n"]

    def test_cfg_from_dict(self):
        """Test CFG deserialization from dictionary."""
        d = {
            'name': 'restored_kernel',
            'entry_block': '.LBB0_0',
            'blocks': {
                '.LBB0_0': {
                    'label': '.LBB0_0',
                    'instructions': [],
                    'successors': ['.LBB0_1'],
                    'predecessors': [],
                    'start_line': 0,
                    'end_line': 10,
                    'raw_lines': {}
                },
                '.LBB0_1': {
                    'label': '.LBB0_1',
                    'instructions': [],
                    'successors': [],
                    'predecessors': ['.LBB0_0'],
                    'start_line': 11,
                    'end_line': 20,
                    'raw_lines': {}
                }
            },
            'header_lines': [],
            'footer_lines': [],
            'block_order': ['.LBB0_0', '.LBB0_1']
        }
        cfg = CFG.from_dict(d)
        assert cfg.name == 'restored_kernel'
        assert cfg.entry_block == '.LBB0_0'
        assert len(cfg.blocks) == 2

    def test_cfg_json_roundtrip(self):
        """Test CFG JSON serialization roundtrip."""
        cfg = CFG(name="json_test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_block(BasicBlock(label=".LBB0_1"))
        cfg.add_edge(".LBB0_0", ".LBB0_1")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cfg.to_json(f.name)
            temp_path = f.name
        
        try:
            restored = CFG.from_json(temp_path)
            assert restored.name == cfg.name
            assert restored.entry_block == cfg.entry_block
            assert len(restored.blocks) == len(cfg.blocks)
        finally:
            os.unlink(temp_path)


# =============================================================================
# Parser Tests
# =============================================================================

class TestAMDGCNParser:
    """Tests for AMDGCNParser."""

    def test_parser_creation(self):
        """Test creating a parser."""
        parser = AMDGCNParser()
        assert parser.blocks == {}
        assert parser.function_name == "unknown"

    def test_parser_is_skip_line_comment(self):
        """Test skip line detection for comments."""
        parser = AMDGCNParser()
        assert parser.is_skip_line("; this is a comment") is True

    def test_parser_is_skip_line_empty(self):
        """Test skip line detection for empty lines."""
        parser = AMDGCNParser()
        assert parser.is_skip_line("") is True
        assert parser.is_skip_line("   ") is True

    def test_parser_is_skip_line_directive(self):
        """Test skip line detection for directives."""
        parser = AMDGCNParser()
        assert parser.is_skip_line("\t.text") is True
        assert parser.is_skip_line("\t.loc 1 10 0") is True

    def test_parser_is_skip_line_label(self):
        """Test that labels are not skipped."""
        parser = AMDGCNParser()
        assert parser.is_skip_line(".LBB0_0:") is False

    def test_parser_is_bb_label(self):
        """Test basic block label detection."""
        parser = AMDGCNParser()
        assert parser.is_bb_label(".LBB0_0:") == ".LBB0_0"
        assert parser.is_bb_label(".LBB12_345:") == ".LBB12_345"
        assert parser.is_bb_label(".Lfunc_begin0:") == ".Lfunc_begin0"

    def test_parser_is_bb_label_debug(self):
        """Test that debug labels are not detected as BB labels."""
        parser = AMDGCNParser()
        assert parser.is_bb_label(".Ltmp0:") is None
        assert parser.is_bb_label(".Ltmp123:") is None

    def test_parser_is_any_label(self):
        """Test any label detection."""
        parser = AMDGCNParser()
        assert parser.is_any_label(".LBB0_0:") == ".LBB0_0"
        assert parser.is_any_label(".Ltmp0:") == ".Ltmp0"
        assert parser.is_any_label("my_function:") == "my_function"

    def test_parser_parse_instruction_simple(self):
        """Test parsing a simple instruction."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\tv_add_f32 v0, v1, v2", 10)
        assert instr is not None
        assert instr.opcode == "v_add_f32"
        assert instr.operands == "v0, v1, v2"
        assert instr.address == 10

    def test_parser_parse_instruction_with_comment(self):
        """Test parsing instruction with comment."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\ts_load_dwordx2 s[2:3], s[0:1], 0x0 ; load pointer", 15)
        assert instr is not None
        assert instr.opcode == "s_load_dwordx2"
        assert "load pointer" not in instr.operands

    def test_parser_parse_instruction_branch(self):
        """Test parsing branch instruction."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\ts_cbranch_scc0 .LBB0_5", 20)
        assert instr is not None
        assert instr.is_branch is True
        assert instr.is_conditional is True
        assert instr.branch_target == ".LBB0_5"

    def test_parser_parse_instruction_terminator(self):
        """Test parsing terminator instruction."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\ts_endpgm", 100)
        assert instr is not None
        assert instr.is_terminator is True

    def test_parser_parse_instruction_label_only(self):
        """Test that label-only lines return None."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction(".LBB0_0:", 5)
        assert instr is None

    def test_parser_parse_instruction_directive(self):
        """Test that directives return None."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\t.loc 1 10 0", 5)
        assert instr is None

    def test_parser_parse_mfma_instruction(self):
        """Test parsing MFMA instruction."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\tv_mfma_f32_16x16x16_bf16 a[0:3], v[120:121], v[56:57], 0", 200)
        assert instr is not None
        assert instr.opcode == "v_mfma_f32_16x16x16_bf16"

    def test_parser_parse_memory_instruction(self):
        """Test parsing memory instruction."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\tglobal_load_dwordx4 v[120:123], v[26:27], off", 150)
        assert instr is not None
        assert instr.opcode == "global_load_dwordx4"


# =============================================================================
# Integration Tests with Real Assembly
# =============================================================================

class TestCFGIntegration:
    """Integration tests using real assembly file."""

    @pytest.fixture
    def parsed_cfg(self):
        """Parse the test assembly file."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test assembly file not found: {TEST_ASSEMBLY_FILE}")
        parser = AMDGCNParser()
        return parser.parse_file(str(TEST_ASSEMBLY_FILE))

    def test_parse_real_file_successful(self, parsed_cfg):
        """Test that real file parses without error."""
        assert parsed_cfg is not None
        assert parsed_cfg.name is not None

    def test_parse_real_file_has_blocks(self, parsed_cfg):
        """Test that parsed file has basic blocks."""
        assert len(parsed_cfg.blocks) > 0

    def test_parse_real_file_has_entry_block(self, parsed_cfg):
        """Test that parsed file has entry block."""
        assert parsed_cfg.entry_block is not None
        assert parsed_cfg.entry_block in parsed_cfg.blocks

    def test_parse_real_file_has_instructions(self, parsed_cfg):
        """Test that blocks have instructions."""
        total_instructions = sum(len(b.instructions) for b in parsed_cfg.blocks.values())
        assert total_instructions > 0

    def test_parse_real_file_has_header_footer(self, parsed_cfg):
        """Test that header and footer are captured."""
        assert len(parsed_cfg.header_lines) > 0

    def test_parse_real_file_block_order(self, parsed_cfg):
        """Test that block order is preserved."""
        assert len(parsed_cfg.block_order) == len(parsed_cfg.blocks)
        for label in parsed_cfg.block_order:
            assert label in parsed_cfg.blocks

    def test_parse_real_file_edges(self, parsed_cfg):
        """Test that edges are created."""
        total_edges = sum(len(b.successors) for b in parsed_cfg.blocks.values())
        assert total_edges > 0

    def test_parse_real_file_terminator(self, parsed_cfg):
        """Test that terminator instruction exists."""
        has_terminator = False
        for block in parsed_cfg.blocks.values():
            for instr in block.instructions:
                if instr.is_terminator:
                    has_terminator = True
                    break
        assert has_terminator

    def test_parse_real_file_mfma_instructions(self, parsed_cfg):
        """Test that MFMA instructions are parsed."""
        has_mfma = False
        for block in parsed_cfg.blocks.values():
            for instr in block.instructions:
                if 'mfma' in instr.opcode.lower():
                    has_mfma = True
                    break
        assert has_mfma

    def test_parse_real_file_waitcnt_instructions(self, parsed_cfg):
        """Test that s_waitcnt instructions are parsed."""
        has_waitcnt = False
        for block in parsed_cfg.blocks.values():
            for instr in block.instructions:
                if instr.opcode.lower() == 's_waitcnt':
                    has_waitcnt = True
                    break
        assert has_waitcnt

    def test_parse_real_file_barrier_instructions(self, parsed_cfg):
        """Test that s_barrier instructions are parsed."""
        has_barrier = False
        for block in parsed_cfg.blocks.values():
            for instr in block.instructions:
                if instr.opcode.lower() == 's_barrier':
                    has_barrier = True
                    break
        assert has_barrier


# =============================================================================
# CFG Regeneration Tests
# =============================================================================

class TestCFGRegeneration:
    """Tests for CFG regeneration to assembly."""

    @pytest.fixture
    def parsed_cfg(self):
        """Parse the test assembly file."""
        if not TEST_ASSEMBLY_FILE.exists():
            pytest.skip(f"Test assembly file not found: {TEST_ASSEMBLY_FILE}")
        parser = AMDGCNParser()
        return parser.parse_file(str(TEST_ASSEMBLY_FILE))

    def test_regenerate_amdgcn_file(self, parsed_cfg):
        """Test regenerating .amdgcn file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.amdgcn', delete=False) as f:
            temp_path = f.name
        
        try:
            parsed_cfg.to_amdgcn(temp_path, keep_debug_labels=False)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_regenerate_preserves_instruction_count(self, parsed_cfg):
        """Test that regeneration preserves instruction count."""
        original_instr_count = sum(len(b.instructions) for b in parsed_cfg.blocks.values())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.amdgcn', delete=False) as f:
            temp_path = f.name
        
        try:
            parsed_cfg.to_amdgcn(temp_path, keep_debug_labels=True)
            
            # Re-parse the regenerated file
            parser = AMDGCNParser()
            regenerated_cfg = parser.parse_file(temp_path)
            regenerated_instr_count = sum(len(b.instructions) for b in regenerated_cfg.blocks.values())
            
            assert regenerated_instr_count == original_instr_count
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# =============================================================================
# DOT Generation Tests
# =============================================================================

class TestDOTGeneration:
    """Tests for DOT format generation."""

    def test_escape_dot_string(self):
        """Test DOT string escaping."""
        assert escape_dot_string('test') == 'test'
        assert escape_dot_string('test"quote') == 'test\\"quote'
        assert escape_dot_string('test<bracket>') == 'test\\<bracket\\>'
        assert escape_dot_string('test|pipe') == 'test\\|pipe'

    def test_truncate_instruction_short(self):
        """Test instruction truncation - short instruction."""
        instr = Instruction(address=1, opcode="s_nop", operands="0", raw_line="\ts_nop 0")
        result = truncate_instruction(instr)
        assert result == "s_nop 0"

    def test_truncate_instruction_long(self):
        """Test instruction truncation - long instruction."""
        instr = Instruction(
            address=1,
            opcode="v_mfma_f32_16x16x16_bf16",
            operands="a[0:3], v[120:121], v[56:57], a[0:3]",
            raw_line="\tv_mfma_f32_16x16x16_bf16 a[0:3], v[120:121], v[56:57], a[0:3]"
        )
        result = truncate_instruction(instr, max_len=40)
        assert len(result) <= 40
        assert result.endswith("...")

    def test_generate_simple_dot(self):
        """Test simple DOT generation."""
        cfg = CFG(name="test_kernel")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        cfg.add_block(BasicBlock(label=".LBB0_1"))
        cfg.add_edge(".LBB0_0", ".LBB0_1")
        
        dot = generate_simple_dot(cfg)
        assert "digraph" in dot
        assert "test_kernel" in dot
        assert "_LBB0_0" in dot
        assert "_LBB0_1" in dot

    def test_generate_dot_with_instructions(self):
        """Test detailed DOT generation with instructions."""
        cfg = CFG(name="test_kernel")
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="v_add_f32", operands="v0, v1, v2",
            raw_line="\tv_add_f32 v0, v1, v2"
        ))
        cfg.add_block(block)
        
        dot = generate_dot(cfg, show_instructions=True)
        assert "digraph" in dot
        assert "v_add_f32" in dot


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_empty_block(self):
        """Test handling of empty block."""
        block = BasicBlock(label=".LBB0_0")
        assert block.is_empty()
        assert block.get_terminator() is None
        assert block.get_last_branch() is None

    def test_single_instruction_block(self):
        """Test block with single instruction."""
        block = BasicBlock(label=".LBB0_0")
        instr = Instruction(address=10, opcode="s_endpgm", operands="",
                          raw_line="\ts_endpgm", is_terminator=True)
        block.instructions.append(instr)
        assert not block.is_empty()
        assert block.get_terminator() == instr

    def test_cfg_single_block(self):
        """Test CFG with single block."""
        cfg = CFG(name="single_block")
        cfg.add_block(BasicBlock(label=".LBB0_0"))
        assert len(cfg.blocks) == 1
        assert cfg.entry_block == ".LBB0_0"

    def test_cfg_no_successors(self):
        """Test block with no successors (exit block)."""
        cfg = CFG(name="test")
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="s_endpgm", operands="",
            raw_line="\ts_endpgm", is_terminator=True
        ))
        cfg.add_block(block)
        assert cfg.blocks[".LBB0_0"].successors == []

    def test_block_with_only_branch(self):
        """Test block containing only a branch."""
        block = BasicBlock(label=".LBB0_0")
        block.instructions.append(Instruction(
            address=10, opcode="s_branch", operands=".LBB0_1",
            raw_line="\ts_branch .LBB0_1", is_branch=True,
            branch_target=".LBB0_1"
        ))
        branch = block.get_last_branch()
        assert branch is not None
        assert branch.opcode == "s_branch"

    def test_instruction_with_no_operands(self):
        """Test instruction with no operands."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\ts_barrier", 50)
        assert instr is not None
        assert instr.opcode == "s_barrier"
        assert instr.operands == ""

    def test_raw_lines_empty(self):
        """Test get_raw_lines_in_order with empty raw_lines."""
        block = BasicBlock(label=".LBB0_0")
        assert block.get_raw_lines_in_order() == []

    def test_instruction_with_offset(self):
        """Test parsing instruction with offset modifier."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\tglobal_load_dwordx4 v[120:123], v[26:27], off offset:16", 100)
        assert instr is not None
        assert "offset:16" in instr.operands

    def test_instruction_with_offen(self):
        """Test parsing instruction with offen modifier."""
        parser = AMDGCNParser()
        instr = parser.parse_instruction("\tbuffer_load_dword v53, off, s[8:11], 0 offen", 100)
        assert instr is not None
        assert "offen" in instr.operands


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

