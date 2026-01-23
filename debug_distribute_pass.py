#!/usr/bin/env python3
"""
Debug Tool for AMDGCN Transform Passes

This tool supports debugging transform passes defined in a JSON pass list file.
It applies passes one by one and tests after each pass to find which one causes failures.

Supported pass types:
- distribute: Distribute instructions evenly (supports two-level debugging)
- move: Move single instruction
- replace_registers: Replace registers

Usage:
    python debug_distribute_pass.py [options]

Options:
    --load-json FILE    Load analysis from JSON file
    --pass-list FILE    Pass list JSON file (e.g., transform_passes_example.json)
    --output FILE       Output .amdgcn file path (optional)
    --output-dir DIR    Output directory for intermediate files (default: amdgcn_edit/debug_distribute)
    --test-cmd CMD      Custom test command (use {FILE} as placeholder)
    --start-pass N      Start from pass N (0-indexed, skip earlier passes)
    --skip-baseline     Skip baseline test
    --verbose           Show verbose output
    
    # Two-level debug for distribute pass in pass list:
    --detail-pass N     Debug specific pass N (must be distribute type) in detail
    --start-step S      Start from step S within --detail-pass (Level 1)
    --detail-step S     Debug step S in detail (Level 2), requires --detail-pass
    --start-move M      Start from move M within --detail-step (Level 2)
    
    # Legacy mode (single distribute pass debug):
    --source FILE       Source .amdgcn file
    --block LABEL       Target block label (default: .LBB0_2)
    --opcode OPCODE     Target instruction opcode (default: global_load_dwordx4)
    --count K           Number of instructions to distribute (default: 16)

Examples:
    # Debug pass list from JSON (test all passes)
    python debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json \\
        --pass-list amdgcn_edit/trans_pass_list.json
    
    # Debug specific distribute pass (Level 1: test each step)
    python debug_distribute_pass.py --load-json analysis.json \\
        --pass-list passes.json --detail-pass 1
    
    # Debug distribute pass starting from step 3
    python debug_distribute_pass.py --load-json analysis.json \\
        --pass-list passes.json --detail-pass 1 --start-step 3
    
    # Debug step 3 of distribute pass in detail (Level 2: test each move)
    python debug_distribute_pass.py --load-json analysis.json \\
        --pass-list passes.json --detail-pass 1 --detail-step 3
    
    # Debug step 3, starting from move 45
    python debug_distribute_pass.py --load-json analysis.json \\
        --pass-list passes.json --detail-pass 1 --detail-step 3 --start-move 45
    
    # Legacy mode: Debug single distribute pass
    python debug_distribute_pass.py --source kernel.amdgcn --block .LBB0_2 --opcode global_load_dwordx4 --count 16
"""

import sys
import os
import subprocess
import shutil
import argparse
import time
import functools
import json

# Enable unbuffered output for real-time printing
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amdgcn_ddg import AnalysisResult, load_analysis_from_json, dump_block_instructions
from amdgcn_passes import (
    MoveInstructionPass, DistributeInstructionPass, RegisterReplacePass,
    get_instruction_cycles, sync_instruction_to_raw_lines
)
from amdgcn_verify import build_global_ddg, verify_optimization, SchedulingVerificationError
from amdgcn_cfg import BasicBlock, Instruction, CFG

# ============================================================================
# GLOBAL CONFIGURATION - Modify these variables as needed
# ============================================================================

# Working directory (where tests are run from)
WORK_DIR = '/mnt/raid0/heyanguang/code/fa_triton/aiter'

# Triton cache directory (cleared before each test)
TRITON_CACHE_DIR = '/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache2'

# Default test command and arguments
TEST_SCRIPT = './op_tests/triton_tests/test_pa_decode_gluon.py'
TEST_ARGS = [
    '-b', '80',
    '-q', '1', 
    '-c', '2048',
    '-n', '16,1',
    '-d', '128',
    '--block_size', '64',
    '--compute_type', 'bf16',
    '--quant_mode', 'per_tensor',
    '--trans_v', 'false',
    '--kv_varlen', 'false',
    '--use_aot_impl', 'false',
    '--quant_q_and_kv', '0,0',
    '--context_partition_size', '1024'
]

# Test timeout in seconds
TEST_TIMEOUT = 180

# Keywords that indicate test failure (case-sensitive)
FAILURE_KEYWORDS = [
    "FAILED",
    "AssertionError", 
    "exceeded the error threshold"
]

# ============================================================================


def get_cumulative_cycle(block: BasicBlock, idx: int) -> int:
    """Calculate cumulative cycle count up to instruction at idx."""
    total = 0
    for i in range(idx + 1):
        instr = block.instructions[i]
        total += get_instruction_cycles(instr.opcode)
    return total


def dump_single_block_instructions(cfg: CFG, block_label: str, output_dir: str) -> None:
    """
    Dump instructions of a single basic block to a text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if block_label not in cfg.blocks:
        print(f"Warning: Block {block_label} not found in CFG")
        return
    
    block = cfg.blocks[block_label]
    
    filename = block_label.lstrip('.') + '.txt'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"; Block: {block_label}\n")
        f.write(f"; Total instructions: {len(block.instructions)}\n")
        f.write(f";\n")
        
        for idx, instr in enumerate(block.instructions):
            if instr.raw_line:
                line = instr.raw_line.strip()
            else:
                line = f"{instr.opcode} {instr.operands}" if instr.operands else instr.opcode
            
            f.write(f"[{idx}] {line}\n")


def save_amdgcn(result: AnalysisResult, output_path: str, block_label: str = None):
    """
    Save the current state as an amdgcn file.
    """
    result.to_amdgcn(output_path)
    
    if block_label:
        dump_dir = output_path.rsplit('.amdgcn', 1)[0]
        dump_single_block_instructions(result.cfg, block_label, dump_dir)


def load_pass_list(pass_list_path: str) -> list:
    """Load and parse pass list from JSON file."""
    with open(pass_list_path, 'r') as f:
        pass_configs = json.load(f)
    
    if not isinstance(pass_configs, list):
        raise ValueError("Pass list JSON must contain a list of pass configurations")
    
    # Filter out empty pass configs
    pass_configs = [p for p in pass_configs if p]
    
    passes = []
    for pass_config in pass_configs:
        pass_type = pass_config.get('type')
        if not pass_type:
            continue
            
        if pass_type == 'move':
            passes.append({
                'type': 'move',
                'block': pass_config['block'],
                'index': pass_config['index'],
                'cycles': pass_config['cycles'],
                'barrier_crossing_opcodes': set(pass_config.get('barrier_crossing_opcodes', []))
            })
        elif pass_type == 'distribute':
            passes.append({
                'type': 'distribute',
                'block': pass_config['block'],
                'opcode': pass_config['opcode'],
                'k': pass_config['k'],
                'barrier_crossing_opcodes': set(pass_config.get('barrier_crossing_opcodes', []))
            })
        elif pass_type == 'replace_registers':
            passes.append({
                'type': 'replace_registers',
                'range_start': pass_config['range_start'],
                'range_end': pass_config['range_end'],
                'registers': pass_config['registers'],
                'alignments': pass_config.get('alignments', [1] * len(pass_config['registers']))
            })
        else:
            print(f"Warning: Unknown pass type '{pass_type}', skipping")
    
    return passes


def format_pass_info(pass_config: dict) -> str:
    """Format pass config into a readable string."""
    pass_type = pass_config['type']
    
    if pass_type == 'move':
        return f"move(block={pass_config['block']}, index={pass_config['index']}, cycles={pass_config['cycles']})"
    elif pass_type == 'distribute':
        return f"distribute(block={pass_config['block']}, opcode={pass_config['opcode']}, k={pass_config['k']})"
    elif pass_type == 'replace_registers':
        return f"replace_registers(range={pass_config['range_start']}-{pass_config['range_end']}, regs={pass_config['registers']})"
    else:
        return f"{pass_type}(?)"


class PassListDebugger:
    """Debugger for testing transform pass lists with two-level distribute debugging."""
    
    def __init__(self, args):
        self.args = args
        self.json_path = args.load_json
        self.pass_list_path = args.pass_list
        self.output_dir = args.output_dir
        self.output_path = args.output
        self.verbose = args.verbose
        self.test_cmd = args.test_cmd
        
        self.skip_test = args.skip_test
        
        self.result = None
        self.passes = []
        
        # For distribute two-level debug
        self.block = None
        self.block_label = None
        self.target_opcode = None
        self.K = 0
        self.ideal_cycles = []
        self.frozen_boundary = 0
        self.protected_instructions = []
        self.barrier_crossing_opcodes = set()
        
    def log(self, msg, force=False):
        """Print message if verbose or forced."""
        if self.verbose or force:
            print(msg)
    
    def run_test(self, amdgcn_path: str) -> tuple:
        """
        Run test with the given amdgcn file.
        Returns (success: bool, output: str)
        """
        # Skip test if --skip-test is set
        if self.skip_test:
            return True, "Test skipped (--skip-test)"
        
        abs_path = os.path.abspath(amdgcn_path)
        
        if self.test_cmd:
            cmd = self.test_cmd.replace("{FILE}", abs_path)
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                output = result.stdout + result.stderr
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, "Test timed out"
            except Exception as e:
                return False, str(e)
        else:
            env = os.environ.copy()
            env['TRITON_CACHE_DIR'] = TRITON_CACHE_DIR
            env['TRITON_OVERRIDE_AMDGCN_FILE'] = abs_path
            
            if os.path.exists(TRITON_CACHE_DIR):
                shutil.rmtree(TRITON_CACHE_DIR)
            
            cmd = ['python', TEST_SCRIPT] + TEST_ARGS
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                       timeout=TEST_TIMEOUT, env=env, cwd=WORK_DIR)
                output = result.stdout + result.stderr
                
                for keyword in FAILURE_KEYWORDS:
                    if keyword in output:
                        return False, output
                return True, output
            except subprocess.TimeoutExpired:
                return False, "Test timed out"
            except Exception as e:
                return False, str(e)
    
    def setup(self):
        """Initialize analysis and load pass list."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loading analysis from {self.json_path}...")
        self.result = load_analysis_from_json(self.json_path)
        
        print(f"Loading pass list from {self.pass_list_path}...")
        self.passes = load_pass_list(self.pass_list_path)
        
        print(f"Loaded {len(self.passes)} passes:")
        for i, p in enumerate(self.passes):
            print(f"  [{i}] {format_pass_info(p)}")
        
        return True
    
    def apply_pass(self, pass_idx: int) -> tuple:
        """
        Apply a single pass.
        Returns (success: bool, error_message: str or None)
        """
        pass_config = self.passes[pass_idx]
        pass_type = pass_config['type']
        
        try:
            if pass_type == 'move':
                move_pass = MoveInstructionPass(
                    block_label=pass_config['block'],
                    instr_index=pass_config['index'],
                    cycles=pass_config['cycles'],
                    verbose=self.verbose,
                    barrier_crossing_opcodes=pass_config.get('barrier_crossing_opcodes', set())
                )
                move_pass.run(self.result)
                return True, None
                
            elif pass_type == 'distribute':
                dist_pass = DistributeInstructionPass(
                    block_label=pass_config['block'],
                    target_opcode=pass_config['opcode'],
                    distribute_count=pass_config['k'],
                    verbose=self.verbose,
                    barrier_crossing_opcodes=pass_config.get('barrier_crossing_opcodes', set())
                )
                dist_pass.run(self.result)
                return True, None
                
            elif pass_type == 'replace_registers':
                replace_pass = RegisterReplacePass(
                    range_start=pass_config['range_start'],
                    range_end=pass_config['range_end'],
                    registers_to_replace=pass_config['registers'],
                    alignments=pass_config['alignments'],
                    verbose=self.verbose
                )
                replace_pass.run(self.result)
                return True, None
            else:
                return False, f"Unknown pass type: {pass_type}"
                
        except Exception as e:
            return False, str(e)
    
    def setup_distribute_debug(self, pass_idx: int) -> bool:
        """
        Setup state for debugging a distribute pass in detail.
        Returns True if successful.
        """
        pass_config = self.passes[pass_idx]
        
        if pass_config['type'] != 'distribute':
            print(f"ERROR: Pass {pass_idx} is not a distribute pass!")
            return False
        
        self.block_label = pass_config['block']
        self.target_opcode = pass_config['opcode']
        self.K = pass_config['k']
        self.barrier_crossing_opcodes = pass_config.get('barrier_crossing_opcodes', set())
        
        self.block = self.result.cfg.blocks.get(self.block_label)
        if not self.block:
            print(f"ERROR: Block {self.block_label} not found!")
            return False
        
        # Find target instructions
        target_indices = [idx for idx, instr in enumerate(self.block.instructions)
                        if instr.opcode.lower() == self.target_opcode.lower()]
        
        if len(target_indices) < self.K:
            print(f"WARNING: Only found {len(target_indices)} {self.target_opcode} instructions, requested {self.K}")
            self.K = len(target_indices)
        
        print(f"Found {len(target_indices)} {self.target_opcode} instructions in {self.block_label}")
        
        # Calculate ideal cycle positions
        branch_boundary = len(self.block.instructions)
        for idx, instr in enumerate(self.block.instructions):
            if instr.is_branch or instr.is_terminator:
                branch_boundary = idx
                break
        
        total_cycles = get_cumulative_cycle(self.block, branch_boundary - 1) if branch_boundary > 0 else 0
        print(f"Block total cycles: {total_cycles}, branch boundary: {branch_boundary}")
        
        if self.K > 1:
            cycle_spacing = total_cycles / self.K
            self.ideal_cycles = [int(cycle_spacing * (i + 1)) for i in range(self.K)]
        else:
            self.ideal_cycles = [total_cycles // 2]
        
        print(f"Ideal cycle positions: {self.ideal_cycles[:5]}{'...' if len(self.ideal_cycles) > 5 else ''}")
        
        self.frozen_boundary = 0
        
        # Collect ALL target instruction objects (same as DistributeInstructionPass.run())
        # This list is used to provide protected_instructions during each step
        self.all_target_instrs = []
        for idx in target_indices:
            self.all_target_instrs.append(self.block.instructions[idx])
        
        return True
    
    def find_target_instruction_index(self, n: int) -> int:
        """Find the current index of the n-th target instruction (0-indexed)."""
        count = 0
        for idx, instr in enumerate(self.block.instructions):
            if instr.opcode.lower() == self.target_opcode.lower():
                if count == n:
                    return idx
                count += 1
        return -1
    
    def apply_distribute_step(self, step_num: int, save_path: str = None) -> bool:
        """
        Apply a single distribution step.
        Returns True if move was successful.
        
        Note: This method should match the behavior of DistributeInstructionPass._move_instruction_toward()
        """
        current_idx = self.find_target_instruction_index(step_num)
        if current_idx < 0:
            return False
        
        target_instr = self.block.instructions[current_idx]
        target_cycle = self.ideal_cycles[step_num]
        
        # Find target_idx based on target_cycle (same as DistributeInstructionPass._cycle_to_index)
        target_idx = 0
        total = 0
        for i, instr in enumerate(self.block.instructions):
            cycles = get_instruction_cycles(instr.opcode)
            total += cycles
            if total >= target_cycle:
                target_idx = i
                break
        
        # Ensure target_idx respects ordering constraints (same as DistributeInstructionPass)
        if step_num > 0:
            prev_idx = self.find_target_instruction_index(step_num - 1)
            if prev_idx >= 0 and target_idx <= prev_idx:
                target_idx = prev_idx + 1
        
        # Ensure target is at least at frozen_boundary (same as DistributeInstructionPass)
        target_idx = max(target_idx, self.frozen_boundary)
        
        # Calculate protected_instructions: all REMAINING targets (same as DistributeInstructionPass)
        # protected_instrs = all_target_instrs[step_num+1:]
        protected_instrs = self.all_target_instrs[step_num + 1:]
        
        # Calculate cycle-based movement (same as DistributeInstructionPass._move_instruction_toward)
        current_cycle = get_cumulative_cycle(self.block, current_idx)
        target_cycle_actual = get_cumulative_cycle(self.block, target_idx)
        cycle_diff = target_cycle_actual - current_cycle
        cycles_to_move = -cycle_diff  # Same as _move_instruction_toward: positive = move up
        
        self.log(f"  Step {step_num + 1}: idx={current_idx} -> {target_idx}, cycle={current_cycle} -> {target_cycle_actual}, move={cycles_to_move}")
        
        if cycles_to_move == 0:
            self.log(f"    No move needed")
        else:
            move_pass = MoveInstructionPass(
                self.block_label,
                current_idx,
                cycles_to_move,
                verbose=False,
                frozen_boundary=self.frozen_boundary,
                protected_instructions=protected_instrs,
                barrier_crossing_opcodes=self.barrier_crossing_opcodes
            )
            move_pass.run(self.result)
            
            if move_pass.total_cycles_moved > 0:
                self.log(f"    Moved {move_pass.total_cycles_moved} cycles")
        
        # Update frozen boundary (same as DistributeInstructionPass)
        new_idx = self.find_target_instruction_index(step_num)
        if new_idx >= 0:
            self.frozen_boundary = new_idx + 1  # +1 to match DistributeInstructionPass
        
        if save_path:
            save_amdgcn(self.result, save_path, self.block_label)
        
        return True
    
    def level1_distribute_debug(self, pass_idx: int, start_step: int = 0) -> int:
        """
        Level 1: Test each distribution step within a distribute pass.
        Returns the first failing step number (1-indexed), or -1 if all pass.
        
        Performs verification after each step to detect scheduling errors.
        """
        print("\n" + "=" * 60)
        print(f"LEVEL 1: Testing each step of distribute pass {pass_idx}")
        print(f"         Block: {self.block_label}, Opcode: {self.target_opcode}, K: {self.K}")
        print("=" * 60)
        
        # Build original GDG for verification (before any steps)
        # Use result.ddgs (same as DistributeInstructionPass.run()) for consistent verification
        original_gdg = build_global_ddg(self.result.cfg, self.result.ddgs)
        
        # Apply steps before start_step without testing
        if start_step > 0:
            print(f"\nSkipping steps 1-{start_step} (applying without testing)...")
            for step in range(start_step):
                self.apply_distribute_step(step)
        
        # Save baseline state
        baseline_path = os.path.join(self.output_dir, f"pass{pass_idx}_step{start_step:02d}_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path, self.block_label)
        
        if not self.args.skip_baseline and start_step == 0:
            print("\nTesting baseline (before this distribute pass)...")
            success, _ = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline test failed!")
                return 0
            print("✓ Baseline passes")
        
        # Test each remaining step
        for step in range(start_step, self.K):
            step_path = os.path.join(self.output_dir, f"pass{pass_idx}_step{step + 1:02d}.amdgcn")
            
            print(f"\nStep {step + 1}/{self.K}:", end=" ")
            
            # Apply the step and catch any verification errors from MoveInstructionPass.run()
            try:
                self.apply_distribute_step(step, step_path)
            except SchedulingVerificationError as e:
                print("✗ VERIFICATION FAILED (during move)")
                print(f"\n{'=' * 60}")
                print(f"FOUND: Step {step + 1} of pass {pass_idx} caused verification to fail!")
                print(f"{'=' * 60}")
                print(f"Error: {e}")
                return step + 1
            
            # Additional verification after this step (in case apply_distribute_step doesn't use MoveInstructionPass)
            print("Verifying...", end=" ", flush=True)
            try:
                verify_optimization(original_gdg, self.result.cfg, 
                                   barrier_crossing_opcodes=self.barrier_crossing_opcodes)
                print("✓ PASS")
            except SchedulingVerificationError as e:
                print("✗ VERIFICATION FAILED")
                print(f"\n{'=' * 60}")
                print(f"FOUND: Step {step + 1} of pass {pass_idx} caused verification to fail!")
                print(f"{'=' * 60}")
                print(f"Error: {e}")
                return step + 1
            
            # Also run custom test if not skipped
            if not self.skip_test:
                print("  Running test...", end=" ", flush=True)
                success, output = self.run_test(step_path)
                if not success:
                    print("✗ TEST FAIL")
                    return step + 1
                print("✓")
        
        print(f"\nAll {self.K} steps passed verification!")
        return -1
    
    def level2_distribute_debug(self, pass_idx: int, failing_step: int, start_move: int = 0) -> dict:
        """
        Level 2: Test each individual instruction move within a failing step.
        
        Performs verification after each move to detect scheduling errors.
        """
        print("\n" + "=" * 60)
        print(f"LEVEL 2: Detailed analysis of Pass {pass_idx}, Step {failing_step}")
        if start_move > 0:
            print(f"         Starting from move #{start_move}")
        print("=" * 60)
        
        # Reset and reload
        self.result = load_analysis_from_json(self.json_path)
        
        # Apply passes before this one
        if pass_idx > 0:
            print(f"\nApplying passes 0-{pass_idx - 1}...")
            for i in range(pass_idx):
                success, error = self.apply_pass(i)
                if not success:
                    print(f"  Pass {i} failed: {error}")
                    return {"error": f"pass_{i}_failed", "message": error}
        
        # Setup distribute debug
        if not self.setup_distribute_debug(pass_idx):
            return {"error": "setup_failed"}
        
        # Build original GDG for verification (before any moves in this step)
        # Use result.ddgs (same as DistributeInstructionPass.run()) for consistent verification
        original_gdg = build_global_ddg(self.result.cfg, self.result.ddgs)
        
        # Apply steps before the failing step
        if failing_step > 1:
            print(f"\nApplying steps 1-{failing_step - 1} (known to pass)...")
            for step in range(failing_step - 1):
                self.apply_distribute_step(step)
        
        # Save baseline state
        baseline_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path, self.block_label)
        
        if start_move == 0:
            print(f"\nVerifying baseline (before step {failing_step})...")
            success, _ = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline before failing step already fails!")
                return {"error": "baseline_fails"}
            print("✓ Baseline passes")
        else:
            print(f"\nSkipping baseline test (starting from move #{start_move})")
        
        # Get details for the failing step
        # Use same calculation as apply_distribute_step for consistency
        current_idx = self.find_target_instruction_index(failing_step - 1)
        target_instr = self.block.instructions[current_idx]
        
        # Find target_idx based on target_cycle (same as apply_distribute_step)
        target_cycle_ideal = self.ideal_cycles[failing_step - 1]
        target_idx = 0
        total = 0
        for i, instr in enumerate(self.block.instructions):
            cycles = get_instruction_cycles(instr.opcode)
            total += cycles
            if total >= target_cycle_ideal:
                target_idx = i
                break
        
        # Ensure target_idx respects ordering constraints (same as apply_distribute_step)
        if failing_step > 1:
            prev_idx = self.find_target_instruction_index(failing_step - 2)
            if prev_idx >= 0 and target_idx <= prev_idx:
                target_idx = prev_idx + 1
        
        target_idx = max(target_idx, self.frozen_boundary)
        
        # Calculate protected_instructions for level2 (same as apply_distribute_step)
        level2_protected = self.all_target_instrs[failing_step:]  # step_num = failing_step - 1, so [step_num + 1:] = [failing_step:]
        
        # Calculate cycle-based movement (same as apply_distribute_step / _move_instruction_toward)
        current_cycle = get_cumulative_cycle(self.block, current_idx)
        target_cycle = get_cumulative_cycle(self.block, target_idx)
        cycle_diff = target_cycle - current_cycle
        cycles_to_move = -cycle_diff  # positive = move up, negative = move down
        
        print(f"\nStep {failing_step} details:")
        print(f"  Target instruction: {self.target_opcode} at idx={current_idx}")
        print(f"  Current cycle: {current_cycle}, Target cycle: {target_cycle}")
        print(f"  Cycles to move: {cycles_to_move} ({'UP' if cycles_to_move > 0 else 'DOWN'})")
        
        if start_move > 0:
            print(f"\nApplying moves 1-{start_move} without testing...")
        print(f"\nTesting individual moves (with verification after each move)...")
        
        move_count = 0
        total_cycles_moved = 0
        
        while total_cycles_moved < abs(cycles_to_move):
            # Find current position of target
            target_idx = -1
            for idx, instr in enumerate(self.block.instructions):
                if instr is target_instr:
                    target_idx = idx
                    break
            
            if target_idx < 0:
                print(f"  Cannot find target instruction")
                break
            
            # Move just 4 cycles (one instruction typically)
            small_move = -4 if cycles_to_move < 0 else 4
            
            move_pass = MoveInstructionPass(
                self.block_label,
                target_idx,
                small_move,
                verbose=False,
                frozen_boundary=self.frozen_boundary,
                protected_instructions=level2_protected,  # Use same protected list as apply_distribute_step
                barrier_crossing_opcodes=self.barrier_crossing_opcodes
            )
            
            # Try to run the move - MoveInstructionPass.run() calls verify_optimization internally
            try:
                success = move_pass.run(self.result)
            except SchedulingVerificationError as e:
                # Move caused verification failure!
                move_count += 1
                print(f"\n  [{move_count}] ✗ VERIFICATION FAILED")
                print(f"\n{'=' * 60}")
                print(f"FOUND: Move #{move_count} in Step {failing_step} of Pass {pass_idx} caused verification to fail!")
                print(f"{'=' * 60}")
                print(f"Error: {e}")
                
                # Save the failing state
                step_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_move{move_count:03d}_FAILED.amdgcn")
                save_amdgcn(self.result, step_path, self.block_label)
                
                if move_count > 1:
                    prev_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_move{move_count - 1:03d}.amdgcn")
                else:
                    prev_path = baseline_path
                
                return {
                    "pass_idx": pass_idx,
                    "failing_step": failing_step,
                    "failing_move": move_count,
                    "total_cycles_moved": total_cycles_moved,
                    "passing_file": prev_path,
                    "failing_file": step_path,
                    "error": str(e)
                }
            
            if not success or move_pass.total_cycles_moved == 0:
                print(f"  Move blocked after {move_count} moves, {total_cycles_moved} cycles")
                break
            
            move_count += 1
            total_cycles_moved += move_pass.total_cycles_moved
            
            # Save this state
            step_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_move{move_count:03d}.amdgcn")
            save_amdgcn(self.result, step_path, self.block_label)
            
            # Skip testing for moves before start_move
            if move_count < start_move:
                print(f"  [{move_count}] +{move_pass.total_cycles_moved} cycles (total={total_cycles_moved}/{abs(cycles_to_move)}) [skipped]")
                continue
            
            print(f"  [{move_count}] +{move_pass.total_cycles_moved} cycles (total={total_cycles_moved}/{abs(cycles_to_move)}) ✓")
            
            # Also run custom test if not skipped
            if not self.skip_test:
                test_success, output = self.run_test(step_path)
                if not test_success:
                    print(f"    ✗ TEST FAIL")
                    
                    # Get the previous file for diff
                    if move_count > 1:
                        prev_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_move{move_count - 1:03d}.amdgcn")
                    else:
                        prev_path = baseline_path
                    
                    print(f"\n{'=' * 60}")
                    print(f"FOUND: Move #{move_count} in Step {failing_step} of Pass {pass_idx} caused the test to fail!")
                    print(f"{'=' * 60}")
                    
                    return {
                        "pass_idx": pass_idx,
                        "failing_step": failing_step,
                        "failing_move": move_count,
                        "total_cycles_moved": total_cycles_moved,
                        "passing_file": prev_path,
                        "failing_file": step_path,
                    }
        
        print(f"\nAll {move_count} moves completed without verification failure.")
        return {"success": True, "total_moves": move_count}
    
    def run_distribute_detail_debug(self, pass_idx: int):
        """Run two-level debug for a specific distribute pass."""
        # Apply passes before this one
        if pass_idx > 0:
            print(f"\nApplying passes 0-{pass_idx - 1}...")
            for i in range(pass_idx):
                success, error = self.apply_pass(i)
                if not success:
                    print(f"  Pass {i} failed: {error}")
                    return {"error": f"pass_{i}_failed", "message": error}
        
        # First, try to apply the distribute pass using DistributeInstructionPass
        # to detect any verification errors (same as normal mode)
        print(f"\nChecking pass {pass_idx} with DistributeInstructionPass...")
        success, error = self.apply_pass(pass_idx)
        pass_verification_failed = not success
        
        if pass_verification_failed:
            print(f"✗ Pass {pass_idx} failed verification: {error}")
            print(f"\n{'=' * 60}")
            print(f"Pass {pass_idx} ({format_pass_info(self.passes[pass_idx])}) failed!")
            print(f"Now entering two-level debug to find the exact step/move...")
            print(f"{'=' * 60}")
        else:
            print(f"✓ Pass {pass_idx} verification passed")
            print(f"\n{'=' * 60}")
            print(f"Continuing with step-by-step testing to ensure consistency...")
            print(f"{'=' * 60}")
        
        # Reload state to start fresh for debugging
        print(f"\nReloading analysis to start fresh...")
        self.result = load_analysis_from_json(self.json_path)
        
        # Re-apply passes before this one
        if pass_idx > 0:
            print(f"Re-applying passes 0-{pass_idx - 1}...")
            for i in range(pass_idx):
                self.apply_pass(i)
        
        # Now enter two-level debug mode
        if not self.setup_distribute_debug(pass_idx):
            return {"error": "setup_failed"}
        
        # Check if we should go directly to level 2
        if self.args.detail_step is not None:
            failing_step = self.args.detail_step
            start_move = self.args.start_move if self.args.start_move is not None else 0
            return self.level2_distribute_debug(pass_idx, failing_step, start_move)
        
        # Level 1: Find failing step
        start_step = self.args.start_step or 0
        failing_step = self.level1_distribute_debug(pass_idx, start_step)
        
        if failing_step < 0:
            if pass_verification_failed:
                print(f"\nNo failing step found in pass {pass_idx}.")
                print(f"The verification error may be in the original code, not caused by any step.")
                return {
                    "failing_pass": pass_idx,
                    "pass_info": format_pass_info(self.passes[pass_idx]),
                    "error": "verification_failed_but_no_failing_step",
                    "message": error
                }
            else:
                print(f"\n✓ All steps passed - step-by-step testing consistent with full pass.")
                return {"success": True, "pass_idx": pass_idx, "consistency_verified": True}
        
        # Level 2: Find exact problematic move
        return self.level2_distribute_debug(pass_idx, failing_step)
    
    def run(self):
        """Run the pass list debugging process."""
        if not self.setup():
            return None
        
        # Check if we're doing detailed debug of a specific pass
        if self.args.detail_pass is not None:
            pass_idx = self.args.detail_pass
            
            if pass_idx >= len(self.passes):
                print(f"ERROR: Pass index {pass_idx} out of range (0-{len(self.passes)-1})")
                return {"error": "invalid_pass_index"}
            
            pass_config = self.passes[pass_idx]
            if pass_config['type'] != 'distribute':
                print(f"ERROR: Pass {pass_idx} is not a distribute pass (type={pass_config['type']})")
                print("Two-level debugging is only supported for distribute passes.")
                return {"error": "not_distribute_pass"}
            
            print(f"\nDetail debugging pass {pass_idx}: {format_pass_info(pass_config)}")
            return self.run_distribute_detail_debug(pass_idx)
        
        # Regular pass list testing
        start_pass = self.args.start_pass or 0
        
        print("\n" + "=" * 60)
        print("Testing Pass List")
        print("=" * 60)
        
        # Save baseline
        baseline_path = os.path.join(self.output_dir, "pass_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path)
        
        # Test baseline
        if not self.args.skip_baseline and start_pass == 0:
            print("\nTesting baseline (before any passes)...")
            success, output = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline test failed!")
                print("Output:", output[:500] if len(output) > 500 else output)
                return {"error": "baseline_fails"}
            print("✓ Baseline passes")
        
        # Apply passes before start_pass without testing
        if start_pass > 0:
            print(f"\nSkipping passes 0-{start_pass - 1} (applying without testing)...")
            for i in range(start_pass):
                success, error = self.apply_pass(i)
                if not success:
                    print(f"  Pass {i} failed: {error}")
                    return {"error": f"pass_{i}_apply_failed", "message": error}
            
            # Save state after skipped passes
            skip_state_path = os.path.join(self.output_dir, f"pass_{start_pass:02d}_before.amdgcn")
            save_amdgcn(self.result, skip_state_path)
        
        # Test each remaining pass
        for pass_idx in range(start_pass, len(self.passes)):
            pass_config = self.passes[pass_idx]
            pass_info = format_pass_info(pass_config)
            
            print(f"\n[Pass {pass_idx}/{len(self.passes)-1}] {pass_info}")
            
            # Apply the pass
            print("  Applying pass...", end=" ")
            success, error = self.apply_pass(pass_idx)
            
            if not success:
                print(f"✗ APPLY FAILED: {error}")
                return {
                    "failing_pass": pass_idx,
                    "pass_info": pass_info,
                    "error": "apply_failed",
                    "message": error
                }
            print("done")
            
            # Save state after pass
            pass_path = os.path.join(self.output_dir, f"pass_{pass_idx:02d}_after.amdgcn")
            save_amdgcn(self.result, pass_path)
            
            # Test
            print("  Testing...", end=" ")
            test_success, output = self.run_test(pass_path)
            
            if test_success:
                print("✓ PASS")
            else:
                print("✗ FAIL")
                
                # Get the previous file for diff
                if pass_idx > 0:
                    prev_path = os.path.join(self.output_dir, f"pass_{pass_idx - 1:02d}_after.amdgcn")
                else:
                    prev_path = baseline_path
                
                print(f"\n{'=' * 60}")
                print(f"FOUND: Pass {pass_idx} ({pass_info}) caused the test to fail!")
                print(f"{'=' * 60}")
                
                print(f"\nPassing file: {prev_path}")
                print(f"Failing file: {pass_path}")
                
                # Show relevant error from output
                for keyword in FAILURE_KEYWORDS:
                    if keyword in output:
                        for line in output.split('\n'):
                            if keyword in line:
                                print(f"\nError: {line.strip()}")
                                break
                        break
                
                # Suggest detailed debug if it's a distribute pass
                if pass_config['type'] == 'distribute':
                    print(f"\nTo debug this distribute pass in detail, run:")
                    print(f"  python debug_distribute_pass.py --load-json {self.json_path} \\")
                    print(f"      --pass-list {self.pass_list_path} --detail-pass {pass_idx}")
                
                return {
                    "failing_pass": pass_idx,
                    "pass_info": pass_info,
                    "passing_file": prev_path,
                    "failing_file": pass_path,
                }
        
        # All passes passed
        print(f"\n{'=' * 60}")
        print(f"All {len(self.passes)} passes completed successfully!")
        print(f"{'=' * 60}")
        
        # Save final output
        if self.output_path:
            save_amdgcn(self.result, self.output_path)
            print(f"\nFinal output saved to: {self.output_path}")
        else:
            final_path = os.path.join(self.output_dir, "final.amdgcn")
            save_amdgcn(self.result, final_path)
            print(f"\nFinal output saved to: {final_path}")
        
        return {"success": True, "total_passes": len(self.passes)}


class DistributeDebugger:
    """Two-level debugger for DistributeInstructionPass (legacy mode)."""
    
    def __init__(self, args):
        self.args = args
        self.source_file = args.source
        self.output_dir = args.output_dir
        self.block_label = args.block
        self.target_opcode = args.opcode
        self.K = args.count
        self.verbose = args.verbose
        self.test_cmd = args.test_cmd
        
        if hasattr(args, 'barrier_crossing') and args.barrier_crossing:
            self.barrier_crossing_opcodes = set(args.barrier_crossing.split(','))
        else:
            self.barrier_crossing_opcodes = set()
        
        self.skip_test = args.skip_test
        
        self.result = None
        self.block = None
        self.ddg = None
        self.ideal_cycles = []
        self.frozen_boundary = 0
        self.protected_instructions = []
        
    def log(self, msg, force=False):
        """Print message if verbose or forced."""
        if self.verbose or force:
            print(msg)
    
    def run_test(self, amdgcn_path: str) -> tuple:
        """
        Run test with the given amdgcn file.
        Returns (success: bool, output: str)
        """
        # Skip test if --skip-test is set
        if self.skip_test:
            return True, "Test skipped (--skip-test)"
        
        abs_path = os.path.abspath(amdgcn_path)
        
        if self.test_cmd:
            cmd = self.test_cmd.replace("{FILE}", abs_path)
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                output = result.stdout + result.stderr
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, "Test timed out"
            except Exception as e:
                return False, str(e)
        else:
            env = os.environ.copy()
            env['TRITON_CACHE_DIR'] = TRITON_CACHE_DIR
            env['TRITON_OVERRIDE_AMDGCN_FILE'] = abs_path
            
            if os.path.exists(TRITON_CACHE_DIR):
                shutil.rmtree(TRITON_CACHE_DIR)
            
            cmd = ['python', TEST_SCRIPT] + TEST_ARGS
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                       timeout=TEST_TIMEOUT, env=env, cwd=WORK_DIR)
                output = result.stdout + result.stderr
                
                for keyword in FAILURE_KEYWORDS:
                    if keyword in output:
                        return False, output
                return True, output
            except subprocess.TimeoutExpired:
                return False, "Test timed out"
            except Exception as e:
                return False, str(e)
    
    def setup(self):
        """Initialize analysis and calculate ideal positions."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Generating analysis from {self.source_file}...")
        gen_cmd = f"python amdgcn_edit/amdgcn_ddg.py {self.source_file} -o {self.output_dir} --stats --inter-deps --waitcnt-deps --json-only"
        result = subprocess.run(gen_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to generate analysis: {result.stderr}")
            return False
        
        json_path = os.path.join(self.output_dir, "analysis.json")
        print(f"Loading analysis from {json_path}...")
        self.result = load_analysis_from_json(json_path)
        
        self.block = self.result.cfg.blocks.get(self.block_label)
        self.ddg = self.result.ddgs.get(self.block_label)
        
        if not self.block:
            print(f"ERROR: Block {self.block_label} not found!")
            return False
        
        target_indices = [idx for idx, instr in enumerate(self.block.instructions)
                        if instr.opcode.lower() == self.target_opcode.lower()]
        
        if len(target_indices) < self.K:
            print(f"WARNING: Only found {len(target_indices)} {self.target_opcode} instructions, requested {self.K}")
            self.K = len(target_indices)
        
        print(f"Found {len(target_indices)} {self.target_opcode} instructions")
        
        branch_boundary = len(self.block.instructions)
        for idx, instr in enumerate(self.block.instructions):
            if instr.is_branch or instr.is_terminator:
                branch_boundary = idx
                break
        
        total_cycles = get_cumulative_cycle(self.block, branch_boundary - 1) if branch_boundary > 0 else 0
        print(f"Block total cycles: {total_cycles}, branch boundary: {branch_boundary}")
        
        if self.K > 1:
            cycle_spacing = total_cycles / self.K
            self.ideal_cycles = [int(cycle_spacing * (i + 1)) for i in range(self.K)]
        else:
            self.ideal_cycles = [total_cycles // 2]
        
        print(f"Ideal cycle positions: {self.ideal_cycles[:5]}{'...' if len(self.ideal_cycles) > 5 else ''}")
        
        self.frozen_boundary = 0
        self.protected_instructions = []
        
        return True
    
    def find_target_instruction_index(self, n: int) -> int:
        """Find the current index of the n-th target instruction (0-indexed)."""
        count = 0
        for idx, instr in enumerate(self.block.instructions):
            if instr.opcode.lower() == self.target_opcode.lower():
                if count == n:
                    return idx
                count += 1
        return -1
    
    def apply_step(self, step_num: int, save_path: str = None) -> bool:
        """
        Apply a single distribution step.
        Returns True if move was successful.
        """
        current_idx = self.find_target_instruction_index(step_num)
        if current_idx < 0:
            return False
        
        target_instr = self.block.instructions[current_idx]
        target_cycle = self.ideal_cycles[step_num]
        current_cycle = get_cumulative_cycle(self.block, current_idx)
        cycles_to_move = -(current_cycle - target_cycle)
        
        self.log(f"  Step {step_num + 1}: idx={current_idx}, cycle={current_cycle} -> {target_cycle}, move={cycles_to_move}")
        
        if cycles_to_move == 0:
            self.log(f"    No move needed")
        else:
            move_pass = MoveInstructionPass(
                self.block_label,
                current_idx,
                cycles_to_move,
                verbose=False,
                frozen_boundary=self.frozen_boundary,
                protected_instructions=self.protected_instructions,
                barrier_crossing_opcodes=self.barrier_crossing_opcodes
            )
            move_pass.run(self.result)
            
            if move_pass.total_cycles_moved > 0:
                self.log(f"    Moved {move_pass.total_cycles_moved} cycles")
        
        new_idx = self.find_target_instruction_index(step_num)
        if new_idx >= 0:
            self.frozen_boundary = new_idx
            self.protected_instructions.append(self.block.instructions[new_idx])
        
        if save_path:
            save_amdgcn(self.result, save_path, self.block_label)
        
        return True
    
    def level1_debug(self, start_step: int = 0) -> int:
        """
        Level 1: Test each distribution step.
        Returns the first failing step number, or -1 if all pass.
        """
        print("\n" + "=" * 60)
        print("LEVEL 1: Testing each distribution step")
        print("=" * 60)
        
        if start_step > 0:
            print(f"Skipping steps 1-{start_step} (applying without testing)...")
            for step in range(start_step):
                self.apply_step(step)
        
        baseline_path = os.path.join(self.output_dir, f"level1_step{start_step:02d}_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path, self.block_label)
        
        if not self.args.skip_baseline and start_step == 0:
            print("\nTesting baseline (original file)...")
            success, _ = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline test failed!")
                return 0
            print("✓ Baseline passes")
        
        for step in range(start_step, self.K):
            step_path = os.path.join(self.output_dir, f"level1_step{step + 1:02d}.amdgcn")
            
            print(f"\nStep {step + 1}/{self.K}:", end=" ")
            self.apply_step(step, step_path)
            
            print("Testing...", end=" ", flush=True)
            success, output = self.run_test(step_path)
            
            if success:
                print("✓ PASS")
            else:
                print("✗ FAIL")
                print(f"\n{'=' * 60}")
                print(f"FOUND: Step {step + 1} caused the test to fail!")
                print(f"{'=' * 60}")
                return step + 1
        
        print(f"\nAll {self.K} steps passed!")
        return -1
    
    def level2_debug(self, failing_step: int, start_move: int = 0) -> dict:
        """
        Level 2: Test each individual instruction move within a failing step.
        """
        print("\n" + "=" * 60)
        print(f"LEVEL 2: Detailed analysis of Step {failing_step}")
        if start_move > 0:
            print(f"         Starting from move #{start_move}")
        print("=" * 60)
        
        self.setup()
        
        if failing_step > 1:
            print(f"Applying steps 1-{failing_step - 1} (known to pass)...")
            for step in range(failing_step - 1):
                self.apply_step(step)
        
        baseline_path = os.path.join(self.output_dir, f"level2_step{failing_step}_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path, self.block_label)
        
        if start_move == 0:
            print(f"\nVerifying baseline (before step {failing_step})...")
            success, _ = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline before failing step already fails!")
                return {"error": "baseline_fails"}
            print("✓ Baseline passes")
        else:
            print(f"\nSkipping baseline test (starting from move #{start_move})")
        
        current_idx = self.find_target_instruction_index(failing_step - 1)
        target_instr = self.block.instructions[current_idx]
        target_cycle = self.ideal_cycles[failing_step - 1]
        current_cycle = get_cumulative_cycle(self.block, current_idx)
        cycles_to_move = -(current_cycle - target_cycle)
        
        print(f"\nStep {failing_step} details:")
        print(f"  Target instruction: {self.target_opcode} at idx={current_idx}")
        print(f"  Current cycle: {current_cycle}, Target cycle: {target_cycle}")
        print(f"  Cycles to move: {cycles_to_move} ({'UP' if cycles_to_move > 0 else 'DOWN'})")
        
        if start_move > 0:
            print(f"\nApplying moves 1-{start_move} without testing...")
        print(f"\nTesting individual moves...")
        
        move_count = 0
        total_cycles_moved = 0
        
        while total_cycles_moved < abs(cycles_to_move):
            target_idx = -1
            for idx, instr in enumerate(self.block.instructions):
                if instr is target_instr:
                    target_idx = idx
                    break
            
            if target_idx < 0:
                print(f"  Cannot find target instruction")
                break
            
            small_move = -4 if cycles_to_move < 0 else 4
            
            move_pass = MoveInstructionPass(
                self.block_label,
                target_idx,
                small_move,
                verbose=False,
                frozen_boundary=self.frozen_boundary,
                protected_instructions=self.protected_instructions,
                barrier_crossing_opcodes=self.barrier_crossing_opcodes
            )
            success = move_pass.run(self.result)
            
            if not success or move_pass.total_cycles_moved == 0:
                print(f"  Move blocked after {move_count} moves, {total_cycles_moved} cycles")
                break
            
            move_count += 1
            total_cycles_moved += move_pass.total_cycles_moved
            
            step_path = os.path.join(self.output_dir, f"level2_step{failing_step}_move{move_count:03d}.amdgcn")
            save_amdgcn(self.result, step_path, self.block_label)
            
            if move_count < start_move:
                print(f"  [{move_count}] +{move_pass.total_cycles_moved} cycles (total={total_cycles_moved}/{abs(cycles_to_move)}) [skipped]")
                continue
            
            print(f"  [{move_count}] +{move_pass.total_cycles_moved} cycles (total={total_cycles_moved}/{abs(cycles_to_move)})", end=" ")
            
            test_success, output = self.run_test(step_path)
            
            if test_success:
                print("✓")
            else:
                print("✗ FAIL")
                
                if move_count > 1:
                    prev_path = os.path.join(self.output_dir, f"level2_step{failing_step}_move{move_count - 1:03d}.amdgcn")
                else:
                    prev_path = baseline_path
                
                print(f"\n{'=' * 60}")
                print(f"FOUND: Move #{move_count} caused the failure!")
                print(f"{'=' * 60}")
                
                print(f"\nDiff between passing and failing files:")
                diff_result = subprocess.run(
                    f"diff {prev_path} {step_path} | head -100",
                    shell=True, capture_output=True, text=True
                )
                print(diff_result.stdout)
                
                return {
                    "failing_step": failing_step,
                    "failing_move": move_count,
                    "total_cycles_moved": total_cycles_moved,
                    "passing_file": prev_path,
                    "failing_file": step_path,
                }
        
        print(f"\nAll {move_count} moves completed without failure.")
        print("The bug might be cumulative or in a different subsystem.")
        return {"error": "no_single_move_failure", "total_moves": move_count}
    
    def run(self):
        """Run the two-level debug process."""
        if not self.setup():
            return
        
        if self.args.detail_step is not None:
            start_move = self.args.start_move if self.args.start_move is not None else 0
            result = self.level2_debug(self.args.detail_step, start_move=start_move)
            return result
        
        failing_step = self.level1_debug(self.args.start_step or 0)
        
        if failing_step < 0:
            print("\nNo failing step found. All tests passed!")
            return None
        
        result = self.level2_debug(failing_step)
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Debug tool for AMDGCN transform passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Pass list mode
    parser.add_argument("--load-json", "-l", default=None,
                       help="Load analysis from JSON file")
    parser.add_argument("--pass-list", "-t", default=None,
                       help="Pass list JSON file (e.g., transform_passes_example.json)")
    parser.add_argument("--output", "-r", default=None,
                       help="Output .amdgcn file path")
    parser.add_argument("--start-pass", type=int, default=0,
                       help="Start from pass N (0-indexed, skip earlier passes)")
    parser.add_argument("--detail-pass", type=int, default=None,
                       help="Debug specific pass N in detail (must be distribute type)")
    
    # Two-level debug options (work with both modes)
    parser.add_argument("--start-step", type=int, default=0,
                       help="Start from step N within distribute pass (Level 1)")
    parser.add_argument("--detail-step", type=int, default=None,
                       help="Debug step N in detail (Level 2)")
    parser.add_argument("--start-move", type=int, default=0,
                       help="Start from move K within --detail-step (Level 2)")
    
    # Legacy mode (single distribute)
    parser.add_argument("--source", default="amdgcn_edit/pa_dot_kernel.v2.amdgcn",
                       help="Source .amdgcn file (legacy mode)")
    parser.add_argument("--block", default=".LBB0_2",
                       help="Target block label (legacy mode)")
    parser.add_argument("--opcode", default="global_load_dwordx4",
                       help="Target instruction opcode (legacy mode)")
    parser.add_argument("--count", type=int, default=16,
                       help="Number of instructions to distribute (legacy mode)")
    
    # Common options
    parser.add_argument("--output-dir", default="amdgcn_edit/debug_distribute",
                       help="Output directory for intermediate files")
    parser.add_argument("--test-cmd", default=None,
                       help="Custom test command (use {FILE} as placeholder for amdgcn path)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline test")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip all tests (only apply passes and generate files)")
    parser.add_argument("--barrier-crossing", default=None,
                       help="Comma-separated opcodes allowed to cross s_barrier (legacy mode)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMDGCN Transform Pass Debugger")
    print("=" * 60)
    
    # Determine mode
    if args.load_json and args.pass_list:
        # Pass list mode
        if args.detail_pass is not None:
            print(f"Mode: Pass List - Detailed Debug of Pass {args.detail_pass}")
            if args.detail_step is not None:
                print(f"       Level 2: Step {args.detail_step}, Start Move: {args.start_move}")
            else:
                print(f"       Level 1: Start Step: {args.start_step}")
        else:
            print(f"Mode: Pass List Debug")
        print(f"JSON: {args.load_json}")
        print(f"Pass List: {args.pass_list}")
        print(f"Output: {args.output or 'auto'}")
        print(f"Output Dir: {args.output_dir}")
        
        debugger = PassListDebugger(args)
    else:
        # Legacy mode
        print(f"Mode: Legacy (Single Distribute)")
        print(f"Source: {args.source}")
        print(f"Block: {args.block}")
        print(f"Opcode: {args.opcode}")
        print(f"Count: {args.count}")
        print(f"Output: {args.output_dir}")
        
        debugger = DistributeDebugger(args)
    
    start_time = time.time()
    result = debugger.run()
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"Debug completed in {elapsed:.1f} seconds")
    print("=" * 60)
    
    if result:
        if "success" in result and result["success"]:
            print(f"\nAll passes/steps completed successfully!")
        elif "failing_pass" in result:
            print(f"\nSummary:")
            print(f"  Failing pass: {result['failing_pass']}")
            print(f"  Pass info: {result.get('pass_info', 'N/A')}")
            if "passing_file" in result:
                print(f"  Passing file: {result['passing_file']}")
                print(f"  Failing file: {result['failing_file']}")
                print(f"\nTo analyze the diff:")
                print(f"  diff {result['passing_file']} {result['failing_file']}")
        elif "failing_move" in result:
            print(f"\nSummary:")
            if "pass_idx" in result:
                print(f"  Pass: {result['pass_idx']}")
            print(f"  Failing step: {result['failing_step']}")
            print(f"  Failing move: #{result['failing_move']}")
            print(f"  Passing file: {result['passing_file']}")
            print(f"  Failing file: {result['failing_file']}")
            print(f"\nTo analyze the diff:")
            print(f"  diff {result['passing_file']} {result['failing_file']}")


if __name__ == "__main__":
    main()
