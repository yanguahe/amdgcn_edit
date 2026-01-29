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
    get_instruction_cycles, sync_instruction_to_raw_lines,
    # New modular interfaces
    MoveExecutor, DistributeStepExecutor, RegisterReplaceExecutor,
    StepResult, SingleMoveInfo, MoveYieldInfo, DistributeContext, RegisterReplaceContext
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
    "exceeded the error threshold",
    "HIP error",
    "illegal memory access",
    "AcceleratorError",
    "RuntimeError",
    "Traceback (most recent call last)"
]

# ============================================================================


# Use canonical implementation from DistributeStepExecutor
def get_cumulative_cycle(block: BasicBlock, idx: int) -> int:
    """Calculate cumulative cycle count up to instruction at idx.
    
    Note: This is a wrapper around DistributeStepExecutor.get_cumulative_cycle
    for backward compatibility.
    """
    return DistributeStepExecutor.get_cumulative_cycle(block, idx)


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
            
            # Use global address instead of local index for consistency with amdgcn_ddg.py
            f.write(f"[{instr.address}] {line}\n")


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
                'cycles': pass_config['cycles']
            })
        elif pass_type == 'distribute':
            passes.append({
                'type': 'distribute',
                'block': pass_config['block'],
                'opcode': pass_config['opcode'],
                'k': pass_config['k']
            })
        elif pass_type == 'replace_registers':
            passes.append({
                'type': 'replace_registers',
                'range_start': pass_config['range_start'],
                'range_end': pass_config['range_end'],
                'registers': pass_config['registers'],
                'alignments': pass_config.get('alignments', [1] * len(pass_config['registers'])),
                'target_opcodes': pass_config.get('target_opcodes', [])
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
        target_ops = pass_config.get('target_opcodes', [])
        ops_str = f", opcodes={target_ops}" if target_ops else ""
        return f"replace_registers(range={pass_config['range_start']}-{pass_config['range_end']}, regs={pass_config['registers']}{ops_str})"
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
                # Check keywords first
                for keyword in FAILURE_KEYWORDS:
                    if keyword in output:
                        return False, output
                # Then check exit code
                if result.returncode != 0:
                    return False, f"Process exited with code {result.returncode}\n{output}"
                return True, output
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
                
                # Check for failure keywords in output
                for keyword in FAILURE_KEYWORDS:
                    if keyword in output:
                        return False, output
                
                # Also check exit code (non-zero indicates failure)
                if result.returncode != 0:
                    return False, f"Process exited with code {result.returncode}\n{output}"
                
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
                    verbose=self.verbose
                )
                move_pass.run(self.result)
                return True, None
                
            elif pass_type == 'distribute':
                dist_pass = DistributeInstructionPass(
                    block_label=pass_config['block'],
                    target_opcode=pass_config['opcode'],
                    distribute_count=pass_config['k'],
                    verbose=self.verbose
                )
                dist_pass.run(self.result)
                return True, None
                
            elif pass_type == 'replace_registers':
                replace_pass = RegisterReplacePass(
                    range_start=pass_config['range_start'],
                    range_end=pass_config['range_end'],
                    registers_to_replace=pass_config['registers'],
                    alignments=pass_config['alignments'],
                    target_opcodes=pass_config.get('target_opcodes', []),
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
        Uses DistributeStepExecutor.create_context for consistent parameter computation.
        Returns True if successful.
        """
        pass_config = self.passes[pass_idx]
        
        if pass_config['type'] != 'distribute':
            print(f"ERROR: Pass {pass_idx} is not a distribute pass!")
            return False
        
        self.block_label = pass_config['block']
        self.target_opcode = pass_config['opcode']
        self.K = pass_config['k']
        
        # Use DistributeStepExecutor.create_context for consistent setup
        self.distribute_ctx = DistributeStepExecutor.create_context(
            self.result,
            self.block_label,
            self.target_opcode,
            self.K
        )
        
        if self.distribute_ctx is None:
            print(f"ERROR: Failed to create distribute context for {self.block_label}")
            return False
        
        # Create the executor
        self.distribute_executor = DistributeStepExecutor(self.result, self.distribute_ctx)
        
        # Export commonly used attributes for backward compatibility
        self.block = self.distribute_ctx.block
        self.K = self.distribute_ctx.K
        self.ideal_cycles = self.distribute_ctx.ideal_cycles
        self.all_target_instrs = self.distribute_ctx.all_target_instrs
        self.frozen_boundary = 0
        
        # Print info
        print(f"Found {self.distribute_ctx.M} {self.target_opcode} instructions in {self.block_label}")
        print(f"Block total cycles: {self.distribute_ctx.total_cycles}, branch boundary: {self.distribute_ctx.branch_boundary}")
        print(f"Ideal cycle positions: {self.ideal_cycles[:5]}{'...' if len(self.ideal_cycles) > 5 else ''}")
        
        return True
    
    def find_target_instruction_index(self, n: int) -> int:
        """Find the current index of the n-th target instruction (0-indexed).
        
        Uses DistributeStepExecutor if available for consistency.
        """
        if hasattr(self, 'distribute_executor'):
            return self.distribute_executor.find_nth_target_index(n)
        
        # Fallback for backward compatibility
        count = 0
        for idx, instr in enumerate(self.block.instructions):
            if instr.opcode.lower() == self.target_opcode.lower():
                if count == n:
                    return idx
                count += 1
        return -1
    
    def apply_distribute_step(self, step_num: int, save_path: str = None) -> bool:
        """
        Apply a single distribution step using DistributeStepExecutor.
        Returns True if move was successful.
        """
        # Sync executor's frozen_boundary with our state
        if hasattr(self, 'distribute_executor'):
            self.distribute_executor.frozen_boundary = self.frozen_boundary
        
        # Get step params for logging
        params = self.distribute_executor.get_step_params(step_num) if hasattr(self, 'distribute_executor') else {}
        
        if params.get("error"):
            self.log(f"  Step {step_num + 1}: Error - {params.get('error')}")
            return False
        
        self.log(f"  Step {step_num + 1}: idx={params.get('current_idx')} -> {params.get('target_idx')}, "
                f"cycle={params.get('current_cycle')} -> {params.get('target_cycle')}, "
                f"move={params.get('cycles_to_move')}")
        
        # Execute the step
        result = self.distribute_executor.execute_step(step_num)
        
        if result.cycles_moved > 0:
            self.log(f"    Moved {result.cycles_moved} cycles")
        elif result.cycles_moved == 0 and params.get('cycles_to_move', 0) == 0:
            self.log(f"    No move needed")
        
        # Sync frozen_boundary back from executor
        self.frozen_boundary = self.distribute_executor.frozen_boundary
        
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
        
        # Get the correct execution order (matches DistributeInstructionPass.run() two-phase strategy)
        # Phase 1: UP moves (forward order), Phase 2: DOWN moves (reverse order)
        execution_order = self.distribute_executor.get_execution_order()
        print(f"Execution order (two-phase): {execution_order}")
        
        # Apply steps before start_step without testing
        if start_step > 0:
            print(f"\nSkipping steps 1-{start_step} (applying without testing)...")
            for order_idx in range(start_step):
                step = execution_order[order_idx]
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
        
        # Test each remaining step using the correct execution order
        for order_idx in range(start_step, self.K):
            step = execution_order[order_idx]
            step_path = os.path.join(self.output_dir, f"pass{pass_idx}_step{order_idx + 1:02d}.amdgcn")
            
            print(f"\nStep {order_idx + 1}/{self.K} (instruction {step}):", end=" ")
            
            # Apply the step and catch any verification errors from MoveInstructionPass.run()
            try:
                self.apply_distribute_step(step, step_path)
            except SchedulingVerificationError as e:
                print("✗ VERIFICATION FAILED (during move)")
                print(f"\n{'=' * 60}")
                print(f"FOUND: Step {order_idx + 1} (instruction {step}) of pass {pass_idx} caused verification to fail!")
                print(f"{'=' * 60}")
                print(f"Error: {e}")
                return order_idx + 1
            
            # Additional verification after this step (in case apply_distribute_step doesn't use MoveInstructionPass)
            print("Verifying...", end=" ", flush=True)
            try:
                verify_optimization(original_gdg, self.result.cfg)
                print("✓ PASS")
            except SchedulingVerificationError as e:
                print("✗ VERIFICATION FAILED")
                print(f"\n{'=' * 60}")
                print(f"FOUND: Step {order_idx + 1} (instruction {step}) of pass {pass_idx} caused verification to fail!")
                print(f"{'=' * 60}")
                print(f"Error: {e}")
                return order_idx + 1
            
            # Also run custom test if not skipped
            if not self.skip_test:
                print("  Running test...", end=" ", flush=True)
                success, output = self.run_test(step_path)
                if not success:
                    print("✗ TEST FAIL")
                    return order_idx + 1
                print("✓")
        
        print(f"\nAll {self.K} steps passed verification!")
        return -1
    
    def level2_distribute_debug(self, pass_idx: int, failing_step: int, start_move: int = 0) -> dict:
        """
        Level 2: Test each logical move unit within a step.
        
        Uses MoveInstructionPass.iter_moves() via DistributeStepExecutor.execute_step_with_iter()
        to get fine-grained control, testing after each logical move unit
        (normal swap, SCC pair skip, or recursive move).
        
        Args:
            pass_idx: Index of the distribute pass being debugged
            failing_step: Execution order position (1-indexed) to debug - this matches
                         what level1_distribute_debug returns
            start_move: Skip testing moves before this number (0 = test all)
            
        Returns:
            Dict with debug results including failing_move if found
        """
        print("\n" + "=" * 60)
        print(f"LEVEL 2: Detailed analysis of Pass {pass_idx}, Step {failing_step}")
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
        
        # Get the correct execution order (matches DistributeInstructionPass.run())
        execution_order = self.distribute_executor.get_execution_order()
        print(f"Execution order (two-phase): {execution_order}")
        
        # Convert failing_step (1-indexed execution position) to actual step number
        exec_idx = failing_step - 1  # Convert to 0-indexed
        if exec_idx >= len(execution_order):
            return {"error": "invalid_step", "message": f"Step {failing_step} out of range"}
        
        actual_step_num = execution_order[exec_idx]
        print(f"Step {failing_step} in execution order = instruction {actual_step_num}")
        
        # Apply steps before the failing step (in execution order)
        if exec_idx > 0:
            print(f"\nApplying steps 1-{exec_idx} in execution order (known to pass)...")
            for order_pos in range(exec_idx):
                step = execution_order[order_pos]
                self.apply_distribute_step(step)
        
        # Save baseline state
        baseline_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path, self.block_label)
        
        print(f"\nVerifying baseline (before step {failing_step})...")
        if not self.skip_test:
            success, _ = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline before failing step already fails!")
                return {"error": "baseline_fails"}
            print("Baseline passes")
        
        # Use actual_step_num (the real step number, not execution order position)
        step_num = actual_step_num
        
        # Sync frozen boundary with executor
        self.distribute_executor.frozen_boundary = self.frozen_boundary
        
        # Use executor to get step parameters
        params = self.distribute_executor.get_step_params(step_num)
        
        current_idx = params['current_idx']
        current_cycle = params['current_cycle']
        target_cycle = params['target_cycle']
        cycles_to_move = params['cycles_to_move']
        
        print(f"\nStep {failing_step} details:")
        print(f"  Target instruction: {self.target_opcode} at idx={current_idx}")
        print(f"  Current cycle: {current_cycle}, Target cycle: {target_cycle}")
        print(f"  Cycles to move: {cycles_to_move} ({'UP' if cycles_to_move > 0 else 'DOWN'})")
        
        if start_move > 0:
            print(f"  Starting from move {start_move} (skipping earlier moves)")
        
        print(f"\nExecuting step {failing_step} with per-move testing...")
        
        # Track state for callback
        failing_move_info = [None]  # Use list to allow modification in nested function
        prev_path = [baseline_path]
        last_move_path = [None]
        
        def on_move(move_info: MoveYieldInfo) -> bool:
            """Callback for each logical move unit."""
            # Save current state
            move_path = os.path.join(
                self.output_dir,
                f"pass{pass_idx}_step{failing_step}_move{move_info.move_num:03d}.amdgcn"
            )
            save_amdgcn(self.result, move_path, self.block_label)
            last_move_path[0] = move_path
            
            # Print move info
            print(f"  Move {move_info.move_num}: type={move_info.move_type}, "
                  f"swaps={move_info.swaps_in_this_move}, "
                  f"cycles={move_info.cycles_this_move}/{move_info.total_cycles_moved}")
            
            # Skip testing if before start_move
            if move_info.move_num < start_move:
                prev_path[0] = move_path
                return True  # Continue
            
            # Run test
            if not self.skip_test:
                test_success, _ = self.run_test(move_path)
                if not test_success:
                    failing_move_info[0] = move_info
                    print(f"    FAILED at move {move_info.move_num}")
                    return False  # Stop iteration
                print(f"    passed")
            
            prev_path[0] = move_path
            return True  # Continue
        
        # Execute step with move-by-move callbacks
        result = self.distribute_executor.execute_step_with_iter(
            step_num=step_num,
            on_move=on_move
        )
        
        # Sync frozen boundary back
        self.frozen_boundary = self.distribute_executor.frozen_boundary
        
        # Check if we found a failing move
        if failing_move_info[0] is not None:
            move_info = failing_move_info[0]
            print(f"\n{'=' * 60}")
            print(f"FOUND: Move {move_info.move_num} of Step {failing_step} (Pass {pass_idx}) failed!")
            print(f"{'=' * 60}")
            print(f"  Move type: {move_info.move_type}")
            print(f"  Swaps in this move: {move_info.swaps_in_this_move}")
            print(f"  Cycles this move: {move_info.cycles_this_move}")
            print(f"  Passing file: {prev_path[0]}")
            print(f"  Failing file: {last_move_path[0]}")
            print(f"\nTo analyze the diff:")
            print(f"  diff {prev_path[0]} {last_move_path[0]}")
            
            return {
                "pass_idx": pass_idx,
                "failing_step": failing_step,
                "failing_move": move_info.move_num,
                "move_type": move_info.move_type,
                "passing_file": prev_path[0],
                "failing_file": last_move_path[0]
            }
        
        # Save final state
        final_path = os.path.join(self.output_dir, f"pass{pass_idx}_level2_step{failing_step}_final.amdgcn")
        save_amdgcn(self.result, final_path, self.block_label)
        
        print(f"\nStep {failing_step} completed: {result.move_count} moves, {result.cycles_moved} cycles")
        print(f"AMDGCN file saved to: {final_path}")
        
        # Compare with Level 1's output to ensure consistency
        level1_path = os.path.join(self.output_dir, f"pass{pass_idx}_step{failing_step:02d}.amdgcn")
        if os.path.exists(level1_path):
            diff_result = subprocess.run(
                ['diff', '-q', level1_path, final_path],
                capture_output=True, text=True
            )
            if diff_result.returncode == 0:
                print(f"\nLevel 2 output is IDENTICAL to Level 1 ({level1_path})")
            else:
                print(f"\nWARNING: Level 2 output DIFFERS from Level 1!")
                print(f"  Level 1 file: {level1_path}")
                print(f"  Level 2 file: {final_path}")
                print(f"  Run 'diff {level1_path} {final_path}' to see differences")
        
        print(f"\nAll {result.move_count} moves completed without failure.")
        return {"success": True, "total_moves": result.move_count, "cycles_moved": result.cycles_moved}
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Debug tool for AMDGCN transform passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument("--load-json", "-l", required=True,
                       help="Load analysis from JSON file")
    parser.add_argument("--pass-list", "-t", required=True,
                       help="Pass list JSON file (e.g., transform_passes_example.json)")
    
    # Optional pass list mode arguments
    parser.add_argument("--output", "-r", default=None,
                       help="Output .amdgcn file path")
    parser.add_argument("--start-pass", type=int, default=0,
                       help="Start from pass N (0-indexed, skip earlier passes)")
    parser.add_argument("--detail-pass", type=int, default=None,
                       help="Debug specific pass N in detail (must be distribute type)")
    
    # Two-level debug options
    parser.add_argument("--start-step", type=int, default=0,
                       help="Start from step N within distribute pass (Level 1)")
    parser.add_argument("--detail-step", type=int, default=None,
                       help="Debug step N in detail (Level 2)")
    parser.add_argument("--start-move", type=int, default=0,
                       help="Start from move K within --detail-step (Level 2)")
    
    # Common options
    parser.add_argument("--output-dir", default="amdgcn_edit/debug_distribute",
                       help="Output directory for intermediate files")
    parser.add_argument("--test-cmd", default=None,
                       help="Custom test command (use {FILE} as placeholder for amdgcn path)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline test")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip all tests (only apply passes and generate files)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMDGCN Transform Pass Debugger")
    print("=" * 60)
    
    # Print mode information
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
