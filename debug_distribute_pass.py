#!/usr/bin/env python3
"""
Two-Level Debug Tool for DistributeInstructionPass

This tool uses a binary-search-like approach to find problematic instruction moves:
1. Level 1: Test each distribution step (coarse-grained)
2. Level 2: For the failing step, test each individual instruction move (fine-grained)

Usage:
    python debug_distribute_pass.py [options]

Options:
    --source FILE       Source .amdgcn file (default: amdgcn_edit/pa_dot_kernel.v2.amdgcn)
    --output-dir DIR    Output directory for intermediate files (default: amdgcn_edit/debug_distribute)
    --block LABEL       Target block label (default: .LBB0_2)
    --opcode OPCODE     Target instruction opcode (default: global_load_dwordx4)
    --count K           Number of instructions to distribute (default: 16)
    --test-cmd CMD      Custom test command (default: built-in test)
    --start-step N      Start from step N (skip earlier steps)
    --detail-step N     Directly run detailed analysis for step N
    --start-move K      Start from move K within --detail-step (skip earlier moves)
    --skip-baseline     Skip baseline test
    --verbose           Show verbose output

Examples:
    # Full two-level debug
    python debug_distribute_pass.py
    
    # Start from step 3
    python debug_distribute_pass.py --start-step 3
    
    # Directly analyze step 3 in detail
    python debug_distribute_pass.py --detail-step 3
    
    # Analyze step 3 starting from move 40
    python debug_distribute_pass.py --detail-step 3 --start-move 40
    
    # Custom parameters
    python debug_distribute_pass.py --block .LBB0_2 --opcode buffer_load_dword --count 8
"""

import sys
import os
import subprocess
import shutil
import argparse
import time
import functools

# Enable unbuffered output for real-time printing
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amdgcn_ddg import AnalysisResult, load_analysis_from_json, dump_block_instructions
from amdgcn_passes import MoveInstructionPass, get_instruction_cycles, sync_instruction_to_raw_lines
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
    
    Args:
        cfg: The CFG containing basic blocks
        block_label: Label of the block to dump (e.g., ".LBB0_2")
        output_dir: Directory to write the block instruction file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if block_label not in cfg.blocks:
        print(f"Warning: Block {block_label} not found in CFG")
        return
    
    block = cfg.blocks[block_label]
    
    # Create filename: .LBB0_2 -> LBB0_2.txt
    filename = block_label.lstrip('.') + '.txt'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"; Block: {block_label}\n")
        f.write(f"; Total instructions: {len(block.instructions)}\n")
        f.write(f";\n")
        
        for idx, instr in enumerate(block.instructions):
            # Format: [idx] opcode operands
            if instr.raw_line:
                # Use raw_line but strip leading whitespace
                line = instr.raw_line.strip()
            else:
                # Fallback to opcode + operands
                line = f"{instr.opcode} {instr.operands}" if instr.operands else instr.opcode
            
            f.write(f"[{idx}] {line}\n")


def save_amdgcn(result: AnalysisResult, output_path: str, block_label: str = None):
    """
    Save the current state as an amdgcn file.
    
    If block_label is provided, also creates a directory with the same name as
    the amdgcn file (without extension) and dumps the block instructions there.
    """
    result.to_amdgcn(output_path)
    
    # If block_label is provided, dump the block instructions
    if block_label:
        # Create directory with same name as amdgcn file (without .amdgcn extension)
        dump_dir = output_path.rsplit('.amdgcn', 1)[0]
        dump_single_block_instructions(result.cfg, block_label, dump_dir)


class DistributeDebugger:
    """Two-level debugger for DistributeInstructionPass."""
    
    def __init__(self, args):
        self.args = args
        self.source_file = args.source
        self.output_dir = args.output_dir
        self.block_label = args.block
        self.target_opcode = args.opcode
        self.K = args.count
        self.verbose = args.verbose
        self.test_cmd = args.test_cmd
        
        self.result = None
        self.block = None
        self.ddg = None
        self.ideal_cycles = []
        self.frozen_boundary = -1
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
        abs_path = os.path.abspath(amdgcn_path)
        
        if self.test_cmd:
            # Custom test command
            cmd = self.test_cmd.replace("{FILE}", abs_path)
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                output = result.stdout + result.stderr
                # Assume non-zero exit code means failure
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, "Test timed out"
            except Exception as e:
                return False, str(e)
        else:
            # Default test command using global configuration
            env = os.environ.copy()
            env['TRITON_CACHE_DIR'] = TRITON_CACHE_DIR
            env['TRITON_OVERRIDE_AMDGCN_FILE'] = abs_path
            
            # Clear triton cache
            if os.path.exists(TRITON_CACHE_DIR):
                shutil.rmtree(TRITON_CACHE_DIR)
            
            # Build test command
            cmd = ['python', TEST_SCRIPT] + TEST_ARGS
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                       timeout=TEST_TIMEOUT, env=env, cwd=WORK_DIR)
                output = result.stdout + result.stderr
                
                # Check for failure keywords
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
        
        # Generate fresh analysis.json
        print(f"Generating analysis from {self.source_file}...")
        gen_cmd = f"python amdgcn_edit/amdgcn_ddg.py {self.source_file} -o {self.output_dir} --stats --inter-deps --waitcnt-deps --json-only"
        result = subprocess.run(gen_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to generate analysis: {result.stderr}")
            return False
        
        # Load analysis
        json_path = os.path.join(self.output_dir, "analysis.json")
        print(f"Loading analysis from {json_path}...")
        self.result = load_analysis_from_json(json_path)
        
        self.block = self.result.cfg.blocks.get(self.block_label)
        self.ddg = self.result.ddgs.get(self.block_label)
        
        if not self.block:
            print(f"ERROR: Block {self.block_label} not found!")
            return False
        
        # Find target instructions
        target_indices = [idx for idx, instr in enumerate(self.block.instructions)
                        if instr.opcode.lower() == self.target_opcode.lower()]
        
        if len(target_indices) < self.K:
            print(f"WARNING: Only found {len(target_indices)} {self.target_opcode} instructions, requested {self.K}")
            self.K = len(target_indices)
        
        print(f"Found {len(target_indices)} {self.target_opcode} instructions")
        
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
        
        self.frozen_boundary = -1
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
                protected_instructions=self.protected_instructions
            )
            move_pass.run(self.result)
            
            if move_pass.total_cycles_moved > 0:
                self.log(f"    Moved {move_pass.total_cycles_moved} cycles")
        
        # Update frozen boundary and protected instructions
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
        
        # Apply steps before start_step without testing
        if start_step > 0:
            print(f"Skipping steps 1-{start_step} (applying without testing)...")
            for step in range(start_step):
                self.apply_step(step)
        
        # Save baseline state
        baseline_path = os.path.join(self.output_dir, f"level1_step{start_step:02d}_baseline.amdgcn")
        save_amdgcn(self.result, baseline_path, self.block_label)
        
        if not self.args.skip_baseline and start_step == 0:
            print("\nTesting baseline (original file)...")
            success, _ = self.run_test(baseline_path)
            if not success:
                print("ERROR: Baseline test failed!")
                return 0
            print("✓ Baseline passes")
        
        # Test each remaining step
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
        
        Args:
            failing_step: The step number to analyze in detail
            start_move: Start testing from move K (skip earlier moves, 0 = start from beginning)
        
        Returns details about the problematic move.
        """
        print("\n" + "=" * 60)
        print(f"LEVEL 2: Detailed analysis of Step {failing_step}")
        if start_move > 0:
            print(f"         Starting from move #{start_move}")
        print("=" * 60)
        
        # Reset and apply steps up to (but not including) the failing step
        self.setup()
        
        if failing_step > 1:
            print(f"Applying steps 1-{failing_step - 1} (known to pass)...")
            for step in range(failing_step - 1):
                self.apply_step(step)
        
        # Save state before the failing step
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
        
        # Get details for the failing step
        current_idx = self.find_target_instruction_index(failing_step - 1)
        target_instr = self.block.instructions[current_idx]
        target_cycle = self.ideal_cycles[failing_step - 1]
        current_cycle = get_cumulative_cycle(self.block, current_idx)
        cycles_to_move = -(current_cycle - target_cycle)
        
        print(f"\nStep {failing_step} details:")
        print(f"  Target instruction: {self.target_opcode} at idx={current_idx}")
        print(f"  Current cycle: {current_cycle}, Target cycle: {target_cycle}")
        print(f"  Cycles to move: {cycles_to_move} ({'UP' if cycles_to_move > 0 else 'DOWN'})")
        
        # Perform fine-grained moves
        if start_move > 0:
            print(f"\nApplying moves 1-{start_move} without testing...")
        print(f"\nTesting individual moves...")
        
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
                protected_instructions=self.protected_instructions
            )
            success = move_pass.run(self.result)
            
            if not success or move_pass.total_cycles_moved == 0:
                print(f"  Move blocked after {move_count} moves, {total_cycles_moved} cycles")
                break
            
            move_count += 1
            total_cycles_moved += move_pass.total_cycles_moved
            
            # Save this state
            step_path = os.path.join(self.output_dir, f"level2_step{failing_step}_move{move_count:03d}.amdgcn")
            save_amdgcn(self.result, step_path, self.block_label)
            
            # Skip testing for moves before start_move
            if move_count < start_move:
                print(f"  [{move_count}] +{move_pass.total_cycles_moved} cycles (total={total_cycles_moved}/{abs(cycles_to_move)}) [skipped]")
                continue
            
            print(f"  [{move_count}] +{move_pass.total_cycles_moved} cycles (total={total_cycles_moved}/{abs(cycles_to_move)})", end=" ")
            
            # Test
            test_success, output = self.run_test(step_path)
            
            if test_success:
                print("✓")
            else:
                print("✗ FAIL")
                
                # Get the previous file for diff
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
        
        if self.args.detail_step:
            # Directly run level 2 for specified step
            start_move = getattr(self.args, 'start_move', 0) or 0
            result = self.level2_debug(self.args.detail_step, start_move=start_move)
            return result
        
        # Level 1: Find failing step
        failing_step = self.level1_debug(self.args.start_step or 0)
        
        if failing_step < 0:
            print("\nNo failing step found. All tests passed!")
            return None
        
        # Level 2: Find exact problematic move
        result = self.level2_debug(failing_step)
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Two-level debug tool for DistributeInstructionPass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--source", default="amdgcn_edit/pa_dot_kernel.v2.amdgcn",
                       help="Source .amdgcn file")
    parser.add_argument("--output-dir", default="amdgcn_edit/debug_distribute",
                       help="Output directory for intermediate files")
    parser.add_argument("--block", default=".LBB0_2",
                       help="Target block label")
    parser.add_argument("--opcode", default="global_load_dwordx4",
                       help="Target instruction opcode")
    parser.add_argument("--count", type=int, default=16,
                       help="Number of instructions to distribute")
    parser.add_argument("--test-cmd", default=None,
                       help="Custom test command (use {FILE} as placeholder for amdgcn path)")
    parser.add_argument("--start-step", type=int, default=0,
                       help="Start from step N (skip earlier steps)")
    parser.add_argument("--detail-step", type=int, default=None,
                       help="Directly run detailed analysis for step N")
    parser.add_argument("--start-move", type=int, default=0,
                       help="Start from move K within --detail-step (skip earlier moves)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline test")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DistributeInstructionPass Two-Level Debugger")
    print("=" * 60)
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
    
    if result and "failing_move" in result:
        print(f"\nSummary:")
        print(f"  Failing step: {result['failing_step']}")
        print(f"  Failing move: #{result['failing_move']}")
        print(f"  Passing file: {result['passing_file']}")
        print(f"  Failing file: {result['failing_file']}")
        print(f"\nTo analyze the diff:")
        print(f"  diff {result['passing_file']} {result['failing_file']}")


if __name__ == "__main__":
    main()

