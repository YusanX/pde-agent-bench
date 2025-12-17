"""Command-line interface for PDEBench."""
import argparse
import json
import sys
from pathlib import Path

from .core.generate import generate
from .core.solve import solve_case
from .core.evaluate import evaluate


def load_case(case_file):
    """Load case specification from JSON file."""
    with open(case_file, 'r') as f:
        return json.load(f)


def cmd_generate(args):
    """Generate phase: build system and reference solution."""
    case_spec = load_case(args.case)
    outdir = Path(args.outdir)
    
    print(f"[generate] Case: {case_spec['id']}")
    print(f"[generate] Output: {outdir}")
    
    generate(case_spec, outdir)


def cmd_solve(args):
    """Solve phase: use baseline solver."""
    case_spec = load_case(args.case)
    outdir = Path(args.outdir)
    
    print(f"[solve] Case: {case_spec['id']}")
    print(f"[solve] Output: {outdir}")
    
    # Parse KSP overrides if provided
    ksp_params = {}
    if args.ksp_type:
        ksp_params['type'] = args.ksp_type
    if args.pc_type:
        ksp_params['pc_type'] = args.pc_type
    if args.ksp_rtol:
        ksp_params['rtol'] = float(args.ksp_rtol)
    
    solve_case(case_spec, outdir, ksp_params if ksp_params else None)


def cmd_evaluate(args):
    """Evaluate phase: compute metrics."""
    case_spec = load_case(args.case)
    outdir = Path(args.outdir)
    
    print(f"[evaluate] Case: {case_spec['id']}")
    print(f"[evaluate] Output: {outdir}")
    
    evaluate(case_spec, outdir)


def cmd_run(args):
    """Run all phases: generate + solve + evaluate."""
    case_spec = load_case(args.case)
    outdir = Path(args.outdir)
    
    print(f"[run] Case: {case_spec['id']}")
    print(f"[run] Output: {outdir}")
    
    # Generate
    print("\n" + "="*60)
    print("PHASE 1: GENERATE")
    print("="*60)
    generate(case_spec, outdir)
    
    # Solve
    print("\n" + "="*60)
    print("PHASE 2: SOLVE")
    print("="*60)
    ksp_params = {}
    if args.ksp_type:
        ksp_params['type'] = args.ksp_type
    if args.pc_type:
        ksp_params['pc_type'] = args.pc_type
    if args.ksp_rtol:
        ksp_params['rtol'] = float(args.ksp_rtol)
    
    solve_case(case_spec, outdir, ksp_params if ksp_params else None)
    
    # Evaluate
    print("\n" + "="*60)
    print("PHASE 3: EVALUATE")
    print("="*60)
    metrics = evaluate(case_spec, outdir)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Validity: {metrics['validity']['pass']}")
    print(f"Reason: {metrics['validity']['reason']}")
    
    if not metrics['validity']['pass']:
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PDEBench: Minimal PDE benchmark with FEniCSx',
        prog='python -m pdebench.cli'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    subparsers.required = True
    
    # Generate command
    parser_gen = subparsers.add_parser('generate', help='Generate phase: build system and reference')
    parser_gen.add_argument('case', help='Path to case JSON file')
    parser_gen.add_argument('--outdir', required=True, help='Output directory')
    parser_gen.set_defaults(func=cmd_generate)
    
    # Solve command
    parser_solve = subparsers.add_parser('solve', help='Solve phase: use baseline solver')
    parser_solve.add_argument('case', help='Path to case JSON file')
    parser_solve.add_argument('--outdir', required=True, help='Output directory')
    parser_solve.add_argument('--ksp-type', help='Override KSP type')
    parser_solve.add_argument('--pc-type', help='Override preconditioner type')
    parser_solve.add_argument('--ksp-rtol', help='Override KSP relative tolerance')
    parser_solve.set_defaults(func=cmd_solve)
    
    # Evaluate command
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate phase: compute metrics')
    parser_eval.add_argument('case', help='Path to case JSON file')
    parser_eval.add_argument('--outdir', required=True, help='Output directory')
    parser_eval.set_defaults(func=cmd_evaluate)
    
    # Run command (all phases)
    parser_run = subparsers.add_parser('run', help='Run all phases')
    parser_run.add_argument('case', help='Path to case JSON file')
    parser_run.add_argument('--outdir', required=True, help='Output directory')
    parser_run.add_argument('--ksp-type', help='Override KSP type')
    parser_run.add_argument('--pc-type', help='Override preconditioner type')
    parser_run.add_argument('--ksp-rtol', help='Override KSP relative tolerance')
    parser_run.set_defaults(func=cmd_run)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

