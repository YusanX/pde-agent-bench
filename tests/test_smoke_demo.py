"""Smoke tests: run all demo cases and check validity."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
import shutil


def get_demo_cases():
    """Get all demo case files."""
    # tests/ is at project root, cases/ is at project root  
    cases_dir = Path(__file__).parent.parent / 'cases' / 'demo'
    return list(cases_dir.glob('*.json'))


@pytest.mark.parametrize('case_file', get_demo_cases(), ids=lambda x: x.stem)
def test_run_demo_case(case_file, tmp_path):
    """Test running a demo case end-to-end."""
    case_id = case_file.stem
    outdir = tmp_path / case_id
    
    # Run the full pipeline
    cmd = [
        sys.executable, '-m', 'pdebench.cli', 'run',
        str(case_file),
        '--outdir', str(outdir)
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent  # Project root
    )
    
    # Check that command succeeded
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    
    assert result.returncode == 0, f"Command failed for {case_id}"
    
    # Check that metrics.json exists
    metrics_file = outdir / 'metrics.json'
    assert metrics_file.exists(), f"metrics.json not found for {case_id}"
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Check validity
    assert 'validity' in metrics, "No validity field in metrics"
    assert metrics['validity']['pass'] == True, f"Validity check failed: {metrics['validity']['reason']}"
    
    # Check rel_res (residual) - should be very small
    if 'rel_res' in metrics:
        assert metrics['rel_res'] < 1e-8, f"Residual too large: {metrics['rel_res']}"
    
    # Check that target metric is met (already checked in validity, but double-check)
    with open(case_file, 'r') as f:
        case_spec = json.load(f)
    
    target_metric = case_spec['targets']['metric']
    target_error = case_spec['targets']['target_error']
    
    if target_metric in metrics:
        achieved_error = metrics[target_metric]
        assert achieved_error <= target_error, \
            f"{target_metric}={achieved_error} exceeds target {target_error}"

