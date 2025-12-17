"""Test that all demo cases validate against the schema."""
import json
from pathlib import Path
import jsonschema
import pytest


def get_demo_cases():
    """Get all demo case files."""
    # tests/ is at project root, cases/ is at project root
    cases_dir = Path(__file__).parent.parent / 'cases' / 'demo'
    return list(cases_dir.glob('*.json'))


def load_schema():
    """Load the case schema."""
    schema_file = Path(__file__).parent.parent / 'cases' / 'schema.case.json'
    with open(schema_file, 'r') as f:
        return json.load(f)


@pytest.mark.parametrize('case_file', get_demo_cases(), ids=lambda x: x.stem)
def test_case_validates_against_schema(case_file):
    """Test that each demo case validates against the JSON schema."""
    schema = load_schema()
    
    with open(case_file, 'r') as f:
        case_data = json.load(f)
    
    # This will raise an exception if validation fails
    jsonschema.validate(instance=case_data, schema=schema)
    
    # If we reach here, validation passed
    assert True

