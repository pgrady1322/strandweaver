# StrandWeaver Test Suite

This directory contains tests for StrandWeaver v0.1.

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-cov click
```

### Run all tests

```bash
# From strandweaver root directory
pytest tests/

# Or with verbose output
pytest tests/ -v
```

### Run specific test categories

```bash
# Unit tests only (fast)
pytest tests/ -m unit

# Skip slow tests
pytest tests/ -m "not slow"

# Integration tests only
pytest tests/ -m integration

# CLI tests only
pytest tests/ -m cli
```

### Run specific test files

```bash
pytest tests/test_kweaver.py
pytest tests/test_edgewarden.py
pytest tests/test_cli.py
```

### Run with coverage report

```bash
pytest tests/ --cov=strandweaver --cov-report=html
# Open htmlcov/index.html in browser to view coverage
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_kweaver.py          # K-mer prediction tests
├── test_edgewarden.py       # Edge quality assessment tests
├── test_read_classification.py  # Read technology detection tests
├── test_sequence_utils.py   # Sequence utility function tests
├── test_cli.py              # Command-line interface tests
├── test_integration.py      # End-to-end pipeline tests
└── data/                    # Test data files (when needed)
```

## Test Categories

### Unit Tests
- Test individual functions and classes
- Fast execution (<1 second per test)
- No external dependencies or file I/O when possible

### Integration Tests
- Test multiple modules working together
- May involve file I/O or subprocess calls
- Marked with `@pytest.mark.integration`

### CLI Tests
- Test command-line interface
- Verify commands run without crashing
- Marked with `@pytest.mark.cli`

### Slow Tests
- Tests that take >5 seconds
- Marked with `@pytest.mark.slow`
- Can be skipped with `pytest -m "not slow"`

## Writing New Tests

### Basic test structure

```python
import pytest
from strandweaver.module import function_to_test

class TestMyFeature:
    """Test description."""
    
    def test_basic_functionality(self):
        """Test that basic case works."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using fixtures

```python
def test_with_temp_dir(temp_output_dir):
    """Use temporary directory fixture."""
    output_file = temp_output_dir / "test.txt"
    output_file.write_text("test")
    assert output_file.exists()
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ -v --cov=strandweaver
```

## Current Test Coverage

As of v0.1:
- **K-Weaver**: Basic functionality tested
- **EdgeWarden**: Scoring and quality assessment tested
- **Read Classification**: Technology detection tested
- **Sequence Utils**: K-mer and GC calculation tested
- **CLI**: Command parsing and help tested
- **Integration**: Minimal pipeline test

## Future Tests (v0.1.1+)

- [ ] Complete assembly pipeline test with real data
- [ ] Benchmark tests for performance validation
- [ ] Multi-technology integration tests
- [ ] Hi-C scaffolding tests
- [ ] SV detection validation tests
- [ ] Ancient DNA correction tests

## Troubleshooting

**Tests fail with import errors:**
- Make sure StrandWeaver is installed: `pip install -e .`

**Tests are very slow:**
- Run without slow tests: `pytest -m "not slow"`

**Coverage report not generating:**
- Install pytest-cov: `pip install pytest-cov`

**Need test data:**
- Minimal synthetic data is generated in fixtures
- Real test datasets will be added in v0.1.1
