#!/usr/bin/env python
"""
Quick installation verification script.

Run this after installing StrandWeaver to verify everything works.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")
    
    try:
        import strandweaver
        print("  ✓ strandweaver package imported")
    except ImportError as e:
        print(f"  ✗ Failed to import strandweaver: {e}")
        return False
    
    try:
        from strandweaver.preprocessing import KWeaverPredictor
        print("  ✓ KWeaver module imported")
    except ImportError as e:
        print(f"  ✗ Failed to import KWeaver: {e}")
        return False
    
    try:
        from strandweaver.assembly_core import EdgeWarden
        print("  ✓ EdgeWarden module imported")
    except ImportError as e:
        print(f"  ✗ Failed to import EdgeWarden: {e}")
        return False
    
    try:
        from strandweaver.cli import main
        print("  ✓ CLI module imported")
    except ImportError as e:
        print(f"  ✗ Failed to import CLI: {e}")
        return False
    
    return True


def test_version():
    """Test that version is accessible."""
    print("\nTesting version...")
    
    try:
        from strandweaver.version import __version__
        print(f"  ✓ StrandWeaver version: {__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to get version: {e}")
        return False


def test_cli_help():
    """Test that CLI --help works."""
    print("\nTesting CLI...")
    
    try:
        from click.testing import CliRunner
        from strandweaver.cli import main
        
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        if result.exit_code == 0:
            print("  ✓ CLI --help works")
            return True
        else:
            print(f"  ✗ CLI --help failed with exit code {result.exit_code}")
            return False
    except Exception as e:
        print(f"  ✗ CLI test failed: {e}")
        return False


def test_dependencies():
    """Test that key dependencies are installed."""
    print("\nTesting dependencies...")
    
    required = {
        'numpy': 'NumPy',
        'networkx': 'NetworkX',
        'biopython': 'BioPython',
        'pysam': 'pysam',
        'click': 'Click',
        'pyyaml': 'PyYAML'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            all_ok = False
    
    return all_ok


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("StrandWeaver Installation Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Version", test_version()))
    results.append(("CLI", test_cli_help()))
    results.append(("Dependencies", test_dependencies()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s} {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! StrandWeaver is ready to use.")
        print("\nNext steps:")
        print("  - Run 'strandweaver --help' to see available commands")
        print("  - See README.md for usage examples")
        print("  - Run 'pytest tests/' to run the full test suite")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("  - Make sure all dependencies are installed")
        print("  - Try reinstalling: pip install -e .")
        print("  - Check that Python version >= 3.9")
        return 1


if __name__ == "__main__":
    sys.exit(main())
