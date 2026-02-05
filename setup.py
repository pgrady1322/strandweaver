#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
StrandWeaver: AI-Powered Multi-Technology Genome Assembler

A genome assembly tool with intelligent error correction, graph-based assembly,
and AI-powered finishing for Ancient DNA, Illumina, ONT, and PacBio HiFi data.

Version: 0.1
License: Dual Academic/Commercial (see LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md)
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure we can import version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strandweaver"))

from version import __version__

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Basic requirements (always installed)
install_requires = read_requirements("requirements.txt")

# Add Hi-C dependencies to default install (scipy and scikit-learn)
install_requires.extend([
    "scipy>=1.9.0",
    "scikit-learn>=1.3.0",
])

# Optional dependencies
extras_require = {
    "dev": read_requirements("requirements-dev.txt"),
    "ai": [
        "torch>=2.0.0",
        "pytorch-geometric>=2.3.0",
        "xgboost>=2.0.0",
    ],
}

# Convenience: install all optional dependencies
extras_require["all"] = (
    extras_require.get("dev", []) +
    extras_require.get("ai", [])
)

setup(
    name="strandweaver",
    version=__version__,
    author="Patrick Grady",
    author_email="dr.pgrady@gmail.com",
    description="AI-Powered Multi-Technology Genome Assembler with GPU Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgrady1322/strandweaver",
    project_urls={
        "Documentation": "https://github.com/pgrady1322/strandweaver/tree/main/docs",
        "Bug Tracker": "https://github.com/pgrady1322/strandweaver/issues",
        "Source Code": "https://github.com/pgrady1322/strandweaver",
    },
    packages=find_packages(exclude=["tests", "benchmarks", "examples", "scripts", "training_runs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "strandweaver=strandweaver.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "strandweaver": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    zip_safe=False,
    keywords="genome assembly bioinformatics long-reads hifi ont pacbio gpu",
)
