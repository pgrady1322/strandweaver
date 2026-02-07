#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
StrandWeaver Setup.py

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

from setuptools import setup, find_packages
import os

# Read version from package
version = {}
with open("strandweaver/version.py") as f:
    exec(f.read(), version)

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="strandweaver",
    version=version["__version__"],
    author="Patrick Grady",
    author_email="dr.pgrady@gmail.com",
    description="AI-Powered Multi-Technology Genome Assembler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgrady1322/strandweaver",
    packages=find_packages(exclude=["tests", "benchmarks", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "ai": ["anthropic>=0.7.0"],
    },
    entry_points={
        "console_scripts": [
            "strandweaver=strandweaver.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "strandweaver": [
            "config/*.yaml",
            "ai/prompt_templates/*.txt",
        ],
    },
    zip_safe=False,
)
