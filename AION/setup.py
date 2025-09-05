#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AION Protocol Setup Script
==========================

This script installs the AION Protocol system and its dependencies.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="aion-protocol",
    version="2.0.0",
    author="Francisco Molina",
    author_email="pako.molina@gmail.com",
    description="AION Protocol for digital asset development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yatrogenesis/Obvivlorum",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aion=aion_cli:main",
        ],
    },
)
