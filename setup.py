"""Setup configuration for the Network Anonymity Threat Modeling SDK."""

from setuptools import setup, find_packages

setup(
    name="threat-modeling-sdk",
    version="1.0.0",
    description=(
        "Network Anonymity Threat Modeling SDK — Analyze anonymity protocols "
        "against adversary capabilities for security research."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Threat Modeling SDK",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "threat-model=threat_modeling.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
