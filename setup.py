"""Setup script for AML-GNN-Deep."""

from setuptools import find_packages, setup

setup(
    name="aml-gnn-deep",
    version="0.1.0",
    description="GNN-based Anti-Money Laundering Detection — Master's Thesis",
    author="Shadab Gada",
    author_email="shadab.gada@student.hu.nl",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
)
