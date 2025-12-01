from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qtmrl",
    version="0.1.0",
    author="QTMRL Team",
    description="基于 A2C 的多资产量化交易强化学习系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qtmrl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "yfinance>=0.2.28",
        "pandas-ta>=0.3.14b0",
        "torch>=2.0.0",
        "pyarrow>=12.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "scikit-learn>=1.3.0",
    ],
)
