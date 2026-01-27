from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="polaris-energy-efficient-gpu",
    version="1.0.0",
    author="Jonathan Ciencias",
    author_email="jonathan.ciencias@email.com",
    description="Energy-Efficient Deep Learning Inference Framework for AMD Polaris GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonatanciencias/polaris-energy-efficient-gpu",
    project_urls={
        "Documentation": "https://github.com/jonatanciencias/polaris-energy-efficient-gpu/docs",
        "Source": "https://github.com/jonatanciencias/polaris-energy-efficient-gpu",
        "Tracker": "https://github.com/jonatanciencias/polaris-energy-efficient-gpu/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
        "Natural Language :: Spanish",
        "Framework :: AsyncIO",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA",
        "Environment :: GPU :: AMD",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.10.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.20.0",
        ],
        "gpu": [
            "pyopencl>=2022.1",
            "cupy>=12.0.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "distributed": [
            "pyzmq>=25.0.0",
            "msgpack>=1.0.0",
            "dask>=2023.0.0",
        ],
        "benchmarking": [
            "pytest-benchmark>=4.0.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "polaris-energy-efficient=src.cli:main",
            "polaris-verify=scripts.deployment.verify_hardware:main",
            "polaris-benchmark=src.benchmarking.comprehensive_performance_validation:main",
            "polaris-paper=docs.paper.paper-energy-efficient-polaris.main:compile_paper",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
