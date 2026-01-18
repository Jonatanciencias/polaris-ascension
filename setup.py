from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legacy-gpu-ai",
    version="0.6.0-dev",
    author="Legacy GPU AI Platform Contributors",
    description="Open-source AI platform for AMD legacy GPUs - Democratizing AI through accessible hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/legacy-gpu-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
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
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
        ],
        "distributed": [
            "pyzmq>=25.0.0",
            "msgpack>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legacy-gpu-ai=src.cli:main",
            "lgai-verify=scripts.verify_hardware:main",
            "lgai-benchmark=scripts.benchmark:main",
        ],
    },
)
