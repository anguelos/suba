from setuptools import setup, find_packages

import re

def _read_version():
    with open('src/suba/version.py') as f:
        m = re.search(r'suba_version\s*=\s*["\']([^"\']+)["\']', f.read())
    if not m:
        raise RuntimeError('Cannot find suba_version in src/suba/version.py')
    return m.group(1)

setup(
    name="suba",
    version=_read_version(),
    description=(
        "Single cell Unidimensional Base Annotation — "
        "genomic data as 1D numpy-like objects"
    ),
    long_description=open("docs/index.md").read(),
    long_description_content_type="text/markdown",
    author="Anguelos Nicolaou",
    author_email="anguelos.nicolaou@gmail.com",
    url="https://github.com/anguelos/suba",
    license="AGPL-3.0-or-later",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "requests",
        "tqdm",
    ],
    extras_require={
        "cli": [
            "fargv",
            "seaborn",
            "matplotlib",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "sphinx",
            "myst-parser",
            "sphinx-autoapi",
        ],
    },
    entry_points={
        "console_scripts": [
            "suba_hilbert_demo=suba.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
