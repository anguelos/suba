from setuptools import setup, find_packages

setup(
    name="suba",
    version="0.1.0",
    description=(
        "Single cell Unidimensional Base Annotation — "
        "genomic data as 1D numpy-like objects"
    ),
    long_description=open("docs/index.md").read(),
    long_description_content_type="text/markdown",
    author="Anguelos Nicolaou",
    author_email="anguelos.nicolaou@gmail.com",
    url="https://github.com/anguelos/suba",
    license="MIT",
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
            "suba=suba.cli:main",
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
