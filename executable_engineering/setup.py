from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="executable_engineering",
    version="0.1.41",
    description="Demonstrative tools for the Executable Engineering textbook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/themintlab/ExecutableEngineering/tree/main/executable_engineering",
    author="Michael Welland",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx",
        "scipy",
        "plotly",
        "pandas",
    ],
    extras_require={
        "docs": ["pyppeteer", "nbconvert"],
        "jupyter": ["ipywidgets", "IPython"],
    },
    python_requires=">=3.9",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
