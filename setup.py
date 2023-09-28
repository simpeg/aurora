#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#    history = history_file.read()

requirements = [
    "dask",
    "deprecated",
    "matplotlib",
    "mth5",
    "mt_metadata",
    "numpy",
    "numba",
    "obspy",
    "psutil",
    "pandas<1.5",
    "scipy",
    "xarray",
    "fortranformat",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Karl Kappler",
    author_email="karl.kappler@berkeley.edu",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Processing Codes for Magnetotelluric Data",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="aurora",
    name="aurora",
    packages=find_packages(include=["aurora", "aurora.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/simpeg/aurora",
    version="0.3.2",
    zip_safe=False,
)
