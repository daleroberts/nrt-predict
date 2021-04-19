"""
nrt-predict
"""

import pathlib
import sys

from setuptools import setup, find_packages, Extension

tests_require = [
    "pytest",
    "joblib",
]

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="nrt-predict",
    packages=find_packages(".", exclude=['tests']),
    include_package_data=True,
    package_data={'': ['models/*.py']},
    setup_requires=["numpy", "gdal"],
    install_requires=["numpy", "gdal"],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
    },
    version="0.1",
    description="Predict on Near-Real-Time Satellite Observations",
    long_description=README,
    long_description_content_type="text/markdown",
    url="http://github.com/daleroberts/nrt-predict",
    author="Dale Roberts",
    author_email="dale.o.roberts@gmail.com",
    license="BSD-3-Clause License",
    zip_safe=False,
)
