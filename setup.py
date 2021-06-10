"""
nrt-predict
"""
from setuptools import setup
from pathlib import Path

tests_require = [
    "pytest",
    "joblib",
]

parent = Path(__file__).parent

REQUIRES = (parent / "requirements.txt").read_text().splitlines()
README = (parent / "README.md").read_text()

breakpoint()

setup(
    name="nrt-predict",
    version="0.1",
    description="Predict on Near-Real-Time Satellite Observations",
    long_description=README,
    long_description_content_type="text/markdown",
    url="http://github.com/daleroberts/nrt-predict",
    author="Dale Roberts",
    author_email="dale.o.roberts@gmail.com",
    license="BSD-3-Clause License",
    tests_require=tests_require,
    install_requires=[
        "numpy",
        "gdal",
        "joblib",
        "psutil",
        "requests",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
    ],
    packages=['nrtmodels'],
    package_dir={'nrtmodels': 'nrtmodels'},
    py_modules=['nrtpredict'],
    entry_points="""
        [console_scripts]
        nrtpredict=nrtpredict:cli_entry
    """,
)
