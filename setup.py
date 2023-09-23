"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from physician_conversion_model import __version__

PACKAGE_REQUIREMENTS = ["pyyaml"]

# packages for local development and unit testing
# please note that these packages are already available in DBR, there is no need to install them on DBR.
LOCAL_REQUIREMENTS = [
    "scikit-learn",
    "pandas"
]

TEST_REQUIREMENTS = [
    # development & testing tools
    "pytest",
    "coverage[toml]",
    "pytest-cov"
]

setup(
    name="physician_conversion_model",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["setuptools","wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"local": LOCAL_REQUIREMENTS, "test": TEST_REQUIREMENTS},
    entry_points = {
        "console_scripts": [
            "feature_pipeline = physician_conversion_model.tasks.feature_pipeline:entrypoint"
    ]},
    version=__version__,
    description="",
    author="",
)