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


setup(
    name="physician_conversion_model",
    install_requires=PACKAGE_REQUIREMENTS,
    entry_points = {
        "console_scripts": [
            "feature_pipeline = physician_conversion_model.tasks.feature_pipeline:entrypoint"
    ]},
    version=__version__,
    description="",
    author="",
)
