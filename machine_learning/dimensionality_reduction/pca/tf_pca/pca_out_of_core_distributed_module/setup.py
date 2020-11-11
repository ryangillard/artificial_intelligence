from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name="pca_out_of_core_distributed",
    version="0.1",
    author="Ryan Gillard",
    author_email="ryangillard@google.com",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="PCA Out-of-Core Distributed",
    requires=[]
)