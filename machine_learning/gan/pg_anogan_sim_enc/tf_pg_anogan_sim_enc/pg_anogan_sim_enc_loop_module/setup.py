from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name="pg_anogan_sim_enc_module",
    version="0.1",
    author="Ryan Gillard",
    author_email="ryangillard@google.com",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Progressively Growing GAN for Anomaly Detection with Simultaneous Encoder custom Estimator model.",
    requires=[]
)