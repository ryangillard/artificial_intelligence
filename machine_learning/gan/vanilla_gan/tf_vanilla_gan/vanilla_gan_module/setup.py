from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name="vanilla_gan_module",
    version="0.1",
    author="Ryan Gillard",
    author_email="ryangillard@google.com",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Vanilla GAN custom Estimator model.",
    requires=[]
)