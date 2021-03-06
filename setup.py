from setuptools import setup

import qit

setup(
    name="qit",
    version=qit.__version__,
    author="Alex Malz, Eric Charles",
    author_email="aimalz@nyu.edu, echarles@slac.stanford.edu",
    url = "https://github.com/LSSTDESC/qit",
    packages=["qit"],
    description="qp-based inference toolkit",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["qp"]
)
