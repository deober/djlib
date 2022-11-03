from setuptools import setup

setup(
    name="djlib",
    version="0.3.0",
    description="General utility package for AVDV people.",
    author="Derick Ober, Jonathan Li",
    license="MIT License",
    packages=["djlib", "djlib.casmcalls", "djlib.clex", "djlib.mc", "djlib.vasputils","djlib.propagation"],
    install_requires=[
        "httpstan",
        "json5>=0.9.6",
        "jsonschema>=4.1.0",
        "numpy",
        "pystan",
        "scikit-learn>=1.0",
        "scipy>=1.7.1",
        "tinc>=0.9.52",
        "tqdm>=4.62.2",
        "arviz",
    ],
    include_package_data=True,
)
