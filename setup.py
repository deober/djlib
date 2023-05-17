from setuptools import setup

setup(
    name="djlib",
    version="0.3.0",
    description="General utility package for AVDV people.",
    author="Derick Ober, Jonathan Li",
    license="MIT License",
    packages=["djlib", "djlib.casmcalls", "djlib.clex", "djlib.mc", "djlib.vasputils","djlib.propagation", "djlib.plotting"],
    install_requires=[
        "httpstan",
        "numpy",
        "pystan",
        "scikit-learn>=1.0",
        "scipy>=1.7.1",
        "arviz",
    ],
    include_package_data=True,
)
