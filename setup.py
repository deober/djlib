from setuptools import setup

setup(
    name="djlib",
    version="0.1.0",
    description="General utility package for AVDV people.",
    author="Derick Ober, Jonathan Li",
    license="MIT License",
    packages=["djlib", "djlib.casmcalls", "djlib.clex", "djlib.mc", "djlib.vasputils"],
    install_requires=[
        "cupy-cuda114>=9.5.0",
        "httpstan",
        "json5>=0.9.6",
        "jsonschema>=4.1.0",
        "matplotlib>=3.4.3",
        "numpy",
        "plotly>=5.1.0",
        "pystan>=3.3.0",
        "scikit-learn>=1.0",
        "scipy>=1.7.1",
        "seaborn>=0.11.2",
        "tinc>=0.9.52",
        "tqdm>=4.62.2",
    ],
    include_package_data=True,
)

