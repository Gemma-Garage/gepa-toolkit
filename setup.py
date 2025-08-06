from setuptools import setup, find_packages

setup(
    name="GEPA",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    python_requires=">=3.7",
)