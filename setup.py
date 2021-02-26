from setuptools import setup, find_packages

setup(
    name='nsdloader',
    version='0.1',
    install_requires=["nsd_access", "pandas", "numpy", "regex"],
    packages=find_packages(),
)
