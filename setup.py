from setuptools import setup

setup(
    name='NSD_DataLoader',
    version='0.1',
    install_requires=["nsd_access", "pandas", "numpy", "regex"],
    scripts=['NSD_DataLoader/NSD_DataLoader.py']
)
