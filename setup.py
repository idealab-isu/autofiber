from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='autofiber',
    version='0.0.1',
    author='Nathan Scheirer',
    description='Accurate, automatic composite fiber placement',
    keywords='auto fiber placement strain energy optimization',
    packages=["autofiberlib"]
)
