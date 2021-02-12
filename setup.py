from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A neural network framework'
LONG_DESCRIPTION = 'Nerve is a neural network framework written purely in python, best for understanding and experimentation'

setup(
        name="nerve",
        version=VERSION,
        author="Gyan Arora",
        author_email="gyansworld@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy",
        ],
        keywords=['python', 'neural', 'networks'],
        classifiers=[
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)