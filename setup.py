#!/usr/bin/env python

from os.path import exists
from setuptools import setup

import versioneer

setup(
    name='provenance',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=('provenance',),
    setup_requires=['pytest>=3.0.0', 'pytest-runner'],
    install_requires=[open('requirements.txt').read().strip().split('\n')],
    test_require=[open('test_requirements.txt').read().strip().split('\n')],
    description="Provenance and caching library for functions",
    long_description=(open('README.rst').read() if exists('README.rst')
                      else ''),
    author="Ben Mabey",
    author_email="ben@benmabey.com",
    url="http://github.com/Savvysherpa/provenance",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
    ],
)
