#!/usr/bin/env python

from os.path import exists
from setuptools import setup

import versioneer

subpackages = {
    'sftp': ['paramiko'],
    'vis':  ['graphviz', 'frozendict']
}

setup(
    name='provenance',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['provenance', 'provenance.sftp', 'provenance.vis'],
    setup_requires=['pytest>=3.0.0', 'pytest-runner'],
    install_requires=[open('requirements.txt').read().strip().split('\n')],
    tests_requires=[open('test_requirements.txt').read().strip().split('\n')],
    extras_require=subpackages,
    include_package_data=True,
    description="Provenance and caching library for functions",
    long_description=(open('README.rst').read() if exists('README.rst')
                      else ''),
    author="Ben Mabey",
    author_email="ben@benmabey.com",
    url="http://github.com/bmabey/provenance",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
    ],
)
