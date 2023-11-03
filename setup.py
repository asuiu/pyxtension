#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 12/1/2015
import os
from os.path import join

__author__ = 'ASU'

# Bump up this version
VERSION = '1.16.6'

from setuptools import setup
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel

py_modules = ['Json', 'streams', 'racelib', 'fileutils', '__init__']

basedir = os.path.dirname(__file__)
dest_package_dir = join(basedir, "pyxtension")

long_description = open('README.md', "rt").read()

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

# load test requirements
with open("requirements-dev.txt") as fp:
    dev_require = fp.read().strip().split("\n")
    dev_require = [s for s in dev_require if not s.startswith(("#", "-"))]

extras_require = {
    'dev':  dev_require,
    'test': dev_require
}
python_requires = '>=3.6, <4'


class InstallCommand(install, object):
    user_options = install.user_options + [('py2', None, "Forces to build Py2 package even if run from Py3")]

    def initialize_options(self):
        super(InstallCommand, self).initialize_options()
        self.py2 = None


class BdistWheelCommand(bdist_wheel, object):
    user_options = bdist_wheel.user_options + [('py2', None, "Forces to build Py2 package even if run from Py3")]

    def initialize_options(self):
        super(BdistWheelCommand, self).initialize_options()
        self.py2 = None

    def finalize_options(self):
        super(BdistWheelCommand, self).finalize_options()
        # self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super(BdistWheelCommand, self).get_tag()
        # We don't contain any python source
        return python, abi, plat


parameters = dict(name='pyxtension',
                  version=VERSION,
                  description='Extension library for Python',
                  long_description=long_description,
                  long_description_content_type="text/markdown",
                  author='Andrei Suiu',
                  author_email='andrei.suiu@gmail.com',
                  url='https://github.com/asuiu/pyxtension',
                  packages=['pyxtension'],
                  python_requires=python_requires,
                  install_requires=install_requires,
                  extras_require=extras_require,
                  data_files=[(".", ["requirements.txt",])],
                  cmdclass={
                      'install':     InstallCommand,
                      'bdist_wheel': BdistWheelCommand
                  },
                  classifiers=[
                      "Development Status :: 5 - Production/Stable",
                      "Intended Audience :: Developers",
                      "Programming Language :: Python :: 2.7",
                      "Programming Language :: Python :: 3.6",
                      "Programming Language :: Python :: 3.7",
                      "Programming Language :: Python :: 3.8",
                      "Programming Language :: Python :: 3.9",
                      "Programming Language :: Python :: 3.10",
                      "Programming Language :: Python :: 3.11",
                      "Programming Language :: Python :: Implementation :: CPython",
                      "Programming Language :: Python :: Implementation :: PyPy", ])

setup(**parameters)
