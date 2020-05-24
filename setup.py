#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 12/1/2015
import os
import sys
from os.path import join
from shutil import copy, rmtree

__author__ = 'ASU'

# Bump up this version
VERSION = '1.13.9'

from setuptools import setup
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel

py_modules = ['Json', 'streams', 'racelib', 'fileutils', '__init__']

basedir = os.path.dirname(__file__)
dest_package_dir = join(basedir, "pyxtension")
try:
    os.makedirs(dest_package_dir)
except os.error:
    pass

pyMajorVersion = str(sys.version_info[0])
if "--py2" in sys.argv:
    pyMajorVersion = '2'

src_dir = join(basedir, 'py' + pyMajorVersion, 'pyxtension')
for fname in [f for f in os.listdir(src_dir) if f.endswith(".py")]:
    copy(join(src_dir, fname), dest_package_dir)

# ToDo: check if there's still BUG in twine, as if falsely reports in README.md
#  line 34: Error: Unexpected indentation.

long_description = open('README.rst', "rt").read()

install_requires = ['tqdm>=4.41.1;python_version>="3"']
extras_require = {
    'dev':  ['mock;python_version<"3"'],
    'test': ['mock;python_version<"3"']
}

if pyMajorVersion == "2":
    python_requires = '>=2.6, <3'
elif pyMajorVersion == "3":
    python_requires = '>=3.6, <4'
else:
    raise Exception("Unknown Python version")


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
        if pyMajorVersion == "2":
            python, abi = 'py2', 'none'
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
                  cmdclass={
                      'install':     InstallCommand,
                      'bdist_wheel': BdistWheelCommand
                  },
                  classifiers=[
                      "Development Status :: 5 - Production/Stable",
                      "Intended Audience :: Developers",
                      "Programming Language :: Python :: 2.6",
                      "Programming Language :: Python :: 2.7",
                      "Programming Language :: Python :: 3.6",
                      "Programming Language :: Python :: 3.7",
                      "Programming Language :: Python :: 3.8",
                      "Programming Language :: Python :: Implementation :: CPython",
                      "Programming Language :: Python :: Implementation :: PyPy", ])

setup(**parameters)

# clean-up
rmtree(dest_package_dir)
