#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 12/1/2015
import os
import sys
from os.path import join
from shutil import copy

__author__ = 'ASU'

# from distutils.core import setup  # this line broke bdist_wheel

from setuptools import setup

py_modules = ['Json', 'streams', 'racelib', 'fileutils', '__init__']

basedir = os.path.dirname(__file__)
dest_package_dir = join(basedir, "pyxtension")
try:
    os.makedirs(dest_package_dir)
except os.error:
    pass

pyMajorVersion = str(sys.version_info[0])

for fname in py_modules:
    copy(join(basedir, 'py' + pyMajorVersion, 'pyxtension', fname + '.py'), dest_package_dir)
parameters = dict(name='pyxtension',
                  version='1.12.1',
                  description='Extension library for Python',
                  author='Andrei Suiu',
                  author_email='andrei.suiu@gmail.com',
                  url='https://github.com/asuiu/pyxtension',
                  packages=['pyxtension'],
                  classifiers=[
                      "Development Status :: 5 - Production/Stable",
                      "Intended Audience :: Developers",
                      "Programming Language :: Python :: 2",
                      "Programming Language :: Python :: 2.6",
                      "Programming Language :: Python :: 2.7",
                      "Programming Language :: Python :: 3",
                      "Programming Language :: Python :: 3.6",
                      "Programming Language :: Python :: 3.7",
                      "Programming Language :: Python :: Implementation :: CPython",
                      "Programming Language :: Python :: Implementation :: PyPy", ])
if pyMajorVersion == '2':
    import pip
    
    requires = ['mock']
    for reqPackage in requires:
        pip.main(['install', reqPackage])
elif pyMajorVersion == '3':
    pass

setup(**parameters)

# clean-up
for fname in os.listdir(dest_package_dir):
    os.unlink(join(dest_package_dir, fname))
os.rmdir(dest_package_dir)
