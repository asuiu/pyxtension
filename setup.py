#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 12/1/2015
import os
from os.path import join
from shutil import copy

__author__ = 'ASU'

from distutils.core import setup

py_modules = ['Json', 'streams', 'racelib', 'fileutils', '__init__']

basedir = os.path.dirname(__file__)
dest_package_dir = join(basedir, "pyxtension")
try:
    os.makedirs(dest_package_dir)
except os.error:
    pass

for fname in py_modules:
    copy(join(basedir, fname + '.py'), dest_package_dir)

setup(name='pyxtension',
      version='1.0',
      description='Python Utilities',
      author='Andrei Suiu',
      author_email='andrei.suiu@gmail.com',
      url='https://github.com/asuiu/pyxtension',
      packages=['pyxtension'],
      classifiers=(
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.2",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy",
      ),
      )

# clean-up
for fname in os.listdir(dest_package_dir):
    os.unlink(join(dest_package_dir, fname))
os.rmdir(dest_package_dir)
