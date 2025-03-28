#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose:
# Created: 11/5/2017
import os
import sys
import unittest

__author__ = "ASU"

if __name__ == "__main__":
    testLoader = unittest.TestLoader()
    testsDir = os.path.join(os.path.dirname(__file__), "pyxtension", "tests")

    trunner = unittest.TextTestRunner(sys.stdout, descriptions=True, verbosity=2)
    testSuite = testLoader.discover(start_dir=testsDir, pattern="test_*.py", top_level_dir=testsDir)
    res = trunner.run(testSuite)

    if res.failures or res.errors:
        sys.exit(1)
    sys.exit(0)
