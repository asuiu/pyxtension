#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 11/5/2017
import os
import sys
import unittest
import io
__author__ = 'ASU'

if __name__ == '__main__':
    testLoader = unittest.TestLoader()
    pymajorVersion = sys.version_info[0]
    packageDir = os.path.join(os.path.dirname(__file__), "py%d" % pymajorVersion, "pyxtension")
    testsDir = os.path.join(packageDir, "tests")
    #sys.path.append(packageDir)

    # textTestResult = unittest.TextTestResult(io.StringIO(),'',verbosity=1)
    trunner = unittest.TextTestRunner(sys.stdout, descriptions=True, verbosity=0)
    testSuite = testLoader.discover(start_dir=testsDir, pattern="test_*.py", top_level_dir=testsDir)
    res = trunner.run(testSuite)

    testSuite = testLoader.discover(start_dir=testsDir, pattern="test_*.py", top_level_dir=testsDir)
    testResult = unittest.TestResult()
    res = testSuite.run(testResult)
    assert not res.errors, "Unittests error: %s" % res.errors
    print(res)
