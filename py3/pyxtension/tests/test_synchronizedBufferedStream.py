#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 11/18/2015
from unittest import TestCase, main

from pyxtension.streams import SynchronizedBufferedStream, slist

__author__ = 'andrei.suiu@gmail.com'


class TestSynchronizedBufferedStream(TestCase):
    def test_nominal(self):
        s = SynchronizedBufferedStream((slist(range(i)) for i in range(1, 4)))
        self.assertListEqual(s.toList(), [0, 0, 1, 0, 1, 2])


if __name__ == '__main__':
    main()
