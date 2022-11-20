#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 11/18/2015
from unittest import TestCase, main

from pyxtension.streams import AbstractSynchronizedBufferedStream, slist

__author__ = 'andrei.suiu@gmail.com'


class TestAbstractSynchronizedBufferedStream(TestCase):
    def test_nominal(self):
        class TestSyncStream(AbstractSynchronizedBufferedStream):
            def __init__(self):
                super(TestSyncStream, self).__init__()
                self._counter = 4

            def _getNextBuffer(self):
                self._counter -= 1
                if self._counter > 0:
                    return slist(range(self._counter))
                return slist()

        test_stream = TestSyncStream()
        self.assertListEqual(test_stream.toList(), [0, 1, 2, 0, 1, 0])

if __name__ == '__main__':
    main()
