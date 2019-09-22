#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 9/22/2019
import io
from unittest import TestCase, main

from fileutils import Progbar
from streams import stream

__author__ = 'ASU'


class TestFileutils(TestCase):
    TEST_VALUES_S = [None,
                     [['key1', 1], ['key2', 1e-4]],
                     [['key3', 1], ['key2', 1e-4]]]

    def test_Progbar_update(self):
        times = [x * 0.1 for x in range(20, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = len(self.TEST_VALUES_S) - 1
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = ("\n0/2 [..............................] - ETA: 0s"
                    "\n1/2 [==============>...............] - ETA: 0s - key1: 1.000 - key2: 1.000e-04"
                    "\n2/2 [==============================] - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000"
                    "\n")
        self.assertEqual(expected, out.getvalue())
        target = None
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = ("\n      0/Unknown - 0s 0us/step"
                    "\n      1/Unknown - 0s 200ms/step - key1: 1.000 - key2: 1.000e-04"
                    "\n      2/Unknown - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000")
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_update_verbose(self):
        times = [x * 0.1 for x in range(20, 4, -1)]
        timer = lambda: times.pop()
        verbose = True
        target = len(self.TEST_VALUES_S) - 1
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = (" - 0s - key1: 1.000 - key2: 1.000e-04 - key3: 1.000\n")
        self.assertEqual(expected, out.getvalue())
        target = None
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = (' - 0s\n'
                    ' - 0s - key1: 1.000 - key2: 1.000e-04\n'
                    ' - 0s - key1: 1.000 - key2: 1.000e-04 - key3: 1.000\n')
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_add(self):
        times = [x * 0.1 for x in range(20, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = len(self.TEST_VALUES_S) - 1
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.add(1, values=values)
        expected = ('\n1/2 [==============>...............] - ETA: 0s\n'
                    '2/2 [==============================] - 0s 100ms/step - key1: 1.000 - key2: 1.000e-04\n\n'
                    '3/2 [=============================================] - 0s 100ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000\n')
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_mapper_use_unknown_size(self):
        times = [x * 0.1 for x in range(200, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = None
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for value in range(3):
            self.assertEqual(value, bar(value))
        expected = ("\n      1/Unknown - 0s 100ms/step\n"
                    "      2/Unknown - 0s 100ms/step\n"
                    "      3/Unknown - 0s 100ms/step")
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_mapper_use_known_size(self):
        times = [x * 0.1 for x in range(200, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = 3
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        self.assertListEqual(list(range(target)), stream(range(target)).map(bar).toList())
        expected = ("\n1/3 [=========>....................] - ETA: 0s\n"
                    "2/3 [===================>..........] - ETA: 0s\n"
                    "3/3 [==============================] - 0s 100ms/step\n")
        self.assertEqual(expected, out.getvalue())


if __name__ == '__main__':
    main()
