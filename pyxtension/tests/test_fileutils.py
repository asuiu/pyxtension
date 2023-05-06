#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 9/22/2019
import gzip
import io
from operator import itemgetter
from pathlib import Path
from unittest import TestCase, main

from pyxtension.fileutils import Progbar, ReversedCSVReader
from pyxtension.streams import stream

__author__ = 'andrei.suiu@gmail.com'


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
        expected = ("0/2 [..............................] - ETA: 0s\n"
                    "1/2 [==============>...............] - ETA: 0s - key1: 1.000 - key2: 1.000e-04\n"
                    "2/2 [==============================] - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000\n\n"
                    )
        self.assertEqual(expected, out.getvalue())
        target = None
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = ("      0/Unknown - 0s 0us/step\n"
                    "      1/Unknown - 0s 200ms/step - key1: 1.000 - key2: 1.000e-04\n"
                    "      2/Unknown - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000\n")
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_update_verbose(self):
        times = [x * 0.1 for x in range(20, 4, -1)]
        timer = lambda: times.pop()
        verbose = True
        target = len(self.TEST_VALUES_S) - 1
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer, dynamic_display=True)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = (
            '\r0/2 [..............................] - ETA: 0s\r1/2 [==============>...............] - ETA: 0s - key1: 1.000 - key2: 1.000e-04          '
            '\r2/2 [==============================] - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000                                          \n'
            '\r - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000 - key1: 1.000 - key2: 1.000e-04 - key3: 1.000'
        )

        self.assertEqual(expected, out.getvalue())
        target = None
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer, dynamic_display=True)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.update(current, values=values)
        expected = ('\r      0/Unknown - 0s 0us/step'
                    '\r - 0s 0us/step               '
                    '\r      1/Unknown - 0s 200ms/step - key1: 1.000 - key2: 1.000e-04'
                    '\r - 0s 200ms/step - key1: 1.000 - key2: 1.000e-04 - key1: 1.000 - key2: 1.000e-04'
                    '\r      2/Unknown - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000                                                                 '
                    '\r - 0s 150ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000 - key1: 1.000 - key2: 1.000e-04 - key3: 1.000')
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_add(self):
        times = [x * 0.1 for x in range(20, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = len(self.TEST_VALUES_S)
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer)
        for current, values in enumerate(self.TEST_VALUES_S):
            bar.add(1, values=values)
        expected = ('1/3 [=========>....................] - ETA: 0s\n'
                    '2/3 [===================>..........] - ETA: 0s - key1: 1.000 - key2: 1.000e-04\n'
                    '3/3 [==============================] - 0s 100ms/step - key1: 1.000 - key2: 1.000e-04 - key3: 1.000\n\n')
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_mapper_use_unknown_size(self):
        times = [x * 0.1 for x in range(200, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = None
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer, dynamic_display=True)
        for value in range(3):
            self.assertEqual(value, bar(value))
        expected = ('\r      1/Unknown - 0s 100ms/step'
                    '\r      2/Unknown - 0s 100ms/step                '
                    '\r      3/Unknown - 0s 100ms/step                '
                    )
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_mapper_use_known_size(self):
        times = [x * 0.1 for x in range(200, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = 3
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer, dynamic_display=True)
        self.assertListEqual(list(range(target)), stream(range(target)).map(bar).toList())
        expected = ('\r1/3 [=========>....................] - ETA: 0s'
                    '\r2/3 [===================>..........] - ETA: 0s          '
                    '\r3/3 [==============================] - 0s 100ms/step          \n'
                    )
        self.assertEqual(expected, out.getvalue())

    def test_Progbar_mapper_use_dynamic_nominal(self):
        times = [x * 0.1 for x in range(200, 4, -1)]
        timer = lambda: times.pop()
        verbose = False
        target = 3
        out = io.StringIO()
        bar = Progbar(target, width=30, verbose=verbose, interval=0.05, stdout=out, timer=timer, dynamic_display=True)
        self.assertListEqual(list(range(target)), stream(range(target)).map(bar).toList())
        expected = ('\r1/3 [=========>....................] - ETA: 0s'
                    '\r2/3 [===================>..........] - ETA: 0s          '
                    '\r3/3 [==============================] - 0s 100ms/step          \n')
        self.assertEqual(expected, out.getvalue())

class TestReversedCSVReader(TestCase):
    TESTS_ROOT = Path(__file__).absolute().parent

    def test_reversed_itr(self):
        reader = ReversedCSVReader(self.TESTS_ROOT / 'data' / 'ADABTC.agg-60s.tick.csv.gz', buf_size=128)
        in_order = list(reversed(list(stream(reader).map(itemgetter('time')))))
        rev_order = stream(reader).reversed().map(itemgetter('time')).toList()
        self.assertListEqual(in_order, rev_order)

    def test_opener_nominal(self):
        opener = lambda filename, mode, newline=None:gzip.open(filename, mode, newline=newline)
        reader = ReversedCSVReader(self.TESTS_ROOT / 'data' / 'ADABTC.agg-60s.tick.csv.gz', buf_size=128, opener=opener)
        in_order = list(reversed(list(stream(reader).map(itemgetter('time')))))
        rev_order = stream(reader).reversed().map(itemgetter('time')).toList()
        self.assertListEqual(in_order, rev_order)


if __name__ == '__main__':
    main()
