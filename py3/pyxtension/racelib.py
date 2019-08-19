#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: Concurrent utility classes (name coming from RACEconditionLIBrary)
# Created: 11/26/2015
import threading
import time

from typing import Union, Callable

__author__ = 'ASU'


class ContextLock():
    def __init__(self, lock: threading.Lock):
        self.__lock = lock

    def __enter__(self):
        self.__lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.__lock.release()
        return False

class TimePerformanceLogger:
    """
    Used to measure the performance of a code block run within a With Statement Context Manager
    """

    def __init__(self, logger: Callable[[float], None] = lambda sec: print("Finished in %.02f sec" % sec)):
        """
        :param logger: logger function that would get number of seconds as argument
        """
        self._logger = logger

    def __enter__(self):
        self._t1 = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self._logger(time.time() - self._t1)
        if exc_type:
            return False
        return True


class CountLogger:
    """
    Used to log partial progress of streams
    """

    def __init__(self, log_interval: int = 1000,
                 msg: str = "\rProcessed %d out of %d in %.01f sec. ETA: %.01f sec",
                 total: int = -1,
                 func: Union[Callable[[str], None], Callable[[str, str], None]] = print,
                 use_end: bool = True):
        self._cnt = 0
        self._n = log_interval
        self._t0 = None
        self._msg = msg
        self._func = func
        self._use_end = use_end
        self._total = total

    def __call__(self, e):
        if self._t0 is None:
            self._t0 = time.time()
        self._cnt += 1
        if self._cnt % self._n == 0:
            elapsed = time.time() - self._t0
            eta = elapsed / self._cnt * (self._total-self._cnt) if self._total>0 else float('NaN')
            msg = self._msg % (self._cnt, self._total, elapsed, eta)
            kwargs = {'end': ''} if self._use_end else {}
            self._func(msg, **kwargs)
        return e
