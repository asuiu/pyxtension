#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: Concurrent utility classes (name coming from RACEconditionLIBrary)
# Created: 11/26/2015
import time

__author__ = 'ASU'


class ContextLock():
    def __init__(self, lock):
        """
        :param lock:
        :type lock: thread.LockType
        """
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

    def __init__(self, logger):
        """
        :param logger: logger function tha would get argument number of seconds
        :type logger: (basestring) -> None
        """
        self._logger = logger

    def __enter__(self):
        self._t1 = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self._logger(time.time() - self._t1)
        if exc_type:
            return False
        return True
