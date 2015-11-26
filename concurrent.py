#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: Concurrent utility classes
# Created: 11/26/2015

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
