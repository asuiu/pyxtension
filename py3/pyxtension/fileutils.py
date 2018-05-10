#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library
import gzip
import io
from bz2 import BZ2File
from gzip import GzipFile

__author__ = 'ASU'


def linereader(f):
    br = io.BufferedReader(f)
    for line in br:
        yield line.decode()
        
def openByExtension(filename, mode='r', buffering=-1, compresslevel=9):
    """
    :param filename: path to filename
    :type filename: basestring
    :type mode: basestring
    :param buffering:
    :type buffering: int
    :return: Returns an opened file-like object, decompressing/compressing data depending on file extension
    :rtype: file | GzipFile | BZ2File
    """
    m = -1
    if 'r' in mode:
        m = 0
    elif 'w' in mode:
        m = 1
    elif 'a' in mode:
        m = 2
    tm = ('r', 'w', 'a')
    bText = 't' in mode

    if filename.endswith('.gz'):
        return gzip.open(filename, tm[m], compresslevel=compresslevel)
    elif filename.endswith('.bz2'):
        mode = tm[m]
        if bText: mode += 'U'
        if buffering <= 1:
            buffering = 0
        return BZ2File(filename, mode, buffering=buffering, compresslevel=compresslevel)
    else:
        return open(filename, mode, buffering=buffering)
