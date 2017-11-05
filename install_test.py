#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 11/6/2017
from pyxtension.Json import Json
from pyxtension.streams import stream

__author__ = 'ASU'

if __name__ == '__main__':
    s = stream([(1,2),(3,4)]).toList()
    j = s.toJson()
    js = str(j)
    j = Json({1:2})