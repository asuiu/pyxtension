#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library

import cPickle
import struct
import collections
from itertools import ifilter, groupby, imap, izip
import sys
import math
from collections import defaultdict

from fileutils import openByExtension

__author__ = 'ASU'


class CallableGeneratorContainer:
    def __init__(self, iterableFunctions):
        self._ifs = iterableFunctions

    def __call__(self):
        return self.__class__.iteratorJoiner(self._ifs)

    @classmethod
    def iteratorJoiner(cls, itrIterables):
        for i in itrIterables:
            # itr = iter(i)
            for obj in i:
                yield obj


class ItrFromFunc:
    def __init__(self, f):
        if callable(f):
            self._f = f
        else:
            raise TypeError(
                "Argument f to %s should be callable, but f.__class__=%s" % (str(self.__class__), str(f.__class__)))

    def __iter__(self):
        return iter(self._f())


class __IStream(collections.Iterable):
    def map(self, f):
        '''

        :param f:
        :type f: lambda
        :return:
        :rtype: stream
        '''
        return stream(ItrFromFunc(lambda: imap(f, self)))

    def enumerate(self):
        return stream(izip(xrange(0, sys.maxint), self))

    @classmethod
    def __flatMapGenerator(cls, itr, f):
        for i in itr:
            for j in f(i):
                yield j

    def flatMap(self, f=None):
        """
        :param f: f is a function that will receive elements of self collection and return an iterable
            By default f is an identity function
        :type f: (self.elementsType)-> collections.Iterable[T]
        :return: will return stream of objects of the same type of elements from the stream returned by f()
        :rtype:
        """
        if f is None:
            f = lambda x: x
        return stream(ItrFromFunc(lambda: self.__class__.__flatMapGenerator(self, f)))

    def filter(self, f):
        return stream(ItrFromFunc(lambda: ifilter(f, self)))
        # return stream(lambda: ifilter(f, self), functionalItr=True)

    def exists(self, f):
        """
        Tests whether a predicate holds for some of the elements of this sequence.
        :param f:
        :type f: (T) -> bool
        :return:
        :rtype: bool
        """
        for e in self:
            if f(e):
                return True
        return False

    def keyBy(self, keyfunc):
        return self.map(lambda h: (keyfunc(h), h))

    def groupBy(self, keyfunc):
        """
        groupBy(keyfunc]) -> create an iterator which returns
        (key, sub-iterator) grouped by each value of key(value).
        """
        # return stream(
        # 	ItrFromFunc(lambda: groupby(sorted(self, key=keyfunc), keyfunc))).map(lambda kv: (kv[0], stream(kv[1])))
        h = defaultdict(slist)
        for v in self:
            h[keyfunc(v)].append(v)
        ##for
        return stream(h.iteritems())

    def groupByToList(self, keyfunc):
        """
        groupBy(keyfunc]) -> create an iterator which returns
        (key, sub-iterator) grouped by each value of key(value).
        """
        return stream(
            ItrFromFunc(lambda: groupby(sorted(self, key=keyfunc), keyfunc))).map(lambda kv: (kv[0], slist(kv[1])))

    def countByValue(self):
        return sdict(collections.Counter(self))

    def distinct(self):
        return self.toSet()

    def reduce(self, f, init=None):
        if init is None:
            return reduce(f, self)
        else:
            return reduce(f, self, init)

    def toSet(self):
        """

        :rtype : sset
        """
        return sset(self)

    def toList(self):
        '''
        :return:
        :rtype: slist
        '''
        return slist(self)

    def sorted(self, key=None, cmp=None, reverse=False):
        return slist(sorted(self, key=key, cmp=cmp, reverse=reverse))

    def toMap(self):
        return sdict(self)

    def toJson(self):
        from Json import Json

        return Json(self)

    def __getitem__(self, i):
        itr = iter(self)
        tk = 0
        while tk < i:
            next(itr)
            tk += 1
        return next(itr)

    def __getslice__(self, i, j):
        def gs(strm):
            itr = iter(strm)
            tk = 0
            while tk < i:
                next(itr)
                tk += 1
            while tk < j:
                yield next(itr)
                tk += 1

        return stream(ItrFromFunc(lambda: gs(self)))

    def __add__(self, other):
        if not isinstance(other, ItrFromFunc):
            othItr = ItrFromFunc(lambda: other)
        else:
            othItr = other
        if isinstance(self._itr, ItrFromFunc):
            i = self._itr
        else:
            i = ItrFromFunc(lambda: self._itr)
        return stream(ItrFromFunc(CallableGeneratorContainer((i, othItr))))

    def __iadd__(self, other):
        if not isinstance(other, ItrFromFunc):
            othItr = ItrFromFunc(lambda: other)
        else:
            othItr = other
        if isinstance(self._itr, ItrFromFunc):
            i = self._itr
        else:
            j = self._itr
            i = ItrFromFunc(lambda: j)

        self._itr = ItrFromFunc(CallableGeneratorContainer((i, othItr)))
        return self

    def size(self):
        try:
            return len(self)
        except:
            return sum(1 for i in iter(self))

    def join(self, f):
        if isinstance(f, basestring):
            return f.join(self)
        else:
            itr = iter(self)
            r = next(itr)
            last = bytearray(r)
            while True:
                try:
                    n = next(itr)
                    r += f(last)
                    last = n
                    r += n
                except StopIteration:
                    pass
            return r

    def mkString(self, c):
        return self.join(c)

    def take(self, n):
        return self[:n]

    def head(self):
        return next(iter(self))

    def sum(self):
        return sum(self)

    def min(self, **kwargs):
        return min(self, **kwargs)

    def max(self, **kwargs):
        return max(self, **kwargs)

    def maxes(self, key=lambda x: x):
        i = iter(self)
        aMaxes = [next(i)]
        mval = key(aMaxes[0])
        for v in i:
            k = key(v)
            if k > mval:
                mval = k
                aMaxes = [v]
            elif k == mval:
                aMaxes.append(v)
                ##if
        return slist(aMaxes)

    def mins(self, key=lambda x: x):
        i = iter(self)
        aMaxes = [next(i)]
        mval = key(aMaxes[0])
        for v in i:
            k = key(v)
            if k < mval:
                mval = k
                aMaxes = [v]
            elif k == mval:
                aMaxes.append(v)
                ##if
        return slist(aMaxes)

    def entropy(self):
        s = self.sum()
        return self.map(lambda x: (float(x) / s) * math.log(s / float(x), 2)).sum()

    def zip(self):
        return stream(izip(*(self.toList())))

    def dumpToPickle(self, fileStream):
        '''
        :param fileStream: should be binary output stream
        :type fileStream: file
        :return: Nothing
        '''
        for el in self:
            s = cPickle.dumps(el, cPickle.HIGHEST_PROTOCOL)
            l = len(s)
            p = struct.pack("<L", l)
            assert len(p) == 4
            fileStream.write(p + s)

    @classmethod
    def __unpickleStreamGenerator(cls, fs, format="<L", statHandler=None):
        count = 0
        sz = 0
        while True:
            s = fs.read(4)
            if s == '':
                return
            elif len(s) != 4:
                raise IOError("Wrong pickled file format")
            l = struct.unpack(format, s)[0]
            s = fs.read(l)
            el = cPickle.loads(s)
            if statHandler is not None:
                count += 1
                sz += 4 + l
                statHandler((count, sz))
            yield el

    @classmethod
    def loadFromPickled(cls, file, format="<L", statHandler=None):
        '''
        :param file: should be path or binary file stream
        :type file: file | str
        :rtype: stream
        '''
        if isinstance(file, basestring):
            file = openByExtension(file, mode='r', buffering=2 ** 12)
        return cls(stream.__unpickleStreamGenerator(file, format, statHandler))


class stream(__IStream):
    def __init__(self, itr=None):
        if itr is None:
            self._itr = []
        else:
            self._itr = itr

    def __iter__(self):
        return iter(self._itr)

    def __repr__(self):
        if isinstance(self._itr, list):
            repr(self._itr)
        else:
            return object.__repr__(self)

    def __str__(self):
        if isinstance(self._itr, list):
            return str(self._itr)
        else:
            return str(list(self._itr))


class sset(set, __IStream):
    def __init__(self, *args, **kwrds):
        set.__init__(self, *args, **kwrds)

    def __iter__(self):
        return set.__iter__(self)

    # Below methods enable chaining and lambda using
    def update(self, *args, **kwargs):
        super(sset, self).update(*args, **kwargs)
        return self

    def intersection_update(self, *args, **kwargs):
        super(sset, self).intersection_update(*args, **kwargs)
        return self

    def difference_update(self, *args, **kwargs):
        super(sset, self).difference_update(*args, **kwargs)
        return self

    def symmetric_difference_update(self, *args, **kwargs):
        super(sset, self).symmetric_difference_update(*args, **kwargs)
        return self

    def clear(self, *args, **kwargs):
        super(sset, self).clear(*args, **kwargs)
        return self

    def remove(self, *args, **kwargs):
        super(sset, self).remove(*args, **kwargs)
        return self

    def add(self, *args, **kwargs):
        super(sset, self).add(*args, **kwargs)
        return self

    def discard(self, *args, **kwargs):
        super(sset, self).discard(*args, **kwargs)
        return self


class slist(list, __IStream):
    def __init__(self, *args, **kwrds):
        list.__init__(self, *args, **kwrds)

    # def __iter__(self):
    #     return list.__iter__(self)

    def __getslice__(self, i, j):
        def gs(strm):
            itr = iter(strm)
            tk = 0
            while tk < i:
                next(itr)
                tk += 1
            while tk < j:
                yield next(itr)
                tk += 1

        return slist(ItrFromFunc(lambda: gs(self)))

    def extend(self, iterable):
        '''
        :param iterable:
        :type iterable:
        :return:
        :rtype: slist
        '''
        list.extend(self, iterable)
        return self

    def append(self, x):
        list.append(self, x)
        return self

    def remove(self, x):
        list.remove(self, x)
        return self

    def insert(self, i, x):
        list.insert(self, i, x)
        return self


class sdict(dict, __IStream):
    def __init__(self, *args, **kwrds):
        dict.__init__(self, *args, **kwrds)

    def __iter__(self):
        return dict.iteritems(self)

    def iteritems(self):
        return stream(dict.iteritems(self))

    def iterkeys(self):
        return stream(dict.iterkeys(self))

    def itervalues(self):
        return stream(dict.itervalues(self))

    def keys(self):
        return slist(dict.keys(self))

    def values(self):
        return slist(dict.values(self))

    def items(self):
        return slist(self.iteritems())

    def update(self, other=None, **kwargs):
        super(sdict, self).update(other, **kwargs)
        return self


class defaultstreamdict(sdict):
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not callable(default_factory)):
            raise TypeError('first argument must be callable')
        super(self.__class__, self).__init__(*a, **kw)
        if default_factory is None:
            self.__default_factory = lambda: object()
        else:
            self.__default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super(self.__class__, self).__getitem__(key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        # if self.__default_factory is None:
        #     raise KeyError(key)
        self[key] = value = self.__default_factory()
        return value

    def __reduce__(self):
        # if self.__default_factory is None:
        #     args = tuple()
        # else:
        args = self.__default_factory,
        return type(self), args, None, None, iter(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.__default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.__default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'defaultdict(%s, %s)' % (self.__default_factory,
                                        super(self.__class__, self).__repr__())


def smap(f, itr):
    return stream(itr).map(f)


def sfilter(f, itr):
    return stream(itr).filter(f)
