#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library
from operator import itemgetter

try:  # in Python 3.x the cPickle do not exist anymore
    import cPickle as pickle
except ImportError:
    import pickle

from Queue import Queue
import struct
import threading
import collections
from itertools import groupby

try:  # Python 3.x doesn't have ifilter, imap
    from itertools import ifilter, imap, izip
except ImportError:
    ifilter = filter
    imap = map
    izip = zip
import sys
import math
from collections import defaultdict

if sys.version_info[0] >= 3:
    xrange = range
from pyxtension.fileutils import openByExtension

__author__ = 'ASU'

_IDENTITY_FUNC = lambda _: _


class ItrFromFunc():
    def __init__(self, f):
        if callable(f):
            self._f = f
        else:
            raise TypeError(
                "Argument f to %s should be callable, but f.__class__=%s" % (str(self.__class__), str(f.__class__)))
    
    def __iter__(self):
        return iter(self._f())


class CallableGeneratorContainer():
    def __init__(self, iterableFunctions):
        self._ifs = iterableFunctions
    
    def __call__(self):
        return iteratorJoiner(self._ifs)


def iteratorJoiner(itrIterables):
    for i in itrIterables:
        for obj in i:
            yield obj


class EndQueue:
    pass


class MapException:
    def __init__(self, exc_info):
        self.exc_info = exc_info


class _IStream(collections.Iterable):
    def map(self, f):
        '''
        :param f:
        :type f: (T) -> V
        :return:
        :rtype: stream
        '''
        return stream(ItrFromFunc(lambda: imap(f, self)))

    @staticmethod
    def __map_thread(f, qin, qout):
        while True:
            el = qin.get()
            if isinstance(el, EndQueue):
                qin.put(el)
                return
            try:
                newEl = f(el)
                qout.put(newEl)
            except:
                qout.put(MapException(sys.exc_info()))

    @staticmethod
    def __fastmap_generator(itr, qin, qout, threadPool):
        while 1:
            try:
                el = next(itr)
            except StopIteration:
                qin.put(EndQueue())
                for t in threadPool:
                    t.join()
                while not qout.empty():
                    newEl = qout.get()
                    if isinstance(newEl, MapException):
                        raise newEl.exc_info[0], newEl.exc_info[1], newEl.exc_info[2]
                    yield newEl
                break
            else:
                qin.put(el)
                newEl = qout.get()
                if isinstance(newEl, MapException):
                    raise newEl.exc_info[0], newEl.exc_info[1], newEl.exc_info[2]
                yield newEl

    @staticmethod
    def __unique_generator(itr, f):
        st = set()
        for el in itr:
            m_el = f(el)
            if m_el not in st:
                st.add(m_el)
                yield el

    def fastmap(self, f, poolSize=16):
        """
        Parallel unordered map using multithreaded pool.
        It spawns at most poolSize threads and applies the f function.
        The elements in the result stream appears in the unpredicted order.
        It's most usefull for I/O or CPU intensive consuming functions.
        :param f:
        :type f: (T) -> V
        :param poolSize: number of threads to spawn
        :type poolSize: int
        :return:
        :rtype: stream
        """
        assert poolSize > 0
        assert poolSize < 2 ** 12
        Q_SZ = poolSize * 4
        qin = Queue(Q_SZ)
        qout = Queue(Q_SZ)
        threadPool = [threading.Thread(target=_IStream.__map_thread, args=(f, qin, qout)) for i in xrange(poolSize)]
        for t in threadPool:
            t.start()
        i = 0
        itr = iter(self)
        hasNext = True
        while i < Q_SZ and hasNext:
            try:
                el = next(itr)
                i += 1
                qin.put(el)
            except StopIteration:
                hasNext = False
        return stream(_IStream.__fastmap_generator(itr, qin, qout, threadPool))

    def enumerate(self):
        return stream(izip(xrange(0, sys.maxint), self))

    @classmethod
    def __flatMapGenerator(cls, itr, f):
        for i in itr:
            for j in f(i):
                yield j

    def flatMap(self, predicate=_IDENTITY_FUNC):
        """
        :param predicate: predicate is a function that will receive elements of self collection and return an iterable
            By default predicate is an identity function
        :type predicate: (self.elementsType)-> collections.Iterable[T]
        :return: will return stream of objects of the same type of elements from the stream returned by predicate()
        :rtype: stream[T]
        """
        return stream(ItrFromFunc(lambda: self.__class__.__flatMapGenerator(self, predicate)))

    def filter(self, predicate=None):
        """
        :param predicate: If predicate is None, return the items that are true.
        :type predicate: None|(T) -> bool
        :rtype: stream
        """
        return stream(ItrFromFunc(lambda: ifilter(predicate, self)))

    def reversed(self):
        try:
            return stream(reversed(self))
        except TypeError:
            raise TypeError("Can not reverse stream")

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

    def keyBy(self, keyfunc=_IDENTITY_FUNC):
        """
        :param keyfunc: function to map values to keys
        :type keyfunc: (V) -> T
        :return: stream of Key, Value pairs
        :rtype: stream[( T, V )]
        """
        return self.map(lambda h: (keyfunc(h), h))

    def keystream(self):
        """
        Applies only on streams of 2-uples
        :return: stream consisted of first element of tuples
        :rtype: stream[T]
        """
        return self.map(itemgetter(0))

    def values(self):
        """
        Applies only on streams of 2-uples
        :return: stream consisted of second element of tuples
        :rtype: stream[T]
        """
        return self.map(itemgetter(1))

    def groupBy(self, keyfunc=_IDENTITY_FUNC):
        """
        groupBy([keyfunc]) -> Make an iterator that returns consecutive keys and groups from the iterable.
        The iterable needs not to be sorted on the same key function, but the keyfunction need to return hasable objects.
        :param keyfunc: [Optional] The key is a function computing a key value for each element.
        :type keyfunc: (T) -> (V)
        :return: (key, sub-iterator) grouped by each value of key(value).
        :rtype: stream[ ( V, slist[T] ) ]
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
        return self.unique()

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

    def toSumCounter(self):
        """
        Elements should be tuples (T, V) where V can be summed
        :return: sdict on stream elements
        :rtype: sdict[ T, V ]
        """
        res = sdict()
        for k, v in self:
            if k in res:
                res[k] += v
            else:
                res[k] = v
        return res

    def toJson(self):
        from pyxtension.Json import JsonList

        return JsonList(self)

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

    def join(self, f=None):
        if f is None:
            return ''.join(self)
        elif isinstance(f, basestring):
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
                    break
            return r

    def mkString(self, c):
        return self.join(c)

    def take(self, n):
        return self[:n]

    def head(self):
        return next(iter(self))

    def sum(self):
        return sum(self)

    def min(self, key=_IDENTITY_FUNC):
        return min(self, key=key)

    def min_default(self, default, key=_IDENTITY_FUNC):
        """
        :param default: returned if there's no minimum in stream (ie empty stream)
        :type default: T
        :param key: the same meaning as used for the builtin min()
        :type key: (T) -> V
        :rtype: T
        """
        try:
            return min(self, key=key)
        except ValueError as e:
            if "empty sequence" in e.message:
                return default
            else:
                raise

    def max(self, **kwargs):
        return max(self, **kwargs)

    def maxes(self, key=_IDENTITY_FUNC):
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

    def mins(self, key=_IDENTITY_FUNC):
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

    def pstddev(self):
        """Calculates the population standard deviation."""
        sm = 0
        n = 0
        for el in self:
            sm += el
            n += 1
        if n < 1:
            raise ValueError('Standard deviation requires at least one data point')
        mean = float(sm) / n
        ss = sum((x - mean) ** 2 for x in self)
        pvar = ss / n  # the population variance
        return pvar ** 0.5

    def mean(self):
        """Return the sample arithmetic mean of data. in one single pass"""
        sm = 0
        n = 0
        for el in self:
            sm += el
            n += 1
        if n < 1:
            raise ValueError('Mean requires at least one data point')
        return sm / float(n)

    def zip(self):
        return stream(izip(*(self.toList())))

    def unique(self, predicate=_IDENTITY_FUNC):
        """
        The stream items should be hashable and comparable.
        :param predicate: optional, maps the elements to comparable objects
        :type predicate: (T) -> U
        :return: Unique elements appearing in the same order. Following copies of same elements will be ignored.
        :rtype: stream[U]
        """
        return stream(ItrFromFunc(lambda: _IStream.__unique_generator(self, predicate)))

    @staticmethod
    def binaryToChunk(binaryData):
        """
        :param binaryData: binary data to transform into chunk with header
        :type binaryData: str
        :return: chunk of data with header
        :rtype: str
        """
        l = len(binaryData)
        p = struct.pack("<L", l)
        assert len(p) == 4
        return p + binaryData

    def dumpToPickle(self, fileStream):
        '''
        :param fileStream: should be binary output stream
        :type fileStream: file
        :return: Nothing
        '''

        for el in self.map(pickle.dumps).map(stream.binaryToChunk):
            fileStream.write(el)

    def dumpPickledToWriter(self, writer):
        '''
        :param writer: should be binary output callable stream
        :type writer: callable
        :return: Nothing
        '''
        for el in self:
            writer(stream._picklePack(el))

    @staticmethod
    def _picklePack(el):
        return stream.binaryToChunk(pickle.dumps(el, pickle.HIGHEST_PROTOCOL))

    def exceptIndexes(self, *indexes):
        """
        Doesn't support negative indexes as the stream doesn't have a length
        :type indexes: list[int]
        :return: the stream with filtered out elements on <indexes> positions
        :rtype: stream [ T ]
        """

        def indexIgnorer(indexSet, _stream):
            i = 0
            for el in _stream:
                if i not in indexSet:
                    yield el
                i += 1

        indexSet = sset(indexes)
        return stream(ItrFromFunc(lambda: indexIgnorer(indexSet, self)))


class stream(_IStream):
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
            return object.__str__(self)

    @staticmethod
    def __binaryChunksStreamGenerator(fs, format="<L", statHandler=None):
        """
        :param fs:
        :type fs: file
        :param format:
        :type format: str
        :param statHandler: statistics handler, will be called before every yield with a tuple (n,size)
        :type statHandler: callable
        :return: unpickled element
        :rtype: T
        """
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
            if statHandler is not None:
                count += 1
                sz += 4 + l
                statHandler((count, sz))
            yield s

    @staticmethod
    def readFromBinaryChunkStream(readStream, format="<L", statHandler=None):
        '''
        :param file: should be path or binary file stream
        :param statHandler: statistics handler, will be called before every yield with a tuple (n,size)
        :type statHandler: callable
        :type file: file | str
        :rtype: stream[T]
        '''
        if isinstance(readStream, basestring):
            readStream = openByExtension(readStream, mode='r', buffering=2 ** 12)
        return stream(stream.__binaryChunksStreamGenerator(readStream, format, statHandler))

    @staticmethod
    def loadFromPickled(file, format="<L", statHandler=None):
        '''
        :param file: should be path or binary file stream
        :param statHandler: statistics handler, will be called before every yield with a tuple (n,size)
        :type statHandler: callable
        :type file: file | str
        :rtype: stream[T]
        '''

        if isinstance(file, basestring):
            file = openByExtension(file, mode='r', buffering=2 ** 12)
        return stream.readFromBinaryChunkStream(file, format, statHandler).map(pickle.loads)


class AbstractSynchronizedBufferedStream(stream):
    """
    Thread-safe buffered stream.
    Just implement the _getNextBuffer() to return a slist() and you are good to go.
    """

    def __init__(self):
        self.__queue = slist()
        self.__lock = threading.RLock()
        self.__idx = -1

    def next(self):
        self.__lock.acquire()
        try:
            val = self.__queue[self.__idx]
        except IndexError:
            self.__queue = self._getNextBuffer()
            assert isinstance(self.__queue, slist)
            if len(self.__queue) == 0:
                raise StopIteration
            val = self.__queue[0]
            self.__idx = 0

        self.__idx += 1
        self.__lock.release()
        return val

    def __iter__(self):
        return self

    def _getNextBuffer(self):
        """
        :return: a list of items for the buffer
        :rtype: slist[T]
        """
        raise NotImplementedError

    def __str__(self):
        return object.__str__(self)

    def __repr__(self):
        return object.__repr__(self)


class SynchronizedBufferedStream(AbstractSynchronizedBufferedStream):
    def __init__(self, iteratorOverBuffers):
        """
        :param bufferGetter: iterator over slist objects
        :type bufferGetter: stream[slist[T]]
        """
        self.__iteratorOverBuffers = iter(iteratorOverBuffers)
        super(SynchronizedBufferedStream, self).__init__()

    def _getNextBuffer(self):
        try:
            return next(self.__iteratorOverBuffers)
        except StopIteration:
            return slist()


class sset(set, _IStream):
    def __init__(self, *args, **kwrds):
        set.__init__(self, *args, **kwrds)

    def __iter__(self):
        return set.__iter__(self)

    # Below methods enable chaining and lambda using
    def update(self, *args, **kwargs):
        # ToDo: Add option to update with iterables, as set.update supports only other set
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


class slist(list, _IStream):
    @property
    def _itr(self):
        return ItrFromFunc(lambda: self)
    
    def __init__(self, *args, **kwrds):
        list.__init__(self, *args, **kwrds)
    
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
    
    def exceptIndexes(self, *indexes):
        """
        Supports negative indexes
        :type indexes: list[int]
        :return: the stream with filtered out elements on <indexes> positions
        :rtype: stream [ T ]
        """

        def indexIgnorer(indexSet, _stream):
            i = 0
            for el in _stream:
                if i not in indexSet:
                    yield el
                i += 1

        sz = self.size()
        indexSet = stream(indexes).map(lambda i: i if i >= 0 else i + sz).toSet()
        return stream(ItrFromFunc(lambda: indexIgnorer(indexSet, self)))
    
    def __iadd__(self, x):
        return list.__iadd__(self, x)

    def __add__(self, other):
        return _IStream.__add__(self, other)
    
    def __getitem__(self, item):
        return list.__getitem__(self, item)


class sdict(dict, _IStream):
    def __init__(self, *args, **kwrds):
        dict.__init__(self, *args, **kwrds)

    def __iter__(self):
        return dict.__iter__(self)

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

    def toJson(self):
        from pyxtension.Json import Json
        return Json(self)


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

    def __str__(self):
        return dict.__str__(self)


def smap(f, itr):
    return stream(itr).map(f)


def sfilter(f, itr):
    return stream(itr).filter(f)
