#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library
import collections
import itertools
import math
import numbers
import pickle
import struct
import sys
import threading
from abc import ABC
from collections import defaultdict
from functools import reduce
from itertools import groupby
from operator import itemgetter
from queue import Queue
from typing import Optional, Union, Callable, TypeVar, Iterable, Iterator, Tuple, BinaryIO, List, Mapping, MutableSet, \
    Dict

ifilter = filter
imap = map
izip = zip
xrange = range
from pyxtension.fileutils import openByExtension

__author__ = 'ASU'

_IDENTITY_FUNC = lambda _: _


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


class EndQueue:
    pass


class MapException:
    def __init__(self, exc_info):
        self.exc_info = exc_info


K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


class _IStream(Iterable[K], ABC):
    def map(self, f: Callable[[K], V]) -> 'stream[K]':
        return stream(ItrFromFunc(lambda: map(f, self)))
    
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
                        raise newEl.exc_info[0](newEl.exc_info[1]).with_traceback(newEl.exc_info[2])
                    yield newEl
                break
            else:
                qin.put(el)
                newEl = qout.get()
                if isinstance(newEl, MapException):
                    raise newEl.exc_info[0](newEl.exc_info[1]).with_traceback(newEl.exc_info[2])
                yield newEl
    
    @staticmethod
    def __unique_generator(itr, f):
        st = set()
        for el in itr:
            m_el = f(el)
            if m_el not in st:
                st.add(m_el)
                yield el
    
    def fastmap(self, f: Callable[[K], V], poolSize: int = 16) -> 'stream[V]':
        """
        Parallel unordered map using multithreaded pool.
        It spawns at most poolSize threads and applies the f function.
        The elements in the result stream appears in the unpredicted order.
        It's most usefull for I/O or CPU intensive consuming functions.
        :param poolSize: number of threads to spawn
        """
        assert poolSize > 0
        assert poolSize < 2 ** 12
        Q_SZ = poolSize * 4
        qin = Queue(Q_SZ)
        qout = Queue(Q_SZ)
        threadPool = [threading.Thread(target=_IStream.__map_thread, args=(f, qin, qout)) for i in range(poolSize)]
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
    
    def enumerate(self) -> 'stream[K]':
        return stream(zip(range(0, sys.maxsize), self))
    
    @classmethod
    def __flatMapGenerator(cls, itr, f):
        for i in itr:
            for j in f(i):
                yield j
    
    def flatMap(self, predicate: Callable[[K], Iterable[V]] = _IDENTITY_FUNC) -> 'stream[V]':
        """
        :param predicate: predicate is a function that will receive elements of self collection and return an iterable
            By default predicate is an identity function
        :return: will return stream of objects of the same type of elements from the stream returned by predicate()
        """
        return stream(ItrFromFunc(lambda: self.__class__.__flatMapGenerator(self, predicate)))
    
    def filter(self, predicate: Optional[Callable[[K], bool]] = None) -> 'stream[K]':
        """
        :param predicate: If predicate is None, return the items that are true.
        """
        return stream(ItrFromFunc(lambda: filter(predicate, self)))
    
    def reversed(self) -> 'stream[K]':
        try:
            return stream(reversed(self))
        except TypeError:
            raise TypeError("Can not reverse stream")
    
    def exists(self, f: Callable[[K], bool]) -> bool:
        """
        Tests whether a predicate holds for some of the elements of this sequence.
        """
        for e in self:
            if f(e):
                return True
        return False
    
    def keyBy(self, keyfunc: Callable[[K], V] = _IDENTITY_FUNC) -> 'stream[Tuple[K,V]]':
        """
        :param keyfunc: function to map values to keys
        :return: stream of Key, Value pairs
        """
        return self.map(lambda h: (keyfunc(h), h))
    
    def keystream(self: 'stream[Tuple[T,V]]') -> 'stream[T]':
        """
        Applies only on streams of 2-uples
        :return: stream consisted of first element of tuples
        """
        return self.map(itemgetter(0))
    
    def values(self: 'stream[Tuple[T,V]]') -> 'stream[V]':
        """
        Applies only on streams of 2-uples
        :return: stream consisted of second element of tuples
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
        return stream(iter(h.items()))
    
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
    
    def reduce(self, f:Callable[[Union[K,V],K],T], init:Optional[Union[K,V]]=None)->'T':
        if init is None:
            return reduce(f, self)
        else:
            return reduce(f, self, init)
    
    def toSet(self) -> 'sset[K]':
        return sset(self)
    
    def toList(self) -> 'slist[K]':
        '''
        :return:
        :rtype: slist
        '''
        return slist(self)
    
    def sorted(self, key=None, reverse=False):
        return slist(sorted(self, key=key, reverse=reverse))
    
    def toMap(self: 'stream[Tuple[T,V]]') -> 'sdict[T,V]':
        return sdict(self)
    
    def toSumCounter(self: 'stream[Tuple[T,V]]') -> 'sdict[T,V]]':
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
    
    def toJson(self) -> 'JsonList':
        from pyxtension.Json import JsonList
        return JsonList(self)
    
    def __getitem__(self, i: Union[slice, int]) -> 'Union[K, stream[K]]':
        if isinstance(i, slice):
            return self.__getslice(i.start, i.stop, i.step)
        else:
            itr = iter(self)
            tk = 0
            while tk < i:
                next(itr)
                tk += 1
            return next(itr)
    
    def __getslice(self, start: Optional[int] = None,
                   stop: Optional[int] = None,
                   step: Optional[int] = None) -> 'stream[K]':
        
        return stream(ItrFromFunc(lambda: itertools.islice(self, start, stop, step)))
    
    def __add__(self, other) -> 'stream[K]':
        if not isinstance(other, ItrFromFunc):
            othItr = ItrFromFunc(lambda: other)
        else:
            othItr = other
        if isinstance(self._itr, ItrFromFunc):
            i = self._itr
        else:
            i = ItrFromFunc(lambda: self._itr)
        return stream(ItrFromFunc(CallableGeneratorContainer((i, othItr))))
    
    def __iadd__(self, other) -> 'stream[K]':
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
    
    def size(self) -> int:
        try:
            return len(self)
        except:
            return sum(1 for i in iter(self))
    
    def join(self, f: Callable[[K], V] = None) -> Union[K, str]:
        if f is None:
            return ''.join(self)
        elif isinstance(f, str):
            return f.join(self)
        else:
            itr = iter(self)
            r = next(itr)
            last = r
            while True:
                try:
                    n = next(itr)
                    r += f(last)
                    last = n
                    r += n
                except StopIteration:
                    break
            return r
    
    def mkString(self, c) -> str:
        return self.join(c)
    
    def take(self, n) -> 'stream[K]':
        return self[:n]
    
    def head(self) -> K:
        return next(iter(self))
    
    def sum(self) -> numbers.Real:
        return sum(self)
    
    def min(self, key: Callable[[K], V] = _IDENTITY_FUNC) -> V:
        return min(self, key=key)
    
    def min_default(self, default: T, key: Callable[[K], V] = _IDENTITY_FUNC) -> Union[V, T]:
        """
        :param default: returned if there's no minimum in stream (ie empty stream)
        :param key: the same meaning as used for the builtin min()
        """
        try:
            return min(self, key=key)
        except ValueError as e:
            if "empty sequence" in e.args[0]:
                return default
            else:
                raise
    
    def max(self, key: Callable[[K], V] = _IDENTITY_FUNC) -> V:
        return max(self, key=key)
    
    def maxes(self, key: Callable[[K], V] = _IDENTITY_FUNC) -> 'slist[V]':
        i = iter(self)
        aMaxes = slist([next(i)])
        mval = key(aMaxes[0])
        for v in i:
            k = key(v)
            if k > mval:
                mval = k
                aMaxes = [v]
            elif k == mval:
                aMaxes.append(v)
        return aMaxes
    
    def mins(self, key: Callable[[K], V] = _IDENTITY_FUNC) -> 'slist[V]':
        i = iter(self)
        aMaxes = slist([next(i)])
        mval = key(aMaxes[0])
        for v in i:
            k = key(v)
            if k < mval:
                mval = k
                aMaxes = [v]
            elif k == mval:
                aMaxes.append(v)
        return aMaxes
    
    def entropy(self) -> numbers.Real:
        s = self.sum()
        return self.map(lambda x: (float(x) / s) * math.log(s / float(x), 2)).sum()
    
    def pstddev(self) -> float:
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
    
    def mean(self) -> float:
        """Return the sample arithmetic mean of data. in one single pass"""
        sm = 0
        n = 0
        for el in self:
            sm += el
            n += 1
        if n < 1:
            raise ValueError('Mean requires at least one data point')
        return sm / float(n)
    
    def zip(self) -> 'stream[V]':
        return stream(zip(*(self.toList())))
    
    def unique(self, predicate: Callable[[K], V] = _IDENTITY_FUNC):
        """
        The stream items should be hashable and comparable.
        :param predicate: optional, maps the elements to comparable objects
        :return: Unique elements appearing in the same order. Following copies of same elements will be ignored.
        :rtype: stream[U]
        """
        return stream(ItrFromFunc(lambda: _IStream.__unique_generator(self, predicate)))
    
    @staticmethod
    def binaryToChunk(binaryData: bytes) -> bytes:
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
        
        for el in self.map(lambda _: pickle.dumps(_, pickle.HIGHEST_PROTOCOL, fix_imports=True)).map(
                stream.binaryToChunk):
            fileStream.write(el)
    
    def dumpPickledToWriter(self, writer: Callable[[bytes], T]) -> None:
        '''
        :param writer: should be binary output callable stream
        '''
        for el in self:
            writer(stream._picklePack(el))
    
    @staticmethod
    def _picklePack(el) -> bytes:
        return stream.binaryToChunk(pickle.dumps(el, pickle.HIGHEST_PROTOCOL))
    
    def exceptIndexes(self, *indexes: List[int]) -> 'stream[K]':
        """
        Doesn't support negative indexes as the stream doesn't have a length
        :return: the stream with filtered out elements on <indexes> positions
        """
        
        def indexIgnorer(indexSet, _stream):
            i = 0
            for el in _stream:
                if i not in indexSet:
                    yield el
                i += 1
        
        indexSet = frozenset(indexes)
        return stream(ItrFromFunc(lambda: indexIgnorer(indexSet, self)))


class stream(_IStream, Iterable[K]):
    def __init__(self, itr: Optional[Iterator[K]] = None):
        if itr is None:
            self._itr = []
        else:
            self._itr = itr
    
    def __iter__(self) -> Iterator[K]:
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
    def __binaryChunksStreamGenerator(fs, format="<L", statHandler: Optional[Callable[[int, int], None]] = None):
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
            if s == b'':
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
    def readFromBinaryChunkStream(readStream: Union[BinaryIO, str],
                                  format: str = "<L",
                                  statHandler: Optional[Callable[[int, int], None]] = None) -> 'stream[V]':
        '''
        :param file: should be path or binary file stream
        :param statHandler: statistics handler, will be called before every yield with a tuple (n,size)
        '''
        if isinstance(readStream, str):
            readStream = openByExtension(readStream, mode='r', buffering=2 ** 12)
        return stream(stream.__binaryChunksStreamGenerator(readStream, format, statHandler))
    
    @staticmethod
    def loadFromPickled(file: Union[BinaryIO, str],
                        format: str = "<L", statHandler: Optional[Callable[[int, int], None]] = None) -> 'stream[V]':
        '''
        :param file: should be path or binary file stream
        :param statHandler: statistics handler, will be called before every yield with a tuple (n,size)
        '''
        if isinstance(file, str):
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
    
    def __next__(self):
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


class sset(set, MutableSet[K], _IStream):
    def __init__(self, *args, **kwrds):
        set.__init__(self, *args, **kwrds)
    
    def __iter__(self):
        return set.__iter__(self)
    
    # Below methods enable chaining and lambda using
    def update(self, *args, **kwargs) -> 'sset[K]':
        # ToDo: Add option to update with iterables, as set.update supports only other set
        super(sset, self).update(*args, **kwargs)
        return self
    
    def intersection_update(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).intersection_update(*args, **kwargs)
        return self
    
    def difference_update(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).difference_update(*args, **kwargs)
        return self
    
    def symmetric_difference_update(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).symmetric_difference_update(*args, **kwargs)
        return self
    
    def clear(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).clear(*args, **kwargs)
        return self
    
    def remove(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).remove(*args, **kwargs)
        return self
    
    def add(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).add(*args, **kwargs)
        return self
    
    def discard(self, *args, **kwargs) -> 'sset[K]':
        super(sset, self).discard(*args, **kwargs)
        return self


class slist(_IStream, List[K]):
    def __init__(self, *args, **kwrds):
        list.__init__(self, *args, **kwrds)
    
    def __getitem__(self, item) -> 'Union[K,slist[K]]':
        if isinstance(item, slice):
            return slist(list.__getitem__(self, item))
        else:
            return list.__getitem__(self, item)
    
    def extend(self, iterable: Iterable[K]) -> 'slist[K]':
        list.extend(self, iterable)
        return self
    
    def append(self, x) -> 'slist[K]':
        list.append(self, x)
        return self
    
    def remove(self, x) -> 'slist[K]':
        list.remove(self, x)
        return self
    
    def insert(self, i, x) -> 'slist[K]':
        list.insert(self, i, x)
        return self
    
    def exceptIndexes(self, *indexes: List[int]) -> 'stream[K]':
        """
        Supports negative indexes
        :type indexes: list[int]
        :return: the stream with filtered out elements on <indexes> positions
        :rtype: stream [ T ]
        """
        
        def indexIgnorer(indexSet: frozenset, _stream: 'stream[K]'):
            i = 0
            for el in _stream:
                if i not in indexSet:
                    yield el
                i += 1
        
        sz = self.size()
        indexSet = frozenset(stream(indexes).map(lambda i: i if i >= 0 else i + sz))
        return stream(ItrFromFunc(lambda: indexIgnorer(indexSet, self)))


class sdict(Dict[K, V], _IStream):
    def __init__(self, *args, **kwrds):
        dict.__init__(self, *args, **kwrds)
    
    def __iter__(self):
        return dict.__iter__(self)
    
    def keys(self) -> stream[K]:
        return stream(dict.keys(self))
    
    def values(self) -> stream[V]:
        return stream(dict.values(self))
    
    def items(self) -> stream[Tuple[K, V]]:
        return stream(dict.items(self))
    
    def update(self, other=None, **kwargs) -> 'sdict[K,V]':
        super(sdict, self).update(other, **kwargs)
        return self
    
    def toJson(self) -> 'sdict[K,V]':
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
    
    def __getitem__(self, key: K) -> V:
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
        args = (self.__default_factory,)
        itms = list(self.items())
        return type(self), args, None, None, iter(itms)
    
    def copy(self) -> Mapping[K, V]:
        return self.__copy__()
    
    def __copy__(self):
        return type(self)(self.__default_factory, self)
    
    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.__default_factory,
                          copy.deepcopy(list(self.items())))
    
    def __repr__(self) -> str:
        return 'defaultdict(%s, %s)' % (self.__default_factory,
                                        super(self.__class__, self).__repr__())
    
    def __str__(self) -> str:
        return dict.__str__(self)


def smap(f, itr: Iterable[K]) -> stream[K]:
    return stream(itr).map(f)


def sfilter(f, itr: Iterable[K]) -> stream[K]:
    return stream(itr).filter(f)
