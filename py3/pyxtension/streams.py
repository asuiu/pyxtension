#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library for >=Python3.6
import collections
import io
import itertools
import math
import numbers
import pickle
import struct
import sys
import threading
from abc import ABC
from collections import defaultdict, abc
from functools import reduce, partial
from itertools import groupby
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from queue import Queue
from types import GeneratorType
from typing import Optional, Union, Callable, TypeVar, Iterable, Iterator, Tuple, BinaryIO, List, \
    Mapping, MutableSet, \
    Dict, Generator, overload, AbstractSet, Set, Any

ifilter = filter
imap = map
izip = zip
xrange = range
from pyxtension.fileutils import openByExtension

from tqdm import tqdm

__author__ = 'ASU'

_K = TypeVar('_K')
_V = TypeVar('_V')
_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)

_IDENTITY_FUNC: Callable[[_T], _T] = lambda _: _


class ItrFromFunc(Iterable[_K]):
    def __init__(self, f: Callable[[], Iterable[_K]]):
        if callable(f):
            self._f = f
        else:
            raise TypeError(
                "Argument f to %s should be callable, but f.__class__=%s" % (str(self.__class__), str(f.__class__)))

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self._f())


class CallableGeneratorContainer(Callable[[], _K]):
    def __init__(self, iterableFunctions: Iterable[ItrFromFunc[_K]]):
        self._ifs = iterableFunctions

    def __call__(self) -> Generator[_K, None, None]:
        return itertools.chain.from_iterable(self._ifs)


class EndQueue:
    pass


class MapException:
    def __init__(self, exc_info):
        self.exc_info = exc_info


class TqdmMapper:

    def __init__(self, *args, **kwargs) -> None:
        """
        :param args: same args that are passed to tqdm
        :param kwargs: same args that are passed to tqdm
        """
        self._tqdm = tqdm(*args, **kwargs)

    def __call__(self, el: _K) -> _K:
        self._tqdm.update()
        return el


class _IStream(Iterable[_K], ABC):
    @staticmethod
    def __fastmap_thread(f, qin, qout):
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
    def __fastFlatMap_thread(f, qin, qout):
        while True:
            itr = qin.get()
            if isinstance(itr, EndQueue):
                qin.put(itr)
                qout.put(EndQueue())
                return
            try:
                newItr = f(itr)
                for el in newItr:
                    qout.put(el)
            except:
                qout.put(MapException(sys.exc_info()))

    def __fastmap_generator(self, f: Callable[[_K], _V], poolSize: int, bufferSize: int):
        qin = Queue(bufferSize)
        qout = Queue(max(bufferSize, poolSize + 1))  # max() is needed to not block when exiting

        threadPool = [threading.Thread(target=_IStream.__fastmap_thread, args=(f, qin, qout)) for _ in range(poolSize)]
        for t in threadPool:
            t.start()

        i = 0
        itr = iter(self)
        hasNext = True
        while i < bufferSize and hasNext:
            try:
                el = next(itr)
                i += 1
                qin.put(el)
            except StopIteration:
                hasNext = False

        try:
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
        finally:
            while not qin.empty():
                qin.get()
            qin.put(EndQueue())
            while not qout.empty() or not qout.empty():
                qout.get()
            for t in threadPool:
                t.join()

    @staticmethod
    def __fastFlatMap_input_thread(itr: Iterator[_K], qin: Queue):
        while 1:
            try:
                el = next(itr)
            except StopIteration:
                qin.put(EndQueue())
                return
            else:
                qin.put(el)

    def __fastFlatMap_generator(self, predicate, poolSize: int, bufferSize: int):
        qin = Queue(bufferSize)
        qout = Queue(bufferSize * 2)
        threadPool = [threading.Thread(target=_IStream.__fastFlatMap_thread, args=(predicate, qin, qout)) for i in
                      range(poolSize)]
        for t in threadPool:
            t.start()
        i = 0
        itr = iter(self)
        hasNext = True
        while i < bufferSize and hasNext:
            try:
                el = next(itr)
                i += 1
                qin.put(el)
            except StopIteration:
                hasNext = False
        inputThread = threading.Thread(target=_IStream.__fastFlatMap_input_thread, args=(itr, qin))
        inputThread.start()

        qout_counter = 0
        while qout_counter < len(threadPool):
            newEl = qout.get()
            if isinstance(newEl, MapException):
                raise newEl.exc_info[0](newEl.exc_info[1]).with_traceback(newEl.exc_info[2])
            if isinstance(newEl, EndQueue):
                qout_counter += 1
                if qout_counter >= len(threadPool):
                    inputThread.join()
                    for t in threadPool:
                        t.join()
                    while not qout.empty():
                        newEl = qout.get()
                        if isinstance(newEl, MapException):
                            raise newEl.exc_info[0](newEl.exc_info[1]).with_traceback(newEl.exc_info[2])
                        yield newEl
            else:
                yield newEl

    def __mp_pool_generator(self, f: Callable[[_K], _V], poolSize: int, bufferSize: int) -> Generator[_V, None, None]:
        p = Pool(poolSize)
        for el in p.imap(f, self, chunksize=bufferSize):
            yield el
        p.close()
        p.join()

    def __mp_fast_pool_generator(self, f: Callable[[_K], _V], poolSize: int, bufferSize: int) -> Generator[_V, None, None]:
        p = Pool(poolSize)
        try:
            for el in p.imap_unordered(f, iter(self), chunksize=bufferSize):
                yield el
        except GeneratorExit:
            p.terminate()
        finally:
            p.close()
            p.join()

    @staticmethod
    def __unique_generator(itr, f):
        st = set()
        for el in itr:
            m_el = f(el)
            if m_el not in st:
                st.add(m_el)
                yield el

    def map(self, f: Callable[[_K], _V]) -> 'stream[_V]':
        return stream(partial(map, f, self))

    def starmap(self, f: Callable[[_K], _V]) -> 'stream[_V]':
        return stream(partial(itertools.starmap, f, self))

    def mpmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(),
              bufferSize: Optional[int] = None) -> 'stream[_V]':
        """
        Parallel ordered map using multiprocessing.Pool.imap
        :param poolSize: number of processes in Pool
        :param bufferSize: passed as chunksize param to imap()
        """
        # Validations
        if not isinstance(poolSize, int) or poolSize <= 0 or poolSize > 2 ** 12:
            raise ValueError("poolSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))
        elif poolSize == 1:
            return self.map(f)
        if bufferSize is None:
            bufferSize = poolSize * 2
        if not isinstance(bufferSize, int) or bufferSize <= 0 or bufferSize > 2 ** 12:
            raise ValueError("bufferSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))

        return stream(self.__mp_pool_generator(f, poolSize, bufferSize))

    def mpfastmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(), bufferSize: Optional[int] = None) -> 'stream[_V]':
        """
        Parallel unordered map using multiprocessing.Pool.imap_unordered
        :param poolSize: number of processes in Pool
        :param bufferSize: passed as chunksize param to imap()
        """
        # Validations
        if not isinstance(poolSize, int) or poolSize <= 0 or poolSize > 2 ** 12:
            raise ValueError("poolSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))
        elif poolSize == 1:
            return self.map(f)
        if bufferSize is None:
            bufferSize = poolSize * 2
        if not isinstance(bufferSize, int) or bufferSize <= 0 or bufferSize > 2 ** 12:
            raise ValueError("bufferSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))

        return stream(self.__mp_fast_pool_generator(f, poolSize, bufferSize))

    def fastmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(), bufferSize: Optional[int] = None) -> 'stream[_V]':
        """
        Parallel unordered map using multithreaded pool.
        It spawns at most poolSize threads and applies the f function.
        The elements in the result stream appears in the unpredicted order.
        It's most usefull for I/O or CPU intensive consuming functions.
        :param poolSize: number of threads to spawn
        """
        if not isinstance(poolSize, int) or poolSize <= 0 or poolSize > 2 ** 12:
            raise ValueError("poolSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))
        elif poolSize == 1:
            return self.map(f)
        if bufferSize is None:
            bufferSize = poolSize
        if not isinstance(bufferSize, int) or bufferSize <= 0 or bufferSize > 2 ** 12:
            raise ValueError("bufferSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))

        return stream(ItrFromFunc(lambda: self.__fastmap_generator(f, poolSize, bufferSize)))

    # ToDo - add fastFlatMap to Python 2.x version
    def fastFlatMap(self, predicate: Callable[[_K], Iterable[_V]] = _IDENTITY_FUNC, poolSize: int = cpu_count(),
                    bufferSize: Optional[int] = None) -> 'stream[_V]':
        if not isinstance(poolSize, int) or poolSize <= 0 or poolSize > 2 ** 12:
            raise ValueError("poolSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))
        elif poolSize == 1:
            return self.flatMap(predicate)
        if bufferSize is None:
            bufferSize = poolSize
        if not isinstance(bufferSize, int) or bufferSize <= 0 or bufferSize > 2 ** 12:
            raise ValueError("bufferSize should be an integer between 1 and 2^12. Received: %s" % str(poolSize))
        return stream(lambda: self.__fastFlatMap_generator(predicate, poolSize, bufferSize))

    def enumerate(self) -> 'stream[Tuple[int,_K]]':
        return stream(zip(range(0, sys.maxsize), self))

    def flatMap(self, predicate: Callable[[_K], Iterable[_V]] = _IDENTITY_FUNC) -> 'stream[_V]':
        """
        :param predicate: predicate is a function that will receive elements of self collection and return an iterable
            By default predicate is an identity function
        :return: will return stream of objects of the same type of elements from the stream returned by predicate()
        """
        if id(predicate) == id(_IDENTITY_FUNC):
            return stream(ItrFromFunc(lambda: itertools.chain.from_iterable(self)))
        return stream(ItrFromFunc(lambda: itertools.chain.from_iterable(self.map(predicate))))

    def filter(self, predicate: Optional[Callable[[_K], bool]] = None) -> 'stream[_K]':
        """
        :param predicate: If predicate is None, return the items that are true.
        """
        return stream(ItrFromFunc(lambda: filter(predicate, self)))

    def reversed(self) -> 'stream[_K]':
        try:
            return self.__reversed__()
        except TypeError:
            raise TypeError("Can not reverse stream")
        except AttributeError:
            raise TypeError("Can not reverse stream")

    def exists(self, f: Callable[[_K], bool]) -> bool:
        """
        Tests whether a predicate holds for some of the elements of this sequence.
        """
        for e in self:
            if f(e):
                return True
        return False

    def keyBy(self, keyfunc: Callable[[_K], _V] = _IDENTITY_FUNC) -> 'stream[Tuple[_K, _V]]':
        """
        :param keyfunc: function to map values to keys
        :return: stream of Key, Value pairs
        """
        return self.map(lambda h: (keyfunc(h), h))

    def keystream(self: 'stream[Tuple[_T,_V]]') -> 'stream[_T]':
        """
        Applies only on streams of 2-uples
        :return: stream consisted of first element of tuples
        """
        return self.map(itemgetter(0))

    def values(self: 'stream[Tuple[_T,_V]]') -> 'stream[_V]':
        """
        Applies only on streams of 2-uples
        :return: stream consisted of second element of tuples
        """
        return self.map(itemgetter(1))

    def groupBy(self, keyfunc: Callable[[_K], _T] = _IDENTITY_FUNC) -> 'slist[Tuple[_T, slist[_K]]]':
        """
        groupBy([keyfunc]) -> Make a slist with consecutive keys and groups from the iterable.
        The iterable needs not to be sorted on the same key function, but the keyfunction need to return hashable objects.
        :param keyfunc: [Optional] The key is a function computing a key value for each element.
        :return: (key, sub-iterator) grouped by each value of key(value).
        """
        h = defaultdict(slist)
        for v in self:
            h[keyfunc(v)].append(v)
        return slist(h.items())

    @staticmethod
    def __stream_on_second_el(t: Tuple[_K, Iterable[_T]]) -> 'Tuple[_K, stream[_T]]':
        return t[0], stream(t[1])

    @staticmethod
    def __slist_on_second_el(t: Tuple[_K, Iterable[_T]]) -> 'Tuple[_K, slist[_T]]':
        return t[0], slist(t[1])

    def groupBySorted(self, keyfunc: Optional[Callable[[_K], _T]] = None) -> 'stream[Tuple[_T, stream[_K]]]':
        """
        Make a stream of consecutive keys and groups (as streams) from the self.
        The iterable needs to already be sorted on the same key function.
        :param keyfunc: a function computing a key value for each element. Defaults to an identity function and returns the element unchanged.
        :return: (key, sub-iterator) grouped by each value of key(value).
        """
        return stream(partial(groupby, iterable=self, key=keyfunc)).map(self.__stream_on_second_el)

    def groupBySortedToList(self, keyfunc: Callable[[_K], _T] = _IDENTITY_FUNC) -> 'stream[Tuple[_T, slist[_K]]]':
        """
        Make a stream of consecutive keys and groups (as streams) from the self.
        The iterable needs to already be sorted on the same key function.
        :param keyfunc: a function computing a key value for each element. Defaults to an identity function and returns the element unchanged.
        :return: (key, sub-iterator) grouped by each value of key(value).
        """
        return stream(partial(groupby, iterable=self, key=keyfunc)).map(self.__slist_on_second_el)

    def countByValue(self) -> 'sdict[_K,int]':
        return sdict(collections.Counter(self))

    def distinct(self) -> 'stream[_K]':
        return self.unique()

    @overload
    def reduce(self, f: Callable[[_K, _K], _K], init: Optional[_K] = None) -> _K:
        ...

    @overload
    def reduce(self, f: Callable[[_T, _K], _T], init: _T = None) -> _T:
        ...

    @overload
    def reduce(self, f: Callable[[Union[_K, _T], _K], _T], init: Optional[_T] = None) -> _T:
        ...

    @overload
    def reduce(self, f: Callable[[Union[_K, _T], _K], _T], init: Optional[_K] = None) -> _T:
        ...

    @overload
    def reduce(self, f: Callable[[_T, _K], _T], init: _T = None) -> _T:
        ...

    def reduce(self, f, init=None):
        if init is None:
            return reduce(f, self)
        else:
            return reduce(f, self, init)

    def toSet(self) -> 'sset[_K]':
        return sset(self)

    def toList(self) -> 'slist[_K]':
        return slist(self)

    def sorted(self, key=None, reverse=False):
        return slist(sorted(self, key=key, reverse=reverse))

    def toMap(self: 'stream[Tuple[_T,_V]]') -> 'sdict[_T,_V]':
        return sdict(self)

    def toSumCounter(self: 'stream[Tuple[_T,_V]]') -> 'sdict[_T,_V]':
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

    @overload
    def __getitem__(self, i: slice) -> 'stream[_K]':
        ...

    @overload
    def __getitem__(self, i: int) -> _K:
        ...

    def __getitem__(self, i: Union[slice, int]):
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
                   step: Optional[int] = None) -> 'stream[_K]':
        # ToDo:fix this for cases where self._itr is generator from fastmap(), so have to be closed()
        return stream(lambda: itertools.islice(self, start, stop, step))

    def __add__(self, other) -> 'stream[_K]':
        if not isinstance(other, ItrFromFunc):
            othItr = ItrFromFunc(lambda: other)
        else:
            othItr = other
        if isinstance(self._itr, ItrFromFunc):
            i = self._itr
        else:
            i = ItrFromFunc(lambda: self._itr)
        return stream(CallableGeneratorContainer((i, othItr)))

    def __iadd__(self, other) -> 'stream[_K]':
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

    def join(self, f: Callable[[_K], _V] = None) -> Union[_K, str]:
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

    def batch(self, size: int) -> 'stream[_K]':

        def batch_gen(itr):
            while True:
                batch = slist(itertools.islice(itr, 0, size))
                if not batch:
                    break
                yield batch

        return stream(lambda: stream(batch_gen(iter(self))))

    # ToDo - add this fix to Python 2.7
    def take(self, n: int) -> 'stream[_K]':
        def gen(other_gen: GeneratorType, n):
            count = 0
            while count < n:
                if count < n:
                    try:
                        el = next(other_gen)
                        count += 1
                        yield el
                    except StopIteration:
                        break
            other_gen.close()

        if isinstance(self._itr, GeneratorType):
            return stream(gen(self._itr, n))
        else:
            return self[:n]

    # ToDo: add tests for takeWhile
    def takeWhile(self, predicate: Callable[[_K], bool]) -> 'stream[_K]':
        def gen(other_gen: Union[GeneratorType, Iterable[_K]], pred: Callable[[_K], bool]):
            isGen = True
            if not isinstance(other_gen, GeneratorType):
                isGen = False
                other_gen = iter(other_gen)
            while True:
                try:
                    el = next(other_gen)
                    if pred(el):
                        yield el
                    else:
                        break
                except StopIteration:
                    break
            if isGen: other_gen.close()

        return stream(gen(self, predicate))

    def dropWhile(self, predicate: Callable[[_K], bool]):
        return stream(partial(itertools.dropwhile, predicate, self))

    def next(self) -> _K:
        if self._itr is not None:
            try:
                n = next(self._itr)
                return n
            except TypeError:
                self._itr = iter(self._itr)
                return next(self._itr)
        else:
            self._itr = iter(self)
            self._f = None
            return next(self._itr)

    def head(self, n: int) -> 'stream[_K]':
        "Return a stream over the first n items"
        return stream(itertools.islice(self, n))

    def tail(self, n: int):
        "Return a steam over the last n items"
        return stream(collections.deque(self, maxlen=n))

    def all_equal(self) -> bool:
        "Returns True if all the elements are equal to each other"
        g = groupby(self)
        return next(g, True) and not next(g, False)

    def quantify(self, predicate: Callable[[_K], bool]) -> int:
        "Count how many times the predicate is true"
        return sum(self.map(predicate))

    def pad_with(self, pad: Any) -> 'stream[Union[Any,_K]]':
        """Returns the sequence elements and then returns None indefinitely.

        Useful for emulating the behavior of the built-in map() function.
        """
        return stream(itertools.chain(self, itertools.repeat(pad)))

    def roundrobin(self) -> 'stream':
        """
        roundrobin('ABC', 'D', 'EF') --> A D E B F C
        Recipe credited to https://docs.python.org/3/library/itertools.html#itertools.chain.from_iterable
        """

        def gen(s: 'stream'):
            num_active = s.size()
            nexts = itertools.cycle(iter(it).__next__ for it in s)
            while num_active:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    # Remove the iterator we just exhausted from the cycle.
                    num_active -= 1
                    nexts = itertools.cycle(itertools.islice(nexts, num_active))

        return stream(lambda: gen(self))

    def sum(self) -> numbers.Real:
        return sum(self)

    def min(self, key: Callable[[_K], _V] = _IDENTITY_FUNC) -> _V:
        return min(self, key=key)

    def min_default(self, default: _T, key: Callable[[_K], _V] = _IDENTITY_FUNC) -> Union[_V, _T]:
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

    def max(self, key: Callable[[_K], _V] = _IDENTITY_FUNC) -> _V:
        return max(self, key=key)

    def maxes(self, key: Callable[[_K], _V] = _IDENTITY_FUNC) -> 'slist[_V]':
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

    def mins(self, key: Callable[[_K], _V] = _IDENTITY_FUNC) -> 'slist[_V]':
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

    def zip(self) -> 'stream[_V]':
        return stream(zip(*(self.toList())))

    def unique(self, predicate: Callable[[_K], _V] = _IDENTITY_FUNC):
        """
        The stream items should be hashable and comparable.
        :param predicate: optional, maps the elements to comparable objects
        :return: Unique elements appearing in the same order. Following copies of same elements will be ignored.
        :rtype: stream[U]
        """
        return stream(lambda: _IStream.__unique_generator(self, predicate))

    def tqdm(self, desc: Optional[str] = None,
             total: Optional[int] = None,
             leave: bool = True,
             file: Optional[io.TextIOWrapper] = None,
             ncols: Optional[int] = None,
             mininterval: float = 0.1,
             maxinterval: float = 10.0,
             ascii: Optional[Union[str, bool]] = None,
             unit: str = 'it',
             unit_scale: Optional[Union[bool, int, float]] = False,
             dynamic_ncols: Optional[bool] = False,
             smoothing: Optional[float] = 0.3,
             initial: int = 0,
             position: Optional[int] = None,
             postfix: Optional[dict] = None,
             gui: bool = False,
             **kwargs) -> 'stream[_K]':
        """
        :param desc: Prefix for the progressbar.
        :param total: The number of expected iterations. If unspecified,
            len(iterable) is used if possible. If float("inf") or as a last
            resort, only basic progress statistics are displayed
            (no ETA, no progressbar).
            If `gui` is True and this parameter needs subsequent updating,
            specify an initial arbitrary large positive integer,
            e.g. int(9e9).
        :param leave: If [default: True], keeps all traces of the progressbar
            upon termination of iteration.
        :param file: Specifies where to output the progress messages
            (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
            methods.  For encoding, see `write_bytes`.
        :param ncols: The width of the entire output message. If specified,
            dynamically resizes the progressbar to stay within this bound.
            If unspecified, attempts to use environment width. The
            fallback is a meter width of 10 and no limit for the counter and
            statistics. If 0, will not print any meter (only stats).
        :param mininterval: Minimum progress display update interval [default: 0.1] seconds.
        :param maxinterval: Maximum progress display update interval [default: 10] seconds.
            Automatically adjusts `miniters` to correspond to `mininterval`
            after long display update lag. Only works if `dynamic_miniters`
            or monitor thread is enabled.
        :param ascii: If unspecified or False, use unicode (smooth blocks) to fill
            the meter. The fallback is to use ASCII characters " 123456789#".
        :param unit: String that will be used to define the unit of each iteration
            [default: it].
        :param unit_scale: If 1 or True, the number of iterations will be reduced/scaled
            automatically and a metric prefix following the
            International System of Units standard will be added
            (kilo, mega, etc.) [default: False]. If any other non-zero
            number, will scale `total` and `n`.
        :param dynamic_ncols: If set, constantly alters `ncols` to the environment (allowing
            for window resizes) [default: False].
        :param smoothing: Exponential moving average smoothing factor for speed estimates
            (ignored in GUI mode). Ranges from 0 (average speed) to 1
            (current/instantaneous speed) [default: 0.3].
        :param initial: The initial counter value. Useful when restarting a progress bar [default: 0].
        :param position: Specify the line offset to print this bar (starting from 0)
            Automatic if unspecified.
            Useful to manage multiple bars at once (eg, from threads).
        :param postfix: Specify additional stats to display at the end of the bar.
            Calls `set_postfix(**postfix)` if possible (dict).
        :param gui: WARNING: internal parameter - do not use.
            Use tqdm_gui(...) instead. If set, will attempt to use
            matplotlib animations for a graphical output [default: False].
        :param kwargs: Params to be sent to tqdm()
        :return: self stream
        """
        return stream(tqdm(iterable=self, desc=desc, total=total, leave=leave,
                           file=file, ncols=ncols, mininterval=mininterval, maxinterval=maxinterval,
                           ascii=ascii, unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, smoothing=smoothing,
                           initial=initial, position=position, postfix=postfix,
                           gui=gui, **kwargs))

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

    def dumpPickledToWriter(self, writer: Callable[[bytes], _T]) -> None:
        '''
        :param writer: should be binary output callable stream
        '''
        for el in self:
            writer(stream._picklePack(el))

    @staticmethod
    def _picklePack(el) -> bytes:
        return stream.binaryToChunk(pickle.dumps(el, pickle.HIGHEST_PROTOCOL))

    def exceptIndexes(self, *indexes: List[int]) -> 'stream[_K]':
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
        return stream(lambda: indexIgnorer(indexSet, self))


class stream(_IStream, Iterable[_K]):
    def __init__(self, itr: Optional[Union[Iterator[_K], Callable[[], Iterable[_K]]]] = None):
        self._f = None
        if itr is None:
            self._itr = []
        elif isinstance(itr, (abc.Iterable, abc.Iterator)) or hasattr(itr, '__iter__') or hasattr(itr, '__getitem__'):
            self._itr = itr
        elif callable(itr):
            self._f = itr
            self._itr = None
        else:
            raise TypeError(
                "Argument f to %s should be callable or iterable, but itr.__class__=%s" % (
                    str(self.__class__), str(itr.__class__)))

    def __iter__(self) -> Iterator[_K]:
        return iter(self.__get_itr())

    def __get_itr(self):
        if self._itr is not None:
            return self._itr
        else:
            return self._f()

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

    def __reversed__(self):
        try:
            return stream(reversed(self.__get_itr()))
        except TypeError:
            try:
                def r():
                    try:
                        n = len(self)
                    except TypeError:
                        raise TypeError("Can not reverse stream")
                    for i in range(n, 0, -1):
                        try:
                            yield self[i - 1]
                        except TypeError:
                            raise TypeError("Can not reverse stream")

                return stream(lambda: r())

            except Exception:
                raise TypeError("Can not reverse stream")

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
                                  statHandler: Optional[Callable[[int, int], None]] = None) -> 'stream[_V]':
        '''
        :param statHandler: statistics handler, will be called before every yield with a tuple (n,size)
        '''
        if isinstance(readStream, str):
            readStream = openByExtension(readStream, mode='r', buffering=2 ** 12)
        return stream(stream.__binaryChunksStreamGenerator(readStream, format, statHandler))

    @staticmethod
    def loadFromPickled(file: Union[BinaryIO, str],
                        format: str = "<L", statHandler: Optional[Callable[[int, int], None]] = None) -> 'stream[_V]':
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
        super().__init__()

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
    def __init__(self, iteratorOverBuffers: 'Iterator[slist[_T]]'):
        self.__iteratorOverBuffers = iter(iteratorOverBuffers)
        super(SynchronizedBufferedStream, self).__init__()

    def _getNextBuffer(self) -> 'slist[_T]':
        try:
            return next(self.__iteratorOverBuffers)
        except StopIteration:
            return slist()


class sset(set, MutableSet[_K], _IStream):
    @property
    def _itr(self):
        return ItrFromFunc(lambda: iter(self))

    def __init__(self, *args, **kwrds):
        set.__init__(self, *args, **kwrds)

    def __iter__(self):
        return set.__iter__(self)

    # Below methods enable chaining and lambda using
    def update(self, *args, **kwargs) -> 'sset[_K]':
        # ToDo: Add option to update with iterables, as set.update supports only other set
        set.update(self, *args, **kwargs)
        return self

    def intersection_update(self, *args, **kwargs) -> 'sset[_K]':
        set.intersection_update(self, *args, **kwargs)
        return self

    def difference_update(self, *args, **kwargs) -> 'sset[_K]':
        set.difference_update(self, *args, **kwargs)
        return self

    def symmetric_difference_update(self, *args, **kwargs) -> 'sset[_K]':
        super(sset, self).symmetric_difference_update(*args, **kwargs)
        return self

    def clear(self, *args, **kwargs) -> 'sset[_K]':
        set.clear(self, *args, **kwargs)
        return self

    def remove(self, *args, **kwargs) -> 'sset[_K]':
        super(sset, self).remove(*args, **kwargs)
        return self

    def add(self, *args, **kwargs) -> 'sset[_K]':
        super(sset, self).add(*args, **kwargs)
        return self

    def discard(self, *args, **kwargs) -> 'sset[_K]':
        super(sset, self).discard(*args, **kwargs)
        return self

    def __reversed__(self):
        raise TypeError("'sset' object is not reversible")

    def __or__(self, s: AbstractSet[_V]) -> Set[Union[_K, _V]]:
        return sset(super().__or__(s))

    def union(self, *s: Iterable[_K]) -> Set[_K]:
        return sset(super().union(*s))

    def tqdm(self, desc: Optional[str] = None, total: Optional[int] = None, leave: bool = True,
             file: Optional[io.TextIOWrapper] = None, ncols: Optional[int] = None, mininterval: float = 0.1,
             maxinterval: float = 10.0, ascii: Optional[Union[str, bool]] = None, unit: str = 'it',
             unit_scale: Optional[Union[bool, int, float]] = False, dynamic_ncols: Optional[bool] = False,
             smoothing: Optional[float] = 0.3, initial: int = 0, position: Optional[int] = None,
             postfix: Optional[dict] = None, gui: bool = False, **kwargs) -> 'stream[_K]':
        if total is None:
            total = self.size()
        return super().tqdm(desc, total, leave, file, ncols, mininterval, maxinterval, ascii, unit, unit_scale,
                            dynamic_ncols, smoothing, initial, position, postfix, gui, **kwargs)

    def __and__(self, other):
        return sset(super().__and__(other))

    def intersection(self, *s: Iterable[object]) -> Set[_K]:
        return sset(super().intersection(*s))

    def __sub__(self, s: AbstractSet[object]) -> Set[_K]:
        return sset(super().__sub__(s))

    def __xor__(self, s: AbstractSet[_V]) -> Set[Union[_K, _V]]:
        return sset(super().__xor__(s))

    def difference(self, *s: Iterable[_V]) -> Set[Union[_K, _V]]:
        return sset(super().difference(*s))

    def symmetric_difference(self, s: Iterable[_V]) -> Set[Union[_K, _V]]:
        return sset(super().symmetric_difference(s))


class slist(List[_K], stream):
    @property
    def _itr(self):
        return ItrFromFunc(lambda: iter(self))

    def __init__(self, *args, **kwrds):
        list.__init__(self, *args, **kwrds)

    def __getitem__(self, item) -> 'Union[_K,slist[_K]]':
        if isinstance(item, slice):
            return slist(list.__getitem__(self, item))
        else:
            return list.__getitem__(self, item)

    def extend(self, iterable: Iterable[_K]) -> 'slist[_K]':
        list.extend(self, iterable)
        return self

    def append(self, x) -> 'slist[_K]':
        list.append(self, x)
        return self

    def remove(self, x) -> 'slist[_K]':
        list.remove(self, x)
        return self

    def insert(self, i, x) -> 'slist[_K]':
        list.insert(self, i, x)
        return self

    def exceptIndexes(self, *indexes: List[int]) -> 'stream[_K]':
        """
        Supports negative indexes
        :type indexes: list[int]
        :return: the stream with filtered out elements on <indexes> positions
        :rtype: stream [ T ]
        """

        def indexIgnorer(indexSet: frozenset, _stream: 'stream[_K]'):
            i = 0
            for el in _stream:
                if i not in indexSet:
                    yield el
                i += 1

        sz = self.size()
        indexSet = frozenset(stream(indexes).map(lambda i: i if i >= 0 else i + sz))
        return stream(lambda: indexIgnorer(indexSet, self))

    def __iadd__(self, other) -> 'stream[_K]':
        list.__iadd__(self, other)
        return self

    def __add__(self, x: List[_K]) -> Union[stream[_K], List[_K]]:
        if not isinstance(x, list) and isinstance(x, stream):
            return stream(self) + x
        return slist(super().__add__(x))

    def tqdm(self, desc: Optional[str] = None, total: Optional[int] = None, leave: bool = True,
             file: Optional[io.TextIOWrapper] = None, ncols: Optional[int] = None, mininterval: float = 0.1,
             maxinterval: float = 10.0, ascii: Optional[Union[str, bool]] = None, unit: str = 'it',
             unit_scale: Optional[Union[bool, int, float]] = False, dynamic_ncols: Optional[bool] = False,
             smoothing: Optional[float] = 0.3, initial: int = 0, position: Optional[int] = None,
             postfix: Optional[dict] = None, gui: bool = False, **kwargs) -> 'stream[_K]':
        if total is None:
            total = self.size()
        return super().tqdm(desc, total, leave, file, ncols, mininterval, maxinterval, ascii, unit, unit_scale,
                            dynamic_ncols, smoothing, initial, position, postfix, gui, **kwargs)


class sdict(Dict[_K, _V], dict, _IStream):
    @property
    def _itr(self):
        return ItrFromFunc(lambda: iter(self))
    
    def __init__(self, *args, **kwrds):
        dict.__init__(self, *args, **kwrds)

    def __iter__(self):
        return dict.__iter__(self)

    def keys(self) -> stream[_K]:
        return stream(dict.keys(self))

    def values(self) -> stream[_V]:
        return stream(dict.values(self))
    
    def items(self) -> stream[Tuple[_K, _V]]:
        return stream(dict.items(self))
    
    def update(self, other=None, **kwargs) -> 'sdict[_K,_V]':
        dict.update(self, other, **kwargs)
        return self
    
    def copy(self) -> 'sdict[_K,_V]':
        return sdict(self.items())
    
    def tqdm(self, desc: Optional[str] = None, total: Optional[int] = None, leave: bool = True,
             file: Optional[io.TextIOWrapper] = None, ncols: Optional[int] = None, mininterval: float = 0.1,
             maxinterval: float = 10.0, ascii: Optional[Union[str, bool]] = None, unit: str = 'it',
             unit_scale: Optional[Union[bool, int, float]] = False, dynamic_ncols: Optional[bool] = False,
             smoothing: Optional[float] = 0.3, initial: int = 0, position: Optional[int] = None,
             postfix: Optional[dict] = None, gui: bool = False, **kwargs) -> 'stream[_K]':
        if total is None:
            total = self.size()
        return super().tqdm(desc, total, leave, file, ncols, mininterval, maxinterval, ascii, unit, unit_scale,
                            dynamic_ncols, smoothing, initial, position, postfix, gui, **kwargs)
    
    def toJson(self) -> 'sdict[_K,_V]':
        from pyxtension.Json import Json
        return Json(self)


class defaultstreamdict(sdict):
    @property
    def _itr(self):
        return ItrFromFunc(lambda: iter(self))

    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not callable(default_factory)):
            raise TypeError('first argument must be callable')
        super(self.__class__, self).__init__(*a, **kw)
        if default_factory is None:
            self.__default_factory = lambda: object()
        else:
            self.__default_factory = default_factory

    def __getitem__(self, key: _K) -> _V:
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

    def copy(self) -> Mapping[_K, _V]:
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


def smap(f, itr: Iterable[_K]) -> stream[_K]:
    return stream(itr).map(f)


def sfilter(f, itr: Iterable[_K]) -> stream[_K]:
    return stream(itr).filter(f)


def iter_except(func: Callable[[], _K], exc: Exception, first: Optional[Callable[[], _K]] = None) -> _K:
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like builtins.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Examples:
        iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator
        iter_except(d.popitem, KeyError)                         # non-blocking dict iterator
        iter_except(d.popleft, IndexError)                       # non-blocking deque iterator
        iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue
        iter_except(s.pop, KeyError)                             # non-blocking set iterator

    """
    try:
        if first is not None:
            yield first()  # For database APIs needing an initial cast to db.first()
        while True:
            yield func()
    except exc:
        pass
