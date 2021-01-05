import io
import pickle
import random
import sys
import time
import traceback
import unittest
from functools import partial
from io import BytesIO
from unittest.mock import MagicMock

from pydantic import validate_arguments, ValidationError

from pyxtension.Json import Json, JsonList
from pyxtension.streams import defaultstreamdict, sdict, slist, sset, stream, TqdmMapper

ifilter = filter
xrange = range

__author__ = 'ASU'


def PICKABLE_DUMB_FUNCTION(x):
    return x


def PICKABLE_SLEEP_FUNC(el):
    time.sleep(0.2)
    return el * el

def PICKABLE_SLEEP_EXACT(t:float):
    time.sleep(t)
    return t


def _rnd_sleep(i):
    time.sleep(i % 10 / 1000)
    return i * i


class SomeCustomException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):  # pragma: no cover
        return 'APIError(code=%s)' % (self.message)


def PICKABLE_FUNCTION_RAISES(x):
    raise SomeCustomException("")


class SlistTestCase(unittest.TestCase):
    def test_slist_str_nominal(self):
        l = [1, 2, 3]
        s = slist(l)
        s1 = str(s)
        self.assertEqual(str(s), str(l))

    def test_slist_repr_nominal(self):
        l = [1, 2, 3]
        s = slist(l)
        self.assertEqual(repr(s), repr(l))

    def test_slist_add_list(self):
        l1 = slist([1, 2])
        l2 = slist([3, 4])
        l3 = (l1 + l2)
        self.assertIsInstance(l3, stream)
        self.assertIsInstance(l3, slist)
        self.assertListEqual(l3.toList(), [1, 2, 3, 4])

    def test_slist_add_stream(self):
        l1 = slist([1, 2])
        l2 = stream([3, 4])
        l3 = (l1 + l2)
        self.assertIsInstance(l3, stream)
        self.assertNotIsInstance(l3, slist)
        self.assertListEqual(l3.toList(), [1, 2, 3, 4])

    def test_slist_iadd(self):
        l1 = slist([1, 2])
        l2 = slist([3, 4])
        l1 += l2
        self.assertIsInstance(l1, slist)
        self.assertListEqual(l1.toList(), [1, 2, 3, 4])

    def testStreamList(self):
        l = lambda: slist((1, 2, 3))
        self.assertEqual(l().toList(), [1, 2, 3])
        self.assertEqual(l()[-1], 3)

    def test_reversedNominal(self):
        s = stream([1, 2, 3])
        self.assertListEqual(s.reversed().toList(), [3, 2, 1])


class SdictTestCase(unittest.TestCase):
    def testSdictToJson(self):
        j = stream((("a", 2), (3, 4))).toMap().toJson()
        self.assertIsInstance(j, Json)
        self.assertEqual(j.a, 2)
        self.assertDictEqual(j, {'a': 2, 3: 4})

    def test_sdict(self):
        d = sdict({1: 2, 3: 4})
        self.assertListEqual(d.items().map(lambda t: t).toList(), [(1, 2), (3, 4)])

    def test_sdict_copy(self):
        d = sdict({1: 2, 3: 4})
        copy = d.copy()
        self.assertIsInstance(copy, sdict)
        self.assertSetEqual(set(d.items()), set(copy.items()))
        d[1] = 3
        self.assertEqual(copy[1], 2)

    def test_reversed_raises(self):
        s = sdict({1: 1, 2: 2})
        with self.assertRaises(TypeError):
            s.reversed().toList()


class SsetTestCase(unittest.TestCase):
    def testStreamSet(self):
        s = lambda: sset([1, 2, 3, 2])
        self.assertEqual(s().size(), 3)
        self.assertEqual(s().map(lambda x: x).toList(), [1, 2, 3])
        self.assertEqual(len(s()), 3)

    def test_sset_updateReturnsSelf(self):
        s = sset((1, 2))
        l = s.update((2, 3))
        self.assertEqual(l, set((1, 2, 3)))

    def test_sset_intersection_updateReturnsSelf(self):
        self.assertEqual(sset((1, 2)).update(set((2, 3))), set((1, 2, 3)))

    def test_ssetChaining(self):
        s = sset().add(0).clear().add(1).add(2).remove(2).discard(3).update(set((3, 4, 5))) \
            .intersection_update(set((1, 3, 4))).difference_update(set((4,))).symmetric_difference_update(set((3, 4)))
        self.assertEqual(s, set((1, 4)))

    def test_reversed_raises(self):
        s = sset(iter(range(1, 4)))
        with self.assertRaises(TypeError):
            s.reversed().toList()

    def test_disjunction(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s3 = s1 | s2
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {1, 2, 3, 4})

    def test_conjunction(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s3 = s1 & s2
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {1, 2})

    def test_sub(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s3 = s1 - s2
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {3})

    def test_xor(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s3 = s1 ^ s2
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {3, 4})

    def test_union(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s2_2 = sset({1, 2, 5})
        s3 = s1.union(s2, s2_2)
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {1, 2, 3, 4, 5})

    def test_intersection(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s2_2 = sset({1, 2, 5})
        s3 = s1.intersection(s2, s2_2)
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {1, 2})

    def test_difference(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s2_2 = sset({1, 2, 5})
        s3 = s1.difference(s2, s2_2)
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {3, })

    def test_symmetric_difference(self):
        s1 = sset({1, 2, 3})
        s2 = sset({1, 2, 4})
        s3 = s1.symmetric_difference(s2)
        self.assertIsInstance(s3, sset)
        self.assertSetEqual(s3, {3, 4})

class StreamTestCase(unittest.TestCase):
    def setUp(self):
        self.s = lambda: stream((1, 2, 3))

    def test_fastFlatMap_reiteration(self):
        l = stream(lambda: (xrange(i) for i in xrange(5))).fastFlatMap()
        self.assertListEqual(sorted(l.toList()), sorted([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))
        self.assertEqual(sorted(l.toList()),
                         sorted([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))  # second time to assert the regeneration of generator

    def test_fastmap_reiteration(self):
        l = stream(lambda: (xrange(i) for i in xrange(5))).fastmap(len)
        self.assertEqual(l.toList(), [0, 1, 2, 3, 4])
        self.assertEqual(l.toList(), [0, 1, 2, 3, 4])  # second time to assert the regeneration of generator

    def testStream(self):
        s = self.s
        self.assertEqual(list(ifilter(lambda i: i % 2 == 0, s())), [2])
        self.assertEqual(list(s().filter(lambda i: i % 2 == 0)), [2])
        self.assertEqual(s().filter(lambda i: i % 2 == 0).toList(), [2])
        self.assertEqual(s()[1], 2)
        self.assertEqual(s()[1:].toList(), [2, 3])
        self.assertEqual(s().take(2).toList(), [1, 2])
        self.assertAlmostEqual(stream((0, 1, 2, 3)).filter(lambda x: x > 0).entropy(), 1.4591479)
        self.assertEqual(stream([(1, 2), (3, 4)]).zip().toList(), [(1, 3), (2, 4)])

    def test_filterFromGeneratorReinstantiatesProperly(self):
        s = stream(lambda: (i for i in xrange(5)))
        s = s.filter(lambda e: e % 2 == 0)
        self.assertEqual(s.toList(), [0, 2, 4])
        self.assertEqual(s.toList(), [0, 2, 4])
        s = stream(xrange(5)).filter(lambda e: e % 2 == 0)
        self.assertEqual(s.toList(), [0, 2, 4])
        self.assertEqual(s.toList(), [0, 2, 4])

    def test_streamExists(self):
        s = stream([0, 1])
        self.assertEqual(s.exists(lambda e: e == 0), True)
        self.assertEqual(s.exists(lambda e: e == 2), False)

    def test_stream_str_doesntChangeStream(self):
        s = stream(iter((1, 2, 3, 4)))
        str(s)
        self.assertListEqual(s.toList(), [1, 2, 3, 4])

    def test_stream_repr_doesntChangeStream(self):
        s = stream(iter((1, 2, 3, 4)))
        repr(s)
        self.assertListEqual(s.toList(), [1, 2, 3, 4])

    def testStreamToJson(self):
        j = stream((("a", 2), (3, 4))).toJson()
        self.assertIsInstance(j, JsonList)
        self.assertListEqual(j, [["a", 2], [3, 4]])

    def testStreamsFromGenerator(self):
        sg = stream(lambda: (i for i in range(4)))
        self.assertEqual(sg.size(), 4)
        self.assertEqual(sg.size(), 4)
        self.assertEqual(sg.filter(lambda x: x > 1).toList(), [2, 3])
        self.assertEqual(sg.filter(lambda x: x > 1).toList(), [2, 3])
        self.assertEqual(sg.map(lambda x: x > 1).toList(), [False, False, True, True])
        self.assertEqual(sg.map(lambda x: x > 1).toList(), [False, False, True, True])
        self.assertEqual(sg.map(lambda i: i ** 2).enumerate().toList(), [(0, 0), (1, 1), (2, 4), (3, 9)])
        self.assertEqual(sg.reduce(lambda x, y: x + y, 5), 11)
        self.assertListEqual(list(sg.batch(2)), [[0, 1], [2, 3]])
        self.assertListEqual(list(sg.batch(2)), [[0, 1], [2, 3]])

    def test_next_from_gen(self):
        # Next consumes from stream
        sg = stream(lambda: (i for i in range(4)))
        self.assertEqual(sg.next(), 0)
        self.assertEqual(sg.next(), 1)
        self.assertListEqual(list(sg), [2, 3])

    def test_next_from_list(self):
        # Next consumes from stream
        sg = stream([i for i in range(4)])
        self.assertEqual(sg.next(), 0)
        self.assertEqual(sg.next(), 1)
        self.assertListEqual(list(sg), [2, 3])

    def testStreamPickling(self):
        sio = BytesIO()
        expected = slist(slist((i,)) for i in xrange(10))
        expected.dumpToPickle(sio)
        sio = BytesIO(sio.getvalue())

        result = stream.loadFromPickled(sio)
        self.assertListEqual(expected, list(result))

    def test_StreamFileReading(self):
        sio = BytesIO()
        expected = slist(slist((i,)) for i in xrange(10))
        expected.dumpToPickle(sio)
        sio = BytesIO(sio.getvalue())

        result = stream.loadFromPickled(sio)
        self.assertEqual(list(expected), list(result))

    def test_starmap(self):
        s = stream([(2, 5), (3, 2), (10, 3)]).starmap(pow)
        self.assertListEqual(s.toList(), [32, 9, 1000])
        self.assertListEqual(s.toList(), [32, 9, 1000])

    def test_flatMap_nominal(self):
        s = stream([[1, 2], [3, 4], [4, 5]])
        self.assertListEqual(s.flatMap().toList(), [1, 2, 3, 4, 4, 5])

    def test_flatMap_withPredicate(self):
        s = stream(({1: 2, 3: 4}, {5: 6, 7: 8}))
        self.assertEqual(s.flatMap(dict.items).toSet(), set(((1, 2), (5, 6), (3, 4), (7, 8))))

    def test_flatMap_reiteration(self):
        l = stream(lambda: (xrange(i) for i in xrange(5))).flatMap()
        self.assertEqual(l.toList(), [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
        self.assertEqual(l.toList(),
                         [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])  # second time to assert the regeneration of generator

    def test_flatMap_defaultIdentityFunction(self):
        l = slist(({1: 2, 3: 4}, {5: 6, 7: 8}))
        self.assertEqual(l.flatMap().toSet(), set((1, 3, 5, 7)))

    def test_reduceUsesInitProperly(self):
        self.assertEqual(slist([sset((1, 2)), sset((3, 4))]).reduce(lambda x, y: x.update(y)), set((1, 2, 3, 4)))
        self.assertEqual(slist([sset((1, 2)), sset((3, 4))]).reduce(lambda x, y: x.update(y), sset()),
                         set((1, 2, 3, 4)))

    def test_transform_nominal(self):
        s = stream(range(4))

        def f(itr):
            for i in itr:
                for j in range(i):
                    yield i

        self.assertListEqual(s.transform(f).toList(), [1, 2, 2, 3, 3, 3])

    def test_shuffle_nominal(self):
        l = list(range(100))
        s = stream(l).shuffle()
        self.assertNotEqual(l, list(s))
        self.assertSetEqual(set(l), s.toSet())

    def test_maxes(self):
        self.assertEqual(stream(['a', 'abc', 'abcd', 'defg', 'cde']).maxes(lambda s: len(s)), ['abcd', 'defg'])

    def test_mins(self):
        self.assertEqual(stream(['abc', 'a', 'abcd', 'defg', 'cde']).mins(lambda s: len(s)), ['a'])

    def test_min_nominal(self):
        self.assertEqual(stream([2, 1]).min(), 1)
        self.assertEqual(stream(['abc', 'a']).min(key=len), 'a')

    def test_min_raises_on_empty_sequence(self):
        with self.assertRaises(ValueError):
            stream().min()

    def test_min_default_nominal(self):
        self.assertEqual(stream([2, 1]).min_default('default'), 1)
        self.assertEqual(stream(['abc', 'a']).min_default('default', key=len), 'a')
        self.assertEqual(stream().min_default('default'), 'default')

    def test_defaultstreamdictBasics(self):
        dd = defaultstreamdict(slist)
        dd[1].append(2)
        self.assertEqual(dd, {1: [2]})

    def test_defaultstreamdictSerialization(self):
        dd = defaultstreamdict(slist)
        dd[1].append(2)
        s = pickle.dumps(dd)
        newDd = pickle.loads(s)
        self.assertEqual(newDd, dd)
        self.assertIsInstance(newDd[1], slist)

    def test_stream_add(self):
        s1 = stream([1, 2])
        s2 = stream([3, 4])
        s3 = s1 + s2
        ll = s3.toList()
        self.assertEqual(s3.toList(), [1, 2, 3, 4])
        self.assertEqual(s3.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug
        s1 = s1 + s2
        self.assertEqual(s1.toList(), [1, 2, 3, 4])
        self.assertEqual(s1.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug

    def test_stream_add_nonstream(self):
        s1 = stream([1, 2])
        s2 = range(3, 5)
        s3 = s1 + s2
        ll = s3.toList()
        self.assertEqual(s3.toList(), [1, 2, 3, 4])
        self.assertEqual(s3.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug
        s1 = s1 + s2
        self.assertEqual(s1.toList(), [1, 2, 3, 4])
        self.assertEqual(s1.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug

    def test_stream_add_with_function_and_generator(self):
        s1 = stream(lambda: range(1, 3))
        s2 = stream(range(3, 5))
        s3 = s1 + s2
        ll = s3.toList()
        self.assertEqual(s3.toList(), [1, 2, 3, 4])
        self.assertEqual(s3.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug
        s1 = s1 + s2
        self.assertEqual(s1.toList(), [1, 2, 3, 4])
        self.assertEqual(s1.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug

    def test_stream_iadd(self):
        s1 = stream([1, 2])
        s1 += [3, 4]
        s1 += stream(xrange(5, 6))  # use xrange to cover the iterator case
        s1 += stream(lambda: (i for i in xrange(6, 7)))  # to cover the lambda
        expected = list(range(1, 7))
        self.assertEqual(s1.toList(), expected)
        self.assertEqual(s1.toList(), expected)  # second time to exclude one time iterator bug

    def test_stream_iadd_generator(self):
        s1 = stream([1, 2])
        s1 += (i for i in xrange(3, 4))
        s1 += stream(i for i in xrange(4, 5))
        expected = list(range(1, 5))
        self.assertEqual(s1.toList(), expected)

    def test_stream_iadd_func_and_xrange(self):
        s1 = stream(lambda: ((i for i in xrange(1, 3))))
        s1 += stream(xrange(3, 4))
        expected = list(range(1, 4))
        self.assertEqual(s1.toList(), expected)

    def test_stream_getitem(self):
        s = stream(i for i in xrange(1))
        self.assertEqual(s[0], 0)

    def test_stream_getitem_withGroupBy_functional(self):
        s = stream(lambda: (i for i in xrange(10)))
        sg = s.groupBySortedToList(lambda _: _ // 3)
        expected = []
        while True:
            try:
                expected.append(sg[0])
            except StopIteration:
                break
        self.assertListEqual(expected, [(0, [0, 1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8]), (3, [9])])

    def test_stream_getitem_withGroupBySortedToList_generator(self):
        s = stream(i for i in xrange(10))
        sg = s.groupBySortedToList(lambda _: _ // 3)
        expected = []
        while True:
            try:
                expected.append(sg[0])
            except StopIteration:
                break
        self.assertListEqual(expected, [(0, [0, 1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8]), (3, [9])])

    def test_stream_getitem_withGroupBySortedToList_next(self):
        s = stream(i for i in xrange(10))
        sg = s.groupBySortedToList(lambda _: _ // 3)
        expected = []
        while True:
            try:
                expected.append(sg.next())
            except StopIteration:
                break
        self.assertListEqual(expected, [(0, [0, 1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8]), (3, [9])])

    def test_stream_getitem_withGroupBySorted_next(self):
        s = stream(i for i in xrange(10))
        sg = s.groupBySorted(lambda _: _ // 3)
        expected = []
        while True:
            try:
                t = sg[0]
                expected.append((t[0], t[1].toList()))
            except StopIteration:
                break
        self.assertListEqual(expected, [(0, [0, 1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8]), (3, [9])])

    def test_fastFlatMap_nominal(self):
        s = stream([[1, 2], [3, 4], [4, 5]])
        self.assertListEqual(s.fastFlatMap(poolSize=2).toList(), [1, 2, 3, 4, 4, 5])

    def test_fastFlatMap_random_sleep_function(self):
        s = stream([1, 2, 5, 3, 4])

        def random_sleep(i):
            time.sleep(random.randrange(0, 10) * 0.01)
            return range(i)

        self.assertListEqual(s.fastFlatMap(random_sleep, poolSize=2).sorted(),
                             [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])

    def test_fastFlatMap_withPredicate(self):
        s = stream(({1: 2, 3: 4}, {5: 6, 7: 8}))
        self.assertEqual(s.fastFlatMap(dict.items).toSet(), set(((1, 2), (5, 6), (3, 4), (7, 8))))

    def test_fastFlatMap_defaultIdentityFunction(self):
        l = slist(({1: 2, 3: 4}, {5: 6, 7: 8}))
        self.assertEqual(l.fastFlatMap().toSet(), set((1, 3, 5, 7)))

    def test_fastmap_time(self):
        def sleepFunc(el):
            time.sleep(0.3)
            return el * el

        s = stream(xrange(100))
        t1 = time.time()
        res = s.fastmap(sleepFunc, poolSize=50).toSet()
        dt = time.time() - t1
        expected = set(i * i for i in xrange(100))
        self.assertSetEqual(res, expected)
        self.assertLessEqual(dt, 1.5)

    def test_mtmap_time(self):
        def sleepFunc(el):
            time.sleep(0.3)
            return el * el

        s = stream(xrange(100))
        t1 = time.time()
        res = s.mtmap(sleepFunc, poolSize=50).toSet()
        dt = time.time() - t1
        expected = set(i * i for i in xrange(100))
        self.assertSetEqual(res, expected)
        self.assertLessEqual(dt, 1.5)

    def test_fastmap_nominal(self):
        s = stream(xrange(100))
        res = s.fastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set(i * i for i in xrange(100))
        self.assertSetEqual(res, expected)

    def test_mtmap_nominal(self):
        s = stream(xrange(100))
        res = s.mtmap(_rnd_sleep, poolSize=8, bufferSize=20).toList()
        expected = [i * i for i in xrange(100)]
        self.assertListEqual(res, expected)

    def test_fastmap_one_el(self):
        s = stream([1, ])
        res = s.fastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set((1,))
        self.assertSetEqual(res, expected)

    def test_mtmap_one_el(self):
        s = stream([1, ])
        res = s.mtmap(lambda x: x * x, poolSize=4).toList()
        expected = [1]
        self.assertListEqual(res, expected)

    def test_fastmap_no_el(self):
        s = stream([])
        res = s.fastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set()
        self.assertSetEqual(res, expected)

    def test_mtmap_no_el(self):
        s = stream([])
        res = s.mtmap(lambda x: x * x, poolSize=4).toList()
        expected = []
        self.assertListEqual(res, expected)

    def test_fastmap_None_el(self):
        s = stream([None])
        res = s.fastmap(lambda x: x, poolSize=4).toSet()
        expected = set([None])
        self.assertSetEqual(res, expected)

    def test_mtmap_None_el(self):
        s = stream([None])
        res = s.mtmap(lambda x: x, poolSize=4).toList()
        expected = [None]
        self.assertListEqual(res, expected)

    def test_fastmap_take_less(self):
        arr = []

        def m(i):
            arr.append(i)
            return i

        s = stream(range(100)).map(m).fastmap(lambda x: x, poolSize=4, bufferSize=5).take(20)
        res = s.toList()
        self.assertLessEqual(len(arr), 30)
        self.assertEqual(len(res), 20)

    def test_mtmap_take_less(self):
        arr = []

        def m(i):
            arr.append(i)
            return i
        s = stream(range(100)).map(m).mtmap(lambda x: x, poolSize=10, bufferSize=5).take(20)
        res = s.toList()
        self.assertLessEqual(len(arr), 25)
        self.assertEqual(len(res), 20)

    def test_traceback_right_when_fastmap_raises_builtin_exception(self):
        s = stream([None])

        def f(x):
            return x * x

        try:
            s.fastmap(f, poolSize=4).toSet()
        except TypeError as e:
            line = traceback.TracebackException.from_exception(e).stack[5].line
            self.assertEqual(line, 'return x * x')
            return
        self.fail("No expected exceptions has been raised")

    def test_traceback_right_when_fastmap_raises_custom_exception(self):
        class SomeCustomException(Exception):
            def __init__(self, message):
                self.message = message

            def __str__(self):  # pragma: no cover
                return 'APIError(code=%s)' % (self.message)

        s = stream([None])

        def f(x):
            raise SomeCustomException("")

        try:
            s.fastmap(f, poolSize=4).toSet()
        except SomeCustomException as e:
            line = traceback.TracebackException.from_exception(e).stack[5].line
            self.assertEqual(line, 'raise SomeCustomException("")')
            return
        self.fail("No expected exceptions has been raised")

    def test_traceback_right_when_mpfastmap_raises_custom_exception(self):
        s = stream([None])
        try:
            s.mpfastmap(PICKABLE_FUNCTION_RAISES, poolSize=4).toSet()
        except SomeCustomException as e:
            line = traceback.TracebackException.from_exception(e).stack[5].line
            self.assertEqual(line, 'raise SomeCustomException("")')
            return
        self.fail("No expected exceptions has been raised")

    def test_traceback_right_when_mpmap_raises_custom_exception(self):
        s = stream([None])
        try:
            s.mpmap(PICKABLE_FUNCTION_RAISES, poolSize=4).toSet()
        except SomeCustomException as e:
            line = traceback.TracebackException.from_exception(e).stack[5].line
            self.assertEqual(line, 'raise SomeCustomException("")')
            return
        self.fail("No expected exceptions has been raised")


    def test_mtmap_raises_exception(self):
        s = stream([None])
        with self.assertRaises(TypeError):
            res = s.mtmap(lambda x: x * x, poolSize=4).toSet()

    def test_traceback_right_when_mtmap_raises_custom_exception(self):
        s = stream([None])
        try:
            s.mtmap(PICKABLE_FUNCTION_RAISES, poolSize=4).toSet()
        except SomeCustomException as e:
            line = traceback.TracebackException.from_exception(e).stack[6].line
            self.assertEqual(line, 'raise SomeCustomException("")')
            return
        self.fail("No expected exceptions has been raised")

    def test_mpfastmap_time(self):
        N = 20
        s = stream(xrange(N))
        t1 = time.time()
        res = s.mpfastmap(PICKABLE_SLEEP_FUNC, poolSize=N).toSet()
        dt = time.time() - t1
        expected = set(i * i for i in xrange(N))
        self.assertSetEqual(res, expected)
        self.assertLessEqual(dt, 2)

    def test_mpfastmap_time_with_sequential_mapping(self):
        N = 15
        t1 = time.time()
        s = stream([0.2]*N+[10.0]*N)
        s = s.map(PICKABLE_SLEEP_FUNC)
        res = s.mpfastmap(PICKABLE_SLEEP_FUNC, poolSize=N).take(N).toSet()
        dt = time.time() - t1
        expected = {(0.2*0.2)*(0.2*0.2),}
        print(res)
        self.assertSetEqual(res, expected)
        self.assertLessEqual(dt, 4)

    def test_mpfastmap_nominal(self):
        s = stream(xrange(10))
        f = partial(pow, 2)
        res = s.mpfastmap(f, poolSize=4).toSet()
        expected = set(f(i) for i in xrange(10))
        self.assertSetEqual(res, expected)

    def test_mpfastmap_one_el(self):
        s = stream([2, ])
        f = partial(pow, 2)
        res = s.mpfastmap(f, poolSize=4).toSet()
        expected = set((4,))
        self.assertSetEqual(res, expected)

    def test_mpfastmap_no_el(self):
        s = stream([])
        res = s.mpfastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set()
        self.assertSetEqual(res, expected)

    def test_mpfastmap_None_el(self):
        s = stream([None])
        res = s.mpfastmap(PICKABLE_DUMB_FUNCTION, poolSize=4).toSet()
        expected = set([None])
        self.assertSetEqual(res, expected)

    # ToDo: fix the Pool.imap not true lazyness
    @unittest.skip(
        "Pool.imap has bug. Workaround: https://stackoverflow.com/Questions/5318936/Python-Multiprocessing-Pool-Lazy-Iteration")
    def test_mpfastmap_take_less(self):
        arr = []

        def m(i):
            arr.append(i)
            return i

        s = stream(range(100)).map(m).mpfastmap(PICKABLE_DUMB_FUNCTION, poolSize=4, bufferSize=5).take(20)
        res = s.toList()
        self.assertLessEqual(len(arr), 30)
        self.assertEqual(len(res), 20)

    def test_mpfastmap_raises_exception(self):
        s = stream([None])
        f = partial(pow, 2)
        with self.assertRaises(TypeError):
            res = s.mpfastmap(f, poolSize=4).toSet()

    def test_mpmap_time(self):
        N = 10
        s = stream(xrange(N))
        t1 = time.time()
        res = s.mpmap(PICKABLE_SLEEP_FUNC, poolSize=10).toSet()
        dt = time.time() - t1
        expected = set(i * i for i in xrange(N))
        self.assertSetEqual(res, expected)
        self.assertLessEqual(dt, 2)

    def test_mpmap_nominal(self):
        s = stream(xrange(10))
        f = partial(pow, 2)
        res = s.mpmap(f, poolSize=4).toSet()
        expected = set(f(i) for i in xrange(10))
        self.assertSetEqual(res, expected)

    def test_mpmap_one_el(self):
        s = stream([2, ])
        f = partial(pow, 2)
        res = s.mpmap(f, poolSize=4).toSet()
        expected = set((4,))
        self.assertSetEqual(res, expected)

    def test_mpmap_no_el(self):
        s = stream([])
        res = s.mpmap(lambda x: x * x, poolSize=4).toSet()
        expected = set()
        self.assertSetEqual(res, expected)

    def test_mpmap_None_el(self):
        s = stream([None])
        res = s.mpmap(PICKABLE_DUMB_FUNCTION, poolSize=4).toSet()
        expected = set([None])
        self.assertSetEqual(res, expected)

    # ToDo: fix the Pool.imap not true lazyness
    @unittest.skip(
        "Pool.imap has bug. Workaround: https://stackoverflow.com/Questions/5318936/Python-Multiprocessing-Pool-Lazy-Iteration")
    def test_mpmap_take_less(self):
        arr = []

        def m(i):
            arr.append(i)
            return i

        s = stream(range(100)).map(m).mpmap(PICKABLE_DUMB_FUNCTION, poolSize=4, bufferSize=5).take(20)
        res = s.toList()
        self.assertLessEqual(len(arr), 30)
        self.assertEqual(len(res), 20)

    def test_mpmap_raises_exception(self):
        s = stream([None])
        f = partial(pow, 2)
        with self.assertRaises(TypeError):
            res = s.mpmap(f, poolSize=4).toSet()

    def test_unique_nominal(self):
        s = stream([1, 2, 3, 1, 2])
        self.assertListEqual(s.unique().toList(), [1, 2, 3])

    def test_unique_mapping(self):
        s = stream(['abc', 'def', 'a', 'b', 'ab'])
        self.assertListEqual(s.unique(len).toList(), ['abc', 'a', 'ab'])

    def test_unique_empty_stream(self):
        s = stream([])
        self.assertListEqual(s.unique().toList(), [])

    def test_unique_generator_stream(self):
        s = stream(lambda: xrange(4))
        u = s.unique()
        self.assertListEqual(u.toList(), [0, 1, 2, 3])
        self.assertListEqual(u.toList(), [0, 1, 2, 3])

    def test_pstddev_nominal(self):
        s = stream([1, 2, 3, 4])
        self.assertAlmostEqual(s.pstddev(), 1.118033988749895)

    def test_pstddev_exception(self):
        with self.assertRaises(ValueError):
            stream([]).pstddev()

    def test_mean(self):
        self.assertAlmostEqual(stream([1, 2, 3, 4]).mean(), 2.5)

    def test_mean_exception(self):
        with self.assertRaises(ValueError):
            stream([]).mean()

    def test_toSumCounter_nominal(self):
        s = stream([('a', 2), ('a', 4), ('b', 2.1), ('b', 3), ('c', 2)])
        self.assertDictEqual(s.toSumCounter(), {'a': 6, 'b': 5.1, 'c': 2})

    def test_toSumCounter_onEmptyStream(self):
        s = stream([])
        self.assertDictEqual(s.toSumCounter(), {})

    def test_toSumCounter_onStrings(self):
        s = stream([('a', 'b'), ('a', 'c')])
        self.assertDictEqual(s.toSumCounter(), {'a': 'bc'})

    def test_keyBy_nominal(self):
        self.assertListEqual(stream(['a', 'bb', '']).keyBy(len).toList(), [(1, 'a'), (2, 'bb'), (0, '')])

    def test_keys_nominal(self):
        self.assertListEqual(stream([(1, 'a'), (2, 'bb'), (0, '')]).keystream().toList(), [1, 2, 0])

    def test_values_nominal(self):
        self.assertListEqual(stream([(1, 'a'), (2, 'bb'), (0, '')]).values().toList(), ['a', 'bb', ''])

    def test_toMap(self):
        self.assertDictEqual(stream(((1, 2), (3, 4))).toMap(), {1: 2, 3: 4})

    def test_joinWithString(self):
        s = "|"
        strings = ('a', 'b', 'c')
        self.assertEqual(stream(iter(strings)).join(s), s.join(strings))

    def test_joinWithNone(self):
        s = ""
        strings = ('a', 'b', 'c')
        self.assertEqual(stream(iter(strings)).join(), s.join(strings))

    def test_joinWithFunction(self):
        class F:
            def __init__(self):
                self.counter = 0

            def __call__(self, *args, **kwargs):
                self.counter += 1
                return str(self.counter)

        strings = ('a', 'b', 'c')
        f = F()
        self.assertEqual(stream(iter(strings)).join(f), "a1b2c")

    def test_mkString(self):
        streamToTest = stream(('a', 'b', 'c'))
        mock = MagicMock()
        joiner = ","
        streamToTest.join = mock
        streamToTest.mkString(joiner)
        mock.assert_called_once_with(joiner)

    def test_batch_nominal(self):
        s = stream(range(10))
        self.assertListEqual(s.batch(3).toList(), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

    def test_batch_is_empty(self):
        s = stream([])
        self.assertListEqual(s.batch(3).toList(), [])

    def test_takeWhile(self):
        s = stream(partial(iter, [1, 4, 6, 4, 1]))
        self.assertListEqual(s.takeWhile(lambda x: x < 5).toList(), [1, 4])
        self.assertListEqual(s.takeWhile(lambda x: x < 5).toList(), [1, 4])

    def test_dropWhile(self):
        s = stream(partial(iter, [1, 4, 6, 4, 1]))
        self.assertListEqual(s.dropWhile(lambda x: x < 5).toList(), [6, 4, 1])
        self.assertListEqual(s.dropWhile(lambda x: x < 5).toList(), [6, 4, 1])

    def test_tail_nominal(self):
        s = stream(range(20))
        self.assertListEqual(s.tail(5).toList(), [15, 16, 17, 18, 19])

    def test_round_robin_nominal(self):
        s = stream(['ABC', 'D', 'EF'])
        self.assertListEqual(s.roundrobin().toList(), ['A', 'D', 'E', 'B', 'F', 'C'])

    def test_pad_with_nominal(self):
        s = stream(range(2))
        self.assertListEqual(s.pad_with(5).take(5).toList(), [0, 1, 5, 5, 5])

    def test_all_equal_nominal(self):
        equal_s = stream('AAAAA')
        distinct_s = stream('AAABAAA')
        self.assertTrue(equal_s.all_equal())
        self.assertFalse(distinct_s.all_equal())

    def test_reversedNominal(self):
        s = stream([1, 2, 3])
        self.assertListEqual(s.reversed().toList(), [3, 2, 1])

    def test_reverse_iterable(self):
        s = stream(range(1, 4))
        self.assertListEqual(s.reversed().toList(), [3, 2, 1])

    def test_reversedIterator(self):
        s = stream(iter(range(1, 4)))
        rev = s.reversed().toList()
        self.assertListEqual(rev, [3, 2, 1])

    def test_iter_continues(self):
        s = stream(iter(range(1, 4)))
        itr = iter(s)
        next(itr)
        b = [i for i in itr]
        self.assertListEqual(b, [2, 3])

    def test_len(self):
        # On iterable as init
        s = stream(range(1, 4))
        with self.assertRaises(TypeError):
            len(s)
        self.assertEqual(3, s.size())
        # On container as init
        s = stream([1, 2, 3])
        with self.assertRaises(TypeError):
            len(s)

    def test_tqdm_nominal(self):
        N = 4
        FLT = r'(\d+\.\d+|\?)'
        PYPY3_ANOMALY = "\x1b\[A\x1b\[A"
        s = stream(range(N))
        out = io.StringIO()
        self.assertListEqual(list(range(N)), s.tqdm(file=out).toList())
        expected = (rf'\r0it \[00:00, {FLT}it/s\]'
                    rf'({PYPY3_ANOMALY})?'
                    rf'\r{N}it \[00:00, {FLT}it/s\]\n')
        self.assertRegex(out.getvalue(), expected)

    def test_tqdm_total(self):
        N = 4
        s = stream(range(N))
        FLT = r'(\d+\.\d+|\?)'
        TM = r'(00:00|\?)'
        PYPY3_ANOMALY = "\x1b\[A\x1b\[A"
        out = io.StringIO()
        self.assertListEqual(list(range(N)), s.tqdm(total=N, file=out).toList())
        expected = (rf'\r  0%\|          \| 0/{N} \[00:00<\?, \?it/s\]'
                    rf'({PYPY3_ANOMALY})?'
                    rf'\r100%\|##########\| {N}/{N} \[00:00<{TM}, {FLT}it/s\]\n')
        self.assertRegex(out.getvalue(), expected)

    def test_tqdm_containers(self):
        if sys.version_info[1] < 7:  # no support for Py3.6
            return
        N = 4
        FLT = r'(\d+\.\d+|\?)'
        TM = r'(00:00|\?)'
        PYPY3_ANOMALY = "\x1b\[A\x1b\[A"
        s = stream(list(range(N)))
        out = io.StringIO()
        self.assertListEqual(list(range(N)), s.toList().tqdm(file=out).toList())
        expected = (rf'\r  0%\|          \| 0/{N} \[00:00<\?, \?it/s\]'
                    rf'({PYPY3_ANOMALY})?'
                    rf'\r100%\|##########\| {N}/{N} \[00:00<{TM}, {FLT}it/s\]\n')
        self.assertRegex(out.getvalue(), expected)

        out = io.StringIO()
        self.assertListEqual(list(range(N)), s.toSet().tqdm(file=out).toList())
        expected = (rf'\r  0%\|          \| 0/{N} \[00:00<\?, \?it/s\]'
                    rf'({PYPY3_ANOMALY})?'
                    rf'\r100%\|##########\| {N}/{N} \[00:00<{TM}, {FLT}it/s\]\n')
        self.assertRegex(out.getvalue(), expected)

        s = stream(((i, i + 1) for i in range(N))).toMap()
        self.assertListEqual([i for i in range(N)], s.tqdm(file=out).toList())
        expected = (rf'\r  0%\|          \| 0/{N} \[00:00<\?, \?it/s\]'
                    rf'({PYPY3_ANOMALY})?'
                    rf'\r100%\|##########\| {N}/{N} \[00:00<{TM}, {FLT}it/s\]\n')
        self.assertRegex(out.getvalue(), expected)

    def test_TqdmMapper_total(self):
        N = 4
        FLT = r'(\d+\.\d+|\?)'
        TM = r'(00:00|\?)'
        s = stream(range(N))
        out = io.StringIO()
        self.assertListEqual(list(range(N)), s.map(TqdmMapper(total=N, file=out)).toList())
        expected = rf'\r  0%\|          \| 0/{N} \[00:00<\?, \?it/s\]' \
                   rf'(\r100%\|##########\| {N}/{N} \[00:00<{TM}, {FLT}it/s\]\n)?'
        self.assertRegex(out.getvalue(), expected)

    def test_TqdmMapper_nominal(self):
        N = 4
        FLT = r'(\d+\.\d+|\?)'
        s = stream(range(N))
        out = io.StringIO()
        self.assertListEqual(list(range(N)), s.map(TqdmMapper(file=out)).toList())
        expected = rf'\r0it \[00:00, {FLT}it/s\]' \
                   rf'(\r{N}it \[00:00, {FLT}it/s\]\n)?'
        self.assertRegex(out.getvalue(), expected)

    def test_pydantic_stream_validation(self):
        @validate_arguments
        def f(x: stream[int]):
            return x

        l = [1, 2]
        s = stream(l)
        self.assertEqual(f(s).toList(), l)
        with self.assertRaises(ValidationError):
            f(l)

    def test_pydantic_slist_validation(self):
        if sys.version_info[1] < 7:  # no support for Py3.6
            return

        @validate_arguments
        def f(x: slist[int]):
            return x

        s = stream([1, 2])
        self.assertEqual(f(s.toList()), [1, 2])
        self.assertEqual(f({1, 2}), [1, 2], "Expect pydantic to convert automatically set to list")

        with self.assertRaises(ValidationError):
            f(dict())
        with self.assertRaises(ValidationError):
            f(range(3))


"""
Allow for these test cases to be run from the command line
"""
if __name__ == '__main__':
    all_tests = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2).run(all_tests)
