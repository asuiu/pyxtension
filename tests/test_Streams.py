from mock import MagicMock

try:  # Python 3.x doesn't have ifilter
    from itertools import ifilter
except ImportError:
    ifilter = filter
from io import BytesIO

try:  # Python 3.x doesn't have cPickle module
    import cPickle as pickle
except ImportError:
    import pickle
import unittest
import sys
import time

if sys.version_info[0] >= 3:
    xrange = range

from streams import stream, slist, sset, sdict, ItrFromFunc, defaultstreamdict

__author__ = 'ASU'


class StreamTestCase(unittest.TestCase):
    def setUp(self):
        self.s = lambda: stream((1, 2, 3))

    def testStream(self):
        s = self.s
        self.assertEquals(list(ifilter(lambda i: i % 2 == 0, s())), [2])
        self.assertEquals(list(s().filter(lambda i: i % 2 == 0)), [2])
        self.assertEquals(s().filter(lambda i: i % 2 == 0).toList(), [2])
        self.assertEquals(s()[1], 2)
        self.assertEquals(s()[1:].toList(), [2, 3])
        self.assertEqual(s().take(2).toList(), [1, 2])
        self.assertAlmostEqual(stream((0, 1, 2, 3)).filter(lambda x: x > 0).entropy(), 1.4591479)
        self.assertEquals(stream([(1, 2), (3, 4)]).zip().toList(), [(1, 3), (2, 4)])

    def test_filterFromGeneratorReinstantiatesProperly(self):
        s = stream(ItrFromFunc(lambda: (i for i in xrange(5))))
        s = s.filter(lambda e: e % 2 == 0)
        self.assertEquals(s.toList(), [0, 2, 4])
        self.assertEquals(s.toList(), [0, 2, 4])
        s = stream(xrange(5)).filter(lambda e: e % 2 == 0)
        self.assertEquals(s.toList(), [0, 2, 4])
        self.assertEquals(s.toList(), [0, 2, 4])

    def test_streamExists(self):
        s = stream(xrange(2))
        self.assertEqual(s.exists(lambda e: e == 0), True)
        self.assertEqual(s.exists(lambda e: e == 2), False)

    def testStreamStr(self):
        s = stream(iter((1, 2, 3, 4)))
        str(s)
        repr(s)
        self.assertListEqual(s.toList(), [1, 2, 3, 4])

    def testStreamToJson(self):
        from Json import Json

        j = stream((("a", 2), (3, 4))).toJson()
        self.assertIsInstance(j, Json)
        self.assertEqual(j.a, 2)

    def testStreamList(self):
        l = lambda: slist((1, 2, 3))
        self.assertEqual(l().toList(), [1, 2, 3])
        self.assertEqual(l()[-1], 3)

    def testStreamSet(self):
        s = lambda: sset([1, 2, 3, 2])
        self.assertEqual(s().size(), 3)
        self.assertEqual(s().map(lambda x: x).toList(), [1, 2, 3])
        self.assertEqual(len(s()), 3)

    def test_sdict(self):
        d = sdict({1: 2, 3: 4})
        self.assertListEqual(d.iteritems().map(lambda t: t).toList(), [(1, 2), (3, 4)])

    def testStreamsFromGenerator(self):
        sg = stream(ItrFromFunc(lambda: (i for i in range(4))))
        self.assertEqual(sg.size(), 4)
        self.assertEqual(sg.size(), 4)
        self.assertEqual(sg.filter(lambda x: x > 1).toList(), [2, 3])
        self.assertEqual(sg.filter(lambda x: x > 1).toList(), [2, 3])
        self.assertEqual(sg.map(lambda x: x > 1).toList(), [False, False, True, True])
        self.assertEqual(sg.map(lambda x: x > 1).toList(), [False, False, True, True])
        self.assertEqual(sg.head(), 0)
        self.assertEqual(sg.head(), 0)
        self.assertEqual(sg.map(lambda i: i ** 2).enumerate().toList(), [(0, 0), (1, 1), (2, 4), (3, 9)])
        self.assertEqual(sg.reduce(lambda x, y: x + y, 5), 11)

    def testStreamPickling(self):
        sio = BytesIO()
        expected = slist(slist((i,)) for i in xrange(10))
        expected.dumpToPickle(sio)
        sio = BytesIO(sio.getvalue())

        result = stream.loadFromPickled(sio)
        self.assertEquals(list(expected), list(result))

    def test_flatMap_basics(self):
        l = stream(({1: 2, 3: 4}, {5: 6, 7: 8}))
        self.assertEquals(l.flatMap(dict.itervalues).toSet(), set((2, 4, 6, 8)))
        self.assertEquals(l.flatMap(dict.iterkeys).toSet(), set((1, 3, 5, 7)))
        self.assertEquals(l.flatMap(dict.iteritems).toSet(), set(((1, 2), (5, 6), (3, 4), (7, 8))))

    def test_flatMap_reiteration(self):
        l = stream(ItrFromFunc(lambda: (xrange(i) for i in xrange(5)))).flatMap()
        self.assertEquals(l.toList(), [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
        self.assertEquals(l.toList(),
                          [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])  # second time to assert the regeneration of generator

    def test_flatMap_defaultIdentityFunction(self):
        l = slist(({1: 2, 3: 4}, {5: 6, 7: 8}))
        self.assertEquals(l.flatMap().toSet(), set((1, 3, 5, 7)))

    def test_sset_updateReturnsSelf(self):
        s = sset((1, 2))
        l = s.update((2, 3))
        self.assertEquals(l, set((1, 2, 3)))

    def test_sset_intersection_updateReturnsSelf(self):
        self.assertEquals(sset((1, 2)).update(set((2, 3))), set((1, 2, 3)))

    def test_reduceUsesInitProperly(self):
        self.assertEquals(slist([sset((1, 2)), sset((3, 4))]).reduce(lambda x, y: x.update(y)), set((1, 2, 3, 4)))
        self.assertEquals(slist([sset((1, 2)), sset((3, 4))]).reduce(lambda x, y: x.update(y), sset()),
                          set((1, 2, 3, 4)))

    def test_ssetChaining(self):
        s = sset().add(0).clear().add(1).add(2).remove(2).discard(3).update(set((3, 4, 5))) \
            .intersection_update(set((1, 3, 4))).difference_update(set((4,))).symmetric_difference_update(set((3, 4)))
        self.assertEquals(s, set((1, 4)))

    def test_maxes(self):
        self.assertEquals(stream(['a', 'abc', 'abcd', 'defg', 'cde']).maxes(lambda s: len(s)), ['abcd', 'defg'])

    def test_mins(self):
        self.assertEquals(stream(['abc', 'a', 'abcd', 'defg', 'cde']).mins(lambda s: len(s)), ['a'])

    def test_defaultstreamdictBasics(self):
        dd = defaultstreamdict(slist)
        dd[1].append(2)
        self.assertEquals(dd, {1: [2]})

    def test_defaultstreamdictSerialization(self):
        dd = defaultstreamdict(slist)
        dd[1].append(2)
        s = pickle.dumps(dd)
        newDd = pickle.loads(s)
        self.assertEquals(newDd, dd)
        self.assertIsInstance(newDd[1], slist)

    def test_stream_add(self):
        s1 = stream([1, 2])
        s2 = stream([3, 4])
        s3 = s1 + s2
        ll = s3.toList()
        self.assertEquals(s3.toList(), [1, 2, 3, 4])
        self.assertEquals(s3.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug
        s1 = s1 + s2
        self.assertEquals(s1.toList(), [1, 2, 3, 4])
        self.assertEquals(s1.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug

    def test_stream_iadd(self):
        s1 = stream([1, 2])
        s1 += [3, 4]
        self.assertEquals(s1.toList(), [1, 2, 3, 4])
        self.assertEquals(s1.toList(), [1, 2, 3, 4])  # second time to exclude one time iterator bug
        self.assertEquals(s1.toList(), [1, 2, 3, 4])

    def test_stream_getitem(self):
        s = stream(i for i in xrange(1))
        self.assertEqual(s[0], 0)

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

    def test_fastmap_nominal(self):
        s = stream(xrange(100))
        res = s.fastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set(i * i for i in xrange(100))
        self.assertSetEqual(res, expected)

    def test_fastmap_one_el(self):
        s = stream([1, ])
        res = s.fastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set((1,))
        self.assertSetEqual(res, expected)

    def test_fastmap_no_el(self):
        s = stream([])
        res = s.fastmap(lambda x: x * x, poolSize=4).toSet()
        expected = set()
        self.assertSetEqual(res, expected)

    def test_fastmap_None_el(self):
        s = stream([None])
        res = s.fastmap(lambda x: x, poolSize=4).toSet()
        expected = set([None])
        self.assertSetEqual(res, expected)

    def test_fastmap_raises_exception(self):
        s = stream([None])
        with self.assertRaises(TypeError):
            res = s.fastmap(lambda x: x * x, poolSize=4).toSet()

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
        s = stream(ItrFromFunc(lambda: xrange(4)))
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

    def test_reversedNominal(self):
        s = slist([1, 2, 3])
        self.assertListEqual(s.reversed().toList(), [3, 2, 1])

    def test_reversedException(self):
        s = stream(xrange(1, 2, 3))
        with self.assertRaises(TypeError):
            s.reversed()

"""
Allow for these test cases to be run from the command line
"""
if __name__ == '__main__':
    all_tests = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2).run(all_tests)
