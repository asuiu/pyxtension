try:  # Python 3.x doesn't have ifilter
    from itertools import ifilter
except ImportError:
    ifilter = filter
from io import BytesIO
from operator import itemgetter

try: # Python 3.x doesn't have cPickle module
    import cPickle as pickle
except ImportError:
    import pickle
import unittest
import sys

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
        self.assertEquals(str(self.s()), str([1, 2, 3]))

    def testGroupBy(self):
        s = stream(iter((1, 2, 3, 4)))
        l = s.groupBy(lambda k: k % 2).sorted(key=itemgetter(0)).toList()
        self.assertEquals([(0, [2, 4]), (1, [1, 3])], l)

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
        s = lambda: sdict({1: 2, 3: 4})
        self.assertListEqual(s().map(lambda t: t).toList(), [(1, 2), (3, 4)])

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


"""
Allow for these test cases to be run from the command line
"""
if __name__ == '__main__':
    all_tests = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2).run(all_tests)
