import copy
import json
import sys
import types
import unittest

from pyxtension.Json import Json, JsonList, FrozenJson
from pyxtension.streams import slist, stream, sdict

__author__ = 'ASU'


class JsonTestCase(unittest.TestCase):
    def setUp(self):
        self.d = Json((("a", 2), (3, 4), ("d", {"d2": 4})))

    def testConstructor(self):
        self.assertEqual(Json('{"a":2,"4":"5"}'), {"a": 2, "4": "5"})

    def testBase(self):
        self.assertEqual(self.d.a, 2)
        self.assertEqual(self.d["a"], 2)
        self.assertEqual(self.d.b.c, {})
        self.assertEqual(self.d.d.d2, 4)
        self.assertIsInstance(self.d.keys(), slist)
        self.assertIsInstance(self.d.iterkeys(), stream)
        self.assertIsInstance(self.d.itervalues(), stream)

    def test_strReturnsSortedMap(self):
        self.assertEqual('{"4":3,"a":"4"}', str(Json({"a": "4", 4: 3})))

    def test_strBasics(self):
        self.assertEqual(json.dumps({"a": "4"}, separators=(',', ':')), str(Json({"a": "4"})))
        self.assertEqual(json.dumps(dict((("a", 2), (3, 4), ("d", {"d2": 4}))), separators=(',', ':'), sort_keys=True),
                         str(Json((("a", 2), (3, 4), ("d", {"d2": 4})))))
        self.assertEqual(json.dumps(dict((("a", 2), (3, 4), ("d", {"d2": 4}))), separators=(',', ':'), sort_keys=True),
                         str(self.d))

    def test_repr_from_dict(self):
        d = {'a': 'a'}
        j = Json(d)
        self.assertEqual(repr(j), repr(d))

    def test_repr_used_setattr(self):
        j = Json()
        j.a = 'a'
        self.assertEqual(repr(j), repr({'a': 'a'}))

    def test_forbiden_attrs(self):
        j = Json()
        with self.assertRaises(AttributeError):
            j.__methods__()

    def testUpdateItems(self):
        d = Json((("a", 2), (3, 4), ("d", {"d2": 4})))
        d.d.d2 = 3
        self.assertEqual(d.d.d2, 3)

    def testSpecialKeys(self):
        d = Json((("__init__", 2), (3, 4), ("d", {"d2": 4})))
        self.assertEqual(d["__init__"], 2)
        self.assertNotEquals(d.__init__, 2)
        self.assertIsInstance(d.__init__, types.MethodType)

    def testIteritems(self):
        b = self.d.iteritems().toList()
        self.assertEqual(self.d.iterkeys().toList(), self.d.toList())
        self.assertEqual(b[2][1].d2, 4)
        self.assertIsInstance(b[2][1], Json)
        self.assertIsInstance(self.d.iteritems(), stream)
        self.assertEqual(self.d.iteritems().toList(), [('a', 2), (3, 4), ('d', {'d2': 4})])
        self.assertEqual(self.d.iteritems()[2][1].d2, 4)
        self.assertIsInstance(self.d.iteritems(), stream)
        self.assertEquals(self.d.iteritems().sorted().toList(), [(3, 4), ('a', 2), ('d', {'d2': 4})])
        self.assertEqual(Json({1: 1, 2: 2, 3: 3}).itervalues().sum(), 6)

    def testJsonList(self):
        jlist = Json({'a': [1, 2, {'b': [{'c': 3}, {'d': 4}]}]})
        self.assertEqual(jlist.a[2], {'b': [{'c': 3}, {'d': 4}]})
        self.assertEqual(jlist.a[2].b[1].d, 4)

    def testJsonSetValues(self):
        self.d.c = "set"
        self.assertEqual(self.d.c, "set")

    def test_toOrigNominal(self):
        j = Json()
        j.a = Json({'b': 'c'})
        j.toString()
        j.toOrig()
        repr(j)
        d = j.toOrig()

        self.assertIsInstance(d, sdict)
        self.assertDictEqual(d, {'a': {'b': 'c'}})

    def test_NoneValueRemainsNone(self):
        j = Json({'a': None})
        self.assertIs(j.a, None)

    def test_ConvertSetToList(self):
        j = Json()
        j.st = set((1, 2))
        d = j.toOrig()
        self.assertIsInstance(d, sdict)
        self.assertDictEqual({'st': set([1, 2])}, d)

    def test_serializeDeserialize(self):
        serialized = '{"command":"put","details":{"cookie":"cookie1","platform":"fb"}}'
        j = Json(serialized)
        self.assertEqual(serialized, j.toString())

TEST_VAL = [1, 2, 3]
TEST_DICT = {'a': {'b': {'c': TEST_VAL}}}
TEST_DICT_STR = str(TEST_DICT)


class TestsFromAddict(unittest.TestCase):
    def test_set_one_level_item(self):
        some_dict = {'a': TEST_VAL}
        prop = Json()
        prop['a'] = TEST_VAL
        self.assertDictEqual(prop, some_dict)

    def test_set_two_level_items(self):
        some_dict = {'a': {'b': TEST_VAL}}
        prop = Json()
        prop['a']['b'] = TEST_VAL
        self.assertDictEqual(prop, some_dict)

    def test_set_three_level_items(self):
        prop = Json()
        prop['a']['b']['c'] = TEST_VAL
        self.assertDictEqual(prop, TEST_DICT)

    def test_set_one_level_property(self):
        prop = Json()
        prop.a = TEST_VAL
        self.assertDictEqual(prop, {'a': TEST_VAL})

    def test_set_two_level_properties(self):
        prop = Json()
        prop.a.b = TEST_VAL
        self.assertDictEqual(prop, {'a': {'b': TEST_VAL}})

    def test_set_three_level_properties(self):
        prop = Json()
        prop.a.b.c = TEST_VAL
        self.assertDictEqual(prop, TEST_DICT)

    def test_init_with_dict(self):
        self.assertDictEqual(TEST_DICT, Json(TEST_DICT))

    def test_init_with_kws(self):
        prop = Json(a=2, b={'a': 2}, c=[{'a': 2}])
        self.assertDictEqual(prop, {'a': 2, 'b': {'a': 2}, 'c': [{'a': 2}]})

    def test_init_with_tuples(self):
        prop = Json((0, 1), (1, 2), (2, 3))
        self.assertDictEqual(prop, {0: 1, 1: 2, 2: 3})

    def test_init_with_list(self):
        prop = Json([(0, 1), (1, 2), (2, 3)])
        self.assertDictEqual(prop, {0: 1, 1: 2, 2: 3})

    def test_init_with_generator(self):
        prop = Json(((i, i + 1) for i in range(3)))
        self.assertDictEqual(prop, {0: 1, 1: 2, 2: 3})

    def test_init_raises(self):
        def init():
            Json(5)

        self.assertRaises(TypeError, init)

    def test_init_with_empty_stuff(self):
        a = Json({})
        b = Json([])
        self.assertDictEqual(a, {})
        self.assertDictEqual(b, {})

    def test_init_with_list_of_dicts(self):
        a = Json({'a': [{'b': 2}]})
        self.assertIsInstance(a.a[0], Json)
        self.assertEqual(a.a[0].b, 2)

    def test_getitem(self):
        prop = Json(TEST_DICT)
        self.assertEqual(prop['a']['b']['c'], TEST_VAL)

    def test_getattr(self):
        prop = Json(TEST_DICT)
        self.assertEqual(prop.a.b.c, TEST_VAL)

    def test_isinstance(self):
        self.assertTrue(isinstance(Json(), dict))

    def test_str(self):
        prop = Json(TEST_DICT)
        self.assertEqual(str(prop), json.dumps(TEST_DICT, separators=(',', ':')))

    def test_delitem(self):
        prop = Json({'a': 2})
        del prop['a']
        self.assertDictEqual(prop, {})

    def test_delitem_nested(self):
        prop = Json(TEST_DICT)
        del prop['a']['b']['c']
        self.assertDictEqual(prop, {'a': {'b': {}}})

    def test_delattr(self):
        prop = Json({'a': 2})
        del prop.a
        self.assertDictEqual(prop, {})

    def test_delattr_nested(self):
        prop = Json(TEST_DICT)
        del prop.a.b.c
        self.assertDictEqual(prop, {'a': {'b': {}}})

    def test_delitem_delattr(self):
        prop = Json(TEST_DICT)
        del prop.a['b']
        self.assertDictEqual(prop, {'a': {}})

    def test_complex_nested_structure(self):
        prop = Json()
        prop.a = [[Json(), 2], [[]], [1, [2, 3], 0]]
        self.assertDictEqual(prop, {'a': [[{}, 2, ], [[]], [1, [2, 3], 0]]})

    def test_tuple_key(self):
        prop = Json()
        prop[(1, 2)] = 2
        self.assertDictEqual(prop, {(1, 2): 2})
        self.assertEqual(prop[(1, 2)], 2)

    def test_set_prop_invalid(self):
        prop = Json()

        def set_keys():
            prop.keys = 2

        def set_items():
            prop.items = 3

        self.assertRaises(AttributeError, set_keys)
        self.assertRaises(AttributeError, set_items)
        self.assertDictEqual(prop, {})

    def test_dir_with_members(self):
        prop = Json({'__members__': 1})
        dir(prop)
        self.assertTrue('__members__' in prop.keys())

    def test_to_dict(self):
        nested = {'a': [{'a': 0}, 2], 'b': {}, 'c': 2}
        prop = Json(nested)
        regular = prop.toOrig()
        self.assertDictEqual(regular, prop)
        self.assertDictEqual(regular, nested)
        self.assertNotIsInstance(regular, Json)
        with self.assertRaises(AttributeError):
            regular.a

        def get_attr_deep():
            return regular['a'][0].a

        self.assertRaises(AttributeError, get_attr_deep)

    def test_to_dict_with_tuple(self):
        nested = {'a': ({'a': 0}, {2: 0})}
        prop = Json(nested)
        regular = prop.toOrig()
        self.assertDictEqual(regular, prop)
        self.assertDictEqual(regular, nested)
        self.assertIsInstance(regular['a'], tuple)
        self.assertNotIsInstance(regular['a'][0], Json)

    def test_update(self):
        old = Json()
        old.child.a = 'old a'
        old.child.b = 'old b'
        old.foo = 'no dict'

        new = Json()
        new.child.b = 'new b'
        new.child.c = 'new c'
        new.foo.now_my_papa_is_a_dict = True

        old.update(new)

        reference = {'foo': {'now_my_papa_is_a_dict': True},
                     'child': {'c': 'new c', 'b': 'new b'}}

        self.assertDictEqual(old, reference)

    def test_update_with_lists(self):
        org = Json()
        org.a = [1, 2, {'a': 'superman'}]
        someother = Json()
        someother.b = [{'b': 123}]
        org.update(someother)

        correct = {'a': [1, 2, {'a': 'superman'}],
                   'b': [{'b': 123}]}

        org.update(someother)
        self.assertDictEqual(org, correct)
        self.assertIsInstance(org.b[0], Json)

    def test_copy(self):
        class MyMutableObject(object):
            def __init__(self):
                self.attribute = None

        foo = MyMutableObject()
        foo.attribute = True

        a = Json()
        a.immutable = 42
        a.mutable = foo

        b = a.copy()

        # immutable object should not change
        b.immutable = 21
        self.assertEqual(a.immutable, 42)

        # mutable object should change
        b.mutable.attribute = False
        self.assertEqual(a.mutable.attribute, b.mutable.attribute)

        # changing child of b should not affect a
        b.child = "new stuff"
        self.assertTrue(isinstance(a.child, Json))

    def test_deepcopy(self):
        class MyMutableObject(object):
            def __init__(self):
                self.attribute = None

        foo = MyMutableObject()
        foo.attribute = True

        a = Json()
        a.child.immutable = 42
        a.child.mutable = foo

        b = copy.deepcopy(a)

        # immutable object should not change
        b.child.immutable = 21
        self.assertEqual(a.child.immutable, 42)

        # mutable object should not change
        b.child.mutable.attribute = False
        self.assertTrue(a.child.mutable.attribute)

        # changing child of b should not affect a
        b.child = "new stuff"
        self.assertTrue(isinstance(a.child, Json))

    def test_equal_objects_nominal(self):
        j1 = Json({'a': 1, 'b': {'c': 'd'}})
        j2 = Json({'a': 1, 'b': {'c': 'd'}})
        j3 = Json({'a': 1, 'b': {'c': 'e'}})
        self.assertEqual(j1, j2)
        self.assertNotEqual(j1, j3)

    def test_JsonList_converts_tuples(self):
        jl = JsonList([(Json(), 2), [[]], [1, (2, 3), 0]])
        self.assertListEqual(jl, [[{}, 2, ], [[]], [1, [2, 3], 0]])

    def test_FrozenJson_nominal(self):
        frozenJson = FrozenJson({'a': 'b'})
        self.assertEqual(frozenJson.a, 'b')
        with self.assertRaises(TypeError):
            frozenJson.a = 'c'
        with self.assertRaises(TypeError):
            frozenJson.b = 'c'

    def test_FrozenJson_hash(self):
        d1 = {'a': 'b'}
        fj1 = FrozenJson(d1)
        d1['b'] = 'c'
        fj2 = FrozenJson(d1)
        del d1['b']
        fj3 = FrozenJson(d1)
        self.assertEqual(fj1, fj3)
        self.assertNotEqual(fj1, fj2)
        self.assertSetEqual({fj1, fj2, fj3}, {fj1, fj2})
        self.assertTrue({fj1, fj2} <= {fj2, fj3})





"""
Allow for these test cases to be run from the command line
"""
if __name__ == '__main__':
    all_tests = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2).run(all_tests)
