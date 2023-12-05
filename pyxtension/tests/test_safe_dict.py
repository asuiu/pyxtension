from unittest import TestCase

from streamerate import slist, stream

from pyxtension import SafeDict


class TestSafeDict(TestCase):
    def test_single_thread_operations(self):
        safe_dict = SafeDict()

        safe_dict["key1"] = "value1"
        self.assertEqual(safe_dict["key1"], "value1")

        safe_dict["key2"] = 42
        self.assertEqual(safe_dict["key2"], 42)

        del safe_dict["key1"]
        self.assertNotIn("key1", safe_dict)

    def test_multi_thread_operations(self):
        safe_dict = SafeDict()
        NUM_THREADS = 10

        def modify_dict(key: int):
            safe_dict[key % 5] = key
            return safe_dict[key % 5]

        res = stream(range(100)).mtmap(modify_dict, poolSize=NUM_THREADS, bufferSize=100).toList()

        self.assertEqual(len(res), 100)
        self.assertEqual(len(safe_dict), 5)
        self.assertDictEqual(safe_dict, {0: 95, 1: 96, 2: 97, 3: 98, 4: 99})

    def test_items_return_slist(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        res = safe_dict.items()
        safe_dict["key1"] = "value2"
        safe_dict["key2"] = 42
        self.assertIsInstance(res, slist)
        self.assertEqual(res, slist([("key1", "value1")]))

    def test_keys_return_slist(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        res = safe_dict.keys()
        safe_dict["key1"] = "value2"
        safe_dict["key2"] = 42
        self.assertIsInstance(res, slist)
        self.assertEqual(res, slist(["key1"]))

    def test_values_return_slist(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        res = safe_dict.values()
        safe_dict["key1"] = "value2"
        safe_dict["key2"] = 42
        self.assertIsInstance(res, slist)
        self.assertEqual(res, slist(["value1"]))

    def test_clear(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        safe_dict.clear()
        self.assertEqual(len(safe_dict), 0)

    def test_ior(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        safe_dict |= [("key3", 43)]
        self.assertEqual(len(safe_dict), 3)
        self.assertEqual(safe_dict["key3"], 43)

    def test_popitem(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        self.assertEqual(len(safe_dict), 2)
        self.assertEqual(safe_dict.popitem(), ("key2", 42))
        self.assertEqual(len(safe_dict), 1)
        self.assertEqual(safe_dict.popitem(), ("key1", "value1"))
        self.assertEqual(len(safe_dict), 0)

    def test_len(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        self.assertEqual(len(safe_dict), 2)
        del safe_dict["key1"]
        self.assertEqual(len(safe_dict), 1)

    def test_copy(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        safe_dict_copy = safe_dict.copy()
        self.assertIsInstance(safe_dict_copy, SafeDict)
        self.assertEqual(len(safe_dict_copy), 2)
        self.assertEqual(safe_dict_copy["key1"], "value1")
        self.assertEqual(safe_dict_copy["key2"], 42)
        del safe_dict_copy["key1"]
        self.assertEqual(len(safe_dict_copy), 1)
        self.assertEqual(len(safe_dict), 2)

    def test_get(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        self.assertEqual(safe_dict.get("key1"), "value1")
        self.assertEqual(safe_dict.get("key2"), 42)
        self.assertEqual(safe_dict.get("key3"), None)
        self.assertEqual(safe_dict.get("key3", "value3"), "value3")

    def test_setdefault(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        self.assertEqual(safe_dict.setdefault("key1"), "value1")

    def test_pop(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        self.assertEqual(safe_dict.pop("key1"), "value1")
        self.assertEqual(len(safe_dict), 1)
        self.assertEqual(safe_dict.pop("key3", "value3"), "value3")
        self.assertEqual(len(safe_dict), 1)

    def test_update(self):
        safe_dict = SafeDict()
        safe_dict.update({"key1": "value1", "key2": 42})
        self.assertEqual(len(safe_dict), 2)
        self.assertEqual(safe_dict["key1"], "value1")
        self.assertEqual(safe_dict["key2"], 42)
        safe_dict.update({"key1": "value2", "key3": 43})
        self.assertEqual(len(safe_dict), 3)
        self.assertEqual(safe_dict["key1"], "value2")
        self.assertEqual(safe_dict["key3"], 43)

    def test_eq(self):
        safe_dict = SafeDict()
        safe_dict["key1"] = "value1"
        safe_dict["key2"] = 42
        safe_dict2 = SafeDict()
        safe_dict2["key1"] = "value1"
        safe_dict2["key2"] = 42
        self.assertEqual(safe_dict, safe_dict2)
        safe_dict2["key3"] = 43
        self.assertNotEqual(safe_dict, safe_dict2)

    def test_from_keys(self):
        safe_dict = SafeDict.fromkeys(["key1", "key2"], "value")
        self.assertIsInstance(safe_dict, SafeDict)
        self.assertEqual(len(safe_dict), 2)
        self.assertEqual(safe_dict["key1"], "value")
        self.assertEqual(safe_dict["key2"], "value")
        self.assertEqual(safe_dict.get("key3"), None)
        self.assertEqual(safe_dict.get("key3", "value3"), "value3")
