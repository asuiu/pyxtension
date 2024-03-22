from unittest import TestCase

from pyxtension.mockutils import UnknownMockArguments, generate_mock_map_func


class TestGenerateMockMapFunc(TestCase):
    def test_generate_mock_map_func_with_only_ordinal(self):
        mf = generate_mock_map_func({("a",): 1, ("b",): "2"})
        self.assertEqual(mf("a"), 1)
        self.assertEqual(mf("b"), "2")

    def test_generate_mock_map_func_with_named_args_and_arg_names(self):
        mf = generate_mock_map_func({("a", "b"): 1, ("a", "c"): "2"}, arg_names=("a1", "a2"))
        # test with ordinal args
        self.assertEqual(mf("a", "b"), 1)
        self.assertEqual(mf("a", "c"), "2")

        # test with named args
        self.assertEqual(mf(a1="a", a2="b"), 1)
        self.assertEqual(mf(a1="a", a2="c"), "2")

    def test_generate_mock_map_func_with_only_named_args(self):
        mf = generate_mock_map_func([({"a1": "a", "a2": "b"}, 1), ({"a1": "a", "a2": "c"}, "2")])
        self.assertEqual(mf(a1="a", a2="b"), 1)
        self.assertEqual(mf(a1="a", a2="c"), "2")

    def test_generate_mock_map_func_throws_exception_on_unknown_args_set(self):
        mf = generate_mock_map_func([({"a1": "a", "a2": "b"}, 1), ({"a1": "a", "a2": "c"}, "2")])

        # Here the argument names are correct, but we don't have a mapping for the arguments
        with self.assertRaises(UnknownMockArguments):
            mf(a1="a", a2="d")

        # Here the argument names are incorrect
        mf = generate_mock_map_func({("a", "b"): 1, ("a", "c"): "2"}, arg_names=("a1", "a2"))
        with self.assertRaises(UnknownMockArguments):
            mf(a1="a", a3="b")
