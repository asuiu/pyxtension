__author__ = 'andrei.suiu@gmail.com'

from unittest import TestCase, main

from pyxtension import frozendict


class TestFrozendict(TestCase):
    def test_immutable(self):
        fd = frozendict(a=1, b=2)
        with self.assertRaises(TypeError):
            fd["a"] = 2
        with self.assertRaises(TypeError):
            fd.update({1:2})
        with self.assertRaises(TypeError):
            del fd["a"]
        with self.assertRaises(TypeError):
            fd.clear()

    def test_empty(self):
        fd_empty = frozendict({})
        self.assertTrue(fd_empty == frozendict([]) == frozendict({}, **{}))

    def test_setattr(self):
        fd = frozendict(a=1, b=2)
        with self.assertRaises(AttributeError):
            fd._initialized = True

    def test_copy(self):
        fd = frozendict(a=1, b=2)
        fd2 = fd.copy()
        self.assertIs(fd, fd2)

    def test_clone(self):
        fd = frozendict(a=1, b=2)
        fd2 = frozendict(dict(fd))
        self.assertEqual(fd, fd2)
        self.assertEqual(hash(fd), hash(fd2))


if __name__ == '__main__':
    main()
