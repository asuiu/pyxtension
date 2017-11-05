#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library
"""
Python module that gives you a dictionary whose values are both gettable and settable using both attribute and getitem syntax
"""
import copy
import json

from pyxtension.streams import *

__author__ = 'ASU'
supermethod = lambda self: super(self.__class__, self)


class JsonList(slist):
    @classmethod
    def __decide(cls, j):
        if isinstance(j, dict):
            return Json(j)
        elif isinstance(j, (list, tuple)) and not isinstance(j, JsonList):
            return JsonList(map(Json._toJ, j))
        elif isinstance(j, stream):
            return JsonList(j.map(Json._toJ).toList())
        else:
            return j

    def __init__(self, *args):
        slist.__init__(self, stream(*args).map(lambda j: JsonList.__decide(j)))

    def toOrig(self):
        return [isinstance(t, (Json, JsonList)) and t.toOrig() or t for t in self]

    def toString(self):
        return json.dumps(self)


class Json(sdict):
    FORBIDEN_METHODS = ('__methods__', '__members__')  # Introduced due to PyCharm debugging accessing these methods

    @classmethod
    def __myAttrs(cls):
        return set(dir(cls))

    @staticmethod
    def load(fp, *args, **kwargs):
        """Deserialize ``fp`` (a ``.read()``-supporting file-like object containing
        a JSON document) to a Python object.

        If the contents of ``fp`` is encoded with an ASCII based encoding other
        than utf-8 (e.g. latin-1), then an appropriate ``encoding`` name must
        be specified. Encodings that are not ASCII based (such as UCS-2) are
        not allowed, and should be wrapped with
        ``codecs.getreader(fp)(encoding)``, or simply decoded to a ``unicode``
        object and passed to ``loads()``

        ``object_hook`` is an optional function that will be called with the
        result of any object literal decode (a ``dict``). The return value of
        ``object_hook`` will be used instead of the ``dict``. This feature
        can be used to implement custom decoders (e.g. JSON-RPC class hinting).

        ``object_pairs_hook`` is an optional function that will be called with the
        result of any object literal decoded with an ordered list of pairs.  The
        return value of ``object_pairs_hook`` will be used instead of the ``dict``.
        This feature can be used to implement custom decoders that rely on the
        order that the key and value pairs are decoded (for example,
        collections.OrderedDict will remember the order of insertion). If
        ``object_hook`` is also defined, the ``object_pairs_hook`` takes priority.

        To use a custom ``JSONDecoder`` subclass, specify it with the ``cls``
        kwarg; otherwise ``JSONDecoder`` is used.
        """
        return Json.loads(fp.read(), *args, **kwargs)

    @staticmethod
    def loads(*args, **kwargs):
        """Deserialize ``s`` (a ``str`` or ``unicode`` instance containing a JSON
        document) to a Python object.

        If ``s`` is a ``str`` instance and is encoded with an ASCII based encoding
        other than utf-8 (e.g. latin-1) then an appropriate ``encoding`` name
        must be specified. Encodings that are not ASCII based (such as UCS-2)
        are not allowed and should be decoded to ``unicode`` first.

        ``object_hook`` is an optional function that will be called with the
        result of any object literal decode (a ``dict``). The return value of
        ``object_hook`` will be used instead of the ``dict``. This feature
        can be used to implement custom decoders (e.g. JSON-RPC class hinting).

        ``object_pairs_hook`` is an optional function that will be called with the
        result of any object literal decoded with an ordered list of pairs.  The
        return value of ``object_pairs_hook`` will be used instead of the ``dict``.
        This feature can be used to implement custom decoders that rely on the
        order that the key and value pairs are decoded (for example,
        collections.OrderedDict will remember the order of insertion). If
        ``object_hook`` is also defined, the ``object_pairs_hook`` takes priority.

        ``parse_float``, if specified, will be called with the string
        of every JSON float to be decoded. By default this is equivalent to
        float(num_str). This can be used to use another datatype or parser
        for JSON floats (e.g. decimal.Decimal).

        ``parse_int``, if specified, will be called with the string
        of every JSON int to be decoded. By default this is equivalent to
        int(num_str). This can be used to use another datatype or parser
        for JSON integers (e.g. float).

        ``parse_constant``, if specified, will be called with one of the
        following strings: -Infinity, Infinity, NaN, null, true, false.
        This can be used to raise an exception if invalid JSON numbers
        are encountered.

        To use a custom ``JSONDecoder`` subclass, specify it with the ``cls``
        kwarg; otherwise ``JSONDecoder`` is used.
        """
        d = json.loads(*args, **kwargs)
        if isinstance(d, dict):
            return Json(d)
        elif isinstance(d, list):
            return JsonList(d)
        else:
            raise NotImplementedError("Unknown JSON format: {}".format(d.__class__))

    @staticmethod
    def fromString(s, *args, **kwargs):
        return Json.loads(s, *args, **kwargs)

    __decide = lambda self, j: isinstance(j, dict) and Json(j) or (isinstance(j, list) and slist(j) or j)

    @classmethod
    def _toJ(cls, j):
        if isinstance(j, Json):
            return j
        elif isinstance(j, dict):
            return Json(j)
        elif isinstance(j, JsonList):
            return j
        elif isinstance(j, list):
            return JsonList(j)
        else:
            return j

    def __init__(self, *args, **kwargs):
        if not kwargs and len(args) == 1 and isinstance(args[0], basestring):
            d = json.loads(args[0])
            assert isinstance(d, dict)
            sdict.__init__(self, d)
        elif len(args) >= 2 and isinstance(args[0], tuple):
            sdict.__init__(self, args)
        else:
            sdict.__init__(self, *args, **kwargs)

    def __getitem__(self, name):
        """
        This is called when the Dict is accessed by []. E.g.
        some_instance_of_Dict['a'];
        If the name is in the dict, we return it. Otherwise we set both
        the attr and item to a new instance of Dict.
        """
        if name in self:
            d = sdict.__getitem__(self, name)
            if isinstance(d, dict) and not isinstance(d, Json):
                j = Json(d)
                sdict.__setitem__(self, name, j)
                return j
            elif isinstance(d, list) and not isinstance(d, JsonList):
                j = JsonList(d)
                sdict.__setitem__(self, name, j)
                return j
            elif isinstance(d, set) and not isinstance(d, sset):
                j = sset(d)
                sdict.__setitem__(self, name, j)
                return j
            else:
                return d
        else:
            j = Json()
            sdict.__setitem__(self, name, j)
            return j

    def __getattr__(self, item):
        if item in self.FORBIDEN_METHODS:
            raise AttributeError("Forbidden methods access to %s. Introduced due to PyCharm debugging problem." % str(
                self.FORBIDEN_METHODS))

        return self.__getitem__(item)

    def __setattr__(self, key, value):
        if key not in self.__myAttrs():
            self[key] = value
        else:
            raise AttributeError("'%s' object attribute '%s' is read-only" % (str(self.__class__), key))

    def __iter__(self):
        return super(Json, self).__iter__()

    def iteritems(self):
        return stream(dict.iteritems(self)).map(lambda kv: (kv[0], Json._toJ(kv[1])))

    def iterkeys(self):
        return stream(dict.iterkeys(self))

    def itervalues(self):
        return stream(dict.itervalues(self)).map(Json._toJ)

    def keys(self):
        return slist(dict.keys(self))

    def values(self):
        return self.itervalues().toList()

    def items(self):
        return self.iteritems().toList()

    def __str__(self):
        return json.dumps(self.toOrig(), separators=(',', ':'), encoding='utf-8', default=lambda k: str(k),
                          sort_keys=True)

    def dump(self, *args, **kwargs):
        """Serialize ``obj`` as a JSON formatted stream to ``fp`` (a
        ``.write()``-supporting file-like object).

        If ``skipkeys`` is true then ``dict`` keys that are not basic types
        (``str``, ``unicode``, ``int``, ``long``, ``float``, ``bool``, ``None``)
        will be skipped instead of raising a ``TypeError``.

        If ``ensure_ascii`` is true (the default), all non-ASCII characters in the
        output are escaped with ``\\uXXXX`` sequences, and the result is a ``str``
        instance consisting of ASCII characters only.  If ``ensure_ascii`` is
        ``False``, some chunks written to ``fp`` may be ``unicode`` instances.
        This usually happens because the input contains unicode strings or the
        ``encoding`` parameter is used. Unless ``fp.write()`` explicitly
        understands ``unicode`` (as in ``codecs.getwriter``) this is likely to
        cause an error.

        If ``check_circular`` is false, then the circular reference check
        for container types will be skipped and a circular reference will
        result in an ``OverflowError`` (or worse).

        If ``allow_nan`` is false, then it will be a ``ValueError`` to
        serialize out of range ``float`` values (``nan``, ``inf``, ``-inf``)
        in strict compliance of the JSON specification, instead of using the
        JavaScript equivalents (``NaN``, ``Infinity``, ``-Infinity``).

        If ``indent`` is a non-negative integer, then JSON array elements and
        object members will be pretty-printed with that indent level. An indent
        level of 0 will only insert newlines. ``None`` is the most compact
        representation.  Since the default item separator is ``', '``,  the
        output might include trailing whitespace when ``indent`` is specified.
        You can use ``separators=(',', ': ')`` to avoid this.

        If ``separators`` is an ``(item_separator, dict_separator)`` tuple
        then it will be used instead of the default ``(', ', ': ')`` separators.
        ``(',', ':')`` is the most compact JSON representation.

        ``encoding`` is the character encoding for str instances, default is UTF-8.

        ``default(obj)`` is a function that should return a serializable version
        of obj or raise TypeError. The default simply raises TypeError.

        If *sort_keys* is ``True`` (default: ``False``), then the output of
        dictionaries will be sorted by key.

        To use a custom ``JSONEncoder`` subclass (e.g. one that overrides the
        ``.default()`` method to serialize additional types), specify it with
        the ``cls`` kwarg; otherwise ``JSONEncoder`` is used.
        """
        return json.dump(self.toOrig(), *args, **kwargs)

    def dumps(self, *args, **kwargs):
        """Serialize ``self`` to a JSON formatted ``str``.

        If ``skipkeys`` is false then ``dict`` keys that are not basic types
        (``str``, ``unicode``, ``int``, ``long``, ``float``, ``bool``, ``None``)
        will be skipped instead of raising a ``TypeError``.

        If ``ensure_ascii`` is false, all non-ASCII characters are not escaped, and
        the return value may be a ``unicode`` instance. See ``dump`` for details.

        If ``check_circular`` is false, then the circular reference check
        for container types will be skipped and a circular reference will
        result in an ``OverflowError`` (or worse).

        If ``allow_nan`` is false, then it will be a ``ValueError`` to
        serialize out of range ``float`` values (``nan``, ``inf``, ``-inf``) in
        strict compliance of the JSON specification, instead of using the
        JavaScript equivalents (``NaN``, ``Infinity``, ``-Infinity``).

        If ``indent`` is a non-negative integer, then JSON array elements and
        object members will be pretty-printed with that indent level. An indent
        level of 0 will only insert newlines. ``None`` is the most compact
        representation.  Since the default item separator is ``', '``,  the
        output might include trailing whitespace when ``indent`` is specified.
        You can use ``separators=(',', ': ')`` to avoid this.

        If ``separators`` is an ``(item_separator, dict_separator)`` tuple
        then it will be used instead of the default ``(', ', ': ')`` separators.
        ``(',', ':')`` is the most compact JSON representation.

        ``encoding`` is the character encoding for str instances, default is UTF-8.

        ``default(obj)`` is a function that should return a serializable version
        of obj or raise TypeError. The default simply raises TypeError.

        If *sort_keys* is ``True`` (default: ``False``), then the output of
        dictionaries will be sorted by key.

        To use a custom ``JSONEncoder`` subclass (e.g. one that overrides the
        ``.default()`` method to serialize additional types), specify it with
        the ``cls`` kwarg; otherwise ``JSONEncoder`` is used.

        """
        return json.dumps(self.toOrig(), *args, **kwargs)

    def toString(self):
        """
        :return: deterministic sorted output string, that can be compared
        :rtype: str
        """
        return str(self)

    """To be removed and make Json serializable"""

    def __eq__(self, y):
        return super(Json, self).__eq__(y)

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __reduce_ex__(self, protocol):
        return str(self)

    def copy(self):
        return Json(super(Json, self).copy())

    def __deepcopy__(self, memo):
        return Json(copy.deepcopy(self.toOrig(), memo))

    def __delattr__(self, name):
        if name in self:
            return supermethod(self).__delitem__(name)
        else:
            raise AttributeError("%s instance has no attribute %s" % (str(self.__class__), name))

    def toOrig(self):
        """
        Converts Json to a native dict
        :return: stream dictionary
        :rtype: sdict
        """
        return sdict(
            self.iteritems().
                map(lambda kv: (kv[0], isinstance(kv[1], (Json, JsonList)) and kv[1].toOrig() or kv[1]))
        )


class FrozenJson(Json):
    def __init__(self, *args, **kwargs):
        super(FrozenJson, self).__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        raise TypeError("Can not update a FrozenJson instance by (key,value): ({},{})".format(key, value))

    def __hash__(self):
        return hash(self.toString())
