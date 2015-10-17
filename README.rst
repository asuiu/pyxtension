========
pyxtension
========
.. image:: https://travis-ci.org/asuiu/pyxtension.svg?branch=master
  :target: https://travis-ci.org/asuiu/pyxtension

pyxtension is a pure Python GNU-licensed library that includes Scala-like streams, Json with attribute access syntax, and other common use stuff.

========
Json
========

Json is a module that provides mapping objects that allow their elements to be accessed both as keys and as attributes::

    > from Json import Json
    > a = Json({'foo': 'bar'})
    > a.foo
    'bar'
    > a['foo']
    'bar'

Attribute access makes it easy to create convenient, hierarchical settings objects::

    with open('settings.yaml') as fileobj:
        settings = Json(yaml.safe_load(fileobj))

    cursor = connect(**settings.db.credentials).cursor()

    cursor.execute("SELECT column FROM table;")

Installation
============
from Github::

    $ git clone https://github.com/asuiu/pyxtension.git
or
	$ git submodule add https://github.com/asuiu/pyxtension.git

Basic Usage
===========
Json comes with two different classes, `Json`, and `JsonList`. 
Json is fairly similar to native dict() as it extends it an is a mutable mapping that allow creating, accessing, and deleting key-value pairs as attributes.
JsonList is similar to native list() as it extends it and offers a way to transform the dict() objects from inside also in Json() instances.

Valid Names
-----------
Any key can be used as an attribute as long as:

#. The key represents a valid attribute (i.e., it is a string comprised only of
   alphanumeric characters and underscores that doesn't start with a number)
#. The key does not shadow a class attribute (e.g., get).

Attributes vs. Keys
-------------------
There is a minor difference between accessing a value as an attribute vs.
accessing it as a key, is that when a dict is accessed as an attribute, it will
automatically be converted to a Json object. This allows you to recursively
access keys::

    > attr = Json({'foo': {'bar': 'baz'}})
    > attr.foo.bar
    'baz'

Relatedly, by default, sequence types that aren't `bytes`, `str`, or `unicode`
(e.g., lists, tuples) will automatically be converted to tuples, with any
mappings converted to Json::

    > attr = Json({'foo': [{'bar': 'baz'}, {'bar': 'qux'}]})
    > for sub_attr in attr.foo:
    >     print(subattr.bar)
    'baz'
    'qux'

To get this recursive functionality for keys that cannot be used as attributes,
you can replicate the behavior by using dict syntax on Json object::

    > attr = AttrDict({1: {'two': 3}})
    > attr[1].two
    3

Classes
-------
AttrDict comes with two different objects, `Json` and `JsonList`.


Json
^^^^
An Attr object that subclasses `dict`. You should be able to use this
absolutely anywhere you can use a `dict`. While this is probably the class you
want to use, there are a few caveats that follow from this being a `dict` under
the hood.

Assignment as keys will still work::

    > attr = AttrDict('foo': {})
    > attr['foo']['bar'] = 'baz'
    > attr.foo
    {'bar': 'baz'}


License
=======
pyxtension is released under a GNU Public license.