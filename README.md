# pyxtension
[![build Status](https://travis-ci.org/asuiu/pyxtension.svg?branch=master)](https://travis-ci.org/asuiu/pyxtension)
[![Coverage Status](https://coveralls.io/repos/asuiu/pyxtension/badge.svg?branch=master&service=github)](https://coveralls.io/github/asuiu/pyxtension?branch=master)

[pyxtension](https://github.com/asuiu/pyxtension) is a pure Python GNU-licensed library that includes Scala-like streams, Json with attribute access syntax, and other common-use stuff.

## Modules overview
### Json.py
##### Json
A `dict` subclass to represent a Json object. You should be able to use this
absolutely anywhere you can use a `dict`. While this is probably the class you
want to use, there are a few caveats that follow from this being a `dict` under
the hood.

### streams.py
#### stream
`streams` subclasses `collections.Iterable`. It's the same Python iterable, but with more added methods.
Used to create stream processing pipelines, similar to those used in [Scala](http://www.scala-lang.org/) and [MapReduce](https://en.wikipedia.org/wiki/MapReduce) programming model.
Those who used [Apache Spark](http://spark.apache.org/) [RDD](http://spark.apache.org/docs/latest/programming-guide.html#rdd-operations) functions will find this model of processing very easy to use.

### [streams](https://github.com/asuiu/pyxtension/blob/master/streams.py)
**Never again will you have to write code like this**:
```python
> lst = xrange(1,6)
> reduce(lambda x, y: x * y, map(lambda _: _ * _, filter(lambda _: _ % 2 == 0, lst)))
64
```
From now on, you may simply write the following lines:
```python
> the_stream = stream( xrange(1,6) )
> the_stream.\
    filter(lambda _: _ % 2 == 0).\
    map(lambda _: _ * _).\
    reduce(lambda x, y: x * y)
64
```

### [Json](https://github.com/asuiu/pyxtension/blob/master/Json.py)

**Never again will you have to write code like this**:
```python
		body = {
			'query': {
				'filtered': {
					'query': {
						'match': {'description': 'addictive'}
					},
					'filter': {
						'term': {'created_by': 'ASU'}
					}
				}
			}
		}
```

From now on, you may simply write the following three lines:
```python
        body = Json()
		body.query.filtered.query.match.description = 'addictive'
		body.query.filtered.filter.term.created_by = 'ASU'
```

[Json](https://github.com/asuiu/pyxtension/blob/master/Json.py) is a module that provides mapping objects that allow their elements to be accessed both as keys and as attributes:
```python
    > from pyxtension.Json import Json
    > a = Json({'foo': 'bar'})
    > a.foo
    'bar'
    > a['foo']
    'bar'
```

Attribute access makes it easy to create convenient, hierarchical settings objects:
```python
    with open('settings.yaml') as fileobj:
        settings = Json(yaml.safe_load(fileobj))

    cursor = connect(**settings.db.credentials).cursor()

    cursor.execute("SELECT column FROM table;")
```

### Installation
from Github::

    $ git clone https://github.com/asuiu/pyxtension.git

or

    $ git submodule add https://github.com/asuiu/pyxtension.git

### Basic Usage

Json comes with two different classes, `Json`, and `JsonList`.
Json is fairly similar to native `dict` as it extends it an is a mutable mapping that allow creating, accessing, and deleting key-value pairs as attributes.
`JsonList` is similar to native `list` as it extends it and offers a way to transform the `dict` objects from inside also in `Json` instances.

#### Construction
###### Directly from a JSON string
```python
> Json('{"key1": "val1", "lst1": [1,2] }')
{u'key1': u'val1', u'lst1': [1, 2]}
```
###### From `tuple`s:
```python
> Json( ('key1','val1'), ('lst1', [1,2]) )
{'key1': 'val1', 'lst1': [1, 2]}
# keep in mind that you should provide at least two tuples with key-value pairs
```
###### As a built-in `dict`
```python
> Json( [('key1','val1'), ('lst1', [1,2])] )
{'key1': 'val1', 'lst1': [1, 2]}

Json({'key1': 'val1', 'lst1': [1, 2]})
{'key1': 'val1', 'lst1': [1, 2]}
```
#### Convert to a `dict`
```python
> json = Json({'key1': 'val1', 'lst1': [1, 2]})
> json.toOrig()
{'key1': 'val1', 'lst1': [1, 2]}
```

#### Valid Names

Any key can be used as an attribute as long as:

1. The key represents a valid attribute (i.e., it is a string comprised only of
   alphanumeric characters and underscores that doesn't start with a number)
2. The key does not shadow a class attribute (e.g., get).

#### Attributes vs. Keys
There is a minor difference between accessing a value as an attribute vs.
accessing it as a key, is that when a dict is accessed as an attribute, it will
automatically be converted to a `Json` object. This allows you to recursively
access keys::
```python
    > attr = Json({'foo': {'bar': 'baz'}})
    > attr.foo.bar
    'baz'
```
Relatedly, by default, sequence types that aren't `bytes`, `str`, or `unicode`
(e.g., `list`s, `tuple`s) will automatically be converted to `tuple`s, with any
mappings converted to `Json`:
```python
    > attr = Json({'foo': [{'bar': 'baz'}, {'bar': 'qux'}]})
    > for sub_attr in attr.foo:
    >     print(sub_attr.bar)
    'baz'
    'qux'
```
To get this recursive functionality for keys that cannot be used as attributes,
you can replicate the behavior by using dict syntax on `Json` object::
```python
    > json = Json({1: {'two': 3}})
    > json[1].two
    3
```
`JsonList` usage examples:
```python
> json = Json('{"lst":[1,2,3]}')
> type(json.lst)
<class 'pyxtension.Json.JsonList'>

> json = Json('{"1":[1,2]}')
> json["1"][1]
2
```


Assignment as keys will still work::
```python
    > json = Json({'foo': {'bar': 'baz'}})
    > json['foo']['bar'] = 'baz'
    > json.foo
    {'bar': 'baz'}
```

### License
pyxtension is released under a GNU Public license.
The idea for [Json](https://github.com/asuiu/pyxtension/blob/master/Json.py) module was inspired from [addict](https://github.com/mewwts/addict>) and [AttrDict](https://github.com/bcj/AttrDict),
but it has a better performance with lower memory consumption.