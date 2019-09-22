# pyxtension
[![build Status](https://travis-ci.org/asuiu/pyxtension.svg?branch=master)](https://travis-ci.org/asuiu/pyxtension)
[![Coverage Status](https://coveralls.io/repos/asuiu/pyxtension/badge.svg?branch=master&service=github)](https://coveralls.io/github/asuiu/pyxtension?branch=master)

[pyxtension](https://github.com/asuiu/pyxtension) is a pure Python GNU-licensed library that includes Scala-like streams, Json with attribute access syntax, and other common-use stuff.

## Install
```
pip install pyxtension
```
or
```
git clone https://github.com/asuiu/pyxtension.git
cd pyxtension
python setup.py install
```
## Modules overview
### Json.py
##### Json
A `dict` subclass to represent a Json object. You should be able to use this
absolutely anywhere you can use a `dict`. While this is probably the class you
want to use, there are a few caveats that follow from this being a `dict` under
the hood.

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
### streams.py
#### stream
`stream` subclasses `collections.Iterable`. It's the same Python iterable, but with more added methods, suitable for multithreading and multiprocess processings.
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

#### A Word Count [Map-Reduce](https://en.wikipedia.org/wiki/MapReduce) naive example using multiprocessing map
```python
corpus = [
    "MapReduce is a programming model and an associated implementation for processing and generating large data sets with a parallel, distributed algorithm on a cluster.",
    "At Google, MapReduce was used to completely regenerate Google's index of the World Wide Web",
    "Conceptually similar approaches have been very well known since 1995 with the Message Passing Interface standard having reduce and scatter operations."]

def reduceMaps(m1, m2):
    for k, v in m2.iteritems():
        m1[k] = m1.get(k, 0) + v
    return m1

word_counts = stream(corpus).\
    mpmap(lambda line: stream(line.lower().split(' ')).countByValue()).\
    reduce(reduceMaps)
```

#### Basic methods
###### **map(f)**
Identic with builtin `map` but returns a stream


###### **mpmap(f, poolSize=16)**
Parallel ordered map using `multiprocessing.Pool.imap()`.

It can replace the `map` when need to split computations to multiple cores, and order of results matters.

It spawns at most `poolSize` processes and applies the `f` function.

The elements in the result stream appears in the same order they appear in the initial iterable.

```
:type f: (T) -> V
:rtype: `stream`
```


###### **mpfastmap(f, poolSize=16)**
Parallel ordered map using `multiprocessing.Pool.imap_unordered()`.

It can replace the `map` when the ordered of results doesn't matter.

It spawns at most `poolSize` processes and applies the `f` function.

The elements in the result stream appears in the unpredicted order.

```
:type f: (T) -> V
:rtype: `stream`
```


###### **fastmap(f, poolSize=16)**
Parallel unordered map using multithreaded pool.
It can replace the `map` when the ordered of results doesn't matter.

It spawns at most `poolSize` threads and applies the `f` function.

The elements in the result stream appears in the unpredicted order.

Because of CPython [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) it's most usefull for I/O or CPU intensive consuming native functions, or on Jython or IronPython interpreters.

:type f: (T) -> V

:rtype: `stream`


###### **flatMap(predicate=_IDENTITY_FUNC)**
:param predicate: is a function that will receive elements of self collection and return an iterable

By default predicate is an identity function

:type predicate: (V)-> collections.Iterable[T]

:return: will return stream of objects of the same type of elements from the stream returned by predicate()

Example:
```python
stream([[1, 2], [3, 4], [4, 5]]).flatMap().toList() == [1, 2, 3, 4, 4, 5]
```


###### **filter(predicate)**
identic with builtin filter, but returns stream


###### **reversed()**
returns reversed stream


###### **exists(predicate)**
Tests whether a predicate holds for some of the elements of this sequence.

:rtype: bool

Example:
```python
stream([1, 2, 3]).exists(0) -> False
stream([1, 2, 3]).exists(1) -> True
```


###### **keyBy(keyfunc = _IDENTITY_FUNC)**
Transforms stream of values to a stream of tuples (key, value)

:param keyfunc: function to map values to keys

:type keyfunc: (V) -> T

:return: stream of Key, Value pairs

:rtype: stream[( T, V )]

Example:
```python
stream([1, 2, 3, 4]).keyBy(lambda _:_ % 2) -> [(1, 1), (0, 2), (1, 3), (0, 4)]
```

###### **groupBy()**
groupBy([keyfunc]) -> Make an iterator that returns consecutive keys and groups from the iterable.

The iterable needs not to be sorted on the same key function, but the keyfunction need to return hasable objects.

:param keyfunc: [Optional] The key is a function computing a key value for each element.

:type keyfunc: (T) -> (V)

:return: (key, sub-iterator) grouped by each value of key(value).

:rtype: stream[ ( V, slist[T] ) ]

Example:
```python
stream([1, 2, 3, 4]).groupBy(lambda _: _ % 2) -> [(0, [2, 4]), (1, [1, 3])]
```

###### **countByValue()**
Returns a collections.Counter of values

Example
```python
stream(['a', 'b', 'a', 'b', 'c', 'd']).countByValue() == {'a': 2, 'b': 2, 'c': 1, 'd': 1}
```

###### **distinct()**
Returns stream of distinct values. Values must be hashable.
```python
stream(['a', 'b', 'a', 'b', 'c', 'd']).distinct() == {'a', 'b', 'c', 'd'}
```


###### **reduce(f, init=None)**
same arguments with builtin reduce() function


###### **toSet()**
returns sset() instance


###### **toList()**
returns slist() instance


###### **toMap()**
returns sdict() instance


###### **sorted(key=None, cmp=None, reverse=False)**
same arguments with builtin sorted()


###### **size()**
returns length of stream. Use carefully on infinite streams.


###### **join(f)**
Returns a string joined by f. Proivides same functionality as str.join() builtin method.

if f is basestring, uses it to join the stream, else f should be a callable that returns a string to be used for join


###### **mkString(f)**
identic with join(f)


###### **take(n)**
    returns first n elements from stream


###### **head()**
    returns first element from stream


###### **zip()**
    the same behavior with itertools.izip()

###### **unique(predicate=_IDENTITY_FUNC)**
    Returns a stream of unique (according to predicate) elements appearing in the same order as in original stream

    The items returned by predicate should be hashable and comparable.


#### Statistics related methods
###### **entropy()**
calculates the Shannon entropy of the values from stream


###### **pstddev()**
Calculates the population standard deviation.


###### **mean()**
returns the arithmetical mean of the values


###### **sum()**
returns the sum of elements from stream


###### **min(key=_IDENTITY_FUNC)**
same functionality with builtin min() funcion


###### **min_default(default, key=_IDENTITY_FUNC)**
same functionality with min() but returns :default: when called on empty streams


###### **max()**
same functionality with builtin max()


###### **maxes(key=_IDENTITY_FUNC)**
returns a stream of max values from stream


###### **mins(key=_IDENTITY_FUNC)**
returns a stream of min values from stream


### Other classes
##### slist
Inherits `streams.stream` and built-in `list` classes, and keeps in memory a list allowing faster index access
##### sset
Inherits `streams.stream` and built-in `set` classes, and keeps in memory the whole set of values
##### sdict
Inherits `streams.stream` and built-in `dict`, and keeps in memory the dict object.
##### defaultstreamdict
Inherits `streams.sdict` and adds functionality  of `collections.defaultdict` from stdlib


### [Json](https://github.com/asuiu/pyxtension/blob/master/Json.py)

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
### Installation
from Github::
```
    $ git clone https://github.com/asuiu/pyxtension.git
    $ python setyp.py install

or

    $ git submodule add https://github.com/asuiu/pyxtension.git
```

### License
pyxtension is released under a GNU Public license.
The idea for [Json](https://github.com/asuiu/pyxtension/blob/master/Json.py) module was inspired from [addict](https://github.com/mewwts/addict>) and [AttrDict](https://github.com/bcj/AttrDict),
but it has a better performance with lower memory consumption.
