# pyxtension
[![Build Status](https://github.com/asuiu/pyxtension/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/asuiu/pyxtension/actions/workflows/python-package.yml)

[pyxtension](https://github.com/asuiu/pyxtension) is a pure Python MIT-licensed library that includes Scala-like streams (using [Fluent Interface pattern](https://en.wikipedia.org/wiki/Fluent_interface)), Json with attribute access syntax, and other common-use stuff.

###### Note:

**Drop support & maintenance for Python 2.x version, due to [Py2 death](https://www.python.org/doc/sunset-python-2/).**

 Although Py2 version will remain in the repository, I won't update PyPi package, so the last Py2 version of the `pyxtension` available at [PyPi](https://pypi.org/project/pyxtension/) will remain [`1.12.7`](https://pypi.org/project/pyxtension/1.12.7/)

Starting with [`1.13.0`](https://pypi.org/project/pyxtension/1.13.0/) I've migrated the packaging & distributing method to [Wheel](https://pythonwheels.com/).

## Installation
```
pip install pyxtension
```
or from Github:
```
git clone https://github.com/asuiu/pyxtension.git
cd pyxtension
python setup.py install
```
or
```
git submodule add https://github.com/asuiu/pyxtension.git
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


###### **mpmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(), bufferSize: Optional[int] = None)**
Parallel ordered map using `multiprocessing.Pool.imap()`.

It can replace the `map` when need to split computations to multiple cores, and order of results matters.

It spawns at most `poolSize` processes and applies the `f` function.

It won't take more than `bufferSize` elements from the input unless it was already required by output, so you can use it with `takeWhile` on infinite streams and not be afraid that it will continue work in background.

The elements in the result stream appears in the same order they appear in the initial iterable.

```
:type f: (T) -> V
:rtype: `stream`
```


###### **mpfastmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(), bufferSize: Optional[int] = None)**
Parallel ordered map using `multiprocessing.Pool.imap_unordered()`.

It can replace the `map` when the ordered of results doesn't matter.

It spawns at most `poolSize` processes and applies the `f` function.

It won't take more than `bufferSize` elements from the input unless it was already required by output, so you can use it with `takeWhile` on infinite streams and not be afraid that it will continue work in background.

The elements in the result stream appears in the unpredicted order.

```
:type f: (T) -> V
:rtype: `stream`
```


###### **fastmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(), bufferSize: Optional[int] = None)**
Parallel unordered map using multithreaded pool.
It can replace the `map` when the ordered of results doesn't matter.

It spawns at most `poolSize` threads and applies the `f` function.

The elements in the result stream appears in the **unpredicted** order.

It won't take more than `bufferSize` elements from the input unless it was already required by output, so you can use it with `takeWhile` on infinite streams and not be afraid that it will continue work in background.

Because of CPython [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) it's most usefull for I/O or CPU intensive consuming native functions, or on Jython or IronPython interpreters.

:type f: (T) -> V

:rtype: `stream`

###### **mtmap(self, f: Callable[[_K], _V], poolSize: int = cpu_count(), bufferSize: Optional[int] = None)**
Parallel ordered map using multithreaded pool.
It can replace the `map` and the order of output stream will be the same as of the input.

It spawns at most `poolSize` threads and applies the `f` function.

The elements in the result stream appears in the **predicted** order.

It won't take more than `bufferSize` elements from the input unless it was already required by output, so you can use it with `takeWhile` on infinite streams and not be afraid that it will continue work in background.

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
    > from pyxtension import Json
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
```
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

### frozendict
`frozendict` is a simple immutable dictionary, where you can't change the internal variables of the class, and they are all immutable objects. Reinvoking `__init__` also doesn't alter the object.

The API is the same as `dict`, without methods that can change the immutability.

`frozendict` is also hashable and can be used as keys for other dictionaries, of course with the condition that all values of the frozendict are also hashable.

```python
>>> from pyxtension import frozendict

>>> fd = frozendict({"A": "B", "C": "D"})
>>> print(fd)
{'A': 'B', 'C': 'D'}

>>> fd["A"] = "C"
TypeError: object is immutable

>>> hash(fd)
-5063792767678978828
```

### License
pyxtension is released under a GNU Public license.
The idea for [Json](https://github.com/asuiu/pyxtension/blob/master/Json.py) module was inspired from [addict](https://github.com/mewwts/addict) and [AttrDict](https://github.com/bcj/AttrDict),
but it has a better performance with lower memory consumption.

### Alternatives
There are other libraries that support Fluent Interface streams as alternatives to Pyxtension, but being much more poor in features for streaming:
- https://pypi.org/project/lazy-streams/
- https://pypi.org/project/pystreams/
- https://pypi.org/project/fluentpy/
- https://github.com/matthagy/scalaps
- https://pypi.org/project/infixpy/ mentioned [here](https://stackoverflow.com/questions/49001986/left-to-right-application-of-operations-on-a-list-in-python3/62585964?noredirect=1#comment111806251_62585964)
- https://github.com/sspipe/sspipe


and something quite different from Fluent patterm, that makes kind of Piping: https://github.com/sspipe/sspipe and https://github.com/JulienPalard/Pipe
