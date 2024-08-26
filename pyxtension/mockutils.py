from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union


class UnknownMockArguments(Exception):
    pass


def generate_mock_map_func(
    args_to_ret: Union[Sequence[Tuple[Dict[str, Any], Any]], Dict[Tuple[Any, ...], Any]],
    arg_names: Tuple[str, ...] = (),
    exc: Type[Exception] = UnknownMockArguments,
) -> Callable[..., Any]:
    """
    This function generates a mock function that returns the value from args_to_ret if the arguments match the keys in args_to_ret.
    The mock basically maps the arguments to the return values.
    Examples of usages:
        mf = generate_mock_map_func({('a',): 1, ('b',): '2'})
        self.assertEqual(mf('a'), 1)
        self.assertEqual(mf('b'), '2')
        ----------------
        mf = generate_mock_map_func({('a', 'b'): 1, ('a', 'c'): '2'}, arg_names=('a1', 'a2'))
        self.assertEqual(mf('a', 'b'), 1)
        self.assertEqual(mf('a', 'c'), '2')

        self.assertEqual(mf(a1='a', a2='b'), 1)
        self.assertEqual(mf(a1='a', a2='c'), '2')
        ----------------
        mf = generate_mock_map_func([({'a1': 'a', 'a2': 'b'}, 1), ({'a1': 'a', 'a2': 'c'}, '2')])
        self.assertEqual(mf(a1='a', a2='b'), 1)
        self.assertEqual(mf(a1='a', a2='c'), '2')
    """
    by_args: Dict[Tuple[Any, ...], Any] = {}
    by_kwargs: Dict[frozenset[Tuple[str, Any]], Any] = {}
    if isinstance(args_to_ret, dict):
        if arg_names:
            for args_key, ret in args_to_ret.items():
                if len(args_key) != len(arg_names):
                    raise exc()

                d = dict(zip(arg_names, args_key))
                by_kwargs[frozenset(d.items())] = ret
        else:
            by_args = args_to_ret
    else:
        args_to_ret = tuple(args_to_ret)
        for args_dict, ret in args_to_ret:
            by_kwargs[frozenset(args_dict.items())] = ret

    def f(*args, **kwargs):
        if args and not arg_names:
            if args not in by_args:
                raise exc()
            return by_args[args]

        if len(args) > len(arg_names):
            raise exc()
        for i, arg in enumerate(args):
            if arg_names[i] in kwargs:
                raise exc()
            kwargs[arg_names[i]] = arg
        frozen_args_key = frozenset(kwargs.items())
        if frozen_args_key not in by_kwargs:
            raise exc()
        return by_kwargs[frozen_args_key]

    return f
