import toolz as t

import provenance.utils as u


def test_fn_info_with_regular_function():
    def inc(x):
        return x + 1

    info = u.fn_info(inc)
    assert info == {
        'name': 'inc',
        'module': 'test_utils',
        'varargs': (),
        'kargs': {}
    }


def test_fn_info_with_partial():
    def mult(x, y):
        return x * y

    double = t.partial(mult, 2)
    info = u.fn_info(double)

    assert info == {
        'name': 'mult',
        'module': 'test_utils',
        'varargs': (),
        'kargs': {
            'x': 2
        }
    }


def test_fn_info_with_partial_of_partial():
    def mult(*args):
        return t.reduce(lambda a, b: a * b, args)

    double = t.partial(mult, 2)
    quad = t.partial(double, 2)
    info = u.fn_info(quad)

    assert info == {
        'name': 'mult',
        'module': 'test_utils',
        'varargs': (2, 2),
        'kargs': {}
    }


def test_fn_info_with_curry():
    @t.curry
    def mult(x, y):
        return x * y

    double = mult(2)
    assert double(2) == 4
    info = u.fn_info(double)

    assert info == {
        'name': 'mult',
        'module': 'test_utils',
        'varargs': (),
        'kargs': {
            'x': 2
        }
    }


def test_fn_info_with_multiple_curries():
    @t.curry
    def mult(a, b, c):
        return a * b * c

    double = mult(2)
    quad = double(2)
    info = u.fn_info(quad)

    assert info == {
        'name': 'mult',
        'module': 'test_utils',
        'varargs': (),
        'kargs': {
            'a': 2,
            'b': 2
        },
    }


def test_with_merged_defaults_basic_merging():
    foo_defaults = {'a': 1, 'b': 2}

    @u.with_merged_defaults()
    def bar(foo=foo_defaults):
        return foo

    assert bar() == {'a': 1, 'b': 2}
    assert bar(foo={'c': 3}) == {'a': 1, 'b': 2, 'c': 3}
    assert bar(foo={'a': 10}) == {'a': 10, 'b': 2}


def test_with_merged_defaults_with_non_dict_args():
    foo_defaults = {'a': 1, 'b': 2}

    @u.with_merged_defaults()
    def bar(a, foo=foo_defaults, baz=None):
        return a, baz, foo

    assert bar(5) == (5, None, {'a': 1, 'b': 2})
    assert bar(5, baz='baz', foo={'c': 3}) == (5, 'baz', {
        'a': 1,
        'b': 2,
        'c': 3
    })


def test_with_merged_defaults_with_args_splat():
    foo_defaults = {'a': 1, 'b': 2}

    @u.with_merged_defaults()
    def bar(*args, foo=foo_defaults):
        return args, foo

    assert bar(5, 10) == ((5, 10), {'a': 1, 'b': 2})
    assert bar() == ((), {'a': 1, 'b': 2})
