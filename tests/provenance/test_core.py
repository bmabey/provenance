import toolz as t
import pytest

import provenance as p
import provenance.blobstores as bs
import provenance.core as pc
import provenance.repos as r
import provenance.utils as u
from provenance.hashing import hash
from conftest import artifact_record

def test_integration_test(repo):
    @p.provenance(version=0, name='initial_data')
    def load_data(filename):
        return [1, 2]


    @p.provenance(repo=repo, remove=('to_remove',))
    def process_data_A(data, process_a_inc, to_remove):
        return [i + process_a_inc for i in data]

    times_called = 0

    @p.provenance(ignore=('to_ignore',))
    def process_data_B(data, process_b_inc, to_ignore):
        nonlocal times_called
        times_called += 1
        return [i + process_b_inc for i in data]

    @p.provenance()
    def combine_processed_data(inc_a, inc_b):
        return [a + b for a, b in zip(inc_a, inc_b)]


    def run_pipeline(filename, to_ignore, to_remove):
        data = load_data(filename) # [1, 2]
        inc_a = process_data_A(data, 1, to_remove) # [2, 3]
        inc_b = process_data_B(data, 5, to_ignore) # [6, 7]
        res = combine_processed_data(inc_a, inc_b) # [8, 10]
        return res

    result = run_pipeline('foo-bar.csv', 'something', 'removed')

    assert result == [8, 10]

    #check initial wrapping
    assert result.artifact.id == hash([8,10])
    artifact = result.artifact

    # check that inputs were removed
    inc_a_artifact = artifact.inputs['kargs']['inc_a'].artifact
    assert inc_a_artifact.inputs == {'kargs': {'data': [1, 2], 'process_a_inc': 1},
                                     'varargs': ()}

    # check metadata
    data_artifact = inc_a_artifact.inputs['kargs']['data'].artifact
    assert data_artifact.name == 'initial_data'
    assert data_artifact.version == 0

    # Check caching
    assert times_called == 1
    new_res = run_pipeline('foo-bar.csv', 'something-different', 'removed-again')
    assert new_res == [8, 10]
    assert times_called == 1

    # Check that the dependencies can be returned
    dependencies = p.dependencies(result.artifact.id)
    assert dependencies == [data_artifact,
                            artifact.inputs['kargs']['inc_b'].artifact,
                            inc_a_artifact,
                            artifact]


def test_fn_with_merged_defaults_set_with_provenance_decorator(repo):

    @p.provenance(merge_defaults=True)
    def add(data, adders={'a': 1, 'b': 2}):
        return {k:[i + inc for i in data] for k,inc in adders.items()}


    assert add([1,2,3], adders={'c': 2, 'b': 0}) == \
        {'a': [2, 3, 4],
         'b': [1, 2, 3],
         'c': [3, 4, 5]}


def test_with_merged_defaults_used_on_fn(repo):

    @u.with_merged_defaults()
    def add(data, adders={'a': 1, 'b': 2}):
        return {k:[i + inc for i in data] for k,inc in adders.items()}

    fetch_add = p.provenance()(add)


    assert add([1,2,3], adders={'c': 2, 'b': 0}) == \
        {'a': [2, 3, 4],
         'b': [1, 2, 3],
         'c': [3, 4, 5]}


    assert fetch_add([1,2,3], adders={'c': 2, 'b': 0}) == \
        {'a': [2, 3, 4],
         'b': [1, 2, 3],
         'c': [3, 4, 5]}

def test_that_curried_values_are_preserved_in_the_inputs(repo):

    @t.curry
    @p.provenance()
    def three_sum(a, b=5, c=3):
        return a + b + c

    my_sum = three_sum(b=5, c=3)

    result = my_sum(2)
    assert result == 10
    assert result.artifact.inputs['kargs'] == {'a': 2, 'b': 5, 'c': 3}


def test_serialization_of_lambdas(repo):

    @p.provenance(serializer='cloudpickle')
    def magnifier(x):
        return lambda y: x * y

    doubler = magnifier(2)
    assert doubler(2) == 4

    doubler_fetched = repo.get_by_id(doubler.artifact.id).value
    assert doubler(5) == 10


def test_composite_artifacts(repo):
    @p.provenance(returns_composite=True,
                  serializer={'a': 'cloudpickle'},
                  load_kwargs={'b': {'memmap': True}})
    def load_data():
        return {'a': 1, 'b': 2, 'c': 3}

    results = load_data()

    assert results['a'] == 1
    assert results['b'] == 2
    assert results['c'] == 3

    # Check that individual artifacts where created
    a_artifact = results['a'].artifact
    assert a_artifact.name == 'load_data_a'
    assert repo.get_by_id(a_artifact.id) == a_artifact

    # Check that the correct serializers is used
    assert a_artifact.serializer == 'cloudpickle'

    # Check that the correct load_kwargs are set
    assert results['b'].artifact.load_kwargs == {'memmap': True}


def test_does_not_allow_argument_modification(repo):
    @p.provenance()
    def append_3_inc(a):
        a.append(3)
        return [n + 1 for n in a]

    msg = "The test_core.append_3_inc function modified arguments: (a)"
    with pytest.raises(pc.ImpureFunctionError, message=msg):
        result = append_3_inc([1,2])
        assert False


def test_set_creation(repo):
    a = artifact_record(id='foo')
    b = artifact_record(id='blah')
    repo.put(a)
    repo.put(b)
    artifact_set = r.create_set([a,b], name='myset')

    assert repo.get_set_by_name('myset') == artifact_set
    assert repo.get_set_by_id(artifact_set.id) == artifact_set


def test_set_renaming(repo):
    a = artifact_record(id='foo')
    b = artifact_record(id='blah')
    repo.put(a)
    repo.put(b)
    named_set = r.create_set([a,b], 'foobar')

    assert repo.get_set_by_name('foobar') == named_set
    renamed_set = r.name_set(named_set, 'baz')

    # we don't delete the old one
    assert repo.get_set_by_name('foobar') == named_set
    assert repo.get_set_by_name('baz') == renamed_set


def test_adding_new_artifact_to_set():
    a = artifact_record(id='foo')
    b = artifact_record(id='blah')
    artifact_set = r.ArtifactSet([a], name='myset')
    updated_set = artifact_set.add(b)
    assert updated_set.name is None

    updated_named_set = artifact_set.add(b, name='myset')
    assert updated_named_set.name == 'myset'


def test_removing_artifact_from_set():
    a = artifact_record(id='foo')
    b = artifact_record(id='blah')
    artifact_set = r.ArtifactSet([a.id, b.id], name='myset')
    updated_set = artifact_set.remove(b)

    assert b not in updated_set
    assert updated_set.name is None

    updated_named_set = artifact_set.remove(b, name='myset')
    assert b not in updated_named_set
    assert updated_named_set.name == 'myset'


def test_set_unions():
    a, b, c, d = [artifact_record(id=l) for l in 'abcd']
    set_one = r.ArtifactSet([a, b], name='set one')
    set_two = r.ArtifactSet([c, d], name='set two')
    set_three = set_one.union(set_two, name='set three')

    assert set_three.name == 'set three'

    assert set_three.artifact_ids == {'a', 'b', 'c', 'd'}
    assert (set_one | set_two).artifact_ids == {'a', 'b', 'c', 'd'}


def test_set_differences():
    a, b, c, d = [artifact_record(id=l) for l in 'abcd']
    set_one = r.ArtifactSet([a, b, c], name='set one')
    set_two = r.ArtifactSet([c, d], name='set two')
    set_three = set_one.difference(set_two, name='set three')


    assert set_three.name == 'set three'
    expected_set = {'a', 'b', 'c'} - {'c', 'd'}
    assert set_three.artifact_ids == expected_set
    # check that the operator works too
    assert (set_one - set_two).artifact_ids == expected_set


def test_set_intersections():
    a, b, c, d = [artifact_record(id=l) for l in 'abcd']
    set_one = r.ArtifactSet([a, b, c], name='set one')
    set_two = r.ArtifactSet([c, d], name='set two')
    set_three = set_one.intersection(set_two, name='set three')


    assert set_three.name == 'set three'

    assert set_three.artifact_ids == {'c'}
    # check that the operator works too
    assert (set_one & set_two).artifact_ids == {'c'}


def test_set_capture(repo):

    @p.provenance()
    def add(a, b):
       return a + b

    @p.provenance()
    def mult(x, y):
       return x * y

    with p.capture_set(name='first'):
        x1 = add(2, 2)
        z1 = mult(x1, 10)

    first_set = repo.get_set_by_name('first')
    assert first_set.artifact_ids == {x1.artifact.id, z1.artifact.id}


    with p.capture_set(name='second'):
        _x = add(2, 2)
        z2 = mult(_x, 20)

    second_set = repo.get_set_by_name('second')
    # note how we check to see if x1 is present!
    assert second_set.artifact_ids == {x1.artifact.id, z2.artifact.id}


def test_set_capture_on_loads(repo):

    @p.provenance()
    def add(a, b):
        return a + b

    @p.provenance()
    def mult(x, y):
        return x * y

    x = add(3, 33)

    with p.capture_set(name='first'):
        x = repo.get_by_id(x.artifact.id).proxy()
        z1 = mult(x, 10)

    first_set = repo.get_set_by_name('first')
    assert first_set.artifact_ids == {x.artifact.id, z1.artifact.id}


def test_set_capture_with_initial_artifacts(repo):

    @p.provenance()
    def add(a, b):
       return a + b

    @p.provenance()
    def mult(x, y):
       return x * y

    x = add(3, 33)

    with p.capture_set(initial_set={x.artifact.id}, name='first'):
        z1 = mult(x, 10)

    first_set = repo.get_set_by_name('first')
    assert first_set.artifact_ids == {x.artifact.id, z1.artifact.id}


def test_provenance_set_decorator(repo):

    @p.provenance()
    def add(a, b):
       return a + b

    @p.provenance()
    def mult(x, y):
       return x * y

    @p.provenance_set(set_name='foobar')
    def my_pipeline(a, b=5, y=10):
        x = add(a, b)
        z = mult(x, y)

    my_set = my_pipeline(2)
    assert my_set.name == 'foobar'
    assert repo.get_set_by_name('foobar').id == my_set.id



def test_provenance_set_decorator_with_provenance(repo):

    @p.provenance()
    def add(a, b):
       return a + b

    @p.provenance()
    def mult(x, y):
       return x * y

    @p.provenance()
    @p.provenance_set(set_name='foobar')
    def my_pipeline(a, b=5, y=10):
        x = add(a, b)
        z = mult(x, y)

    my_set = my_pipeline(2)
    assert my_set.name == 'foobar'
    assert (my_set.artifact.inputs['kargs'] ==
            {'a': 2, 'b':5, 'y': 10})

    assert repo.get_set_by_name('foobar').id == my_set.id


def test_provenance_set_decorator_being_named_with_fn(repo):

    @p.provenance()
    def add(a, b):
       return a + b

    @p.provenance()
    def mult(x, y):
       return x * y

    @p.provenance_set(set_name_fn=lambda a, b, y: 'pipeline_{}_{}'.format(a,b))
    def my_pipeline(a, b=5, y=10):
        x = add(a, b)
        z = mult(x, y)

    my_set = my_pipeline(2)
    assert my_set.name == 'pipeline_2_5'

def test_provenance_set_decorator_being_named_with_fn_used_with_curry(repo):

    @p.provenance()
    def add(a, b):
       return a + b

    @p.provenance()
    def mult(x, y):
       return x * y

    @t.curry
    @p.provenance()
    @p.provenance_set(set_name_fn=lambda a, b, y: 'pipeline_{}_{}_{}'.format(a,b,y))
    def my_pipeline(a, b, y=30):
        x = add(a, b)
        z = mult(x, y)

    pipeline_variant = my_pipeline(10)

    my_set = pipeline_variant(20)
    assert my_set.name == 'pipeline_10_20_30'


def test_lazy_dict():
    lazy_dict = p.lazy_dict({'foo': lambda: 'bar',
                             'baz': lambda: 'qux'})
    initial_hash = hash(lazy_dict)

    # check that keys can be fetched
    assert lazy_dict['foo'] == 'bar'
    assert lazy_dict['baz'] == 'qux'

    # check that the hash remains the same as values are realized
    assert hash(lazy_dict) == initial_hash

    # check that it raises correctly
    with pytest.raises(KeyError):
        lazy_dict['bar']

    del lazy_dict['foo']

    with pytest.raises(KeyError):
        lazy_dict['foo']


def test_lazy_proxy_dict_prevents_creation_with_artifacts_of_same_name(repo):
    foo = repo.put(artifact_record(name='foo', value=42))
    foo2 = repo.put(artifact_record(name='foo', value=100))

    msg = """Only artifacts with distinct names can be used in a lazy_proxy_dict.
Offending names: {'foo': 2}
"""
    with pytest.raises(ValueError, message=msg):
        p.lazy_proxy_dict([foo, foo2])
