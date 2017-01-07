import provenance.blobstores as bs
import provenance.repos as r
import provenance._config as c
import conftest as ct

def test_atomic_blobstore_config_reading():
    config = {'type': 'disk',
              'cachedir': '.artifacts/',
              'read': True,
              'write': True,
              'read_through_write': False,
              'delete': True}
    store = c.blobstore_from_config(config)
    assert type(store) == bs.DiskStore
    assert store.cachedir == bs._abspath(config['cachedir'])
    assert store._read == config['read']
    assert store._write == config['write']
    assert store._delete == config['delete']
    assert store._read_through_write == config['read_through_write']

def test_prototypes_are_merged():
    config = {'local_disk':
              {'type': 'disk',
               'cachedir': '.artifacts/',
               'read': True,
               'write': True,
               'read_through_write': False,
               'delete': True},
              'local_read_only':
              {'prototype': 'local_disk',
               'write': False,
               'delete': False},
              'local_read_through_write':
              {'prototype': 'local_read_only',
               'read_through_write': True}}


    stores = c.blobstores_from_config(config)
    store = stores['local_read_through_write']
    assert type(store) == bs.DiskStore
    assert store.cachedir == bs._abspath('.artifacts/')
    assert store._read
    assert not store._write
    assert not store._delete
    assert store._read_through_write


def test_blobstores_config_reading():
    config = {'local_disk':
              {'type': 'disk',
               'cachedir': '.artifacts/',
               'read': True,
               'write': True,
               'read_through_write': False,
               'delete': True},
              'mem':
              {'type': 'memory',
               'read': True,
               'write': True,
               'read_through_write': False,
               'delete': True},
              'shared_s3':
              {'type': 's3',
               'cachedir': '/tmp/foo',
               'basepath': 'mybucket/blobs',
               'delete': False,
               's3_config': {'anon': True}},
              'chained': {'type': 'chained',
                          'stores': ['local_disk', 'mem', 'shared_s3']}}

    stores = c.blobstores_from_config(config)
    chained = stores['chained']
    assert isinstance(chained, bs.ChainedStore)
    assert [type(s) for s in chained.stores] == [bs.DiskStore,
                                                 bs.MemoryStore,
                                                 bs.S3Store]

def test_from_config():
    config = {'blobstores':
              {'mem':
               {'type': 'memory',
                'read': True,
                'write': True,
                'read_through_write': False,
                'delete': True}},
              'artifact_repos':
              {'db':
               {'type': 'postgres',
                'db': ct.db_conn_str(),
                'store': 'mem'}}}
    objs = c.from_config(config)
    repo = objs['repos']['db']
    assert isinstance(repo, r.PostgresRepo)
    assert isinstance(repo.blobstore, bs.MemoryStore)
