from collections import namedtuple
import toolz as t
import cloudpickle
import joblib

from .hashing import hash

def cloudpickle_dump(obj, filename, **kargs):
    with open(filename, 'wb') as f:
        return cloudpickle.dump(obj, f, **kargs)


def cloudpickle_load(filename, **kargs):
    with open(filename, 'rb') as f:
        return cloudpickle.load(f, **kargs)


Serializer = namedtuple('Serializer', 'name, dump, load')


serializers = {'joblib': Serializer('joblib', joblib.dump, joblib.load),
               'cloudpickle': Serializer('cloudpickle',
                                         cloudpickle_dump,
                                         cloudpickle_load),}


def register_serializer(name, dump, load):
    serializers[name] = Serializer(name, dump, load)

@t.memoize(key=lambda *args: hash(args))
def partial_serializer(serializer_name, dump_kwargs, load_kwargs):
    s = serializers[serializer_name]
    return Serializer(s.name,
                      t.partial(s.dump, **dump_kwargs) if dump_kwargs else s.dump,
                      t.partial(s.load, **load_kwargs) if load_kwargs else s.load)


def serializer(artifact):
    return partial_serializer(artifact.serializer,
                              artifact.dump_kwargs,
                              artifact.load_kwargs)

DEFAULT_VALUE_SERIALIZER = serializers['joblib']
DEFAULT_INPUT_SERIALIZER = serializers['joblib']
