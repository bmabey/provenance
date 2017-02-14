import sys

from . import hashing as h
from . import repos as r


def _save(obj, ids):
    if isinstance(obj, r.Artifact):
        ids.add(obj.id)
    if r.is_proxy(obj):
        ids.add(obj.artifact.id)


class ArtifactIdHasher(h.Hasher):
    def __init__(self, ids=None, hash_name='sha1'):
        if ids is None:
            ids = set()

        self.ids = ids
        h.Hasher.__init__(self, hash_name=hash_name)

    def save(self, obj):
        _save(obj, self.ids)
        h.Hasher.save(self, obj)

    def hash(self, obj):
        return (h.Hasher.hash(self, obj), frozenset(self.ids))


class NumpyArtifactIdHasher(h.NumpyHasher):
    def __init__(self, ids=None, hash_name='sha1', coerce_mmap=True):
        if ids is None:
            ids = set()

        self.ids = ids
        h.NumpyHasher.__init__(self, hash_name=hash_name, coerce_mmap=coerce_mmap)

    def save(self, obj):
        _save(obj, self.ids)
        h.NumpyHasher.save(self, obj)

    def hash(self, obj):
        return (h.NumpyHasher.hash(self, obj), frozenset(self.ids))


def artifact_id_hasher(*args, **kwargs):
    if 'numpy' in sys.modules:
        return NumpyArtifactIdHasher(*args, **kwargs)
    else:
        return ArtifactIdHasher(*args, **kwargs)
