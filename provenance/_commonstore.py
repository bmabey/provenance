import toolz as t
import operator as op


class PermissionError(Exception):
    def __init__(self, action, store, permission):
        message = "A `{}` operation was attempted on {} and {} is set to `False`!".\
                  format(action, store, permission)
        self.action = action
        self.store = store
        self.permission = permission
        Exception.__init__(self, message)


class KeyExistsError(Exception):
    def __init__(self, key, store):
        msg = "The key {} is already present in {}, you can not overwrite it!".\
              format(key, store)
        self.key = key
        self.store = store
        Exception.__init__(self, msg)


class InconsistentKeyError(Exception):
    def __init__(self, key, store, value):
        msg = "The key {} already represents a different value in {}".format(key, store)
        self.key = key
        self.store = store
        self.value = value
        Exception.__init__(self, msg)


def ensure_read(obj, action='get'):
    if not obj._read:
        raise PermissionError(action, obj, 'read')


def ensure_write(obj, action='put'):
    if not obj._write:
        raise PermissionError(action, obj, 'write')

ensure_contains = t.partial(ensure_read, action='contains')

def ensure_present(obj, id):
    if id not in obj:
        raise KeyError(id, obj)

def ensure_delete(obj, id=None, check_contains=True):
    if not obj._delete:
        raise PermissionError('delete', obj, 'delete')
    if check_contains and id is not None and id not in obj:
        raise KeyError(id, obj)


def ensure_put(obj, id, read_through=None, check_contains=True):
    if read_through:
        if not obj._read_through_write:
            raise PermissionError('read_through_put', obj, 'read_through_write')
    elif not obj._write:
        raise PermissionError('put', obj, 'write')
    if check_contains and id in obj:
        raise KeyExistsError(id, obj)

def chained_contains(chained, id, contains=op.contains):
    stores_with_read = [s for s in chained.stores if s._read]
    if len(stores_with_read) == 0:
        raise PermissionError('contains', chained, 'read')

    for store in stores_with_read:
        if store._read and contains(store, id):
            return True
    return False

def chained_put(chained, id, value, put=None, contains=op.contains, **kargs):
    stores_with_write = [s for s in chained.stores if s._write]
    if len(stores_with_write) == 0:
        raise PermissionError('put', chained, 'write')

    record = None
    putin = []
    for store in stores_with_write:
        if not contains(store, id):
            if put:
                record = put(store, id, value, **kargs)
            else:
                record = store.put(id, value, **kargs)
            putin.append(store)

    if (len(putin) == 0 and
        len(stores_with_write) > 0):
        raise KeyExistsError(id, chained)

    return record


def chained_get(chained, get, id, put=None):
    stores_with_read = [s for s in chained.stores if s._read]
    if len(stores_with_read) == 0:
        raise KeyError(id, chained)

    pushback = []
    for store in stores_with_read:
        try:
            value = get(store, id)
            break
        except KeyError:
            if store._read_through_write:
                pushback.append(store)
    else:
        raise KeyError(id, chained)

    for store in pushback:
        if put:
            put(store, id, value, read_through=True)
        else:
            store.put(id, value, read_through=True)
    return value


def chained_delete(chained, id, delete=None, contains=op.contains):
    stores_with_delete = [s for s in chained.stores if s._delete]
    if len(stores_with_delete) == 0:
        raise PermissionError('delete', chained, 'delete')

    foundin = []
    for store in stores_with_delete:
        if contains(store, id):
            foundin.append(store)
            if delete:
                delete(store, id)
            else:
                store.delete(id)
    if len(foundin) == 0:
        raise KeyError(id, chained)
    else:
        return foundin
