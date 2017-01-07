from collections import namedtuple, OrderedDict, defaultdict
import toolz as t
from boltons import funcutils as bfu
import time
import datetime

import multiprocessing
import os
import platform
import psutil

from ._dependencies import dependencies
from .hashing import hash
from . import repos as repos
from . import serializers as s
from .serializers import DEFAULT_VALUE_SERIALIZER
from . import utils


class ImpureFunctionError(Exception):
    pass

def get_metadata(f):
    if hasattr(f, '_provenance_metadata'):
        return f._provenance_metadata
    if hasattr(f, 'func'):
        return get_metadata(f.func)
    else:
        return {}


@t.memoize()
def host_info():
    return {'machine': platform.machine(),
            'nodename': platform.node(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': multiprocessing.cpu_count(),
            'release': platform.release(),
            'system': platform.system(),
            'version': platform.version()}


@t.memoize()
def process_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    return {'cmdline': p.cmdline(),
            'cwd': p.cwd(),
            'exe': p.exe(),
            'name': p.name(),
            'num_fds': p.num_fds(),
            'num_threads': p.num_threads()}


artifact_properties = ['id', 'input_id', 'inputs', 'fn_module', 'fn_name', 'value',
                       'name', 'version', 'composite', 'input_id_duration',
                       'serializer', 'load_kwargs', 'dump_kwargs',
                       'compute_duration', 'hash_duration', 'computed_at', 'host',
                       'process', 'custom_fields']

ArtifactRecord = namedtuple('ArtifactRecord', artifact_properties)


def fn_info(f):
    info = utils.fn_info(f)
    metadata = get_metadata(f)
    info['identifiers'] = {'name': metadata['name'],
                           'version': metadata['version'],
                           'input_hash_fn': metadata['input_hash_fn']}
    info['input_process_fn'] = metadata['input_process_fn']
    info['composite'] = metadata.get('returns_composite', False)
    info['custom_fields'] = t.dissoc(metadata,
                                     'name', 'version', 'input_hash_fn',
                                     'input_process_fn', 'returns_composite',
                                     'serializer', 'load_kwargs', 'dump_kwargs')

    if info['composite']:
        info['serializer'] = metadata.get('serializer', {})
        info['load_kwargs'] = metadata.get('load_kwargs', {})
        info['dump_kwargs'] = metadata.get('dump_kwargs', {})
        valid_serializer = isinstance(info['serializer'], dict)
        for serializer in info['serializer'].values():
            valid_serializer = valid_serializer and serializer in s.serializers
            if not valid_serializer:
                break
    else:
        info['serializer'] = metadata.get('serializer', DEFAULT_VALUE_SERIALIZER.name)
        info['load_kwargs'] = metadata.get('load_kwargs', None)
        info['dump_kwargs'] = metadata.get('dump_kwargs', None)
        valid_serializer = info['serializer'] in s.serializers

    if not valid_serializer:
        msg = 'Invalid serializer option "{}" for artifact "{}", available serialziers: {} '.\
              format(info['serializer'], info['name'], tuple(s.serializers.keys()))
        raise ValueError(msg)

    return info


def hash_inputs(inputs):
    return {'kargs': t.valmap(hash, inputs['kargs']),
            'varargs': tuple(hash(arg) for arg in inputs['varargs'])}


def create_input_id(input_hashes, input_hash_fn, name, version):
    return t.thread_first(input_hashes,
                          input_hash_fn,
                          (t.merge, {'name': name, 'version': version}),
                          hash)


@t.curry
def composite_artifact(repo, inputs, input_hashes, input_hash_fn, artifact_info,
                       compute_duration, computed_at, key, value):
    start_input_id_time = time.time()
    info = artifact_info.copy()
    info['composite'] = False
    info['name'] = '{}_{}'.format(info['name'], key)
    info['serializer'] = info['serializer'].get(key, DEFAULT_VALUE_SERIALIZER.name)
    info['load_kwargs'] = info['load_kwargs'].get(key, None)
    info['dump_kwargs'] = info['dump_kwargs'].get(key, None)

    input_id = create_input_id(input_hashes, input_hash_fn, info['name'], info['version'])
    input_id_duration = time.time() - start_input_id_time

    start_hash_time = time.time()
    id = hash(value)
    hash_duration = time.time() - start_hash_time

    record = ArtifactRecord(id=id, input_id=input_id, value=value,
                            input_id_duration=input_id_duration,
                            compute_duration=compute_duration,
                            hash_duration=hash_duration, computed_at=computed_at,
                            inputs=inputs, **info)
    return repo.put(record)

def _base_fn(f):
    if utils.is_curry_func(f):
        return utils.inner_function(f)
    else:
        return f

@t.curry
def provenance_wrapper(repo, f):
    base_fn = _base_fn(f)
    extract_args = utils.args_extractor(base_fn, merge_defaults=True)
    func_info = fn_info(f)
    input_process_fn = func_info['input_process_fn']

    artifact_info = {'name': func_info['identifiers']['name'],
                     'version': func_info['identifiers']['version'],
                     'fn_name': func_info['name'], 'fn_module': func_info['module'],
                     'custom_fields': func_info['custom_fields'],
                     'serializer': func_info['serializer'],
                     'load_kwargs': func_info['load_kwargs'],
                     'dump_kwargs': func_info['dump_kwargs'],
                     'composite': func_info['composite'],
                     'host': host_info(), 'process': process_info()}

    @bfu.wraps(f)
    def _provenance_wrapper(*args, **kargs):
        r = repos.get_default_repo() if repo is None else repo
        info = artifact_info
        start_input_id_time = time.time()
        varargs, argsd = extract_args(args, kargs)
        inputs = input_process_fn({'varargs': varargs + func_info['varargs'],
                                   'kargs': t.merge(argsd, func_info['kargs'])})
        input_hashes = hash_inputs(inputs)
        input_id = create_input_id(input_hashes, **func_info['identifiers'])
        input_id_duration = time.time() - start_input_id_time

        try:
            artifact = r.get_by_input_id(input_id)
        except KeyError:
            artifact = None

        if artifact is None:
            start_compute_time = time.time()
            computed_at = datetime.datetime.utcnow()
            value = f(*varargs, **argsd)
            compute_duration = time.time() - start_compute_time

            post_input_hashes = hash_inputs(inputs)
            if input_id != create_input_id(post_input_hashes, **func_info['identifiers']):
                modified_inputs = []
                kargs = input_hashes['kargs']
                varargs = input_hashes['varargs']
                for name, _hash in post_input_hashes['kargs'].items():
                    if _hash != kargs[name]:
                        modified_inputs.append(name)
                for i, _hash in enumerate(post_input_hashes['varargs']):
                    if _hash != varargs[i]:
                        modified_inputs.append("varargs[{}]".format(i))
                msg = "The {}.{} function modified arguments: ({})".format(
                    func_info['module'], func_info['name'], ",".join(modified_inputs))
                raise ImpureFunctionError(msg)

            if artifact_info['composite']:
                input_hash_fn = func_info['identifiers']['input_hash_fn']
                ca = composite_artifact(r, inputs, input_hashes, input_hash_fn,
                                        artifact_info, compute_duration, computed_at)
                value = {k: ca(k, v) for k, v in value.items()}
                artifact_info['serializer'] = DEFAULT_VALUE_SERIALIZER.name
                artifact_info['load_kwargs'] = None
                artifact_info['dump_kwargs'] = None

            start_hash_time = time.time()
            id = hash(value)
            hash_duration = time.time() - start_hash_time

            record = ArtifactRecord(id=id, input_id=input_id, value=value,
                                    input_id_duration=input_id_duration,
                                    compute_duration=compute_duration,
                                    hash_duration=hash_duration,
                                    computed_at=computed_at,
                                    inputs=inputs, **artifact_info)
            artifact = r.put(record)

        return artifact.proxy()

    if utils.is_curry_func(f):
        fb = bfu.FunctionBuilder.from_func(utils.inner_function(f))
        for arg in f.args + tuple(f.keywords.keys()):
            fb.remove_arg(arg)
        param_info = utils.param_info(f)
        args = []
        defaults = []
        for arg, value in param_info.items():
            args.append(arg)
            if value != utils.UNSPECIFIED_ARG:
               defaults.append(value)
        arg_inv = ['{}={}'.format(arg,arg) for arg in args]
        fb.body = 'return _provenance_wrapper(%s)' % ", ".join(arg_inv)
        fb.args = args
        fb.defaults = tuple(defaults)
        execdict = {'_provenance_wrapper': _provenance_wrapper}
        ret = fb.get_func(execdict, with_dict=True)
        return ret
    return _provenance_wrapper


def remove_inputs_fn(to_remove):
    def remove_inputs(inputs):
        kargs = t.keyfilter(lambda k: k not in to_remove, inputs['kargs'])
        return {'kargs': kargs, 'varargs': inputs['varargs']}
    return remove_inputs


def provenance(version=0, repo=None, name=None, merge_defaults=None,
               ignore=None, input_hash_fn=None, remove=None, input_process_fn=None,
               _provenance_wrapper=provenance_wrapper, **kargs):
    if ignore and input_hash_fn:
        raise ValueError("You cannot provide both ignore and input_hash_fn")

    if ignore:
        ignore = frozenset(ignore)
        input_hash_fn = remove_inputs_fn(ignore)

    if not input_hash_fn:
        input_hash_fn = lambda inputs: inputs

    if remove and input_process_fn:
        raise ValueError("You cannot provide both remove and input_process_fn")

    if remove:
        remove = frozenset(remove)
        input_process_fn = remove_inputs_fn(remove)

    if not input_process_fn:
        input_process_fn = lambda inputs: inputs

    def wrapped(f):
        _name = name
        if _name is None:
            _name = f.__name__
        f._provenance_metadata = t.merge(kargs,
                                         {'version': version,
                                          'name': _name,
                                          'input_hash_fn': input_hash_fn,
                                          'input_process_fn': input_process_fn})
        f.__merge_defaults__ = merge_defaults
        return _provenance_wrapper(repo, f)

    return wrapped


def provenance_set(set_name=None, initial_set=None, set_name_fn=None):
    if set_name and set_name_fn:
        raise ValueError("You cannot provide both set_name and set_name_fn.")

    def make_wrapper(f):
        if set_name_fn:
            base_fn = _base_fn(f)
            extract_args = utils.args_extractor(base_fn, merge_defaults=True)
            func_info = utils.fn_info(f)

        @bfu.wraps(f)
        def wrapper(*fargs, **fkargs):
            if set_name_fn:
                varargs, argsd = extract_args(fargs, fkargs)
                varargs += func_info['varargs']
                argsd.update(func_info['kargs'])
                name = set_name_fn(*varargs, **argsd)
            else:
                name = set_name

            with repos.capture_set(name=name, initial_set=initial_set) as result:
                f(*fargs, **fkargs)
            return result[0]

        return wrapper

    return make_wrapper


def promote(artifact_or_id, to_repo, from_repo=None):
    from_repo = from_repo if from_repo else repos.get_default_repo()
    artifact = repos.coerce_to_artifact(artifact_or_id, repo=from_repo)
    for a in dependencies(artifact):
        if a not in to_repo:
            to_repo.put(a)
