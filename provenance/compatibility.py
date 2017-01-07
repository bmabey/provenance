import builtins
import inspect
import sys

PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2


if PY3:
    getargspec = inspect.getfullargspec
else:
    getargspec = inspect.getargspec

if hasattr(builtins, 'basestring'):
    string_type = basestring
else:
    string_type = str
