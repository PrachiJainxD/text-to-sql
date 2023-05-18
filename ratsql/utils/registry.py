'''
defines a registry of callable objects - that can be registered and looked up by:
kind and name
'''

import collections
import collections.abc
import inspect
import sys

_REGISTRY = collections.defaultdict(dict)


def register(kind, name):
    '''
    takes kind and name as input arguments
    returns: a decorator function that can be used to register a callable object

    The registered object is added to a dictionary called _REGISTRY which is a defaultdict of dictionaries, 
    with the first-level keys being the different kinds of objects, 
    and the second-level keys being the names of the objects
    '''
    kind_registry = _REGISTRY[kind]

    def decorator(obj):
        if name in kind_registry:
            raise LookupError(f'{name} already registered as kind {kind}')
        kind_registry[name] = obj
        return obj

    return decorator


def lookup(kind, name):
    '''
    takes two arguments, kind and name, 
    and returns the registered object associated with that kind and name.
    '''
    if isinstance(name, collections.abc.Mapping):
        name = name['name']

    if kind not in _REGISTRY:
        raise KeyError(f'Nothing registered under "{kind}"')
    return _REGISTRY[kind][name]


def construct(kind, config, unused_keys=(), **kwargs):
    '''
    takes three arguments, kind, config, and unused_keys, 
    and returns the result of instantiating the registered object associated with that kind 
    using the instantiate function. 
    The config argument is a dictionary of keyword arguments to pass to the callable object 
    during instantiation. 
    The unused_keys argument is a tuple of keys that should be ignored in the config dictionary. 
    The function first looks up the registered object associated with the kind and name in the registry, 
    then merges the config dictionary with any additional keyword arguments passed in via kwargs. 
    It then uses the inspect.signature function to determine the expected arguments of the callable object 
    and raises an error if any positional or variable positional arguments are found. 
    If the callable object accepts variable keyword arguments, the merged config and kwargs dictionary 
    is passed to it directly. 
    Otherwise, any keys in the merged dictionary that do not correspond to expected arguments 
    of the callable object are removed, and a warning is printed if any such keys are found. 
    Finally, the merged dictionary is passed to the callable object during instantiation
    '''
    return instantiate(
            lookup(kind, config),
            config,
            unused_keys + ('name',),
            **kwargs)


def instantiate(callable, config, unused_keys=(), **kwargs):
    '''
    takes four arguments, callable, config, unused_keys, and kwargs, and 
    returns the result of calling the callable with the merged config and kwargs dictionaries. 
    This function is called by construct to actually instantiate the callable object with the merged configuration.
    It raises an error if any positional or variable positional arguments are found, 
    and handles any unexpected keys in the merged dictionary by removing them and printing a warning.
    '''
    merged = {**config, **kwargs}
    signature = None
    try:
        signature = inspect.signature(callable)
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
                raise ValueError(f'Unsupported kind for param {name}: {param.kind}')
    except:
        signature = inspect.signature(callable.__init__)
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
                raise ValueError(f'Unsupported kind for param {name}: {param.kind}')
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable(**merged)

    missing = {}
    for key in list(merged.keys()):
        if key not in signature.parameters:
            if key not in unused_keys:
                missing[key] = merged[key]
            merged.pop(key)
    if missing:
        print(f'WARNING {callable}: superfluous {missing}', file=sys.stderr)
    return callable(**merged)
