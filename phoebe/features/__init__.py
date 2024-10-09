from .dataset_features import *
from .component_features import *
import inspect

def get_code_for_cls(cls, ignore=[]):
    code = inspect.getsource(cls.__init__)
    for name, member in inspect.getmembers_static(cls):
        if name in ignore:
            continue
        if name.startswith('__'):
            continue
        if isinstance(member, property):
            raise ValueError("input class cannot contain any properties (pass to ignore to skip during export)")
        if hasattr(member, '__call__') or hasattr(member, '__wrapped__'):
            code += inspect.getsource(member)
        else:
            value = getattr(cls, name)
            if isinstance(value, str):
                value = f"'{value}'"
            code += f"    {name} = {value}\n"
    return code

def get_class_from_code(code):
    code = f"class ClassFromCode:\n" + code
    exec(code)
    return locals()["ClassFromCode"]