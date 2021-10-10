import numpy as np
import sympy as sp
from copy import copy
from typing import Any, get_type_hints, Dict
from inspect import getclosurevars, getsource, getargs
import ast
from ast import parse, get_source_segment

from .numpy import NumPy
from .torch import torch_defs


defines = {}
defines.update(torch_defs)


def check_type(item, target):
    assert item == target


def exec_lines(source: str, body, loc: Dict[str, Any], glob: Dict[str, Any], ret: Any):
    def get_value(v):
        if isinstance(v, ast.BinOp):
            a = get_value(v.left)
            b = get_value(v.right)
            return a
        elif isinstance(v, ast.Name):
            return loc.get(v.id)
        elif isinstance(v, ast.Call):
            args = [get_value(a) for a in v.args]
            func = loc.get(v.func.id, None) or glob.get(v.func.id, None)
            return func(*args)
        elif isinstance(v, ast.List):
            return [get_value(e) for e in v.elts]
        elif isinstance(v, ast.Constant):
            return v.value
        seg = get_source_segment(source, v)
        return eval(seg, glob, loc)

    for line in body:
        if isinstance(line, ast.Return):
            value = get_value(line.value)
            check_type(value, ret)
        elif isinstance(line, ast.If):
            loc1, loc2 = copy(loc), copy(loc)
            exec_lines(source, line.body, loc1, glob, ret)
            exec_lines(source, line.orelse, loc2, glob, ret)
        elif isinstance(line, ast.Assign):
            value = get_value(line.value)
            t = line.targets
        else:
            exec(get_source_segment(source, line), glob, loc)


def check(func):
    args = getargs(func.__code__)
    hints = get_type_hints(func)
    cv = getclosurevars(func)
    loc_vars = {n: Any for n in args.args}
    ret = hints.pop('return') if 'return' in hints else None
    loc_vars.update(hints)
    glob_vars = {}
    for k, v in cv.globals.items():
        if v is np:
            glob_vars[k] = NumPy()
        else:
            glob_vars[k] = defines.get(v, None) or v
    source = getsource(func)
    f_ast = parse(source).body[0]
    body = f_ast.body
    exec_lines(source, body, loc_vars, glob_vars, ret)
    defines[func] = 1
    return func
