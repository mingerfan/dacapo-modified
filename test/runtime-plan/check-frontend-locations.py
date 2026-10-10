#!/usr/bin/env python3
"""Direct frame lookup preserves the source positions used by native calls."""
import argparse
import ast
import inspect
from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--source-dir", type=Path, required=True)
args = parser.parse_args()
source = args.source_dir / "python/hecate/hecate/expr.py"
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("_frameInfo", "getCallerFrame", "getProperFrame")]
assert len(functions) == 3
namespace = {"sys": sys, "inspect": inspect}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
bridges = """
def old_binary():
    return inspect.stack()[1]
def new_binary():
    return getCallerFrame()
def old_proper():
    thisname = inspect.stack()[0].filename
    for frame in inspect.stack():
        if frame.filename != thisname:
            return frame
def new_proper():
    return getProperFrame()
def old_nested():
    return old_proper()
def new_nested():
    return new_proper()
"""
exec(compile(bridges, str(source), "exec"), namespace)
for old, new in (("old_binary", "new_binary"), ("old_proper", "new_proper"),
                 ("old_nested", "new_nested")):
    observed = []
    for method in (old, new):
        exec(compile(f"location = {method}()", "user-model.py", "exec"), namespace)
        observed.append(tuple(namespace["location"])[1:4])
    assert observed[0] == observed[1] == ("user-model.py", 1, "<module>"), observed
print("Direct and nested frontend source locations match inspect.stack")
