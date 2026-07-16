from importlib import import_module

from .expr import *
# from SimFHE.SimFHE_HEVM import run as simulate 


_RUNNER_EXPORTS = {
    "HEVM", "reinit_lw", "run_hardware", "run_library", "setLibnHW"
}


def __getattr__(name):
    if name in _RUNNER_EXPORTS:
        return getattr(import_module(".runner", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ ="0.0.1"
__all__ = ["setMain","Plain", "Model", "sigmoid", "sqrt", "inverse" ,
        "sum" , "mean", "variance", "func", "compile", 
        "dump", "loadModule", "getFunctionInfo","loadContext",
        "encrypt", "decrypt", "toggleDebug", "precision_cast",
        "PlainMat", "reduce", "BackendType", "setBound", "load_mlir", "pprint",
        "hecate_dir", "removeCtxt", "Empty", "bootstrap", "save", "SimFHE"
        ]
