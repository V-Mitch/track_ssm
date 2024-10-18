# Reloader #

import importlib
import sys
import inspect

# reload a function using only its name, no matter what module it comes from
def r_f(func):
    module_object = sys.modules[func.__module__]
    importlib.reload(module_object)
    
    # Update the global function to the reloaded version
    globals()[func.__name__] = getattr(module_object, func.__name__)
  
# clear the environment of all global variables
def c_e():
  for name in dir():
    if not name.startswith("_"):  # Avoid deleting built-in variables
        del globals()[name]

# view a function
def v_f(func):
    print(inspect.getsource(func))
