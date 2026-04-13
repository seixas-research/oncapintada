# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Onça-pintada. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
warnings.filterwarnings("ignore")

import os
import platform
from socket import gethostname
from sys import version as __python_version__
from sys import executable as __python_executable__
from ase import __version__ as __ase_version__
from ase import __file__ as __ase_file__
from numpy import __version__ as __numpy_version__
from numpy import __file__ as __numpy_file__
from pandas import __version__ as __pandas_version__
from pandas import __file__ as __pandas_file__
from pytest import __version__ as __pytest_version__
from pytest import __file__ as __pytest_file__
from yaml import __version__ as __yaml_version__
from yaml import __file__ as __yaml_file__

from ase.parallel import parprint as print

from ._version import __version__

# from .bonds_counter import BondsCounter
# from .bonds_model import BondsModel
# from .disordered_alloy import DisorderedAlloyGenerator, DisorderedAlloyConfig
# from .phase_diagram import PhaseDiagram
# from .qca import QCABinary
# from .subregular_model import BinaryAlloy, MultiComponentAlloy



def banner():
    print("                                           ")
    print("          ▄▖          ▘  ▗    ▌            ")
    print("          ▌▌▛▌▛▘▀▌▄▖▛▌▌▛▌▜▘▀▌▛▌▀▌          ")
    print("          ▙▌▌▌▙▖█▌  ▙▌▌▌▌▐▖█▌▙▌█▌          ")
    print("                    ▌                      ")
    print("                                           ")
    print(f"    version: {__version__}                 ")
    print("    developed by: Leandro Seixas Rocha       ")
    print("    homepage: https://github.com/seixas-research/oncapintada")
    print("                                                  ")
    print("------------------------------------------------------------")
    print("                                                  ")
    print("System:")
    print(f"├── architecture: {platform.machine()}")
    print(f"├── platform: {platform.system()}")
    print(f"├── user: {os.environ['USER']}")
    print(f"├── hostname: {gethostname()}")
    print(f"├── cwd: {os.getcwd()}")
    print(f"└── PID: {os.getpid()}")
    print("                                               ")
    print("Python:")
    print(f"├── version: {__python_version__}      ")
    print(f"└── executable: {__python_executable__}      ")
    print("                                               ")
    print("Dependencies:")
    print(f"├── ase version: {__ase_version__}    [{__ase_file__[:-11]}]")
    print(f"├── numpy version: {__numpy_version__}    [{__numpy_file__[:-11]}]")
    print(f"├── pandas version: {__pandas_version__}    [{__pandas_file__[:-11]}]")
    print(f"├── pytest version: {__pytest_version__}    [{__pytest_file__[:-11]}]")
    # print(f"├── yaml version: {__yaml_version__}    [{__yaml_file__[:-11]}]")
    print("                                               ")


banner()

# if __name__ == "__main__":
    # banner()