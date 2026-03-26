# -*- coding: utf-8 -*-
# file: mixing.py

# This code is part of Onça-pintada.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.seixas@proton.me> 
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

# To calculate the enthalpy of mixing based on DSI model.

class MixingModel:
    '''Base class for mixing models of binary alloys.'''
    def __init__(self):
        pass
    
    def enthalpy_of_mixing(self):
        '''Calculate the enthalpy of mixing.'''
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def entropy_of_mixing(self):
        '''Calculate the entropy of mixing.'''
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def free_energy_of_mixing(self):
        '''Calculate the free energy of mixing.'''
        raise NotImplementedError("This method should be implemented by subclasses.")