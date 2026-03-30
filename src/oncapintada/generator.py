# -*- coding: utf-8 -*-
# file: sqs.py

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

class SQSGenerator:
    '''Special Quasirandom Structure (SQS) generator for modeling disordered alloys.

    Parameters
    ----------
    structure : str
        The crystal structure of the alloy (e.g., "fcc", "bcc", "hcp").
    size : tuple
        The size of the supercell to generate (e.g., (2, 2, 2) for a 2x2x2 supercell).
    composition : dict
        A dictionary specifying the composition of the alloy (e.g., {"A": 0.5, "B": 0.5} for a 50-50 binary alloy).
    '''

    def __init__(self, structure: str, size: tuple, composition: dict):
        self.structure = structure
        self.size = size
        self.composition = composition

    def generate(self):
        '''
        Generate the SQS structure based on the specified parameters.
        '''
        raise NotImplementedError("SQS generation algorithm is not implemented yet.")
    
    
    def calculate_correlation_functions(self):
        '''
        Calculate the correlation functions for the generated SQS structure.
        '''
        raise NotImplementedError("Correlation function calculation is not implemented yet.")
    

# To generate structures with a given target correlation function using Monte Carlo sampling.
class ReverseMonteCarlo:
    def __init__(self, atoms: Optional[Atoms] = None, composition=None, target=None):
        self.atoms = atoms
        self.composition = composition
        self.target = target

# To generate SQS structures for a given composition and lattice type.
class SQS:
    def __init__(self, atoms: Optional[Atoms] = None, composition=None, method='monte_carlo'):
        self.atoms = atoms
        self.composition = composition
        self.method = method