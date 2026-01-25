# Copyright (C) 2025 Jure Cerar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path
import numpy as np
import json
import inspect
from typing import Callable
from dataclasses import dataclass, field


@dataclass
class Atom():
    """Class representing atomic form factors (AAF) as defined in the `International Tables for Crystallography`_
    The atomic form factor describes the scattering amplitude of an atom as a function of the scattering vector q.
    It is expressed as a sum of Gaussian functions plus a constant term:

    .. math::

        f(q) = \sum_{i=1}^{N} a_i \exp \left[-b_i \left( \frac{q}{4\pi} \right)^2 \right] + c

    Attributes
    ----------
    element : str
        Chemical symbol of the element (e.g., 'C' for carbon).
    z : int
        Atomic number of the element.
    charge : int
        Net charge of the atom.
    method : str
        Method or reference for the form factor parameters.
    a : list[float]
        List of 'a' coefficients for the exponential terms.
    b : list[float]
        List of 'b' coefficients for the exponential terms.
    c : float
        Constant term in the form factor equation.

    Raises
    ------
    ValueError: If the lengths of the 'a' and 'b' coefficient lists do not match.

    .. _International Tables for Crystallography:
        http://dx.doi.org/10.1107/97809553602060000600
    """
    element: str = ""
    z: int = 0
    charge: int = 0
    method: str = ""
    a: list[float] = field(default_factory=list)
    b: list[float] = field(default_factory=list)
    c: float = 0.

    def __post_init__(self):
        if len(self.a) != len(self.b):
            raise ValueError("coefficient length mismatch")

    def form(self, q: np.ndarray) -> np.ndarray:
        """Calculates the form factor for a given array of q values.

        Arguments
        ---------
        q : np.ndarray
            An array of q vector values for which the form factor is to be calculated.
            
        Returns
        -------
        : np.ndarray:
            An array containing the calculated form factor values corresponding to each input q.
        """
        # If s-vector is smaller than 2.0 1/A (or 8*pi) the AAF equations are not valid
        if np.any(q < 0):
            raise ValueError("Scattering vector must be positive numbers")
        s = q / (4 * np.pi)
        if np.any(s > 2.0):
            raise NotImplementedError()
        y = np.zeros_like(q)
        for a, b in zip(self.a, self.b):
            y += a * np.exp(-b * s ** 2)
        y += self.c
        return y


class Pseudo():
    """Class representing form factors for pseudo-atoms. For pseudo-atoms, the form factor
    is calculated using the Debye equation, which accounts for the interference effects
    between constituent atoms:

    .. math::

        f(q)^2 = \sum_{i} \sum_{j} f_i(q) f_j(q) \frac {\sin(qr_{ij})} {qr_{ij}}

    Attributes
    ----------
    elements : list[str]
        List of atom types representing the constituent atoms of the pseudo-atom.
    coords : np.ndarray 
        A 2D numpy array of shape (N, 3) containing the 3D coordinates of the constituent atoms.
    form_dict : dict
        A dictionary mapping element symbols to their corresponding Atom instances.
    """
    def __init__(self, elements: list, coords: np.ndarray, form_dict: dict[Atom]):
        self.elements = list(elements)
        self.coords = np.array(coords)
        # Check input
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must be a 2D ndarray of shape (N, 3)")
        if len(self.elements) != self.coords.shape[0]:
            raise ValueError("elements and coords must be same size")
        for e in elements:
            if e not in form_dict:
                raise ValueError(f"unknown element: {e}")
            if not isinstance(form_dict[e], Atom):
                raise ValueError("element in space is not an Atom instance")
        # Get aff functions from scope
        self._form_func = dict()
        for e in set(elements):
            self._form_func[e] = form_dict[e].form
    
    def __repr__(self):
        return f"Pseudo(elements={self.elements}, xyz={self.coords.tolist()})"

    def form(self, q: np.ndarray) -> np.ndarray:
        """Calculates the form factor for a given array of q values.

        Arguments
        ---------
        q (np.ndarray):
            An array of q vector values for which the form factor is to be calculated.
            
        Returns
        -------
        np.ndarray:
            An array containing the calculated form factor values corresponding to each input q.
        """
        y = np.zeros_like(q)
        # Calculate distance matrix
        diffs = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)
        # Calculate atomic form factors
        aff = [self._form_func[e](q) for e in self.elements]
        # Calculate pseudo atom form factor according to Debye equation
        rows, cols = np.triu_indices(len(self.elements))
        for i, j in zip(rows, cols):
            if i == j:
                y += aff[i] ** 2
            else:
                y += 2 * aff[i] * aff[j] * np.sinc(dmat[i, j] / np.pi)
        return np.sqrt(y)
    

class Custom():
    """Custom is a wrapper class for user-defined form factor functions.

    Attributes
    ----------
    expr : Callable[[float], float]
        A user-defined function that takes a single float and returns a float.
    """
    def __init__(self, expr: Callable[[float], float]): 
        # Check if callable and inspect signature
        if not callable(expr):
            raise TypeError("input must be callable")
        sig = inspect.signature(expr)
        if len(sig.parameters) != 1:
            raise ValueError("Function must take exactly one argument")
        self.expr = expr
    
    def __repr__(self):
        return f"Custom(expr={self.expr})"
    
    def form(self, q: np.ndarray) -> np.ndarray:
        """Calculates the form factor for a given array of q values.

        Arguments
        ---------
        q (np.ndarray):
            An array of q vector values for which the form factor is to be calculated.
            
        Returns
        -------
        np.ndarray:
            An array containing the calculated form factor values corresponding to each input q.
        """
        return np.vectorize(self.expr)(q)
        

class FormDict(dict):
    """A specialized dictionary to store Atom, Pseudo, and Custom form factor instances.

    Attributes
    ----------
    Inherits from dict.
    """
    def __setitem__(self, key, value):
        if not isinstance(value, (Atom, Pseudo, Custom)):
            raise TypeError("Value must be an instance of Atom, Pseudo, or Custom")
        super().__setitem__(key, value)

    def copy(self):
        return type(self)(self)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self.__setitem__(k, v)

    def add_atom(self, key, *args, **kwargs):
        """Adds a Atom instance to the dictionary. 
        """ + (Atom.__doc__ or "")
        super().__setitem__(key, Atom(*args, **kwargs))

    def add_pseudo(self, key, *args, **kwargs):
        """
        Adds a Pseudo instance to the dictionary. 
        """ + (Pseudo.__doc__ or "")
        super().__setitem__(key, Pseudo(*args, **kwargs, form_dict=self))

    def add_custom(self, key, *args, **kwargs):
        """
        Adds a Custom instance to the dictionary. 
        """ + (Custom.__doc__ or "")
        super().__setitem__(key, Custom(*args, **kwargs))
    
    def read_json(self, path):
        """Reads form factor definitions from a JSON file and populates the dictionary.

        Arguments
        ---------
        path : str or Path
            Path to the JSON file containing form factor definitions.
        """
        with open(path) as f:
            df = json.load(f)
        for k, v in df.get("atoms", {}).items():
            self.add_atom(k, **v)
        for k, v in df.get("pseudo", {}).items():
            self.add_pseudo(k, **v)
        return self


# Load values form a file
formfactors = FormDict().read_json(
    Path(__file__).parent / "formfactor.json"
)
"""
A dictionary containing predefined form factors. Keys are element symbols or
custom names, and values are instances of :class:`formfactor.Atom`,
:class:`formfactor.Pseudo`, or :class:`formfactor.Custom` classes.
"""

