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

import numpy as np
import matplotlib.pyplot as plt

import pytest

import csscat
import csscat.formfactor


@pytest.fixture
def qrange(qmin: float = 0.0, qmax: float = 8 * np.pi, qpoints: int = 1_000):
    """Construct a q-vector range for formfactor calculations"""
    return np.linspace(qmin, qmax, qpoints, endpoint=True)


class TestFormFactor(object):

    def test_formfactors(self):
        """Test if formfactors loads correctly from JSON"""
        formfactors = csscat.formfactors
        assert formfactors is not None
        for k, v in formfactors.items():
            print(k, v)

    def test_formfactors_atom(self, qrange):
        """Test Atom class functionality"""
        # Try different initializers
        csscat.formfactor.Atom()
        csscat.formfactor.Atom(element="C")
        csscat.formfactor.Atom(z=1)
        csscat.formfactor.Atom(method="UNK")
        csscat.formfactor.Atom(a=[1], b=[1])
        # A and B length mismatch should raise error
        with pytest.raises(ValueError):
            csscat.formfactor.Atom(a=[1, 1], b=[1])
        with pytest.raises(ValueError):
            csscat.formfactor.Atom(a=[1], b=[1, 1])
        csscat.formfactor.Atom(c=0)
        # Test form factor calculation
        atom = csscat.formfactors["C"]
        fq = atom.form(qrange)
        assert fq.shape == qrange.shape
        assert np.all(fq >= 0.0)
        # Should fail for negative numbers
        with pytest.raises(ValueError):
            atom.form(np.array([-1.0]))
        # TODO: Currently not implemented for large q-vector values
        with pytest.raises(NotImplementedError):
            atom.form(np.array([9 * np.pi]))


    def test_formfactor_pseudo(self, qrange):
        """Test Pseudo class functionality"""
        # Example of CO molecule definition
        form_dict = csscat.formfactors
        elements = ["C", "O"]
        xyz = [
            [0., 0., 0.],
            [1.128, 0., 0.],
        ]
        # Try different initializers
        csscat.formfactor.Pseudo(elements, xyz, form_dict=form_dict)
        with pytest.raises(ValueError):
            csscat.formfactor.Pseudo(elements, xyz[:1], form_dict=form_dict)
        with pytest.raises(ValueError):
            csscat.formfactor.Pseudo(elements[:1], xyz, form_dict=form_dict)
        # Test form factor calculation
        pseudo = csscat.formfactor.Pseudo(elements, xyz, form_dict=form_dict)
        fq = pseudo.form(qrange)
        assert fq.shape == qrange.shape
        assert np.all(fq >= 0.0)


    def test_formfactor_custom(self, qrange):
        """Test Custom class functionality"""
        # Example of dummy function
        def dummy(x: np.ndarray) -> np.ndarray: 
            return np.exp(0.1 * x ** 2)
        # Try different initializers
        csscat.formfactor.Custom(dummy)
        # Fail from non-callable input
        with pytest.raises(TypeError):
            csscat.formfactor.Custom("not a function")
        # Fail for functions with wrong signature
        with pytest.raises(ValueError):
            csscat.formfactor.Custom(lambda x, y: x + y)
        # Test form factor calculation
        custom = csscat.formfactor.Custom(dummy)
        fq = custom.form(qrange)
        assert fq.shape == qrange.shape
        assert np.all(fq >= 0.0)
        assert np.allclose(fq, dummy(qrange))


    def test_formfactors_modification(self):
        """Test modification of formfactors dictionary"""
        # Try adding custom classes to dictionary
        formfactors = csscat.formfactors.copy()
        assert isinstance(formfactors, csscat.formfactor.FormDict)
        formfactors["X"] = csscat.formfactor.Atom("X", a=[1.0], b=[1.0], c=0.0)
        formfactors["Y"] = csscat.formfactor.Pseudo(["C", "H"], [[0, 0, 0], [0, 0, 1]], form_dict=formfactors)
        formfactors["Z"] = csscat.formfactor.Custom(lambda x: 2.0)
        assert len(formfactors) == len(csscat.formfactors) + 3
        assert "X" in formfactors
        assert "Y" in formfactors
        assert "Z" in formfactors
        # Try adding form factors intended way
        formfactors = csscat.formfactors.copy()
        formfactors.add_atom("X", a=[1.0], b=[1.0], c=0.0)
        formfactors.add_pseudo("Y", ["C", "H"], [[0, 0, 0], [0, 0, 1]])
        formfactors.add_custom("Z", lambda x: 2.0)
        assert len(formfactors) == len(csscat.formfactors) + 3
        assert "X" in formfactors
        assert "Y" in formfactors
        assert "Z" in formfactors


    @pytest.mark.visual
    def test_formfactor_plots(self, qrange):
        """Plot form factors for visual inspection"""
        # Plot for selected subset
        plt.subplot(1, 2, 1)
        for k in ["H", "C", "CH", "CH2", "CH3"]:
            plt.plot(qrange, csscat.formfactors[k].form(qrange), label=k)
        plt.legend()
        # Plot for all elements
        plt.subplot(1, 2, 2)
        for k in csscat.formfactors:
            plt.plot(qrange, csscat.formfactors[k].form(qrange), label=k)
        plt.legend()
        plt.show()
