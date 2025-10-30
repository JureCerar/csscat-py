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
import warnings
import MDAnalysis as mda

import pytest

import csscat
import csscat.core


@pytest.fixture
def univ(nAtoms: int = 1_000):
    """Generate a random configuration of argon atoms"""
    density = 1.7837 # kg/m3
    atomicMass = 39.948 # g/mol
    boxSize = ((nAtoms * atomicMass) / (density * 6.022e23)) ** (1/3) * 1e9  # in Angstroms
    warnings.filterwarnings("ignore", category=UserWarning)
    u = mda.Universe.empty(nAtoms, trajectory=True)
    u.add_TopologyAttr("elements", ["Ar"] * nAtoms)
    u.atoms.positions = boxSize * np.random.rand(nAtoms, 3)
    u.dimensions = [boxSize, boxSize, boxSize, 90.0, 90.0, 90.0]
    return u


class TestCore(object):


    def test_core(self):
        """Test some core functionality"""
        # Check for available devices
        num_devices = csscat.core.num_devices()
        assert isinstance(num_devices, int)
        csscat.core.check_devices()
    

    @pytest.mark.parametrize("backend", ["cpu", "gpu", "auto"])
    def test_hist_ii(self, univ, backend):
        """Test hist_ii function with simple input"""
        if backend == "gpu" and csscat.core.num_devices() < 0:
            pytest.skip("No GPU devices available")
        # Prepare parameters
        nbins = 1_000
        nAtoms = univ.atoms.n_atoms
        box = univ.dimensions[:3]
        xyz = univ.atoms.positions
        binsize = np.linalg.norm(box / 2) / nbins
        # Compute histogram
        hist = csscat.core.hist_ii(nbins, binsize, xyz, box, backend=backend)
        assert hist.shape == (nbins,)
        assert np.sum(hist) == nAtoms * (nAtoms - 1) // 2


    @pytest.mark.parametrize("backend", ["cpu", "gpu", "auto"])
    def test_hist_ij(self, univ, backend):
        """Test hist_ij function with simple input"""
        if backend == "gpu" and csscat.core.num_devices() == 0:
            pytest.skip("No GPU devices available")
        # Prepare parameters
        nbins = 1_000
        nAtoms = univ.atoms.n_atoms
        box = univ.dimensions[:3]
        xyz = univ.atoms.positions
        binsize = np.linalg.norm(box / 2) / nbins
        # Compute histogram
        hist = csscat.core.hist_ij(nbins, binsize, xyz, xyz, box, backend=backend)
        assert hist.shape == (nbins,)
        assert np.sum(hist) == nAtoms ** 2
