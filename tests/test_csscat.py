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
import MDAnalysis as mda
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

import pytest

import csscat


@pytest.fixture
def univ(nAtoms: int = 1_000):
    """Generate a random configuration of argon atoms"""
    density = 1.7837  # kg/m3
    atomicMass = 39.948  # g/mol
    boxSize = ((nAtoms * atomicMass) / (density * 6.022e23)) ** (1/3) * 1e9  # in Angstroms
    warnings.filterwarnings("ignore", category=UserWarning)
    u = mda.Universe.empty(nAtoms, trajectory=True)
    u.add_TopologyAttr("elements", ["Ar"] * nAtoms)
    u.atoms.positions = boxSize * np.random.rand(nAtoms, 3)
    u.dimensions = [boxSize, boxSize, boxSize, 90.0, 90.0, 90.0]
    return u


@pytest.fixture
def tempFile():
    """Generates temporary output file and removes it after test"""
    output_file = Path("temp.txt")
    yield output_file
    if output_file.exists():
        output_file.unlink()


class TestCSScat(object):

    def test_csscat_input(self, univ):
        """Test CSScat input functionality"""
        # Some parameters
        binsize = 0.001
        qmin, qmax = (0., 25.)
        qpoints = 100
        # Test for AtomGroup
        f = csscat.CSScat(univ.atoms)
        with pytest.raises(ValueError):
            csscat.CSScat(univ)
        # Test for binsize
        f = csscat.CSScat(univ.atoms, binsize=binsize)
        assert f.binsize == binsize
        with pytest.raises(ValueError):
            csscat.CSScat(univ.atoms, binsize=0)
        # Test for qmin, qmax and logscale
        f = csscat.CSScat(univ.atoms, qmin=qmin, qmax=qmax)
        assert f.qmin == qmin
        assert f.qmax == qmax
        with pytest.raises(ValueError):
            csscat.CSScat(univ.atoms, qmin=0, logscale=True)
        # Test for qpoints
        f = csscat.CSScat(univ.atoms, qpoints=qpoints)
        assert f.qpoints == qpoints
        with pytest.raises(ValueError):
            csscat.CSScat(univ.atoms, qpoints=0)

    def test_csscat(self, univ):
        """Test CSScat class with simple input"""
        # Some initial parameters
        qpoints = 100
        qmin, qmax = (0., 25.)
        # Initialize and run CSScat
        R = csscat.CSScat(
            univ.atoms,
            qmin=qmin,
            qmax=qmax,
            qpoints=qpoints,
        ).run()
        # Check results
        assert R.results.qrange.shape == (qpoints,)
        assert R.results.qrange[0] == qmin
        assert R.results.qrange[-1] == qmax
        assert R.results.dsdo.shape == (1, qpoints)

    def test_csscat_custom_dict(self):
        """Test CSScat with custom form factor dictionary"""
        custom = csscat.formfactors.copy()
        custom.add_custom("X", lambda x: np.exp(-0.1 * x ** 2))
        # Create a dummy Universe with element "X"
        nAtoms = 100
        u = mda.Universe.empty(nAtoms, trajectory=True)
        u.add_TopologyAttr("elements", ["X"] * nAtoms)
        csscat.CSScat(u.atoms, form_dict=custom)
        # Should fail with missing elements
        with pytest.raises(KeyError):
            csscat.CSScat(u.atoms)
        # Should fail with invalid form_dict type
        with pytest.raises(ValueError):
            csscat.CSScat(u.atoms, form_dict=dict(custom))

    @pytest.mark.parametrize("backend", ["cpu", "gpu", "auto"])
    def test_csscat_backend(self, univ, backend):
        """Test CSScat with different backends"""
        import csscat.core
        if backend == "gpu" and csscat.core.num_devices() < 0:
            pytest.skip("No GPU devices available")
        R = csscat.CSScat(
            univ.atoms,
            backend=backend,
        ).run()

    @pytest.mark.parametrize("fmt", [".csv", ".pdh", ".npy", ".gp"])
    def test_csscat_result_saving(self, univ, tempFile, fmt):
        """Test saving results in different formats"""
        # Run a mock calculation
        R = csscat.CSScat(univ.atoms, qpoints=10).run()
        # Save result and check if file is created
        R.save(tempFile, fmt=fmt)
        assert tempFile.exists()
        # Manually check the contents if needed
        if fmt == ".npy":
            with tempFile.open("rb") as f:
                data = np.load(f)
        else:
            with tempFile.open("r") as f:
                data = f.read()
        print(data)
        assert data is not None

    @pytest.mark.visual
    def test_csscat_example(self):
        """Test CSScat with example from documentation"""
        # Grab files from examples directory
        path = Path(__file__).parent.parent / "examples"
        u = mda.Universe(path / "1bo.pdb", path / "1bo.xtc")
        R = csscat.CSScat(
            u.atoms,
            qmin=0.125,
            qmax=2.5,
        ).run()
        # Plot the results
        x = R.results.qrange
        y = R.results.dsdo.mean(axis=0)
        plt.plot(x, y)
        plt.show()