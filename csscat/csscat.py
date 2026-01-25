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

from types import SimpleNamespace
from pathlib import Path
import logging

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core import groups

import numpy as np

from . import formfactor


logger = logging.getLogger(__name__)


RE2 = 0.28179403227 ** 2
"""float: Classic electron radius squared [m²]."""


class CSScat(AnalysisBase):
    """
    A class for computing small-angle X-ray scattering (SAXS) intensities from atomic coordinates
    using the `complemented system scattering`_. Atomic form factors are applied based on element types and formal
    charges that are defined in the `formfactor` module.

    Arguments
    ---------
    ag : groups.AtomGroup
        The AtomGroup instance containing atomic coordinates and properties.
    binsize :  float, optional
        The bin size for distance histogramming [Å], must be positive.
    qmin : float, optional
        Minimum value of the q-vector range [Å⁻¹].
    qmax : float, optional
        Maximum value of the q-vector range [Å⁻¹].
    qpoints : int, optional
        Number of points in the q-vector scale, must be at least 1
    logscale : bool, optional
        Use logarithmic spacing for the q-vector scale; otherwise, use linear spacing (default).
    form_dict : dict, optional
        Optional dictionary of form factors to override or extend the defaults.
    **kwargs :
        Additional keyword arguments passed to the parent class initializer.

    Notes
    -----
    The `AtomGroup` must have an `elements` attribute, and optionally a `formalcharges`
    attribute. The form factor dictionary is constructed by updating the default form factors
    with any user-provided values.

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> import csscat
    >>> u = mda.Universe("topology.pdb", "trajectory.dcd")
    >>> R = csscat.CSScat(
    ...     u.atoms,
    ...     binsize=0.002, # [Å]
    ...     qmin=0.0, # [Å⁻¹]
    ...     qmax=2.5, # [Å⁻¹]
    ...     qpoints=251,
    ... ).run()
    >>> qrange = R.results.qrange
    >>> dsdo = R.results.dsdo.mean(axis=0)
    >>> R.save("output.pdh")  # Save results in PDH format

    .. _complemented system scattering:
        https://doi.org/10.1107/S1600576714009366
    """
    # NOTE: Parallelization is already done within the analysis core functions 
    _analysis_algorithm_is_parallelizable = False

    @classmethod
    def get_supported_backends(cls):
        """Return a tuple of supported backends for this analysis"""
        return ("serial",)

    def __init__(self, ag, binsize: float = 0.002, qmin: float = 0.0, qmax: float = 2.5, qpoints: int = 251,
                 logscale: bool = False, form_dict: dict = None, *, backend: str = "auto", **kwargs):
        # HACK: Delay core import to save startup time
        from . import core

        super(CSScat, self).__init__(ag.universe.trajectory, **kwargs)

        # Check input
        if not isinstance(ag, groups.AtomGroup):
            raise ValueError("Invalid AtomGroup instance")
        if form_dict and not isinstance(form_dict, formfactor.FormDict):
            raise ValueError("Invalid FormDict instance")
        if binsize <= 0.0:
            raise ValueError("Invalid binsize")
        if qpoints < 1:
            raise ValueError("Invalid number of q-vector points")
        if backend.lower() not in ["cpu", "gpu", "auto"]:
            raise ValueError("Invalid backend")

        # Set locals
        self.ag = ag
        self.binsize = float(binsize)
        self.qmin, self.qmax = float(qmin), float(qmax)
        self.qpoints = int(qpoints)
        self.logscale = bool(logscale)
        self.backend = backend

        # Check for available devices
        core.check_devices()

        # Local form-factor dictionary and update with user provided one
        formfactors = formfactor.FormDict()
        formfactors.update(formfactor.formfactors)
        if form_dict:
            formfactors.update(form_dict)

        # Construct q-vector scale
        if logscale:
            self.qrange = np.geomspace(qmin, qmax, qpoints, endpoint=True)
        else:
            self.qrange = np.linspace(qmin, qmax, qpoints, endpoint=True)

        # AtomGroup must contain elements where formal charge is optional
        attr = ["elements"]
        if not hasattr(self.ag.atoms, "elements"):
            raise ValueError("AtomGroup has no 'elements' property")
        if hasattr(self.ag.atoms, "formalcharges"):
            attr.append("formalcharges")
        
        # Structure groups and calculate atomic-form factors
        logger.info(f"Grouping by: {attr}")
        self.groups = list()
        for key, val in ag.groupby(attr).items():
            # Construct name from element and charge
            elem, *charge = key + (0,)
            charge = str(charge[0])[::-1] if charge[0] else ""
            name = elem + charge
            if name not in formfactors:
                raise KeyError(f"Unsupported atom type: {name}")
            # Construct a group
            logger.info(f"Group: {name} ({val.n_atoms})")
            self.groups.append(
                SimpleNamespace(
                    name=name,
                    ag=val,
                    aff=formfactors[name].form(self.qrange)
                )
            )

    def _prepare(self):
        """Initialize data structures before analysis."""
        self.results.dsdo = list()
        self.results.qrange = self.qrange

    def _single_frame(self):
        """Compute scattering intensity for a single frame."""
        # HACK: Delay core import to save startup time
        from . import core

        dsdo = np.zeros_like(self.qrange)
        box = np.array(self.ag.universe.dimensions[0:3])
        if self.qmin < 4 * np.pi / box.min():
            logger.warning("qmin is smaller than lowest valid q-vector")
        nbins = np.ceil(np.linalg.norm(box / 2) / self.binsize).astype(int)

        # Calculate scattering for each group combinations
        indices = np.triu_indices(len(self.groups))
        for i, j in zip(*indices):
            logger.info(f"Calculating scattering for: {i}-{j}")
            if i == j:
                hist = core.hist_ii(
                    nbins,
                    self.binsize,
                    self.groups[i].ag.positions,
                    box,
                    backend=self.backend,
                )
            else:
                hist = core.hist_ij(
                    nbins,
                    self.binsize,
                    self.groups[i].ag.positions,
                    self.groups[j].ag.positions,
                    box,
                    backend=self.backend,
                )

            # Calculate longest valid distance (cutoff) and apply the Heaviside
            # function aka. cut the distance histogram to max valid distance
            cutoff = (np.min(box / 2) // self.binsize) * self.binsize
            imax = np.floor(cutoff / self.binsize).astype(int)
            hist = hist[:imax]

            # Calculating cross-scattering term
            logger.info("Calculating cross-scattering term")
            r = np.arange(len(hist)) * self.binsize + self.binsize / 2
            qr = np.sinc(np.outer(r, self.qrange) / np.pi)
            qrh = np.sum(qr * hist[:, None], axis=0)
            dsdo += 2 * self.groups[i].aff * self.groups[j].aff * qrh

        # Calculate self scattering terms
        logger.info("Calculating self-scattering term")
        dsdo += np.sum([g.ag.n_atoms * g.aff ** 2 for g in self.groups], axis=0)

        # Calculating complemented-scattering term:
        # NOTE: At q = 0 expression evaluates to:
        # $$ \frac {4} {3} \pi r_c^3 $$
        logger.info("Calculating complemented-scattering term")
        volume = np.prod(box)
        cplt = np.sum([g.ag.n_atoms * g.aff for g in self.groups], axis=0)
        idx = (self.qrange != 0)
        qrc = self.qrange[idx] * cutoff
        dsdo[~idx] -= cplt[~idx] ** 2 / volume * (4 * np.pi * cutoff ** 3 / 3)
        dsdo[idx] -= cplt[idx] ** 2 / volume * (4 * np.pi / self.qrange[idx] ** 3) * \
                     (np.sin(qrc) - qrc * np.cos(qrc))

        # Normalize to scattering intensity
        dsdo *= RE2 / volume

        # Add result to array
        self.results.dsdo.append(dsdo)

    def _conclude(self):
        """Finalize analysis after all frames have been processed."""
        self.results.dsdo = np.array(self.results.dsdo)

    def save(self, fname: str, fmt: str = None):
        """Exports the computed results to a file in the specified format.

        Arguments
        ---------
        fname : str or Path
            The output file path where the results will be written.
        fmt : str, optional
            The file format to use for output. Format is inferred from
            the file extension if not provided. Supported formats are:
            * ".pdh": Primary Data Handling format
            * ".gp": GNUPlot format
            * ".npy": NumPy binary format
            * ".csv": CSV (Comma Separated Values) format (this is default).  
        """
        logger.info(f"Writing output to: {fname}")

        # Check inputs
        if getattr(self.results, "dsdo", None) is None:
            raise RuntimeError("No results to write")

        X = np.vstack((
            self.results.qrange,
            self.results.dsdo.mean(axis=0),
            self.results.dsdo.std(axis=0),
        ))

        # Get file format
        if not fmt:
            fmt = Path(fname).suffix

        match fmt.lower():
            case ".pdh":
                # Write in PDH (Primary Data Handling) format
                def formatter(fmt, iterable, sep=" "):
                    formatted = [fmt.format(_) for _ in iterable]
                    return sep.join(formatted)

                with open(fname, "w") as f:
                    print(f"{'Calculated with CSScat-py':80}", file=f)
                    print(formatter("{:4}", ["SAXS", "CALC"] + [""] * 14), file=f)
                    print(formatter("{:9}", [X.shape[1], 0, 0, 0, 0, 0, 0, 0]), file=f)
                    print(formatter("{:14.6e}", [0., 0., 0., 1., 0.]), file=f)
                    print(formatter("{:14.6e}", [0., 0., 0., 0., 0.]), file=f)
                    for line in X.T:
                        print(formatter("{:14.6e}", line), file=f)

            case ".gp":
                # Write in GP (GNUPlot) format
                colors = [ "#444444", "#F8F4C0"]
                with open(fname, "w") as f:
                    print("# Calculated with CSScat-py", file=f)
                    print("$DATA << END", file=f)
                    for line in X.T:
                        print(*line, file=f)
                    print("END", file=f)
                    print(f"LINECOLOR='{colors[0]}'", file=f)
                    print(f"FILLCOLOR='{colors[1]}'", file=f)
                    print("set xlabel 'q [Å⁻¹]' enhanced", file=f)
                    print("set ylabel 'd{/Symbol S}(q)/d{/Symbol W} [cm⁻¹]' enhanced", file=f)
                    print("set grid linestyle 1 linecolor 'gray' linewidth 1", file=f)
                    print("plot $DATA using 1: ($2 +$3): ($2 -$3) with filledcurve "
                          "title 'stdev' fillcolor rgb FILLCOLOR, \\", file=f)
                    print("'' using 1:2 with line notitle linecolor rgb LINECOLOR "
                        "linewidth 1.5", file=f)
            
            case ".npy":
                # Write in NPY (NumPy binary) format
                with open(fname, "wb") as f:
                    np.save(f, X)

            case _:
                # Write in CSV (Comma Separated Values) format
                with open(fname, "w") as f:
                    print("q[Å⁻¹]", "dS/dO[cm⁻¹]", "stdev", sep=",", file=f)
                    for line in X.T:
                        print(*line, sep=",", file=f)
