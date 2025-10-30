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

"""
Calculate small-angle X-ray scattering (SAXS) intensities from atomic coordinates
using the Complemented System Scattering [1, 2].

The resulting SAXS intensities are obtained on an absolute scale as a function
of the magnitude of the scattering vector, q=4π⋅sin(Θ)/λ, where 2Θ is the
scattering angle and λ is wavelength of the radiation. Regardless of the units in
the input files, the output units are [cm⁻¹] and [Å⁻¹] for the scattering
intensity and the magnitude of scattering vector, respectively.

The atomic form factors (AFFs) are applied based on element types and formal charges
from the topology. They are calculated using the analytical expressions from the
literature [3]. A wide range of common atoms and ions (from H to Xe) are natively
supported, with form factors sourced from. For pseudo-atoms AFFs are computed 
internally using the Debye equation.

The input structure can be provided as a single frame structure file (e.g. PDB, GRO),
or as a trajectory file (e.g. XTC, DCD) together with a structure/topology file.
If a trajectory is provided, the scattering intensity is averaged over the selected
frames. The user can specify the range of frames to be read from the trajectory,
as well as the step between frames. The output is written to a file in one of the
supported formats (PDH, CSV, NumPy binary, or gnuplot). The output file contains 
three columns: the scattering vector values, the scattering intensity values, 
and the standard deviation values (for trajectory input).

For more information see:

[1] A. Lajovic, M. Tomsic, A. Jamnik: The complemented system approach:
A novel method for calculating the x-ray scattering from computer simulations
Journal of Chemical Physics, 2010, 133, 174123
DOI: https://dx.doi.org/10.1063/1.3502683

[2] J. Cerar, A. Jamnik, I. Pethes, L. Temleitner, L. Pusztai, M. Tomsic:
Structural, rheological and dynamic aspects of hydrogen-bonding molecular liquids
Journal of Colloid and Interface Science, 2020, 560, 730-742
DOI: https://doi.org/10.1016/j.jcis.2019.10.094

[3] P.J. Brown, et al.: Intensity of diffracted intensities
International Tables for Crystallography Volume C, 2004, 554-595
DOI: https://doi.org/10.1107/97809553602060000600
"""

import argparse
import logging
import warnings

import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar

from . import csscat
from . import __version__

logger = logging.getLogger(__name__)


class ProgressHandler(logging.Handler):
    """Custom logging handler to work with MDAnalysis progress bar."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            # Ensures logs don't overwrite progress bar
            msg = self.format(record)
            ProgressBar.write(msg)  
            self.flush()
        except Exception:
            self.handleError(record)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    """
    Custom formatter to combine ArgumentDefaultsHelpFormatter
    and RawDescriptionHelpFormatter
    """
    pass


class LT3Action(argparse.Action):
    """Trigger for LT3 mode (for internal use only)"""
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        csscat.core.lt3()


def main():
    # Setup logger
    logger = logging.getLogger()
    handler = ProgressHandler()
    formatter = logging.Formatter(
        "[%(asctime)s]:%(name)s:%(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Custom argument parser
    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class=lambda prog: CustomFormatter(
            prog,
            max_help_position=42,
        ),
        epilog=f"CSScat-py v{__version__}\n"
        "Copyright (C) 2025 Jure Cerar\n"
        "This is free software; see the source for copying conditions.  There is NO\n"
        "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n",
    )
    # Define command line arguments
    parser.add_argument("-V", "-version", action="version", version=f"CSScat-py v{__version__}",
                        help="show version information and exit")
    parser.add_argument("-v", "-verbose", dest="verbose", action="store_true",
                        help="enable verbose output")
    parser.add_argument("-lt3",  dest="lt3", action=LT3Action, nargs=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("-qpoints", dest="qpoints",  type=int, default=251,
                        help="number of points in the q-vector scale")
    parser.add_argument("-logscale", dest="logscale", action="store_true",
                        help="calculate for LOG q-scale")
    parser.add_argument("-qmin", dest="qmin", type=float, default=0.0,
                        help="minimum value of the q-vector range [Å⁻¹]")
    parser.add_argument("-qmax", dest="qmax", type=float, default=2.50,
                        help="maximum value of the q-vector range [Å⁻¹]")
    parser.add_argument("-binsize", dest="binsize", type=float, default=0.002,
                        help="bin size for distance histogramming [Å]")
    parser.add_argument("-first", dest="first", type=int, default=None,
                        help="first frame to read from trajectory")
    parser.add_argument("-last", dest="last", type=int, default=None,
                        help="last frame to read from trajectory")
    parser.add_argument("-step", dest="step", type=int, default=None,
                        help="steps between trajectory frames")
    parser.add_argument("-s", "-struct", dest="struct", type=str, required=True,
                        help="input structure or topology path")
    parser.add_argument("-f", "-traj", dest="traj", type=str, default=None,
                        help="input trajectory path")
    parser.add_argument("-o", "-output", dest="output", type=str, default="out.csv",
                        help="output file path; supported formats: csv, pdh, npy, gp")
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.INFO)

    # Ignore warnings from MDAnalysis
    warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
    warnings.filterwarnings("ignore", category=FutureWarning, module="MDAnalysis")

    # Create universe
    if args.traj:
        univ = mda.Universe(args.struct, args.traj)
    else:
        univ = mda.Universe(args.struct)

    # Guess elements from atom names if not present
    if not hasattr(univ.atoms, "elements"):
        logger.info("Guessing elements from atom names...")
        univ.guess_TopologyAttrs(to_guess=["elements"])

    # Calculate scattering
    R = csscat.CSScat(
        univ.atoms,
        args.binsize,
        args.qmin,
        args.qmax,
        args.qpoints,
        args.logscale,
        verbose=True,
    ).run(args.first, args.last, args.step)
    
    # Write output
    R.save(args.output)


if __name__ == "__main__":
    main()