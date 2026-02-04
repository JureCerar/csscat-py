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
This packages helps with calculating Small-Angle X-Ray Scattering (SAXS) 
from molecular simulation data using the Complemented System Approach.

For more information see:

A. Lajovic, M. Tomsic, A. Jamnik: The complemented system approach:
A novel method for calculating the x-ray scattering from computer simulations
Journal of Chemical Physics, 2010, 133, 174123
DOI: https://dx.doi.org/10.1063/1.3502683
"""

from .csscat import CSScat
from .formfactor import formfactors

__author__ = "Jure Cerar"
__copyright__ = "Copyright (C) 2025-2026 Jure Cerar"
__license__ = "GNU GPL v3.0"
__version__ = "0.8.14"
__all__ = [
    "CSScat",
    "formfactors",
]



