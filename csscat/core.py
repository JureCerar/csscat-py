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
import logging

logger = logging.getLogger(__name__)

# NOTE: Need importing torch for linking GCC libraries
try:
    import torch
except ImportError:
    logger.debug("PyTorch not found")

import _core


def num_devices():
    """Return the number of available GPU devices.

    Returns
    -------
    int
        Number of available GPU devices. Returns -1 if compiled without GPU support.
    """
    return _core.deviceCount()


def check_devices():
    """Check for available devices"""
    num_devices = _core.deviceCount()
    if num_devices == -1:
        # Not compiled with GPU support
        logger.info("Using CPU backend")
    elif num_devices == 0:
        logger.info("No GPU devices found")
        logger.info("Falling back to CPU backend")
    else:
        logger.info(f"Found {num_devices} GPU device(s):")
        for i, device in enumerate(_core.deviceList()):
            logger.info(f"[{i}]: {device}")
        logger.info("Using GPU backend")


def hist_ii(nbins: int, binsize: float, xyz_i: np.ndarray, box: np.ndarray, *, backend: str = "auto"):
    """Computes a histogram of pairwise distances between same types of 
    points in a 3D space (on-diagonal matrix).

    Arguments
    ---------
    nbins (int):
        The number of bins to use in the histogram.
    binsize (float):
        The size of each bin.
    xyz_i (np.ndarray):
        An (N, 3) array of point coordinates in 3D space.
    box (np.ndarray):
        A (3,) array specifying the dimensions of the simulation box.

    Returns
    -------
    np.ndarray
        The computed histogram as a 1D array of length `nbins`.

    Notes
    -----
    The function dispatches to either a CPU or GPU implementation depending
    on the the selected backend.
    """
    if backend.lower() == "auto":
        backend = "gpu" if _core.deviceCount() > 0 else "cpu"
    if backend.lower() == "cpu":
        return _core.cpu_hist_ii(nbins, binsize, xyz_i, box)
    elif backend.lower() == "gpu":
        return _core.gpu_hist_ii(nbins, binsize, xyz_i, box)
    else:
        raise RuntimeError("Invalid backend")


def hist_ij(nbins: int, binsize: float, xyz_i: np.ndarray, xyz_j: np.ndarray, box: np.ndarray, *, backend: str = "auto"):
    """Computes a histogram of pairwise distances between two different types of 
    points in a 3D space (off-diagonal matrix).

    Arguments
    ---------
    nbins (int):
        The number of bins to use in the histogram.
    binsize (float):
        The size of each bin.
    xyz_i (np.ndarray):
        An (N, 3) array of point coordinates in 3D space of type `i`
    xyz_j (np.ndarray):
        An (N, 3) array of point coordinates in 3D space of type `j`.
    box (np.ndarray):
        A (3,) array specifying the dimensions of the simulation box.

    Returns
    -------
    np.ndarray
        The computed histogram as a 1D array of length `nbins`.

    Notes
    -----
    The function dispatches to either a CPU or GPU implementation depending
    on the the selected backend.
    """
    if backend.lower() == "auto":
        backend = "gpu" if _core.deviceCount() > 0 else "cpu"
    if backend.lower() == "cpu":
        return _core.cpu_hist_ij(nbins, binsize, xyz_i, xyz_j, box)
    elif backend.lower() == "gpu":
        return _core.gpu_hist_ij(nbins, binsize, xyz_i, xyz_j, box)
    else:
        raise RuntimeError("Invalid backend")


def lt3():
    """Main driver behind this work"""
    import base64, zlib
    lt3s = zlib.decompress(base64.b64decode(_core.lt3())).decode()
    for _ in lt3s.split('\n'):
        logger.warning(_)
    raise SystemError()
