/**
 * Copyright (C) 2025 Jure Cerar
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <list>

namespace py = pybind11;


/**
 * Computes a histogram of pairwise distances between same types of points
 *
 * @param nbins The number of equal-width bins in the given range
 * @param binsize Width on a histogram bin
 * @param xyz_i Container of 3D points
 * @param box Container of box dimensions
 * @return The values of the histogram
 */
py::array_t<int> cpu_hist_ii(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> box);


/**
 * Computes a histogram of pairwise distances between two different types of points
 *
 * @param nbins The number of equal-width bins in the given range
 * @param binsize Width on a histogram bin
 * @param xyz_i Container of 3D points of type `i`
 * @param xyz_j Container of 3D points of type `j`
 * @param box Container of box dimensions
 * @return The values of the histogram
 */
py::array_t<int> cpu_hist_ij(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> xyz_j, py::array_t<float> box);


/**
 * Computes a histogram of pairwise distances between same types of points on GPU
 *
 * @param nbins The number of equal-width bins in the given range
 * @param binsize Width on a histogram bin
 * @param xyz_i Container of 3D points
 * @param box Container of box dimensions
 * @return The values of the histogram
 */
py::array_t<int> gpu_hist_ii(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> box);


/**
 * Computes a histogram of pairwise distances between two different types of points on GPU
 *
 * @param nbins The number of equal-width bins in the given range
 * @param binsize Width on a histogram bin
 * @param xyz_i Container of 3D points of type `i`
 * @param xyz_j Container of 3D points of type `j`
 * @param box Container of box dimensions
 * @return The values of the histogram
 */
py::array_t<int> gpu_hist_ij(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> xyz_j, py::array_t<float> box);


/**
 * Retrun number of cuda capable devices found on host
 * Return -1 if not compiled with CUDA support
 * 
 * @return Number of CUDA capable devices on host
 */
int deviceCount();


/**
 * List all cuda capable devices on host.
 *
 * @return List of all CUDA capable devices on host
 */
std::list<std::string> deviceList();


/**
 * Main driver behind this work
 *
 * @return const char* compressed string
 */
static const char* lt3();
