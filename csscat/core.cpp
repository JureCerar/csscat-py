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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <string>
#include <list>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <core.h>

namespace py = pybind11;


// Coordinate vector
struct float3 {
    float x, y, z;
};


/**
 * Calculate distance between two points according to minimal image convention
 *
 * @param u Coordinates of a first point
 * @param v Coordinates of a second point
 * @param box dimensions of the simulation box
 * @return distance between two points
 */
float distance(float3 u, float3 v, float3 box) {
    float3 dr;
    // Calculate distance
    dr.x = u.x - v.x;
    dr.y = u.y - v.y;
    dr.z = u.z - v.z;
    // Reduce to minimal image
    dr.x -= round(dr.x / box.x) * box.x;
    dr.y -= round(dr.y / box.y) * box.y;
    dr.z -= round(dr.z / box.z) * box.z;
    // Calculate vector distance
    return sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
}

/**
 * Computes a histogram of pairwise distances between same types of points
 *
 * @param nbins The number of equal-width bins in the given range
 * @param binsize Width on a histogram bin
 * @param xyz_i Container of 3D points
 * @param box Container of box dimensions
 * @return The values of the histogram
 */
py::array_t<int> cpu_hist_ii(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> box) {
    // Request buffer info
    py::buffer_info buf_xyz_i = xyz_i.request();
    py::buffer_info buf_box = box.request();

    // TODO: add some exception checking

    // Allocate output array
    auto hist = py::array_t<int>(nbins);
    py::buffer_info buf_hist = hist.request();

    // Initialize array
    std::memset(hist.mutable_data(), 0, nbins * sizeof(int));

    // Cast numpy to pointers
    float3 *ptr_box = reinterpret_cast<float3 *>(buf_box.ptr);
    float3 *ptr_xyz_i = reinterpret_cast<float3 *>(buf_xyz_i.ptr);
    int *ptr_hist = static_cast<int *>(buf_hist.ptr);

    // Size of the array
    size_t size_i = buf_xyz_i.shape[0];

    // Because NumPy is row-major, we can just loop over all elements
    #pragma omp parallel for reduction(+:ptr_hist[:nbins])
    for (size_t i = 0; i < size_i; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            double dist = distance(ptr_xyz_i[i], ptr_xyz_i[j], *ptr_box);
            int bin = static_cast<int>(dist / binsize);
            ptr_hist[bin]++;
        }
    }

    return hist;
}


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
py::array_t<int> cpu_hist_ij(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> xyz_j, py::array_t<float> box) {
    // Request buffer info
    py::buffer_info buf_xyz_i = xyz_i.request();
    py::buffer_info buf_xyz_j = xyz_j.request();
    py::buffer_info buf_box = box.request();

    // TODO: add some exception checking

    // Allocate output array
    auto hist = py::array_t<int>(nbins);
    py::buffer_info buf_hist = hist.request();

    // Initialize array
    std::memset(hist.mutable_data(), 0, nbins * sizeof(int));

    // Cast numpy to pointers
    float3 *ptr_box = reinterpret_cast<float3 *>(buf_box.ptr);
    float3 *ptr_xyz_i = reinterpret_cast<float3 *>(buf_xyz_i.ptr);
    float3 *ptr_xyz_j = reinterpret_cast<float3 *>(buf_xyz_j.ptr);
    int *ptr_hist = static_cast<int *>(buf_hist.ptr);

    // Size of the arrays
    size_t size_i = buf_xyz_i.shape[0];
    size_t size_j = buf_xyz_j.shape[0];

    // Because NumPy is row-major, we can just loop over all elements
    #pragma omp parallel for reduction(+:ptr_hist[:nbins])
    for (size_t i = 0; i < size_i; i++)
    {
        for (size_t j = 0; j < size_j; j++)
        {
            double dist = distance(ptr_xyz_i[i], ptr_xyz_j[j], *ptr_box);
            int bin = static_cast<int>(dist / binsize);
            ptr_hist[bin]++;
        }
    }

    return hist;
}

/**
 * Main driver behind this work
 *
 * @return const char* compressed string
 */
static const char* lt3(void) {
    return "eJzzy8xLVMhKVSgrTk1PVMguyixLVOSSjrY0tDbMVdBJsbBI0oNS0tEGuTAZoAAIlEGoJGQpiBBQBMj2AxqOLIysTj3SAgEC1JGlFBTgkugSUClMYaBEGFgQAC3+Muc=";
}

#ifndef _GPU

    /**
     * Dummy function if using non-GPU build
     */
    py::array_t<int> gpu_hist_ii(int nbins, float binsize, py::array_t<float> xyz, py::array_t<float> box) {
        throw std::runtime_error("GPU support not enabled");
    }

    /**
     * Dummy function if using non-GPU build
     */
    py::array_t<int> gpu_hist_ij(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> xyz_j, py::array_t<float> box) {
        throw std::runtime_error("GPU support not enabled");
    }

    /**
     * Retrun number of cuda capable devices found on host
     * Return -1 if not compiled with CUDA support
     *
     * @return Number of CUDA capable devices on host
     */
    int deviceCount() {
        return -1;
    }

    /**
     * Dummy function if using non-GPU build
     */
    std::list<std::string> deviceList() {
        return {};
    }

#endif

// Bind function to Python module
PYBIND11_MODULE(_core, m)
{
    m.def("cpu_hist_ii", &cpu_hist_ii);
    m.def("cpu_hist_ij", &cpu_hist_ij);
    m.def("gpu_hist_ii", &gpu_hist_ii);
    m.def("gpu_hist_ij", &gpu_hist_ij);
    m.def("deviceCount", &deviceCount);
    m.def("deviceList", &deviceList);
    m.def("lt3", &lt3);
}