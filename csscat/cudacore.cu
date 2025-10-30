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

#include <vector_types.h>

namespace py = pybind11;

// DEBUG mode
// #define _DEBUG


// GPU constant memory
const int cmem_max_bytes = 64000; // 64 kB
__constant__ char cmem_data[cmem_max_bytes];


// Macro wrapper for CUDA error checking
#define cudaCheck(__call) { \
    cudaError_t __err = __call; \
    if (__err != cudaSuccess) { \
        char buffer[1024]; \
        snprintf(buffer, sizeof(buffer), "CUDA error: %s: %d: %s", \
        __FILE__, __LINE__, cudaGetErrorString(__err)); \
        throw std::runtime_error(buffer); \
    } \
}


/**
 * Calculate distance between two points according to minimal image convention
 *
 * @param u Coordinates of a first point
 * @param v Coordinates of a second point
 * @param box dimensions of the simulation box
 * @return distance between two points
 */
__device__ __forceinline__ float distance(float3 u, float3 v, float3 box) {
    float3 dr;
    // Calculate distance
    dr.x = u.x - v.x; // 1 FLOP
    dr.y = u.y - v.y;
    dr.z = u.z - v.z;
    // Reduce to minimal image
    dr.x -= round(dr.x / box.x) * box.x; // 4 FLOP (ish?)
    dr.y -= round(dr.y / box.y) * box.y;
    dr.z -= round(dr.z / box.z) * box.z;
    // Calculate vector distance
    return sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z); // 5 FLOP
}


/**
 * CUDA kernel for computing a histogram of pairwise distances between same types of points
 *
 * @param devId Current device ID 
 * @param devCount Total number of devices
 * @param nbins The number of equal-width bins in the given range
 * @param hist The device's values of the histogram
 * @param binsize Width on a histogram bin
 * @param size_i Number of coordinates
 * @param xyz_i Array of coordinates
 * @param cmem_size Size of device constant memory tile
 * @param cmem_offset Current offset in device constant memory
 * @param box dimensions of the simulation box
 */
__global__ void kernel_ii(int devId, int devCount, int nbins, int *hist, float binsize, 
                          int size_i, float3 *xyz_i, int cmem_size, int cmem_offset, float3 box) {
    // Place in global memory based of device and block ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    i = i * devCount + devId;
    // Constant memory data
    float3 *cmem_xyz = (float3*)cmem_data;
    // Calculate the corresponding bin from minimal image distance
    if (i < size_i) {
        for (int j = 0; j < cmem_size; j++) {
            if (j < i - cmem_offset) {
                int idx = round(distance(xyz_i[i], cmem_xyz[j], box) / binsize);
                // Atomic increment the coresponding bin
                atomicAdd(&hist[idx], 1);
            }
        }
    }
    return;
}


/**
 * CUDA kernel for computing a histogram of pairwise distances between two different types of points
 *
 * @param devId Current device ID 
 * @param devCount Total number of devices
 * @param nbins The number of equal-width bins in the given range
 * @param hist The device's values of the histogram
 * @param binsize Width on a histogram bin
 * @param size_j Number of coordinates
 * @param xyz_j Array of coordinates
 * @param cmem_size Size of device constant memory tile
 * @param box dimensions of the simulation box
 */
__global__ void kernel_ij(int devId, int devCount, int nbins, int *hist, float binsize,
                          int size_j, float3 *xyz_j, int cmem_size, float3 box) {
    // Place in global memory based of device and block ID
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    j = j * devCount + devId;
    // Constant memory data
    float3 *cmem_xyz = (float3*)cmem_data;
    // Calculate the corresponding bin from minimal image distance
    if (j < size_j) {
        for (int i = 0; i < cmem_size; i++) {
            int idx = round(distance(xyz_j[j], cmem_xyz[i], box) / binsize);
            atomicAdd(&hist[idx], 1);
        }
    }
    return;
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
py::array_t<int> gpu_hist_ii(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> box) {
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

    // Size of the arrays
    size_t size_i = buf_xyz_i.shape[0];

    // Get number of devices available
    int devCount = 0;
    cudaGetDeviceCount(&devCount);

    // If multiple devices are detected, spawn a thread for each device
    #pragma omp parallel if(devCount > 1) for reduction(+:ptr_hist[:nbins]) num_threads(devCount)
    {
        // Get thread ID (which is also a device ID)
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        // Get device proterties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, tid);

        // Allocate and copy data to device
        float3 *dev_xyz_i;
        cudaCheck(cudaMalloc(&dev_xyz_i, size_i * sizeof(float3)));
        cudaCheck(cudaMemcpy(dev_xyz_i, ptr_xyz_i, size_i * sizeof(float3), cudaMemcpyHostToDevice));

        // Alloccate and initialize device histogram data
        int *dev_hist;
        cudaCheck(cudaMalloc(&dev_hist, nbins * sizeof(int)));
        cudaCheck(cudaMemset(dev_hist, 0, nbins * sizeof(int)));

        // Set block and grid dimmnensions
        dim3 dimBlock = dim3(512, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);

        // The kernel will have to iterate if the amount of work exceeds grid size
        dimGrid.x = (unsigned)ceil(double(size_i / devCount) / double(dimBlock.x));
        dimGrid.x = min(dimGrid.x, prop.maxGridSize[0]);

        // Num. of coor that will fit into constant memory tile
        const int cmem_max_tile_size = cmem_max_bytes / sizeof(float3);

        // Num. of completly filled constant memory tiles and remmaining tile size
        const int cmem_ntiles        = size_i / cmem_max_tile_size;
        const int cmem_rem_tile_size = size_i % cmem_max_tile_size;

        // DEBUG: Some generic info
        #ifdef _DEBUG
            printf("CUDA kernel launch:\n");
            printf("thread= %d dev= %d\n", tid, prop.pciDeviceID);
            printf("dimBlock={ %u %u %u }\n", dimBlock.x, dimBlock.y, dimBlock.z);
            printf("dimGrid={ %u %u %u }\n", dimGrid.x, dimGrid.y, dimGrid.z);
            printf("cmem_ntiles= %d\n", cmem_ntiles);
            printf("cmem_tile_size= %d\n", cmem_max_tile_size);
            printf("cmem_tile_rem= %d\n", cmem_rem_tile_size);
        #endif

        // Loop over constant memory tiles
        for (size_t i = 0; i <= cmem_ntiles; i++) {
            // Size of current constant memory tile in bytes
            size_t cmem_tile_size = (i < cmem_ntiles ? cmem_max_tile_size : cmem_rem_tile_size);
            size_t cmem_bytes     = cmem_tile_size * sizeof(float3);

            // Copy data to device constant memory
            int cmem_tile_offset = i * cmem_max_tile_size;
            cudaCheck(cudaMemcpyToSymbol(cmem_data, &ptr_xyz_i[cmem_tile_offset], cmem_bytes));

            // Launch cuda kernel
            kernel_ii<<<dimGrid, dimBlock>>>(tid, devCount, nbins, dev_hist, binsize, size_i, dev_xyz_i, cmem_tile_size, cmem_tile_offset, *ptr_box);
        }
        
        // Wait for all devices to finish and copy back to host
        cudaDeviceSynchronize();
        cudaCheck(cudaMemcpy(ptr_hist, dev_hist, nbins * sizeof(int), cudaMemcpyDeviceToHost));

        // Clean-up memory
        cudaCheck(cudaFree(dev_xyz_i));
        cudaCheck(cudaFree(dev_hist));
    }

    // DEBUG: Sanity check
    #ifdef _DEBUG
        long hist_sum = 0;
        for (size_t i = 0; i < nbins; i++) {
            hist_sum += ptr_hist[i];
        }
        assert(hist_sum == (long)size_i * ((long)size_i - 1) / 2);
    #endif

    // Return result
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
py::array_t<int> gpu_hist_ij(int nbins, float binsize, py::array_t<float> xyz_i, py::array_t<float> xyz_j, py::array_t<float> box) {
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

    // Get number of devices available
    int devCount = 0;
    cudaGetDeviceCount(&devCount);

    // If multiple devices are detected, spawn a thread for each device
    #pragma omp parallel if(devCount > 1) for reduction(+:ptr_hist[:nbins]) num_threads(devCount)
    {
        // Get thread ID (which is also a device ID)
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        // Get device proterties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, tid);

        // Allocate and copy data to device
        // NOTE: xyz_i data will be copied to constant memory so we only need xyz_j data 
        float3 *dev_xyz_j;
        cudaCheck(cudaMalloc(&dev_xyz_j, size_j * sizeof(float3)));
        cudaCheck(cudaMemcpy(dev_xyz_j, ptr_xyz_j, size_j * sizeof(float3), cudaMemcpyHostToDevice));

        // Alloccate and initialize device histogram data
        int *dev_hist;
        cudaCheck(cudaMalloc(&dev_hist, nbins * sizeof(int)));
        cudaCheck(cudaMemset(dev_hist, 0, nbins * sizeof(int)));

        // Set block and grid dimmnensions
        dim3 dimBlock = dim3(512, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);

        // The kernel will have to iterate if the amount of work exceeds grid size
        dimGrid.x = (unsigned)ceil(double(size_j / devCount) / double(dimBlock.x));
        dimGrid.x = min(dimGrid.x, prop.maxGridSize[0]);

        // Num. of coor that will fit into constant memory tile
        const int cmem_max_tile_size = cmem_max_bytes / sizeof(float3);

        // Num. of completly filled constant memory tiles and remmaining tile size
        const int cmem_ntiles        = size_i / cmem_max_tile_size;
        const int cmem_rem_tile_size = size_i % cmem_max_tile_size;

        // Debug info
        #ifdef _DEBUG
            printf("CUDA kernel launch:\n");
            printf("thread= %d dev= %d\n", tid, prop.pciDeviceID);
            printf("dimBlock={ %u %u %u }\n", dimBlock.x, dimBlock.y, dimBlock.z);
            printf("dimGrid={ %u %u %u }\n", dimGrid.x, dimGrid.y, dimGrid.z);
            printf("cmem_ntiles= %d\n", cmem_ntiles);
            printf("cmem_tile_size= %d\n", cmem_max_tile_size);
            printf("cmem_tile_rem= %d\n", cmem_rem_tile_size);
        #endif

        // Loop over constant memory tiles
        for (size_t i = 0; i <= cmem_ntiles; i++) {
            // Byte size of current constant memory tile
            size_t cmem_tile_size = (i < cmem_ntiles ? cmem_max_tile_size : cmem_rem_tile_size);
            size_t cmem_bytes     = cmem_tile_size * sizeof(float3);

            // Copy coor data to device constant memory
            int cmem_tile_offset = i * cmem_max_tile_size;
            cudaCheck(cudaMemcpyToSymbol(cmem_data, &ptr_xyz_i[cmem_tile_offset], cmem_bytes));

            // Launch CUDA kernel
            kernel_ij<<<dimGrid, dimBlock>>>(tid, devCount, nbins, dev_hist, binsize, size_j, dev_xyz_j, cmem_tile_size, *ptr_box);
        }
        
        // Wait for all devices to finish and copy back to host
        cudaDeviceSynchronize();
        cudaCheck(cudaMemcpy(ptr_hist, dev_hist, nbins * sizeof(int), cudaMemcpyDeviceToHost));

        // Clean-up memory
        cudaCheck(cudaFree(dev_xyz_j));
        cudaCheck(cudaFree(dev_hist));
    }

    // DEBUG: Sanity check
    #ifdef _DEBUG
        long hist_sum = 0;
        for (size_t i = 0; i < nbins; i++) {
            hist_sum += ptr_hist[i];
        }
        assert(hist_sum == (long)size_i * (long)size_j);
    #endif

    // Return result
    return hist;
}

/**
* Retrun number of cuda capable devices found on host
*
* @return Number of CUDA capable devices on host
*/
int deviceCount() {
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    return devCount;
}

/**
* List all cuda capable devices on host.
* 
* @return List of all CUDA capable devices on host
*/
std::list<std::string> deviceList()
{
    std::list<std::string> devices = {};
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if ( devCount != 0 ) {
        for (int i = 0; i < devCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            devices.push_back(prop.name);
        }
    }
    return devices;
}
