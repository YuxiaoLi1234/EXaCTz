#include "release_paths.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "./UnionFind.h"
#include "SZ3/api/sz.hpp"
#include "zfp.h"
#include "mgard/compress_x.hpp"
#include <cuSZp.h>
// #include <parallel/algorithm>  
// #include <omp.h>
#include <set>
#include <cfloat>
#include <bitset>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
// #include <thrust/device_vector.h>
// #include <thrust/sequence.h>
// #include <thrust/find.h>
// #include <thrust/merge.h>
// #include <thrust/sort.h>
// #include <thrust/device_ptr.h>
// #include <thrust/unique.h>
std::vector<std::vector<float>> time_counter;
std::string filename, compressor_id;
// Key 102: (51, 4)
// Key 153: (12, 4)
// Key 277: (0, 4)
// Key 413: (4, 6)
// Key 554: (63, 4)
// Key 622: (55, 6)
// Key 1050: (15, 4)
// Key 1179: (13, 6)
// Key 1594: (31, 6)
// Key 2085: (48, 4)
// Key 2151: (49, 6)
// Key 2357: (16, 6)
// Key 5462: (3, 7)
// Key 5463: (1, 8)
// Key 5470: (7, 8)
// Key 5494: (19, 8)
// Key 5599: (5, 10)
// Key 6014: (23, 10)
// Key 7543: (17, 10)
// Key 10921: (60, 7)
// Key 10923: (61, 8)
// Key 10925: (52, 8)
// Key 10937: (28, 8)
// Key 10991: (53, 10)
// Key 11197: (20, 10)
// Key 11963: (29, 10)
// Key 16383: (21, 14)

#define MAX_NEIGHBORS 14
#define MAX_V3 6 
#define NUM_LOOKUP 14 // 14 * 14 种 (v1, v2) 组合
#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE + 2)  // 这里为 10
int width_host, height_host, depth_host;
int *adjacency_host;
int nSaddle2_host = 0;
int nMin_host = 0;
int nMax_host = 0;
int nSaddle1_host = 0;
int host_wrong_min_counter = 0, host_wrong_min_counter_2 = 0, host_wrong_saddle_counter = 0, host_wrong_saddle_counter_join = 0, host_wrong_max_counter_2 = 0, host_wrong_max_counter = 0;
__device__ int table[64];
// __device__ int lookupTable[NUM_LOOKUP][NUM_LOOKUP][3];
__device__ int lookupTable[NUM_LOOKUP][MAX_V3][3];

int lookupTable_host[NUM_LOOKUP][MAX_V3][3];
int maxNeighbors_host = 14;
float additional_time, compression_time;
int vertex_types[27] = {277, 153, 1050, 1179, 413, 2085, 10921, 554, 10923, 10925, 2357, 10937, 1594, 11963, 11197, 5462, 5463, 5470, 5599, 102, 2151, 622, 10991, 5494, 7543, 6014, 16383};
__constant__ int neighborOffsets[14][3] = {
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1}, {-1,1,0}, {1,-1,0},
    {0,1,1}, {0,-1,-1}, {-1,0,1}, {1,0,-1},
    {-1,1,1}, {1,-1,-1}
};
int neighborOffsets_host[14][3] = {
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1}, {-1,1,0}, {1,-1,0},
    {0,1,1}, {0,-1,-1}, {-1,0,1}, {1,0,-1},
    {-1,1,1}, {1,-1,-1}
};

__device__ const int keys[] = {
    277, 153, 1050, 2085, 102, 554,  // value = 4
    413, 1179, 2357, 1594, 2151, 622, // value = 6
    5462, 10921,  // value = 7
    5463, 5470, 5494, 10937, 10925, 10923,     // value = 8
    5599, 7543, 11197, 6014, 11963, 10991,   // value = 10
    16383  // value = 14
};
__device__ int edit_mode = 0;
int edit_mode_host = 0;
__device__ const int lutOffsets[] = {
    0, (1 << 4), (2 << 4), (3 << 4), (4 << 4), (5 << 4),  // value = 4
    6 * (1 << 4), 6 * (1 << 4) + (1 << 6), 6 * (1 << 4) + 2 * (1 << 6),
    6 * (1 << 4) + 3 * (1 << 6), 6 * (1 << 4) + 4 * (1 << 6), 6 * (1 << 4) + 5 * (1 << 6),  // value = 6
    6 * (1 << 4) + 6 * (1 << 6), 6 * (1 << 4) + 6 * (1 << 6) + (1 << 7),  // value = 7
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7), 6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 2 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 3 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 4 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 5 * (1 << 8),  // value = 8
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8),  // value = 10
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 2 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 3 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 4 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 5 * (1 << 10),  // value = 10
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 6 * (1 << 10)  // value = 14
};
const int NUM_KEYS = sizeof(keys) / sizeof(keys[0]);

const int LUT_SIZE = 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 6 * (1<<10) + (1 << 14);

__device__ int LUT_cuda[LUT_SIZE];
int q_host, threshold_host, edit_type_host;
__device__ int q, threshold, edit_type;
__device__ double delta;
int LUT[6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 6 * (1<<10) + (1 << 14)];
// nvcc -ccbin g++-13 ExtreM_cuda.cu -o ExtreM_cuda $EXTREMZ_NVCC_FLAGS
// salloc --nodes 1 --qos interactive --time 00:10:00 --constraint "gpu" --ntasks-per-node=1 --gpus-per-task=1 --account m4259
struct OffsetComparator {
    double* d_offset;
    __host__ __device__ OffsetComparator(double* offset) : d_offset(offset) {}

    __host__ __device__ bool operator()(int v, int u) const {
        if (v < 0 || u < 0) return false;
        return d_offset[v] > d_offset[u] || 
               ((abs(d_offset[v] - d_offset[u]) == 0) && (v > u));
    }
};

int initialValue = 0, num_Elements_host;
__device__ double *d_deltaBuffer, *input_data, *decp_data, bound, thr, *decp_data_copy_d;
__device__ int count_f_max, count_f_min, count_f_saddle, count_f_dir, count_f_order_saddle, count_f_cases,
                width, height, depth, maxNeighbors, num_Elements, numFaces, num_false_cases, num_false_cases1;
__device__ int *dec_vertex_type, *vertex_type, *vertex_cells, *dec_saddlerank, *saddlerank,
                *dec_saddle1rank, *saddle1rank, 
                *edits, *lowerStars, *dec_lowerStars, 
                *upperStars, *dec_upperStars, *adjacency,
                *AS_M, *DS_M,
                *minimum, *saddles1, *saddles2, *maximum,
                *dec_minimum, *dec_saddles1, *dec_saddles2, *dec_maximum,
                *saddleTriplets, *dec_saddleTriplets, *saddle1Triplets, *dec_saddle1Triplets,
                *tempArray, *dec_tempArray, *tempArrayMin, *dec_tempArrayMin, *largestSaddlesForMax, *dec_largestSaddlesForMax,
                *smallestSaddlesForMin, *dec_smallestSaddlesForMin, 
                *flattened_max2saddles, *max_offsets;
__device__ uint8_t *delta_counter;
__device__ int *or_saddle_max_map;
__device__ int *or_saddle_min_map;
__device__ int *simplified, *max_index;
__device__ int nSaddle2 = 0, nSaddle1 = 0, nMin = 0, nMax = 0;
__device__ int number_of_false_cases = 0, wrong_max_counter = 0, wrong_saddle_counter = 0, wrong_saddle_counter_join = 0, wrong_max_counter_2 = 0, globalMin = 0, dec_globalMin = 0;
__device__ int number_of_false_cases1 = 0, wrong_min_counter = 0, wrong_min_counter_2 = 0;

int *d_max_offsets, *d_flattened_max2saddles, *d_flattened_min1saddles, *d_min_offsets;
__device__ int directions[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1,0,0,-1,
-1,1,0,1,-1,0, 
0,1,1,0,-1,-1,  
-1,0,1,1,0,-1,
-1,1,1,1,-1,-1};

int directions_host[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1,0,0,-1,
-1,1,0,1,-1,0, 
0,1,1,0,-1,-1,  
-1,0,1,1,0,-1,
-1,1,1,1,-1,-1};

struct CompareByDataThenIndexAsc {
    const double* data;
    __host__ __device__
    CompareByDataThenIndexAsc(const double* d) : data(d) {}

    __host__ __device__
    bool operator()(int a, int b) const {
        double da = data[a];
        double db = data[b];
        if (da < db) return true;
        if (da > db) return false;
        // data 相同：按 index 升序
        return a < b;
    }
};

void sort_indexs_by_data(int* d_indexs, const double* d_data, int N) {
    thrust::device_ptr<int> idx(d_indexs);
    thrust::sort(idx, idx + N, CompareByDataThenIndexAsc(d_data));
}

size_t cmpSize = 0;
std::string file_path;
double maxValue, minValue, er, host_bound, host_thre, host_sim;
std::vector<int> saddleTriplets_d, saddle1Triplets_d, max_index_d;
void getdata(const std::string &filename, double *input_data_host, double *&decp_data_host, double *&decp_data_copy, 
            const double er, double &bound, int data_size, int compression) {
        
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }

    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }

    cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));
    
    SZ3::Config conf(static_cast<size_t>(depth_host),
                 static_cast<size_t>(height_host),
                 static_cast<size_t>(width_host));

    // conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_REL;
    conf.relErrorBound = er; 

    
    
    char *compressedData = SZ_compress(conf, h_buffer.data(), cmpSize);

    
    decp_data_copy = new double[data_size];
    SZ_decompress(conf, compressedData, cmpSize, decp_data_copy);
    
    
    cudaMemcpy(decp_data_host, decp_data_copy, data_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*));
    delete[] compressedData;

    
    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    host_thre = (maxValue - minValue) * host_sim;
    bound = (maxValue - minValue) * er;
    // bound = er;
    
    std::cout << "Data read, compressed, and decompressed successfully." << std::endl;
}
void getdata_zfp(const std::string &filename, double *input_data_host, double *decp_data_host, double *&decp_data_copy,
                 const double er, double &bound, int data_size, int compression) {

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Cannot open file: " << filename << std::endl; return; }
    std::streamsize size = file.tellg(); file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl; return;
    }
    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) { std::cerr << "Error reading file." << std::endl; return; }

    cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));

    // compute value range → absolute tolerance
    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    double abs_tol = (maxValue - minValue) * er;

    // ── ZFP compress ────────────────────────────────────────────────────────
    zfp_field *field = zfp_field_3d(h_buffer.data(), zfp_type_double,
                                     (uint)width_host, (uint)height_host, (uint)depth_host);
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, abs_tol);   // absolute error bound

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    std::vector<char> zbuf(bufsize);
    bitstream *bstr = stream_open(zbuf.data(), bufsize);
    zfp_stream_set_bit_stream(zfp, bstr);
    zfp_stream_rewind(zfp);
    cmpSize = zfp_compress(zfp, field);
    if (!cmpSize) { std::cerr << "ZFP compression failed." << std::endl; return; }

    // ── ZFP decompress ──────────────────────────────────────────────────────
    std::vector<double> d_buffer(data_size);
    zfp_field *field_out = zfp_field_3d(d_buffer.data(), zfp_type_double,
                                         (uint)width_host, (uint)height_host, (uint)depth_host);
    zfp_stream_rewind(zfp);
    if (!zfp_decompress(zfp, field_out)) { std::cerr << "ZFP decompression failed." << std::endl; return; }

    zfp_field_free(field); zfp_field_free(field_out);
    zfp_stream_close(zfp); stream_close(bstr);
    decp_data_copy = new double[data_size];
    cudaMemcpy(decp_data_host, d_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(decp_data_copy, decp_data_host, data_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(decp_data,        &decp_data_host,  sizeof(double*));
    

    host_thre = (maxValue - minValue) * host_sim;
    bound     = abs_tol;

    std::cout << "Data read, ZFP-compressed, and decompressed successfully." << std::endl;
}

void getdata_zfp_cuda(const std::string &filename,
                      double *input_data_host,      // device ptr
                      double *decp_data_host,       // device ptr
                      double *&decp_data_copy,      // host ptr (output)
                      const double er, double &bound,
                      int data_size, int compression,
                      double target_cr = 10.0) {
    auto CHECK_CUDA = [&](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) {
            std::cerr << "[ZFP-CUDA] " << msg << ": " << cudaGetErrorString(e) << std::endl;
            throw std::runtime_error(msg);
        }
    };

    // 1) read file to host
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Cannot open file: " << filename << std::endl; return; }
    std::streamsize fsz = file.tellg(); file.seekg(0, std::ios::beg);
    if (fsz != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl; return;
    }
    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char*>(h_buffer.data()), fsz);
    if (!file) { std::cerr << "Error reading file." << std::endl; return; }

    // 2) range (kept for diagnostics / host_thre)
    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    double abs_tol = (maxValue - minValue) * er;

    // upload original
    CHECK_CUDA(cudaMemcpy(input_data_host, h_buffer.data(),
                          data_size * sizeof(double), cudaMemcpyHostToDevice), "H2D input_data_host");
    CHECK_CUDA(cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*)), "toSymbol input_data");

    // 3) device buffers for zfp
    double *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  data_size * sizeof(double)), "cudaMalloc d_in");
    CHECK_CUDA(cudaMalloc(&d_out, data_size * sizeof(double)), "cudaMalloc d_out");
    CHECK_CUDA(cudaMemcpy(d_in, h_buffer.data(),
                          data_size * sizeof(double), cudaMemcpyHostToDevice), "H2D d_in");

    // 4) ZFP setup (CUDA exec)
    zfp_field *field = zfp_field_3d(d_in, zfp_type_double,
                                    (uint)width_host, (uint)height_host, (uint)depth_host);
    zfp_stream *zfp = zfp_stream_open(nullptr);
    // CUDA path uses fixed-rate.
    // For double: rate(bits/value) = 64 / target_cr.
    if (target_cr <= 0.0) target_cr = 10.0;
    double rate = 64.0 / target_cr;
    zfp_stream_set_rate(zfp, rate, zfp_type_double, 3, 0);

    if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
        std::cerr << "[ZFP-CUDA] failed to set CUDA execution policy." << std::endl;
        zfp_field_free(field); zfp_stream_close(zfp);
        cudaFree(d_in); cudaFree(d_out);
        return;
    }

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void *buffer = malloc(bufsize);
    if (!buffer) {
        std::cerr << "[ZFP-CUDA] malloc bitstream buffer failed." << std::endl;
        zfp_field_free(field); zfp_stream_close(zfp);
        cudaFree(d_in); cudaFree(d_out);
        return;
    }

    bitstream *bstr = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, bstr);
    zfp_stream_rewind(zfp);

    // 5) compress
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t zsize = zfp_compress(zfp, field);
    CHECK_CUDA(cudaDeviceSynchronize(), "sync after zfp_compress");
    auto t1 = std::chrono::high_resolution_clock::now();

    if (zsize == 0) {
        std::cerr << "[ZFP-CUDA] compression failed." << std::endl;
        stream_close(bstr); free(buffer);
        zfp_field_free(field); zfp_stream_close(zfp);
        cudaFree(d_in); cudaFree(d_out);
        return;
    }

    cmpSize = zsize;
    compression_time = std::chrono::duration<double>(t1 - t0).count();

    // 6) decompress
    zfp_field *field_out = zfp_field_3d(d_out, zfp_type_double,
                                        (uint)width_host, (uint)height_host, (uint)depth_host);
    zfp_stream_rewind(zfp);

    if (!zfp_decompress(zfp, field_out)) {
        std::cerr << "[ZFP-CUDA] decompression failed." << std::endl;
        zfp_field_free(field_out);
        stream_close(bstr); free(buffer);
        zfp_field_free(field); zfp_stream_close(zfp);
        cudaFree(d_in); cudaFree(d_out);
        return;
    }
    CHECK_CUDA(cudaDeviceSynchronize(), "sync after zfp_decompress");

    // 7) copy back
    if (!decp_data_copy) decp_data_copy = new double[data_size];
    CHECK_CUDA(cudaMemcpy(decp_data_copy, d_out,
                          data_size * sizeof(double), cudaMemcpyDeviceToHost), "D2H decp_data_copy");
    CHECK_CUDA(cudaMemcpy(decp_data_host, d_out,
                          data_size * sizeof(double), cudaMemcpyDeviceToDevice), "D2D decp_data_host");
    CHECK_CUDA(cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*)), "toSymbol decp_data");

    host_thre = (maxValue - minValue) * host_sim;
    // In fixed-rate mode, use observed max abs error as bound.
    double max_abs_diff = 0.0;
    for (int i = 0; i < data_size; ++i) {
        double diff = std::abs(decp_data_copy[i] - h_buffer[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    bound = max_abs_diff;

    // cleanup
    zfp_field_free(field_out);
    stream_close(bstr);
    free(buffer);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Data read, ZFP(CUDA)-compressed, and decompressed successfully. "
              << "cmpSize=" << cmpSize << ", bound=" << bound << ", compression time " << compression_time<< std::endl;
}

// void getdata_zfp(const std::string &filename, double *input_data_host, double *decp_data_host, double *&decp_data_copy,
//                  const double er, double &bound, int data_size, int compression) {

//     std::ifstream file(filename, std::ios::binary | std::ios::ate);
//     if (!file) { std::cerr << "Cannot open file: " << filename << std::endl; return; }
//     std::streamsize size = file.tellg(); file.seekg(0, std::ios::beg);
//     if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
//         std::cerr << "File size does not match expected data size." << std::endl; return;
//     }
//     std::vector<double> h_buffer(data_size);
//     file.read(reinterpret_cast<char *>(h_buffer.data()), size);
//     if (!file) { std::cerr << "Error reading file." << std::endl; return; }

//     cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));

//     // compute value range → absolute tolerance
//     minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
//     maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
//     double abs_tol = (maxValue - minValue) * er;

//     // ── ZFP compress (CUDA backend) ────────────────────────────────────────
//     zfp_field *field = zfp_field_3d(h_buffer.data(), zfp_type_double,
//                                      width_host, height_host, depth_host);
//     zfp_stream *zfp = zfp_stream_open(nullptr);
//     zfp_stream_set_accuracy(zfp, abs_tol);   // absolute error bound


//     size_t bufsize = zfp_stream_maximum_size(zfp, field);
//     std::vector<char> zbuf(bufsize);
//     bitstream *bstr = stream_open(zbuf.data(), bufsize);
//     zfp_stream_set_bit_stream(zfp, bstr);
//     zfp_stream_rewind(zfp);
//     size_t zfpsize = zfp_compress(zfp, field);
//     cmpSize = zfpsize;
//     cudaDeviceSynchronize();
//     if (!cmpSize) { 
//         std::cout<<abs_tol<<std::endl;
//         std::cerr << "ZFP compression failed." << std::endl; return; }

//     // ── ZFP decompress ──────────────────────────────────────────────────────

//     zfp_field *field_out = zfp_field_3d(decp_data_host, zfp_type_double,
//                                          (uint)width_host, (uint)height_host, (uint)depth_host);
//     zfp_stream_rewind(zfp);
//     if (!zfp_decompress(zfp, field_out)) { std::cerr << "ZFP decompression failed." << std::endl; return; }
//     cudaDeviceSynchronize();

//     zfp_field_free(field); zfp_field_free(field_out);
//     zfp_stream_close(zfp); stream_close(bstr);

//     cudaMemcpy(decp_data_copy, decp_data_host, data_size * sizeof(double), cudaMemcpyDeviceToHost);
    
//     cudaMemcpyToSymbol(decp_data,        &decp_data_host,  sizeof(double*));
    

//     host_thre = (maxValue - minValue) * host_sim;
//     bound     = abs_tol;

//     std::cout << "Data read, ZFP(CUDA)-compressed, and decompressed successfully." << std::endl;
// }

std::string extractFilename(const std::string& path) {
    
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

    size_t dotPos = filename.find_last_of('.');
    std::string name = (dotPos == std::string::npos) ? filename : filename.substr(0, dotPos);

    return name;
}

void getdata_tthresh(const std::string &filename, double *input_data_host, double *decp_data_host, double *&decp_data_copy,
                     const double er, double &bound, int data_size, int compression) {
    (void)compression;

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }

    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }

    cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));

    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    host_thre = (maxValue - minValue) * host_sim;
    bound = (maxValue - minValue) * er;  // tthresh uses abs error here

    std::string base = extractFilename(filename) + "_" +
                       std::to_string(width_host) + "_" +
                       std::to_string(height_host) + "_" +
                       std::to_string(depth_host);
    std::string tmp_comp = (exactz_scratch() + "/tmp/") + base + ".tthresh.cmp";
    std::string tmp_decp = (exactz_scratch() + "/tmp/") + base + ".tthresh.dec.bin";

    const char *bin_env = std::getenv("TTHRESH_BIN");
    std::string tthresh_bin = (bin_env && std::strlen(bin_env) > 0)
        ? std::string(bin_env)
        : "tthresh";

    std::string cmd =
        "\"" + tthresh_bin + "\""
        " -i \"" + filename + "\""
        " -t double"
        " -c \"" + tmp_comp + "\""
        " -o \"" + tmp_decp + "\""
        " -s " + std::to_string(width_host) +
        " " + std::to_string(height_host) +
        " " + std::to_string(depth_host) +
        " -r " + std::to_string(er);

    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = std::system(cmd.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    compression_time = static_cast<float>(std::chrono::duration<double>(t1 - t0).count());
    if (ret != 0) {
        std::cerr << "[tthresh] command failed with code " << ret << std::endl;
        std::cerr << "[tthresh] cmd: " << cmd << std::endl;
        return;
    }

    if (!std::filesystem::exists(tmp_comp) || !std::filesystem::exists(tmp_decp)) {
        std::cerr << "[tthresh] output files not found." << std::endl;
        return;
    }
    cmpSize = std::filesystem::file_size(tmp_comp);

    std::ifstream decfile(tmp_decp, std::ios::binary | std::ios::ate);
    if (!decfile) {
        std::cerr << "[tthresh] cannot open decompressed file: " << tmp_decp << std::endl;
        return;
    }
    std::streamsize dec_size = decfile.tellg();
    decfile.seekg(0, std::ios::beg);
    if (dec_size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "[tthresh] decompressed file size mismatch: " << dec_size
                  << " vs expected " << data_size * sizeof(double) << std::endl;
        return;
    }
    if (!decp_data_copy) decp_data_copy = new double[data_size];
    decfile.read(reinterpret_cast<char*>(decp_data_copy), dec_size);
    if (!decfile) {
        std::cerr << "[tthresh] failed to read decompressed data." << std::endl;
        return;
    }
    decfile.close();

    cudaMemcpy(decp_data_host, decp_data_copy, data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*));
    double max_abs_diff = 0.0;
    for (int i = 0; i < data_size; ++i) {
        double diff = std::abs(decp_data_copy[i] - h_buffer[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    bound = max_abs_diff;
    std::cout<<bound<<std::endl;
    std::filesystem::remove(tmp_comp);
    std::filesystem::remove(tmp_decp);

    std::cout << "Data read, tthresh-compressed, and decompressed successfully."
              << " cmpSize=" << cmpSize
              << ", bound=" << bound
              << ", compression time " << compression_time << " s" << std::endl;
}

void getdata_sperr(const std::string &filename, double *input_data_host, double *decp_data_host, double *&decp_data_copy,
                   const double er, double &bound, int data_size, int compression) {
    (void)compression;

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }

    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }

    cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));

    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    host_thre = (maxValue - minValue) * host_sim;
    double abs_tol = (maxValue - minValue) * er;
    bound = abs_tol;

    std::string base = extractFilename(filename) + "_" +
                       std::to_string(width_host) + "_" +
                       std::to_string(height_host) + "_" +
                       std::to_string(depth_host);

    std::string tmp_dir = "/tmp/";
    if (const char* tmp_env = std::getenv("TMPDIR")) {
        if (std::strlen(tmp_env) > 0) tmp_dir = std::string(tmp_env);
    }
    if (!tmp_dir.empty() && tmp_dir.back() != '/') tmp_dir.push_back('/');

    std::string tmp_comp = tmp_dir + base + ".sperr";
    std::string tmp_decp = tmp_dir + base + ".sperr.dec.bin";

    const char *bin_env = std::getenv("SPERR_BIN");
    std::string sperr_bin = (bin_env && std::strlen(bin_env) > 0)
        ? std::string(bin_env)
        : "sperr3d";
    const char *threads_env = std::getenv("SPERR_THREADS");
    std::string sperr_threads = (threads_env && std::strlen(threads_env) > 0)
        ? std::string(threads_env)
        : "1";
    std::string omp_prefix = "OMP_NUM_THREADS=" + sperr_threads + " ";

    char tol_buf[128];
    std::snprintf(tol_buf, sizeof(tol_buf), "%.17f", abs_tol);

    std::string cmd_compress =
        omp_prefix +
        "\"" + sperr_bin + "\""
        " -c"
        " --ftype 64"
        " --dims " + std::to_string(width_host) + " " + std::to_string(height_host) + " " + std::to_string(depth_host) +
        " --chunks " + std::to_string(width_host) + " " + std::to_string(height_host) + " " + std::to_string(depth_host) +
        " --pwe " + std::string(tol_buf) +
        " --bitstream \"" + tmp_comp + "\""
        " \"" + filename + "\"";

    std::string cmd_decompress =
        omp_prefix +
        "\"" + sperr_bin + "\""
        " -d"
        " --decomp_d \"" + tmp_decp + "\""
        " \"" + tmp_comp + "\"";

    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = std::system(cmd_compress.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    compression_time = static_cast<float>(std::chrono::duration<double>(t1 - t0).count());
    if (ret != 0) {
        std::cerr << "[sperr] compress command failed with code " << ret << std::endl;
        std::cerr << "[sperr] cmd: " << cmd_compress << std::endl;
        return;
    }

    ret = std::system(cmd_decompress.c_str());
    if (ret != 0) {
        std::cerr << "[sperr] decompress command failed with code " << ret << std::endl;
        std::cerr << "[sperr] cmd: " << cmd_decompress << std::endl;
        return;
    }

    if (!std::filesystem::exists(tmp_comp) || !std::filesystem::exists(tmp_decp)) {
        std::cerr << "[sperr] output files not found." << std::endl;
        return;
    }
    cmpSize = std::filesystem::file_size(tmp_comp);

    std::ifstream decfile(tmp_decp, std::ios::binary | std::ios::ate);
    if (!decfile) {
        std::cerr << "[sperr] cannot open decompressed file: " << tmp_decp << std::endl;
        return;
    }
    std::streamsize dec_size = decfile.tellg();
    decfile.seekg(0, std::ios::beg);
    if (dec_size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "[sperr] decompressed file size mismatch: " << dec_size
                  << " vs expected " << data_size * sizeof(double) << std::endl;
        return;
    }
    if (!decp_data_copy) decp_data_copy = new double[data_size];
    decfile.read(reinterpret_cast<char*>(decp_data_copy), dec_size);
    if (!decfile) {
        std::cerr << "[sperr] failed to read decompressed data." << std::endl;
        return;
    }
    decfile.close();

    cudaMemcpy(decp_data_host, decp_data_copy, data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*));

    double max_abs_diff = 0.0;
    for (int i = 0; i < data_size; ++i) {
        double diff = std::abs(decp_data_copy[i] - h_buffer[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    bound = max_abs_diff;
    std::cout << "[sperr] max|decp-original| = " << bound << std::endl;

    std::filesystem::remove(tmp_comp);
    std::filesystem::remove(tmp_decp);

    std::cout << "Data read, sperr-compressed, and decompressed successfully."
              << " cmpSize=" << cmpSize
              << ", bound=" << bound
              << ", compression time " << compression_time << " s" << std::endl;
}

void getdata_mgard(const std::string &filename, double *input_data_host, double *decp_data_host, double *&decp_data_copy,
                   const double er, double &bound, int data_size, int compression) {

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Cannot open file: " << filename << std::endl; return; }
    std::streamsize size = file.tellg(); file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl; return;
    }
    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) { std::cerr << "Error reading file." << std::endl; return; }

    cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));

    // compute value range for diagnostics
    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    double tol = er;

    std::cout << "[MGARD-CLI] start, tol=" << tol
              << ", dims=" << width_host << "x" << height_host << "x" << depth_host
              << std::endl;
    std::cout.flush();

    std::string base = extractFilename(filename) + "_" +
                       std::to_string(width_host) + "_" +
                       std::to_string(height_host) + "_" +
                       std::to_string(depth_host);
    std::string tmp_comp = (exactz_scratch() + "/tmp/") + base + ".mgard";
    std::string tmp_decp = (exactz_scratch() + "/tmp/") + base + ".mgard.dec.bin";

    std::string cmd_compress =
        "mgard-x -z -i \"" + filename +
        "\" -o \"" + tmp_comp +
        "\" -dt d -dim 3 " +
        std::to_string(width_host) + " " +
        std::to_string(height_host) + " " +
        std::to_string(depth_host) +
        " -em rel -e " + std::to_string(tol) +
        " -s 0 -l huffman-zstd -d cuda -v 2 2>&1";

    auto run_and_capture = [](const std::string &cmd,
                              const std::string &time_key,
                              double *parsed_time) -> int {
        FILE *pipe = popen(cmd.c_str(), "r");
        if (!pipe) return -1;

        char line[4096];
        if (parsed_time) *parsed_time = -1.0;
        while (fgets(line, sizeof(line), pipe)) {
            std::string s(line);
            std::cout << s;
            if (parsed_time && s.find(time_key) != std::string::npos) {
                size_t pos = s.find(':');
                if (pos != std::string::npos) {
                    std::string tail = s.substr(pos + 1);
                    size_t s_pos = tail.find('s');
                    if (s_pos != std::string::npos) tail = tail.substr(0, s_pos);
                    try { *parsed_time = std::stod(tail); } catch (...) {}
                }
            }
        }
        return pclose(pipe);
    };

    auto t_comp_begin = std::chrono::high_resolution_clock::now();
    double parsed_comp_time = -1.0;
    int ret = run_and_capture(cmd_compress, "High-level compression time", &parsed_comp_time);
    auto t_comp_end = std::chrono::high_resolution_clock::now();
    if (ret != 0) {
        std::cerr << "[MGARD-CLI] compress command failed with code " << ret << std::endl;
        return;
    }
    if (parsed_comp_time > 0) {
        compression_time = static_cast<float>(parsed_comp_time);
        std::cout << "[MGARD-CLI] using parsed High-level compression time: "
                  << compression_time << " s" << std::endl;
    } else {
        compression_time = static_cast<float>(std::chrono::duration<double>(t_comp_end - t_comp_begin).count());
        std::cout << "[MGARD-CLI] parsed compression time not found, fallback wall time: "
                  << compression_time << " s" << std::endl;
    }

    std::string cmd_decompress =
        "mgard-x -x -i \"" + tmp_comp +
        "\" -o \"" + tmp_decp + "\" -d cuda -v 2 2>&1";
    double parsed_decomp_time = -1.0;
    ret = run_and_capture(cmd_decompress, "High-level decompression time", &parsed_decomp_time);
    if (ret != 0) {
        std::cerr << "[MGARD-CLI] decompress command failed with code " << ret << std::endl;
        return;
    }

    cmpSize = std::filesystem::file_size(tmp_comp);
    std::ifstream decfile(tmp_decp, std::ios::binary | std::ios::ate);
    if (!decfile) {
        std::cerr << "[MGARD-CLI] cannot open decompressed file: " << tmp_decp << std::endl;
        return;
    }
    std::streamsize dec_size = decfile.tellg();
    decfile.seekg(0, std::ios::beg);
    if (dec_size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "[MGARD-CLI] decompressed file size mismatch: " << dec_size
                  << " vs expected " << data_size * sizeof(double) << std::endl;
        return;
    }
    if (!decp_data_copy) decp_data_copy = new double[data_size];
    decfile.read(reinterpret_cast<char*>(decp_data_copy), dec_size);
    if (!decfile) {
        std::cerr << "[MGARD-CLI] failed to read decompressed file data." << std::endl;
        return;
    }
    decfile.close();

    cudaMemcpy(decp_data_host, decp_data_copy, data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*));

    std::filesystem::remove(tmp_comp);
    std::filesystem::remove(tmp_decp);

    host_thre = (maxValue - minValue) * host_sim;
    double max_abs_diff = 0.0;
    for (int i = 0; i < data_size; ++i) {
        double diff = std::abs(decp_data_copy[i] - h_buffer[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    bound = max_abs_diff;
    std::cout << "[MGARD-CLI] max|decp-original| = " << bound << std::endl;

    std::cout << "Data read, MGARD-compressed, and decompressed successfully." << std::endl;
}

void getdata_cuszp(const std::string &filename, double *input_data_host, double *decp_data_host, double *&decp_data_copy,
                   const double er, double &bound, int data_size, int compression) {
    auto check_cuda = [](cudaError_t e, const char *msg) -> bool {
        if (e != cudaSuccess) {
            std::cerr << "[cuSZp] " << msg << ": " << cudaGetErrorString(e) << std::endl;
            return false;
        }
        return true;
    };

    if (input_data_host == nullptr || decp_data_host == nullptr) {
        std::cerr << "[cuSZp] input_data_host or decp_data_host is nullptr. "
                  << "Please cudaMalloc both buffers before calling getdata_cuszp." << std::endl;
        return;
    }

    cudaPointerAttributes in_attr{}, out_attr{};
    cudaError_t attr_in = cudaPointerGetAttributes(&in_attr, input_data_host);
    cudaError_t attr_out = cudaPointerGetAttributes(&out_attr, decp_data_host);
    if (!check_cuda(attr_in, "cudaPointerGetAttributes(input_data_host)") ||
        !check_cuda(attr_out, "cudaPointerGetAttributes(decp_data_host)")) {
        return;
    }
    if (in_attr.type != cudaMemoryTypeDevice || out_attr.type != cudaMemoryTypeDevice) {
        std::cerr << "[cuSZp] input_data_host and decp_data_host must be device pointers." << std::endl;
        return;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Cannot open file: " << filename << std::endl; return; }
    std::streamsize size = file.tellg(); file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
        std::cerr << "File size does not match expected data size." << std::endl; return;
    }
    std::vector<double> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) { std::cerr << "Error reading file." << std::endl; return; }

    // compute value range → absolute tolerance
    minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    double abs_tol = (maxValue - minValue) * er;

    // upload original data to device
    if (!check_cuda(cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D input_data_host")) return;
    if (!check_cuda(cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*)),
                    "cudaMemcpyToSymbol(input_data)")) return;

    auto t0 = std::chrono::high_resolution_clock::now();
    
    
    // ── cuSZp compress (GPU) ────────────────────────────────────────────────
    double *d_oriData = nullptr;
    if (!check_cuda(cudaMalloc(&d_oriData, data_size * sizeof(double)), "cudaMalloc d_oriData")) return;
    if (!check_cuda(cudaMemcpy(d_oriData, h_buffer.data(), data_size * sizeof(double), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D d_oriData")) {
        cudaFree(d_oriData);
        return;
    }

    size_t max_cmp_bytes = data_size * sizeof(double);
    unsigned char *d_cmpBytes = nullptr;
    if (!check_cuda(cudaMalloc(&d_cmpBytes, max_cmp_bytes), "cudaMalloc d_cmpBytes")) {
        cudaFree(d_oriData);
        return;
    }

    uint3 dims = make_uint3(static_cast<unsigned>(width_host),
                            static_cast<unsigned>(height_host),
                            static_cast<unsigned>(depth_host));
    size_t comp_size = 0;
    cudaStream_t stream;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        return;
    }

    cuSZp_compress(d_oriData, d_cmpBytes, static_cast<size_t>(data_size), &comp_size,
                   static_cast<float>(abs_tol),
                   CUSZP_DIM_3D, dims, CUSZP_TYPE_DOUBLE, CUSZP_MODE_PLAIN, stream);
    if (!check_cuda(cudaStreamSynchronize(stream), "cuSZp_compress stream sync")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);
        return;
    }
    if (!check_cuda(cudaGetLastError(), "cuSZp_compress kernel")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);
        return;
    }
    cmpSize = comp_size;
    if (cmpSize == 0) {
        std::cerr << "[cuSZp] compression failed: compressed size is 0." << std::endl;
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);
        return;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    compression_time = std::chrono::duration<double>(t1 - t0).count();

    // ── cuSZp decompress (GPU) ──────────────────────────────────────────────
    double *d_decData = nullptr;
    if (!check_cuda(cudaMalloc(&d_decData, data_size * sizeof(double)), "cudaMalloc d_decData")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);
        return;
    }
    if (!check_cuda(cudaMemset(d_decData, 0, data_size * sizeof(double)), "cudaMemset d_decData")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }

    cuSZp_decompress(d_decData, d_cmpBytes, static_cast<size_t>(data_size), cmpSize,
                     static_cast<float>(abs_tol),
                     CUSZP_DIM_3D, dims, CUSZP_TYPE_DOUBLE, CUSZP_MODE_PLAIN, stream);
    if (!check_cuda(cudaStreamSynchronize(stream), "cuSZp_decompress stream sync")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }
    if (!check_cuda(cudaGetLastError(), "cuSZp_decompress kernel")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }

    // copy decompressed result back to host and to decp_data_host
    if (!decp_data_copy) decp_data_copy = new double[data_size];
    if (!check_cuda(cudaMemcpy(decp_data_copy, d_decData, data_size * sizeof(double), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H decp_data_copy")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }
    if (!check_cuda(cudaMemcpy(decp_data_host, d_decData, data_size * sizeof(double), cudaMemcpyDeviceToDevice),
                    "cudaMemcpy D2D decp_data_host")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }
    if (!check_cuda(cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*)),
                    "cudaMemcpyToSymbol(decp_data)")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after D2D copy")) {
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaFree(d_decData);
        cudaStreamDestroy(stream);
        return;
    }

    // cleanup
    cudaFree(d_oriData);
    cudaFree(d_cmpBytes);
    cudaFree(d_decData);
    cudaStreamDestroy(stream);

    host_thre = (maxValue - minValue) * host_sim;
    double max_abs_diff = 0.0;
    for (int i = 0; i < data_size; ++i) {
        double diff = std::abs(decp_data_copy[i] - h_buffer[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    bound = max_abs_diff;


    std::cout << "Data read, cuSZp-compressed, and decompressed successfully." << cmpSize << ", "<<bound << std::endl;
}

int calculateNeighborIndex(int nx, int ny, int nz){
    for(int i = 0; i<maxNeighbors_host; i++){
        if(nx == neighborOffsets_host[i][0] && ny == neighborOffsets_host[i][1] && nz == neighborOffsets_host[i][2]) return i;
    }
    return -1;
}

void loadLUTToGPU() {
    // 1. 在 CPU 上分配内存并读取 LUT.bin
    int* h_LUT = new int[LUT_SIZE];
    std::ifstream file("LUT.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Failed to open LUT.bin" << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(h_LUT), LUT_SIZE * sizeof(int));
    file.close();

    // 2. 将 CPU 的 LUT 复制到 __device__ 变量
    cudaMemcpyToSymbol(LUT_cuda, h_LUT, LUT_SIZE * sizeof(int));

    // 3. 释放 CPU 内存
    delete[] h_LUT;
}


int calculateLUT(const double* heightMap, int i, int tableSize, int startIndex, int type=0) {

    int y = (i / (width_host)) % height_host; // Get the x coordinate
    int x = i % width_host; // Get the y coordinate
    int z = (i / (width_host * height_host)) % depth_host;
    
    for(int config = 0; config<tableSize;config++){
        int neighbor_size = 0;
        std::bitset<14> binary(config); 
        
        std::vector<std::vector<int>> *upperComponents = nullptr;
        std::vector<std::vector<int>> *lowerComponents = nullptr;

        std::vector<std::vector<int>> localUpperComponents;
        std::vector<std::vector<int>> localLowerComponents;
        if(upperComponents == nullptr) {
            upperComponents = &localUpperComponents;
        }
        if(lowerComponents == nullptr) {
            lowerComponents = &localLowerComponents;
        }
        
        std::vector<int > lowerStar, upperStar;
        
        int neighbor[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        for (int d =0;d<maxNeighbors_host;d++) {
            // if(adjacency_host[vertexId*maxNeighbors_host+d]==-1) continue;
            // int r = adjacency_host[i*maxNeighbors_host+d];
            int dx = directions_host[3 * d];
            int dy = directions_host[3 * d + 1];
            int dz = directions_host[3 * d + 2];

            int nx = x + dx;
            int ny = y + dy;
            int nz = z + dz;
            if(nx < 0 || nx >= width_host || ny < 0 || ny >= height_host || nz < 0 || nz >= depth_host) continue;
            int r = x + dx + (y + dy + (z + dz) * height_host) * width_host;
            neighbor[neighbor_size] = r;
            if (binary[neighbor_size] == 0) {
                lowerStar.emplace_back(r);
            } 
            else{
                upperStar.emplace_back(r);
            }
            neighbor_size++;
        }
        // if(neighbor_size == 8 && config == 44){
        //     // printf("unmatched here !!!%d: %d %d %d %d\n", g_idx, result, LUT_result, neighbor_size, decimal_value);
        //     std::cout<<binary<<", "<< i<<std::endl;
        //     printf("\n");
        //     for(auto i:upperStar) printf("%d ",i);
        //     printf("\n");
        // }
        std::vector<UnionFind> lowerSeeds(lowerStar.size());
        std::vector<UnionFind> upperSeeds(upperStar.size());
        std::vector<UnionFind*> lowerList(lowerStar.size());
        std::vector<UnionFind*> upperList(upperStar.size());

        std::vector<int> lowerNeighbors, upperNeighbors;
        upperNeighbors = upperStar;
        lowerNeighbors = lowerStar;
    
        for(int i = 0; i < (int)lowerList.size(); i++)
            lowerList[i] = &(lowerSeeds[i]);
        for(int i = 0; i < (int)upperList.size(); i++)
            upperList[i] = &(upperSeeds[i]);
        
        int neighbor_idx = 0;
        for(int t = 0; t < 14; t++){
            int dx = directions_host[ t * 3 + 0];
            int dy = directions_host[ t * 3 + 1];
            int dz = directions_host[ t * 3 + 2];

            int nx = x + dx;
            int ny = y + dy;
            int nz = z + dz;
            int v2 = nx + (ny + (nz) * height_host) * width_host;
            
            if(nx >= width_host || ny >= height_host || nz >= depth_host || nx < 0 ||
                ny < 0 || nz < 0 || v2 >= num_Elements_host || v2 < 0) continue;
            bool const lower0 = binary[neighbor_idx] == 0;
            neighbor_idx++;
            for(int j = 0; j < MAX_V3; j++){
                int dx1 = lookupTable_host[t][j][0];
                int dy1 = lookupTable_host[t][j][1];
                int dz1 = lookupTable_host[t][j][2];
                // int cellId = vertex_cells[i * 42 + t];
                if(dx1 == -2 || dy1 == -2 || dz1 == -2) break;
                int nx1 = x + dx1;
                int ny1 = y + dy1;
                int nz1 = z + dz1;
                int v3 = nx1 + (ny1 + (nz1) * height_host) * width_host;
                
                if(nx1 >= width_host || ny1 >= height_host || nz1 >= depth_host || nx1 < 0 ||
                    ny1 < 0 || nz1 < 0 || v3 >= num_Elements_host || v3 < 0 || v3 < v2) continue;
                int idx;
                for(int k = 0; k<neighbor_size; k++){
                    if(v3 == neighbor[k]){
                        idx = k;
                        break;
                    }
                }
                bool const lower1 = binary[idx] == 0;
                std::vector<int> *neighbors = &lowerNeighbors;
                std::vector<UnionFind *> *seeds = &lowerList;

                if(!lower0) {
                    neighbors = &upperNeighbors;
                    seeds = &upperList;
                }

                if(lower0 == lower1) {
                    // connect their union-find sets!
                    int lowerId0 = -1, lowerId1 = -1;
                    for(int l = 0; l < (int)neighbors->size(); l++) {
                        if((*neighbors)[l] == v2) {
                            
                            lowerId0 = l;
                        }
                        if((*neighbors)[l] == v3) {
                            lowerId1 = l;
                        }
                    }
                    if((lowerId0 != -1) && (lowerId1 != -1)) {
                        // if(vertexId == 0) cout<<lowerId0<<", "<<lowerId1<<endl;
                        (*seeds)[lowerId0] = UnionFind::makeUnion(
                            (*seeds)[lowerId0], (*seeds)[lowerId1]);
                            (*seeds)[lowerId1] = (*seeds)[lowerId0];
                        }
                }
            }
        }
        
        // update the UF if necessary
        for(int i = 0; i < (int)lowerList.size(); i++)
            lowerList[i] = lowerList[i]->find();
        for(int i = 0; i < (int)upperList.size(); i++)
        {
            upperList[i] = upperList[i]->find();
        }
            

        std::unordered_map<UnionFind *, std::vector<int>>::iterator it;
        std::unordered_map<UnionFind *, std::vector<int>>
            upperComponentId{};
        std::unordered_map<UnionFind *, std::vector<int>>
            lowerComponentId{};
        // We retrieve the lower and upper components if we want them
        for(int i = 0; i < (int)upperNeighbors.size(); i++) {
            
            it = upperComponentId.find(upperList[i]);
            if(it != upperComponentId.end()) {
                upperComponentId[upperList[i]].push_back(upperNeighbors[i]);
            } else {
            upperComponentId[upperList[i]]
                = std::vector<int>(1, upperNeighbors[i]);
            }
        }

        
        for(auto elt : upperComponentId) {
            upperComponents->push_back(std::vector<int>());
            
            for(int i = 0; i < (int)elt.second.size(); i++) {
                upperComponents->back().push_back(elt.second.at(i));
            }
        }
    
        

        for(int i = 0; i < (int)lowerNeighbors.size(); i++) {
            it = lowerComponentId.find(lowerList[i]);
            if(it != lowerComponentId.end()) {
                lowerComponentId[lowerList[i]].push_back(lowerNeighbors[i]);
            } else {
            lowerComponentId[lowerList[i]]
                = std::vector<int>(1, lowerNeighbors[i]);
            }
        }

        for(auto elt : lowerComponentId) {
            lowerComponents->push_back(std::vector<int>());
            for(int i = 0; i < (int)elt.second.size(); i++) {
                lowerComponents->back().push_back(elt.second.at(i));
            }
        }
        
        int lowerComponentNumber = lowerComponents->size();
        int upperComponentNumber = upperComponents->size();

        // minimum: 0 1-saddle: 1, 2-saddle: 2, regular: 5 maximum: 4;
        if(lowerComponentNumber == 0 && upperComponentNumber == 1) LUT[startIndex + config]=0;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 0) LUT[startIndex + config]=4;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 1) LUT[startIndex + config]=5;
        else if(lowerComponentNumber > 1 && upperComponentNumber > 1) LUT[startIndex + config]=3;
        else if(lowerComponentNumber > 1) LUT[startIndex + config]=1;
        else if(upperComponentNumber > 1) LUT[startIndex + config]=2;
    }      
    return 0;
}

__device__ void getVerticesFromTriangleID_2d(
        int triangleID,
        int &v1, 
        int &v2, 
        int &v3) {


        int baseID, min_x, min_y, min_z;
        if (triangleID < 2 * (width - 1) * (height - 1)) {
            baseID = triangleID / 2;
            min_z = 0;
            min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
            min_x = baseID % (width - 1);

            // 三角形类型（上下三角形）
            if (triangleID % 2 == 0) {
                v1 = min_x + min_y * width + min_z * width * height;
                v2 = (min_x + 1) + min_y * width + min_z * width * height;
                v3 = min_x + (min_y + 1) * width + min_z * width * height;
            } else {
                v1 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
                v2 = (min_x) + (min_y + 1)* width + min_z * width * height;
                v3 = min_x + 1 + (min_y) * width + min_z * width * height;
            }

            if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
            if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
            if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
            return;
            
        }

        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        
        return; 
}

__device__ void getVerticesFromTriangleID_3d(
    int triangleID,
    int &v1, 
    int &v2, 
    int &v3) {

    int baseID, min_x, min_y, min_z;
    // int starID = triangleID;
    
    if (triangleID < 2 * depth * (width - 1) * (height - 1)) {
        baseID = triangleID / 2;
        min_z = baseID / ((width - 1) * (height - 1));
        min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
        min_x = baseID % (width - 1);

        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + min_z * width * height;
        } else {
            v1 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + min_z * width * height;
        }

        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        
        return;
        
    }

    triangleID -= 2 * depth * (width - 1) * (height - 1);
    if (triangleID < 2 * width * (height - 1) * (depth - 1)) {
        baseID = triangleID / 2;
        min_x = baseID / ((height - 1) * (depth - 1));
        min_y = (baseID % ((height - 1) * (depth - 1))) / (depth - 1);
        min_z = baseID % (depth - 1);
        
        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = min_x + (min_y + 1) * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z + 1) * width * height;
        } else {
            v1 = min_x + (min_y) * width + (min_z) * width * height;
            v2 = min_x + (min_y) * width + (min_z + 1) * width * height;
            v3 = min_x + (min_y+1) * width + (min_z + 1) * width * height;
        }
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type2: %d %d %d %d %d\n", v1, v2, v3, starID, triangleID);
        return;
    }

    // XZ 平面的三角形
    triangleID -= 2 * width * (height - 1) * (depth - 1);
    if (triangleID < 2 * height * (width - 1) * (depth - 1)) {
        baseID = triangleID / 2;
        min_y = baseID / ((width - 1) * (depth - 1));
        min_z = (baseID % ((width - 1) * (depth - 1))) / (width - 1);
        min_x = baseID % (width - 1);

        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + min_y * width + (min_z + 1) * width * height;
        } else {
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + min_y * width + (min_z + 1) * width * height;
            v3 = (min_x+1) + min_y * width + (min_z + 1) * width * height;
        }
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type3: %d %d %d\n", v1, v2, v3, starID);
        return;
    }

    // 对角线方向的三角形
    triangleID -= 2 * height * (width - 1) * (depth - 1);
    baseID = triangleID / 6;
    min_z = baseID / ((width - 1) * (height - 1));
    min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
    min_x = baseID % (width - 1);
    // int y2 = triangleID / ((width - 1) * (depth - 1));
    // int z2 = (triangleID % ((width - 1) * (depth - 1))) / (depth - 1);
    // int x2 = triangleID % (depth - 1);

    int subTriangleType = triangleID % 6;
    
    switch (subTriangleType) {
        
        case 0:
            v1 = min_x + (min_y) * width + (min_z) * width * height;
            v2 = min_x+1 + min_y * width + min_z * width * height;
            v3 = (min_x ) + (min_y +1)* width + (min_z+1) * width * height;
            break;
        case 1:
            v1 = (min_x +1) + (min_y ) * width + (min_z) * width * height;
            v2 = min_x + (min_y + 1) * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z+1)* width * height;
            break;
        case 2:
            v1 = min_x + 1 + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
            v3 = (min_x) + (min_y +1)* width + (min_z +1) * width * height;
            break;
        case 3:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + min_y * width + ( min_z+1) * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z + 1) * width * height;
            break;
        case 4:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x + 1) + (min_y) * width +( min_z +1 ) * width * height;
            v3 = (min_x) + (min_y + 1) * width + (min_z + 1)* width * height;
            break;
        case 5:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + (min_y + 1) * width + (min_z + 1) * width * height;
            v3 = (min_x + 1) + (min_y + 1) * width +( min_z+1) * width * height;
            break;
    }

    // if(starID == 2945) printf("type4: %d %d %d %d %d %d\n", v1, v2, v3, starID, triangleID, subTriangleType);
    if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
    if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
    if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
    
    return;
    
}



__global__ void computeAdjacency() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    
    int x = i % width; // Get the y coordinate
    int y = (i / (width)) % height; // Get the x coordinate
    int z = (i / (width * height)) % depth;
    int neighborIdx = 0;
    // int binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            adjacency[i * maxNeighbors + neighborIdx] = r;
            neighborIdx++;
            // binary[13 - d] = 1;
        }
        
    }

    // int decimal_value = 0;
    // for (int i = 0; i < maxNeighbors; i++) {
    //     decimal_value = (decimal_value << 1) | binary[i]; // 左移并加上当前位
    // }
    
    // Fill the remaining slots with -1 or another placeholder value
    for (int j = neighborIdx; j < maxNeighbors; ++j) {
        adjacency[i * maxNeighbors + j] = -1;
    }
    
    // if((x == 0 || y == 0 || z == 0 || x == width - 1 || y == height - 1 || z == depth - 1) && (x != width - 2 && y != height - 2 && z != depth - 2) ){
    //     printf("%d %d %d %d %d \n", i, neighborIdx, x, y, z);
    //     // for(int j = 0; j<maxNeighbors; j++) printf("%d ", adjacency[i * maxNeighbors + j]);
        
    // }
    // printf("%d %d: %d\n", decimal_value,i,neighborIdx );
}

__global__ void printsortedsaddle() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=1) return;
    for(int id = 0; id<nSaddle2; id++){
        int saddle = saddles2[id];
        if(id == 0 && decp_data[saddle] > decp_data[saddles2[1]]) printf("wrong!! 1\n");
        else if(id == nSaddle2 - 1 & decp_data[saddle] < decp_data[saddles2[nSaddle2 - 2]]) printf("wrong!! 2\n");
        else if(id > 0 && id < nSaddle2 -1 ){
            if(decp_data[saddle] > decp_data[saddles2[id + 1]]) printf("wrong!! 3 %d %d\n", id, saddle);
            if(decp_data[saddle] < decp_data[saddles2[id - 1]]) printf("wrong!! 4\n");
        }
        printf("%d %.17f %17f\n", saddle, decp_data[saddle], bound);
    }
}


template <typename T>
__global__ void applyDeltaBuffer(size_t num_Elements, 
                                uint8_t *changed_vertex_h) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_Elements) {
        if(edits[tid] > 0){
            int s = edits[tid];
            if(delta_counter[tid] + s <= q && fabs(input_data[tid]-(decp_data[tid] - s * delta)) <= bound){
                decp_data[tid] -= s * delta; 
                delta_counter[tid] += s;
            }
            else{
                delta_counter[tid] = q;
                decp_data[tid] = input_data[tid] - bound;
                // edits[tid] = q;
            }
            
            
            int x = tid % width;
            int y = (tid / width) % height;
            int z = (tid / (width * height)) % depth;
            // if(x <= 1 || x >= width - 2 || y <= 1 || y >= height - 2 || z <= 1 || z >= depth - 2 ) *send_flag = 1;
            for (int d = 0; d < maxNeighbors; d++) {
        
                int dirX = directions[d * 3];     
                int dirY = directions[d * 3 + 1]; 
                int dirZ = directions[d * 3 + 2]; 
                int newX = x + dirX;
                int newY = y + dirY;
                int newZ = z + dirZ;
                int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < num_Elements && newZ<depth && newZ>=0) {
                    changed_vertex_h[r] = 1;
                }
            }
            changed_vertex_h[tid] = 1;
            
        }    
    }
        
}



template <typename T>
__device__ void applyDeltaBuffer_local(uint8_t *delta_counter,
                                T *decp_data, T delta, T *input_data, 
                                T bound, int q, size_t tid, uint8_t *changed_vertex_h) {
    
    if(abs(input_data[tid]-(decp_data[tid] - delta)) <= bound && delta_counter[tid] < q - 1){
        decp_data[tid] -= delta; 
        delta_counter[tid] += 1;
    }
    else{
        delta_counter[tid] = q;
        decp_data[tid] = input_data[tid] - bound;
    }

    int x = tid % width;
    int y = (tid / width) % height;
    int z = (tid / (width * height)) % depth;
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < num_Elements && newZ<depth && newZ>=0) {
            changed_vertex_h[r] = 1;
        }
    }
    changed_vertex_h[tid] = 1;
      
}

__device__ bool islarger(const int v, const int u, const double *offset){
    return offset[v] > offset[u] || (fabs(offset[v] - offset[u]) == 0 && v > u);
}

__device__ bool islarger_shared(const int v, const int u, double value_v1, double value_v2){
    return value_v1 > value_v2 || (value_v1 == value_v2 && v > u);
}

__device__ bool isless(const int v, const int u, const double* offset){
    return offset[v] < offset[u] || (offset[v] == offset[u] && v < u);
}

__device__ bool isless_shared(const int v, const int u, double value_v1, double value_v2){
    return value_v1 < value_v2 || (value_v1 == value_v2 && v < u);
}

__global__ void preserveSaddleOrder(uint8_t *changed_vertex_h, int saddletype = 0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nSaddle = nSaddle1;
    int *saddles = saddles1;
    if(saddletype == 1){
        nSaddle = nSaddle2;
        saddles = saddles2;
    }
    if(nSaddle < 2) return;
    if(i>=nSaddle) return;
    int id = saddles[i];
    double value = decp_data[id];
    if(i == 0){
        double nextValue = decp_data[saddles[1]];
        if(isless_shared(saddles[1], id, nextValue, value)){
            count_f_cases += 1;
            if(edit_mode != 0){
                int step = (int)floor((value - nextValue) / delta) + 1;
                if (step >= 5) step = 5;
                atomicMax(&edits[id], step);
            }
           
            else if (atomicCAS(&edits[id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, id, changed_vertex_h);
            }
        }
    }
    else if(i == nSaddle - 1){
        int prev_id = saddles[i - 1];
        double prevValue = decp_data[prev_id];
        if(isless_shared(id, prev_id,  value, prevValue)){
            count_f_cases += 1;
            if(edit_mode != 0){
                int step = (int)floor((prevValue - value) / delta) + 1;
                if (step >= 5) step = 5;
                atomicMax(&edits[prev_id], step);
            }
            else if (atomicCAS(&edits[prev_id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, prev_id, changed_vertex_h);
            }
        }
    }
    else{
        int prev_id = saddles[i - 1];
        int next_id = saddles[i + 1];
        double nextValue = decp_data[next_id];
        double prevValue = decp_data[prev_id];
        if(isless_shared(next_id, id, nextValue, value)){
            count_f_cases += 1;
            if(edit_mode != 0){
                int step = (int)floor((value - nextValue) / delta) + 1;
                if (step >= 5) step = 5;
                atomicMax(&edits[id], step);
            }
            else if (atomicCAS(&edits[id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, id, changed_vertex_h);
            }
        }

        if(isless_shared(id, prev_id,  value, prevValue)){
            count_f_cases += 1;
            if(edit_mode != 0){
                int step = (int)floor((prevValue - value) / delta) + 1;
                if (step >= 5) step = 5;
                atomicMax(&edits[prev_id], step);
            }
            else if (atomicCAS(&edits[prev_id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, prev_id, changed_vertex_h);
            }
        }

    }
}


__device__ int getTriangleID_internal(int v1, int v2, int v3, int width, int height, int depth, int direction = 0) {
    int id = -1;
    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    
    
    int min_x = min(v1%width, min(v2%width, v3%width));
    int min_y = min((v1/width)%height, min((v2/width)%height, (v3/width)%height));
    int min_z = min((v1 / (width * height)) % depth, min((v2 / (width * height)) % depth, (v3 / (width * height)) % depth));
    
    // xy palne
    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    if(direction == 0){

        int cell_id = min_y * (width - 1) + min_x;
        if(sortedV3_x == sortedV1_x && sortedV1_x == min_x && sortedV3_y == sortedV2_y && sortedV2_y == min_y){
            return cell_id * 2 + 2 * min_z * (width - 1) * (height - 1);
        }

        else{
            return cell_id * 2 + 1 + 2 * min_z * (width - 1) * (height - 1);
        }
    }
    // yz plane
    else if(direction == 1){
        int cell_id = min_y * (depth - 1) + min_z;
        if(sortedV2_y == sortedV1_y && sortedV1_y == min_y + 1 && sortedV3_z == sortedV2_z && sortedV2_z == min_z){
            return cell_id * 2 + 2 * depth  * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
        else{
            return cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
    }
    //  xz plane
    else if(direction == 2){
        int cell_id = min_z * (width - 1) + min_x;
        if(sortedV2_z == sortedV3_z && sortedV2_z == min_z && sortedV1_x == sortedV3_x && sortedV1_x == min_x){
            id = cell_id * 2 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
            return id;
        }
        else{
            return cell_id * 2 + 1 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
        }

    }

    else if(direction == 3){
        int cell_id = min_z * (width - 1) * (height - 1) + min_y * (width - 1) + min_x;
        int current_id = -1;
        if(sortedV3_x == min_x && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1&& sortedV2_y == min_y && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 0;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y + 1 && sortedV1_z == min_z+1 
        ) {current_id = 1;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 2;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 3;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 4;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x + 1 && sortedV1_y== min_y + 1 && sortedV1_z == min_z +1 
        ) {current_id = 5;}
        id = cell_id * 6 + current_id + 2 * depth * (width - 1) * (height - 1) + 2 * width * (height - 1) * (depth - 1) +  2 * height * (width - 1) * (depth - 1);
       
        return id;
    }
    return -1;
    

}

__device__ int find(int* parent, int x) {
    int root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (x != root) {
        int old = parent[x];
        parent[x] = root;  // 一次性路径压缩
        x = old;
    }
    return root;
}

__device__ void unionSets(int *parent, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);
    if (rootX != rootY) {
        parent[rootY] = rootX; // 连接两个集合
    }
}

__device__ void union_sets(int *parent, int *rank, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);
    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX] += 1;
        }
    }
}


__device__ bool areVectorsDifferent_local(int i, int *dec_stars, int lowerCount){
    int *stars = lowerStars + (maxNeighbors+1) * i;
    if(stars[maxNeighbors] != lowerCount){
        return true;
    }
    for (size_t j = 0; j < lowerCount; j++) {
        // if(i == 782) printf("%d %d\n", stars[(maxNeighbors+1) * i +j], dec_stars[(maxNeighbors+1) * i +j]);
        if (stars[j] != dec_stars[j]) {
            return true; 
        }
    }
    return false;
}

__device__ int binarySearchLUT(const int* keys, int numKeys, int target) {

    for(int i = 0; i< numKeys; i++){
        if(target == keys[i]) return i;
    }
    return -1;
}






__device__ int getTriangleID_2d(int v1, int v2, int v3, int width, int height, int depth){

    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    

    if(sortedV1_z == sortedV2_z && sortedV2_z == sortedV3_z){
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth) ;
        return id;
    }
    // yz plane
    if(sortedV1_x == sortedV2_x && sortedV2_x == sortedV3_x){
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth, 1) ;
        return id;
    }

    
    
    if(sortedV1_y == sortedV2_y && sortedV2_y == sortedV3_y){
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth, 2);
        return id;
    }
    
    else{
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth, 3);
        return id;
    }

    return -1;
}

__device__ int getTriangleID_3d_internel(int v1, int v2, int v3, int width, int height, int depth, int direction = 0) {
    int id = -1;
    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    
    
    int min_x = min(v1%width, min(v2%width, v3%width));
    int min_y = min((v1/width)%height, min((v2/width)%height, (v3/width)%height));
    int min_z = min((v1 / (width * height)) % depth, min((v2 / (width * height)) % depth, (v3 / (width * height)) % depth));
    
    // xy palne
    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    if(direction == 0){

        int cell_id = min_y * (width - 1) + min_x;
        if(sortedV3_x == sortedV1_x && sortedV1_x == min_x && sortedV3_y == sortedV2_y && sortedV2_y == min_y)
        {
            
            return cell_id * 2 + 2 * min_z * (width - 1) * (height - 1);
        }

        else
        {
            return cell_id * 2 + 1 + 2 * min_z * (width - 1) * (height - 1);
        }
    }
    // yz plane
    else if(direction == 1){
        int cell_id = min_y * (depth - 1) + min_z;
        if(sortedV2_y == sortedV1_y && sortedV1_y == min_y + 1 && sortedV3_z == sortedV2_z && sortedV2_z == min_z)
        {
            
            return cell_id * 2 + 2 * depth  * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
        else
        {
            // printf("%d\n",cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1));
            
            
            return cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
    }
    //  xz plane
    // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
    else if(direction == 2){
        int cell_id = min_z * (width - 1) + min_x;
        // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
        if(sortedV2_z == sortedV3_z && sortedV2_z == min_z && sortedV1_x == sortedV3_x && sortedV1_x == min_x)
        {
            id = cell_id * 2 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
            // if(id == 4858) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
            
            return id;
        }
        else
        {
            
            // printf("%d\n", cell_id * 2 + 1 + 2*(depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1)  + min_y*(width-1)*(depth-1)));
            return cell_id * 2 + 1 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
        }

    }

    else if(direction == 3){
        int cell_id = min_z * (width - 1) * (height - 1) + min_y * (width - 1) + min_x;
        
        int current_id = -1;
        if(sortedV3_x == min_x && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1&& sortedV2_y == min_y && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 0;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y + 1 && sortedV1_z == min_z+1 
        ) {current_id = 1;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 2;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 3;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 4;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x + 1 && sortedV1_y== min_y + 1 && sortedV1_z == min_z +1 
        ) {current_id = 5;}
        id = cell_id * 6 + current_id + 2 * depth * (width - 1) * (height - 1) + 2 * width * (height - 1) * (depth - 1) +  2 * height * (width - 1) * (depth - 1);
        // printf("%d\n", id);
        return id;
        
    }
    return -1;
    

}

__device__ int getTriangleID_3d(int v1, int v2, int v3, int width, int height, int depth){
    // xy plane

    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    

    if(sortedV1_z == sortedV2_z && sortedV2_z == sortedV3_z)
    {
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth) ;
        
        return id;
    }
    // yz plane
    if(sortedV1_x == sortedV2_x && sortedV2_x == sortedV3_x)
    {
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth, 1) ;
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_x * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) );
        return id;
    }

    
    
    if(sortedV1_y == sortedV2_y && sortedV2_y == sortedV3_y)
    {
        // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth, 2);
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_y * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) + width * 2*(width-1)*(height-1) );
        return id;
    }
    
    else
    {
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth, 3);
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_y * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) + width * 2*(width-1)*(height-1) );
        return id;

    }

    return -1;
}





__global__ void get_vertex_traingle(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=num_Elements) return;

    int x = index % width;
    int y = (index / (width)) % height;
    int z = (index / (width * height)) % depth;
    int numTriangles = 0;
    for (int k = 0; k < maxNeighbors; k++) {
        int dx = directions[k * 3 + 0];
        int dy = directions[k * 3 + 1];
        int dz = directions[k * 3 + 2];

        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;

        if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
        int v2 = nx + ny * (width) + nz * (width * height);

        int v2_x = nx;
        int v2_y = ny;
        int v2_z = nz;
        for(int i=0;i<maxNeighbors;i++){
            int nx = x + directions[i * 3 + 0];
            int ny = y + directions[i * 3 + 1];
            int nz = z + directions[i * 3 + 2];
            for(int j=0;j<maxNeighbors;j++){
                int nx1 = v2_x + directions[j * 3 + 0];
                int ny1 = v2_y + directions[j * 3 + 1];
                int nz1 = v2_z + directions[j * 3 + 2];
                int neighbor = nx + ny * width + nz* (height * width);
                
                if(nx == nx1 && ny == ny1 && nz == nz1 && 
                    nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 
                    && nz<depth && neighbor < num_Elements && neighbor >=0 
                    && neighbor!=index && neighbor != v2 && neighbor > v2){
                    int triangleID = getTriangleID_3d(index, v2, neighbor, width, height, depth);
                    vertex_cells[index*42 + numTriangles++] = triangleID;
                }
            }
            // for (int j = 0; j < maxNeighbors; j++) {
            //     int dx = directions[j * 3 + 0];
            //     int dy = directions[j * 3 + 1];
            //     int dz = directions[j * 3 + 2];

            //     int nx = v2_x + dx;
            //     int ny = v2_y + dy;
            //     int nz = v2_z + dz;
            //     if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
            //     int v3 = nx + ny * (width) + nz * (width * height);
            //     int triangleID = getTriangleID(i, v2, v3, width, height, depth );
            //     vertex_cells[i*maxNeighbors + numTriangles++] = triangleID;
            // }
        }

        
    }
    
}

__global__ void ComputeDirection(int type = 0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;

    double *offset = input_data;
    int *DS_M_t = DS_M;
    int *AS_M_t = AS_M;
    
    int largest_neighbor = i;
    int smallest_neighbor = i;
    int x = i%width;
    int y = (i/width) % height;
    int z = (i / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            // int neighbor_id = adjacency[maxNeighbors * i + j];
            // int r = neighbor_id;
            // if(neighbor_id == -1) continue;
            if(offset[largest_neighbor] < offset[r] || (offset[largest_neighbor] == offset[r] && largest_neighbor < r)) largest_neighbor = r;
            if(offset[smallest_neighbor] > offset[r] || (offset[smallest_neighbor] == offset[r] && smallest_neighbor > r)) smallest_neighbor = r;
        }
    }
    // if(i == 60111) std::cout<<"id is:"<<largest_neighbor<<std::endl;
    DS_M_t[i] = largest_neighbor;
    AS_M_t[i] = smallest_neighbor;
    
    // for(int j = 0; j< maxNeighbors; j++)
    // {
    //     int neighbor_id = adjacency[ maxNeighbors * i + j];
    //     if(neighbor_id == -1) continue;
        
    // }

    
    return;
};

__global__ void ComputeDescendingManifold(int type = 0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    
    int *DS_M_t = DS_M;
    
    
    int v = i;
    bool is_saddle_neighbor = false;
    int x = i%width;
    int y = (i/width) % height;
    int z = (i / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            if(vertex_type[3*r] == 2 || vertex_type[3*r] == 3){
                is_saddle_neighbor = true;
                break;
            }
        }
    }


    if(!is_saddle_neighbor) return;
    
    while(true){
        int u = DS_M_t[v];
        int w = DS_M_t[u];
        if (u == w) break;
        DS_M_t[v] = w;
    }
    return;
}; 

__global__ void ComputeAscendingManifold(int type = 0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;

    
    int *AS_M_t = AS_M;
    

    int v = i;
    bool is_saddle_neighbor = false;
    
    int x = i%width;
    int y = (i/width) % height;
    int z = (i / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            
            if(vertex_type[3*r] == 1 || vertex_type[3*r] == 3){
                is_saddle_neighbor = true;
                break;
            }
        }
    }
    is_saddle_neighbor = true;
    if(!is_saddle_neighbor) return;
    while(true){
        int u = AS_M_t[v];
        int w = AS_M_t[u];
        if (u == w) break;
        AS_M_t[v] = w;
    }
    return;
}; 



__global__ void init_Manifold(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    
    int *DS_M_t = DS_M;
    int *AS_M_t = AS_M;

    DS_M_t[i] = i;
    AS_M_t[i] = i;

}

__global__ void ExtractCP(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=num_Elements) return;

    int *minimum_t = minimum;
    int *saddles1_t = saddles1;
    int *saddles2_t = saddles2;
    int *maximum_t = maximum;
    int *nSaddle2_t = &nSaddle2;
    int *nSaddle1_t = &nSaddle1;
    int *nMax_t = &nMax;
    int *nMin_t = &nMin;
    
    int type = vertex_type[3*idx];

    if (type == 2 || type == 3) {
        int pos = atomicAdd(nSaddle2_t, 1);
        saddles2_t[pos] = idx;
    }

    if (type == 1 || type == 3) {
        int pos = atomicAdd(nSaddle1_t, 1);
        saddles1_t[pos] = idx;
    }

    if (type == 4) {
        int pos = atomicAdd(nMax_t, 1);
        maximum_t[pos] = idx;
    }

    if (type == 0) {
        int pos = atomicAdd(nMin_t, 1);
        minimum_t[pos] = idx;
    }

    
}

__global__ void ComputeTempArray(int direction = 0, int type = 0){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if((idx>=nMin && direction == 0) || (idx>=nMax && direction == 1) || (idx>=nSaddle1 && direction == 2) || (idx>=nSaddle2 && direction == 3)) return;

    int *tempArray_t = dec_tempArray;
    int *tempArrayMin_t = dec_tempArrayMin;
    int *saddleindex_t = dec_saddlerank;
    int *saddle1index_t = dec_saddle1rank;
    int *minimum_t = dec_minimum;
    int *maximum_t = dec_maximum;
    int *saddles2_t = dec_saddles2;
    int *saddles1_t = dec_saddles1;
    if(type!=0){
        tempArray_t = tempArray;
        tempArrayMin_t = tempArrayMin;
        minimum_t = minimum;
        maximum_t = maximum;
        saddleindex_t = saddlerank;
        saddle1index_t = saddle1rank;
        saddles2_t = saddles2;
        saddles1_t = saddles1;

    }
    if(direction == 0) tempArrayMin_t[minimum_t[idx]] = idx;
    else if(direction == 1){
        tempArray_t[maximum_t[idx]] = idx;
    }
    else if(direction == 2){
        saddle1index_t[saddles1_t[idx]] = idx;
    }
    else{
        saddleindex_t[saddles2_t[idx]] = idx;
    }
    
}



__device__ void insertionSort(int *arr, int n, double *offset){
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && isless(arr[j], key, offset)) {
            
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

__device__ void reversed_insertionSort(int *arr, int n, double *offset){
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        /* Move elements of arr[0..i-1], that are
           greater than key, to one position ahead
           of their current position */
        
        while (j >= 0 && islarger(arr[j], key, offset)) {
            
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}


__global__ void findAscPaths(int type = 0){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle2) return;

    int *saddleTriplets_t = saddleTriplets;
    int *desManifold = DS_M;
    double *offset = input_data;
    
    int temp_Saddle_tripplets[46];
   
    
    temp_Saddle_tripplets[44] = 0;
    const int vertexId = saddles2[index];
    int x = vertexId%width;
    int y = (vertexId/width) % height;
    int z = (vertexId / (width * height)) % depth;
    double vertexValue = offset[vertexId];
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            
            int maxi = desManifold[r];
            if(islarger(r, vertexId, offset) && fabs(offset[maxi] - vertexValue) >= thr) {
                temp_Saddle_tripplets[temp_Saddle_tripplets[44]] = desManifold[r];
                temp_Saddle_tripplets[44]++;
            }
        }
    }
   
    temp_Saddle_tripplets[45] = vertexId;

   
    int ItemSize = temp_Saddle_tripplets[44];

    
    insertionSort(temp_Saddle_tripplets, ItemSize, offset);
    for(int i = 0; i < ItemSize; i++){
        saddleTriplets_t[index * 46 + i] = temp_Saddle_tripplets[i];
    }
    saddleTriplets_t[index * 46 + 44] = temp_Saddle_tripplets[44];
    saddleTriplets_t[index * 46 + 45] = temp_Saddle_tripplets[45];
}




__global__ void findDescPaths(int type = 0){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle1) return;
    int *saddles1_t = saddles1;
    int *saddle1Triplets_t = saddle1Triplets;
    int *asManifold = AS_M;
    double *offset = input_data;
    int temp_Saddle_tripplets[46];

    
    temp_Saddle_tripplets[44] = 0;
    const int vertexId = saddles1_t[index];
    int x = vertexId%width;
    int y = (vertexId/width) % height;
    int z = (vertexId / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){

        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            if(isless(r, vertexId, offset)) {

                temp_Saddle_tripplets[temp_Saddle_tripplets[44]] = asManifold[r];
                temp_Saddle_tripplets[44]++;
            }
        }
    }
    

    int ItemSize = temp_Saddle_tripplets[44];
    temp_Saddle_tripplets[45] = saddles1_t[index];
    reversed_insertionSort(temp_Saddle_tripplets, ItemSize, offset);
   
    for(int i = 0; i < ItemSize; i++){
        saddle1Triplets_t[index * 46 + i] = temp_Saddle_tripplets[i];
    }
    saddle1Triplets_t[index * 46 + 44] = temp_Saddle_tripplets[44];
    saddle1Triplets_t[index * 46 + 45] = temp_Saddle_tripplets[45];
    
}



__device__ int computeMaxLabel(int i, const double *offset){
        
    int current_id = i;
    int largest_neighbor = current_id;  
    int next_largest_neighbor;
    while (true) {
        
        next_largest_neighbor = largest_neighbor;
        
        // if(i == 880) printf("%d\n", next_largest_neighbor);
        for (int j = 0; j < maxNeighbors; j++) {
            int neighbor_id = adjacency[maxNeighbors * current_id + j];
            if (neighbor_id == -1) continue;  

            
            if (offset[next_largest_neighbor] < offset[neighbor_id] || 
            (offset[next_largest_neighbor] == offset[neighbor_id] && next_largest_neighbor < neighbor_id)) {
                next_largest_neighbor = neighbor_id;
            }
        }

        if (next_largest_neighbor == largest_neighbor) break;

        current_id = next_largest_neighbor;
        largest_neighbor = next_largest_neighbor;
    }

    return current_id;

}

__device__ int computeMinLabel(int i, const double *offset){
        
    int current_id = i;
    int largest_neighbor = current_id;  

    while (true) {
        int next_largest_neighbor = largest_neighbor;
        
        for (int j = 0; j < maxNeighbors; j++) {
            int neighbor_id = adjacency[maxNeighbors * current_id + j];
            if (neighbor_id == -1) continue;  

            
            if (offset[next_largest_neighbor] > offset[neighbor_id] || 
            (offset[next_largest_neighbor] == offset[neighbor_id] && next_largest_neighbor > neighbor_id)) {
                next_largest_neighbor = neighbor_id;
            }
        }

        if (next_largest_neighbor == largest_neighbor) break;

        current_id = next_largest_neighbor;
        largest_neighbor = next_largest_neighbor;
    }

    return current_id;

}

__global__ void compute_Max_for_Saddle(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle2) return;
    int saddle = saddles2[index];
    int label_count = 0;
    for(int j = 0; j< maxNeighbors; j++){
        int neighborId = adjacency[maxNeighbors * saddle + j];
        if(neighborId == -1) continue;
        if(islarger(neighborId, saddle, input_data)){
            
            int l = computeMaxLabel(neighborId, input_data);
            or_saddle_max_map[index * 4 + label_count] = l;
            label_count++;
        }

    }
    return;
}

__global__ void compute_Min_for_Saddle(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle1) return;
    int label_count = 0;
    int saddle = saddles1[index];
    for(int j = 0; j< maxNeighbors; j++){
        int neighborId = adjacency[ maxNeighbors * saddle + j];
        if(neighborId == -1) continue;
        if(isless(neighborId, saddle, input_data)){
            int l = computeMinLabel(neighborId, input_data);
            or_saddle_min_map[index * 4 + label_count] = l;
            label_count++;
        }

    }
    return;
}



__global__ void init_edits(uint8_t* changed_vertex_h){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    edits[i] = 0;
    changed_vertex_h[i] = 0;
}




__device__ double atomicCASDouble(double* address, double val) {
   
    uint64_t* address_as_ull = (uint64_t*)address;
    
    uint64_t old_val_as_ull = *address_as_ull;
    uint64_t new_val_as_ull = __double_as_longlong(val);
    uint64_t assumed;


    assumed = old_val_as_ull;
    
    // return atomicCAS((unsigned long long int*)address, (unsigned long long int)compare, (unsigned long long int)val);
    
    old_val_as_ull = atomicCAS((unsigned long long int*)address_as_ull, (unsigned long long int)assumed, (unsigned long long int)new_val_as_ull);
    // } while (assumed != old_val_as_ull);
    return __longlong_as_double(old_val_as_ull);
}



__device__ int findDifferences(const int* vec1, int size1, 
                               const int* vec2, int size2, 
                               int index, int* differences) {
    int count = 0;
    
    
    for (int i = 0; i < size2; i++) {
        int val = vec2[index * (maxNeighbors + 1) + i];
        bool found = false;

        // 遍历 `vec1`，检查 `val` 是否存在
        for (int j = 0; j < size1; j++) {
            if (vec1[index * (maxNeighbors + 1) + j] == val) {
                found = true;
                break;
            }
        }

        // 如果 `val` 不在 `vec1`，则存入 `differences`
        if (!found) {
            differences[count++] = val;
            
        }
    }
    return count; // 返回 `differences` 数组的大小
}

__device__ int findDifferences_local(const int* vec1, int size1, const int* vec2, int size2, int* differences) {
    int count = 0;
    
    for (int i = 0; i < size2; i++) {
        int val = vec2[i];
        bool found = false;

        // 遍历 `vec1`，检查 `val` 是否存在
        for (int j = 0; j < size1; j++) {
            if (vec1[j] == val) {
                found = true;
                break;
            }
        }

        // 如果 `val` 不在 `vec1`，则存入 `differences`
        if (!found) {
            differences[count++] = val;
            
        }
    }
    return count; // 返回 `differences` 数组的大小
}


__device__ void fix_saddle_local(int i, int *o_lowerStar_start, int o_size1, uint8_t *changed_vertex_h){
    
    
    double delta;
    if (i < 0 || i >= num_Elements) return;
    if (maxNeighbors > MAX_NEIGHBORS) return;
    if (o_size1 < 0) o_size1 = 0;
    if (o_size1 > MAX_NEIGHBORS) o_size1 = MAX_NEIGHBORS;
    int lowerCount = 0;
    int lowerStar[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
   

    int gx = i % width;
    int gy = (i/width) % height;
    int gz = (i/(width*height)) % depth;
    double currentHeight = decp_data[i];
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = gx + dirX;
        int newY = gy + dirY;
        int newZ = gz + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            
            
            double neighbor_value = decp_data[r];

            if (neighbor_value < currentHeight || (neighbor_value == currentHeight && r < i)) {
                
                lowerStar[lowerCount] = r;
                lowerCount++;
                
            } 
        }
    }
    int diff[MAX_NEIGHBORS];
    double datavalue = decp_data[i];
    // find id in olowerstar but not in lowerstars
    int count = findDifferences_local(lowerStar, lowerCount, o_lowerStar_start, o_size1, diff);
    delta = -bound / (q);
    for(int index = 0; index<count; index++){
        const int id = diff[index];
        if (id < 0 || id >= num_Elements) continue;
        if(edit_mode != 0){
            int step = (int)floor((decp_data[id] - datavalue) / delta) + 1;
            if (step >= 5) step = 5;
            atomicMax(&edits[id], step);
        }
        else if (atomicCAS(&edits[id], 0, 1) == 0){         
            applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, id, changed_vertex_h);
        }
    }
    
    int diff1[MAX_NEIGHBORS];
    // find id in lowerstar but not in olowerstars
    count = findDifferences_local(o_lowerStar_start, o_size1, lowerStar, lowerCount, diff1);
    if(count > 0){
        
        if(edit_mode != 0){
            if (diff1[0] < 0 || diff1[0] >= num_Elements) return;
            double min_v = decp_data[diff1[0]];
            int min_id = diff1[0];
            for(int index = 1; index<count; index++){
                const int id = diff1[index];
                if (id < 0 || id >= num_Elements) continue;
                if(min_v > decp_data[id] || (min_v == decp_data[id] && min_id > id) ){
                    min_v = decp_data[id];
                    min_id = id;
                }
            }
            int step = (int)floor((datavalue - min_v) / delta) + 1;
            if (step >= 5) step = 5;
            atomicMax(&edits[i], step);
        }
        else if (atomicCAS(&edits[i], 0, 1) == 0){
            applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, i, changed_vertex_h);
        }
    }
    
    // }
    return;
}



__global__ void get_wrong_index_max(uint8_t *changed_vertex_h){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nSaddle2 ||saddleTriplets[i*46 + 44] == 0) return;

    int numberOfMax = saddleTriplets[i * 46 + 44] - 1;
    int ori_largest = saddleTriplets[i * 46 + numberOfMax];
    int dec_largest = ori_largest;
    double dec_largest_value = decp_data[ori_largest];
    // if(saddles2[i]==105645) printf("%.17f\n", ori_largest_value);
    for(int d = 0; d<numberOfMax+1; d++){
        int id = saddleTriplets[i * 46 + d];
        double decp_value = decp_data[id];
        // if(saddles2[i]==105645)printf("%.17f %.17f %.17f %.17f %d %d %d %d\n", input_data[id], decp_data[id], dec_largest_value, ori_largest_value, id, ori_largest, numberOfMax);
        if(isless_shared(id, dec_largest, decp_value, dec_largest_value)){
            dec_largest = id;
            dec_largest_value = decp_value;
        }
    }

    if(ori_largest != dec_largest){
        count_f_cases+=1;

        if(edit_mode != 0){
            int step = (int)floor((decp_data[ori_largest] - dec_largest_value) / delta) + 1;
            if (step >= 5) step = 5;
            atomicMax(&edits[ori_largest], step);
        }
        else if (atomicCAS(&edits[ori_largest], 0, 1) == 0) {
            applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, ori_largest, changed_vertex_h);
        }
    }
        
    
}

__global__ void get_wrong_index_min(uint8_t *changed_vertex_h){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nSaddle1 || saddle1Triplets[i*46 + 44] == 0) return;

    int numberOfMin = saddle1Triplets[i * 46 + 44] - 1;
    int ori_largest = saddle1Triplets[i * 46 + numberOfMin];
    int dec_largest = ori_largest;
    double dec_largest_value = decp_data[ori_largest];
    
    for(int d = 0; d<numberOfMin+1; d++){
        int id = saddle1Triplets[i * 46 + d];
        double decp_value = decp_data[id];  
        
        if(islarger_shared(id, dec_largest, decp_value, dec_largest_value)){
            dec_largest = id;
            dec_largest_value = decp_value;
        }
    }
    
    if(ori_largest != dec_largest){
        count_f_cases+=1;
        if(edit_mode != 0){
            int step = (int)floor((dec_largest_value - decp_data[ori_largest]) / delta) + 1;
            if (step >= 5) step = 5;
            atomicMax(&edits[dec_largest], step);
        }
        else if (atomicCAS(&edits[dec_largest], 0, 1) == 0) {
            applyDeltaBuffer_local<double>(delta_counter,
                                decp_data, delta, input_data,
                                bound, q, dec_largest, changed_vertex_h);
        }
    }
        
    
}


__device__ bool isCommonNeighbor(int dx, int dy, int dz) {
    
    for(int i = 0; i<14; i++){
        if(dx == directions[ 3 * i + 0] && dy == directions[3 * i + 1] && dz == directions[3 * i + 2]) return true;
    }
    return false;
}

__global__ void generateLookupTable() {

    for (int i = 0; i < 14; i++) {
        int dx = directions[ i * 3 + 0];
        int dy = directions[ i * 3 + 1];
        int dz = directions[ i * 3 + 2];
        int v3_count = 0;
        for (int j = 0; j < 14; j++) {
            int dx1 = directions[ j * 3 + 0];
            int dy1 = directions[ j * 3 + 1];
            int dz1 = directions[ j * 3 + 2];
            
            int dx2 = dx + dx1;
            int dy2 = dy + dy1;
            int dz2 = dz + dz1;
            // if(i == 0 && j == 10) printf("%d %d %d %d %d %d\n", )
            if(isCommonNeighbor(dx2, dy2, dz2)){
                lookupTable[i][v3_count][0] = dx2;
                lookupTable[i][v3_count][1] = dy2;
                lookupTable[i][v3_count][2] = dz2;
                v3_count++;
            }
            // else{
            //     lookupTable[i][j][0] = -2;
            //     lookupTable[i][j][1] = -2;
            //     lookupTable[i][j][2] = -2;
            // }
            
        }   
        
        while (v3_count < MAX_V3) {
            lookupTable[i][v3_count][0] = -2;
            lookupTable[i][v3_count][1] = -2;
            lookupTable[i][v3_count][2] = -2;
            v3_count++;
        }

    }
}



__global__ void update_unique_sizes(int *saddleTriplets_t, int *unique_sizes, int nSaddle2) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nSaddle2) {
        saddleTriplets_t[index * 46 + 44] = unique_sizes[index];
    }
}

__global__ void count_vulnerable_live_kernel(const double* data,
                                             double xi_abs,
                                             unsigned long long* vulnerable_edges,
                                             unsigned long long* total_unique_pairs,
                                             unsigned long long* all_neighbors_vulnerable_points) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_Elements) return;

    int gx = g_idx % width;
    int gy = (g_idx / width) % height;
    int gz = (g_idx / (width * height)) % depth;

    double fu = data[g_idx];
    int total_neighbors = 0;
    int vulnerable_neighbors = 0;
    double thr2 = 2.0 * xi_abs;

    for (int d = 0; d < maxNeighbors; d++) {
        int dx = directions[d * 3];
        int dy = directions[d * 3 + 1];
        int dz = directions[d * 3 + 2];

        int nx = gx + dx;
        int ny = gy + dy;
        int nz = gz + dz;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;

        int v = nx + ny * width + nz * (width * height);
        double fv = data[v];
        double diff = fabs(fu - fv);

        total_neighbors++;
        if (diff <= thr2) vulnerable_neighbors++;
        if (g_idx < v) atomicAdd(total_unique_pairs, 1ULL);
        if (fu > fv && diff <= thr2) atomicAdd(vulnerable_edges, 1ULL);
    }

    if (total_neighbors > 0 && vulnerable_neighbors == total_neighbors) {
        atomicAdd(all_neighbors_vulnerable_points, 1ULL);
    }
}

__global__ void relax_neighbor_levels_live_kernel(const double* data,
                                                  double xi_abs,
                                                  int* levels,
                                                  int* changed) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_Elements) return;

    int gx = g_idx % width;
    int gy = (g_idx / width) % height;
    int gz = (g_idx / (width * height)) % depth;

    double fv = data[g_idx];
    double thr2 = 2.0 * xi_abs;
    int best = 1;

    for (int d = 0; d < maxNeighbors; d++) {
        int dx = directions[d * 3];
        int dy = directions[d * 3 + 1];
        int dz = directions[d * 3 + 2];

        int nx = gx + dx;
        int ny = gy + dy;
        int nz = gz + dz;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;

        int u = nx + ny * width + nz * (width * height);
        double fu = data[u];
        double diff = fabs(fu - fv);
        if (fu > fv && diff <= thr2) {
            int cand = levels[u] + 1;
            if (cand > best) best = cand;
        }
    }

    if (best > levels[g_idx]) {
        atomicMax(&levels[g_idx], best);
        atomicExch(changed, 1);
    }
}


// void sortTripletsOnGPU(int *saddleTriplets_t, int nSaddle2, double *offset, int type = 0) {
//     thrust::device_ptr<int> d_ptr(saddleTriplets_t);
    
//     // 申请一个 `device_vector` 来存储 unique_size 结果
//     thrust::device_vector<int> unique_sizes(nSaddle2);

//     // 并行排序 & 去重
//     for (int index = 0; index < nSaddle2; index++) {
//         int sort_size;
//         cudaMemcpy(&sort_size, &saddleTriplets_t[index * 46 + 44], sizeof(int), cudaMemcpyDeviceToHost);

//         thrust::sort(d_ptr + index * 46, d_ptr + index * 46 + sort_size, OffsetComparator(offset));
//         auto new_end = thrust::unique(d_ptr + index * 46, d_ptr + index * 46 + sort_size);
        
//         // 计算 unique_size
//         unique_sizes[index] = thrust::distance(d_ptr + index * 46, new_end);
//     }

//     // 用 CUDA 内核更新 `saddleTriplets_t[index * 46 + 44]`
//     int *d_unique_sizes = thrust::raw_pointer_cast(unique_sizes.data());
//     update_unique_sizes<<<(nSaddle2 + 255) / 256, 256>>>(saddleTriplets_t, d_unique_sizes, nSaddle2);
// }

// void sortCP(int *minimum, int *maximum,
//             int *saddles1, int *saddles2,
//             int *saddlerank, int *saddle1rank,
//             double *offset, int nMin, int nMax, 
//             int nSaddle1, int nSaddle2, int type = 1){
//     thrust::device_ptr<int> minimum_ptr(minimum);
//     thrust::device_ptr<int> maximum_ptr(maximum);
//     thrust::device_ptr<int> saddles1_ptr(saddles1);
//     thrust::device_ptr<int> saddles2_ptr(saddles2);
//     thrust::device_ptr<int> indices1_ptr(saddle1rank);
//     thrust::device_ptr<int> indices2_ptr(saddlerank);

//     thrust::sort(minimum_ptr, minimum_ptr + nMin, OffsetComparator(offset));
//     thrust::sort(maximum_ptr, maximum_ptr + nMax, OffsetComparator(offset));
//     thrust::sort(saddles1_ptr, saddles1_ptr + nSaddle1, OffsetComparator(offset));
//     thrust::sort(saddles2_ptr, saddles2_ptr + nSaddle2, OffsetComparator(offset));

//     thrust::sequence(indices1_ptr, indices1_ptr + nSaddle1);
//     thrust::sequence(indices2_ptr, indices2_ptr + nSaddle2);


//     // thrust::sort_by_key(saddles1_ptr, saddles1_ptr + nSaddle1, indices1_ptr, OffsetComparator(offset));
//     // thrust::sort_by_key(saddles2_ptr, saddles2_ptr + nSaddle2, indices2_ptr, OffsetComparator(offset));

// }

template<typename T>
void saveArrayToBin(const T* arr, size_t size, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(arr), size * sizeof(T));

    outFile.close();
}


__global__ void classifyVertex_CUDA(uint8_t *changed_vertex_h, uint8_t *changed_vertex_copy, int ite, int type = 0, int local = 1) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= num_Elements ) return;
    // 

    double* heightMap = input_data;
    int* lowerStars_t = lowerStars;

    if(type == 1){
        heightMap = decp_data;
    }
    
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(g_idx >= num_Elements) return;
    if(maxNeighbors > MAX_NEIGHBORS) return;
    int gx = g_idx % width;
    int gy = (g_idx / width) % height;
    int gz = (g_idx / (width * height)) % depth;
    if (!(gx < width && gy < height && gz < depth))  return;
    if(type == 1 && ite != 1 && changed_vertex_copy[g_idx] == 0){
        return;
    }

    
    double data_value = heightMap[g_idx];
    int vertexId = g_idx;

    int lowerCount = 0, upperCount = 0;
    int lowerStar[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int upperStar[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    int largest_index = g_idx;
    double largest_value = data_value;
    
    int smallest_index = g_idx;
    double smallest_value = data_value;
    

    int neighbor_size = 0;
    int binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int vertex_binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = gx + dirX;
        int newY = gy + dirY;
        int newZ = gz + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < num_Elements && newZ<depth && newZ>=0) {
    
            
            
            double neighbor_value = heightMap[r];
            if (neighbor_value < data_value || (neighbor_value == data_value && r < g_idx)) {
                if(type==0) lowerStars_t[vertexId*(maxNeighbors+1) + lowerCount] = r;
                binary[maxNeighbors - 1 - neighbor_size] = 0;
                lowerStar[lowerCount] = r;
                lowerCount++;
                
            } else if (neighbor_value > data_value || (neighbor_value == data_value && r > g_idx)) {
                
                upperStar[upperCount] = r;
                binary[maxNeighbors - 1 - neighbor_size] = 1;
                upperCount++;
                
            }
            
            
            vertex_binary[13 - d] = 1;
            neighbor_size++;
            if(islarger_shared(r, largest_index, neighbor_value, largest_value)){
                largest_index = r;
                largest_value = neighbor_value;
            }

            
            if(isless_shared(r, smallest_index, neighbor_value, smallest_value)){
                smallest_index = r;
                smallest_value = neighbor_value;

            }
        }
    }

    if(type==0) lowerStars_t[vertexId*(maxNeighbors+1) + maxNeighbors] = lowerCount;
    
    int decimal_value = 0;
    for (int i = 0; i < maxNeighbors; i++) {
        decimal_value = (decimal_value << 1) | binary[i]; // 左移并加上当前位
    }

    int vertex_types = 0;
    for (int i = 0; i < maxNeighbors; i++) {
        vertex_types= (vertex_types << 1) | vertex_binary[i]; // 左移并加上当前位
    }
    int keyIndex = binarySearchLUT(keys, NUM_KEYS, vertex_types);
    
    int LUT_result = 5;
    if(upperCount == 0) LUT_result = 4;
    else if(lowerCount == 0) LUT_result = 0;
    else if (keyIndex != -1) {
        LUT_result = LUT_cuda[lutOffsets[keyIndex] + decimal_value];
    } else {
        LUT_result = -1; // 未找到
    }

    
    if(type==0){
        vertex_type[3 * g_idx] = LUT_result;
        vertex_type[3 * g_idx + 1] = smallest_index;
        vertex_type[3 * g_idx + 2] = largest_index;

    }
    
    else{

        int ori_type = vertex_type[3 * g_idx];
        
        // if(LUT_result != ori_type){
        //         // maximum
        //         if((LUT_result==4 and ori_type!=4) or (LUT_result!=4 and ori_type==4)){
                    
        //             count_f_cases+=1;
        //             if (ori_type != 4) {
        //                 if (atomicCAS(&edits[g_idx], 0, 1) == 0){
                            
        //                     applyDeltaBuffer_local<double>(delta_counter,
        //                                     decp_data, delta, input_data,
        //                                     bound, q, g_idx, changed_vertex_h);
        //                 }
                        
        //             } else {
        //                 if (atomicCAS(&edits[largest_index], 0, 1) == 0) {
        //                     applyDeltaBuffer_local<double>(delta_counter,
        //                                     decp_data, delta, input_data,
        //                                     bound, q, largest_index, changed_vertex_h);
        //                 }
                        
        //             }
        //         } 
        //         if((LUT_result==0 and ori_type!=0) or (LUT_result!=0 and ori_type==0)){
                    
        //             count_f_cases+=1;
        //             if (ori_type != 0) {
        //                 int original_smallest_index = vertex_type[3 * g_idx + 1];
        //                 if (atomicCAS(&edits[original_smallest_index], 0, 1) == 0) {
        //                     applyDeltaBuffer_local<double>(delta_counter,
        //                                     decp_data, delta, input_data,
        //                                     bound, q, original_smallest_index, changed_vertex_h);
        //                 }
        //             } else{
        //                 if (atomicCAS(&edits[g_idx], 0, 1) == 0) {
        //                     applyDeltaBuffer_local<double>(delta_counter,
        //                                     decp_data, delta, input_data,
        //                                     bound, q, g_idx, changed_vertex_h);
        //                 }
                        
        //             }
        //         }

        if((LUT_result==3 and ori_type!=3) or (LUT_result!=3 and ori_type==3) or (LUT_result==1 and ori_type!=1) or (LUT_result!=1 and ori_type==1) or 
            (LUT_result==2 and ori_type!=2) or (LUT_result!=2 and ori_type==2)){
            // printf("saddle diff %d\n", g_idx);
            count_f_cases+=1;
            fix_saddle_local(g_idx, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors], changed_vertex_h);
        }


        // }

        if(LUT_result==2 and ori_type==2 || LUT_result== 1 and ori_type==1 || LUT_result== 3 and ori_type==3){
                
            if(areVectorsDifferent_local(g_idx, lowerStar, lowerCount)){
                count_f_cases += 1;
                // printf("saddle neighbor diff %d\n", g_idx);
                fix_saddle_local(g_idx, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors], changed_vertex_h);
            }
        }

        int original_largest_index  = vertex_type[3 * g_idx + 2];
        if (original_largest_index < 0 || original_largest_index >= num_Elements) return;
        
        if(largest_index != original_largest_index){
            count_f_cases+=1;
            // printf("original_largest_index %d\n", g_idx);
            if(edit_mode != 0){
                int step = (int)floor((decp_data[largest_index] - decp_data[original_largest_index]) / delta) + 1;
                if (step >= 5) step = 5;
                atomicMax(&edits[largest_index], step);
            }
            else if (atomicCAS(&edits[largest_index], 0, 1) == 0) {
                applyDeltaBuffer_local<double>(delta_counter,
                                decp_data, delta, input_data,
                                bound, q, largest_index, changed_vertex_h);
            }
            
            
        }

        int original_smallest_index  = vertex_type[3 * g_idx + 1];
        if (original_smallest_index < 0 || original_smallest_index >= num_Elements) return;
        
        if(smallest_index != original_smallest_index){
            // printf("original_smallest_index %d\n", g_idx);
            count_f_cases+=1;
            if(edit_mode != 0){
                
                int step = (int)floor((decp_data[original_smallest_index] - decp_data[smallest_index]) / delta) + 1;
                // if(g_idx == 108681) printf("original_smallest_index %d: %.17f %.17f, %d: %.17f %.17f %d\n", original_smallest_index,decp_data[original_smallest_index], input_data[original_smallest_index] - bound, smallest_index, decp_data[smallest_index], input_data[smallest_index] - bound, step);
                if (step >= 5) step = 5;
                atomicMax(&edits[original_smallest_index], step);
            }
            else if (atomicCAS(&edits[original_smallest_index], 0, 1) == 0) {
                applyDeltaBuffer_local<double>(delta_counter,
                                decp_data, delta, input_data,
                                bound, q, original_smallest_index, changed_vertex_h);
            }
            
        }
        return;

    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void c_loops(int &ite, dim3 gridSize, dim3 blockSize, int host_count_f_max, int host_count_f_min, int host_count_f_saddle, int host_count_f_dir, int host_count_f_order_saddle, int host_count_f_cases, double delta){
    float c_loopt = 0.0;
    
    dim3 blockDim1(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridDim1((width_host + TILE_SIZE - 1) / TILE_SIZE,
                 (height_host + TILE_SIZE - 1) / TILE_SIZE,
                 (depth_host + TILE_SIZE - 1) / TILE_SIZE);

    
    
    cudaEvent_t start1, stop, start;

    cudaEventCreate(&start1);
    cudaEventCreate(&stop);
    cudaEventCreate(&start);
    cudaEventRecord(start1, 0);
    cudaEventRecord(start, 0);

    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_order_saddle, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    
    cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(int), 0, cudaMemcpyDeviceToHost);
    int cnt = 0;
    
    host_count_f_cases = 1;
    float elapsedTime = 0.0;
    dim3 gridSize_1saddle((nSaddle1_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_2saddle((nSaddle2_host + blockSize.x - 1) / blockSize.x);
    uint8_t *changed_vertex_copy, *changed_vertex_h;
    cudaMalloc(&changed_vertex_copy, num_Elements_host * sizeof(uint8_t));
    cudaMemset(changed_vertex_copy, 0,  num_Elements_host * sizeof(uint8_t));

    cudaMalloc(&changed_vertex_h, num_Elements_host * sizeof(uint8_t));
    cudaMemset(changed_vertex_h, 0,  num_Elements_host * sizeof(uint8_t));
    checkCudaError(cudaDeviceSynchronize(), "before cloops 1");
    // while(host_count_f_max>0 or host_count_f_min>0 or host_count_f_saddle>0 or host_count_f_dir>0 or host_count_f_order_saddle > 0){
    while(host_count_f_cases > 0){
        // std::cout<<"c_loops: "<<host_count_f_cases<<", "<<host_count_f_min<<", "<<host_count_f_saddle<<", "<<host_count_f_dir << "," << host_count_f_order_saddle <<std::endl;
        cnt++;
        
        std::vector<float> temp;
        
        
        cudaMemcpy(changed_vertex_copy, changed_vertex_h, num_Elements_host * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        init_edits<<<gridSize, blockSize>>>(changed_vertex_h);
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        classifyVertex_CUDA<<<gridSize, blockSize>>>(changed_vertex_h, changed_vertex_copy, ite, 1, 0);
        checkCudaError(cudaDeviceSynchronize(), "classifyVertex_CUDA");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        temp.push_back(elapsedTime/1000);

        // cudaMemcpyFromSymbol(&host_count_f_cases, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize(); 
        // std::cout<<"after vertex: "<<host_count_f_cases<<std::endl;

        // cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaEventRecord(start, 0);
        if (nSaddle1_host > 0) {
            preserveSaddleOrder<<<gridSize_1saddle, blockSize>>>(changed_vertex_h);
            cudaDeviceSynchronize();
        }
        if (nSaddle2_host > 0) {
            preserveSaddleOrder<<<gridSize_2saddle, blockSize>>>(changed_vertex_h, 1);
            cudaDeviceSynchronize();
        }
        // cudaMemcpyFromSymbol(&host_count_f_cases, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize(); 
        // std::cout<<"after preserveSaddleOrder: "<<host_count_f_cases<<std::endl;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        temp.push_back(elapsedTime/1000);

        cudaEventRecord(start, 0);
        if (nSaddle2_host > 0) {
            get_wrong_index_max<<<gridSize_2saddle, blockSize>>>(changed_vertex_h);
            cudaDeviceSynchronize();
        }
        if (nSaddle1_host > 0) {
            get_wrong_index_min<<<gridSize_1saddle, blockSize>>>(changed_vertex_h);
            cudaDeviceSynchronize();
        }
        // cudaMemcpyFromSymbol(&host_count_f_cases, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize(); 
        // std::cout<<"after get_wrong_index_min: "<<host_count_f_cases<<std::endl;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        temp.push_back(elapsedTime/1000);
        
        if(edit_mode_host == 1){
            applyDeltaBuffer<double><<<gridSize, blockSize>>>(num_Elements_host, 
                                changed_vertex_h);
            cudaDeviceSynchronize();
        }
        
        cudaMemcpyFromSymbol(&host_count_f_cases, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
        // cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
        // cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
        // cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(int));
        // cudaMemcpyToSymbol(count_f_order_saddle, &initialValue, sizeof(int));
        cudaDeviceSynchronize();
        time_counter.push_back(temp);
        ite++;
    }

    std::ofstream outFilep("EXaCTz_breakdown_"+compressor_id+"_"+filename+".txt", std::ios::app);
        // 检查文件是否成功打开
    if (!outFilep) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return; // 返回错误码
    }
    int c1 = 0;
    outFilep << "ite: "<<ite <<std::endl;
    for (const auto& row : time_counter) {
        outFilep << "iteration: "<<c1<<": ";
        for (size_t i = 0; i < row.size(); ++i) {
            outFilep << row[i];
            if (i != row.size() - 1) { 
                outFilep << ", ";
            }
        }
        
        outFilep << std::endl;
        c1+=1;
    }
    outFilep << "\n"<< std::endl;
    outFilep.close();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start1, stop);

    std::cout<<"cloops only" <<elapsedTime/1000<<std::endl;
}



double calculateMSE(const double* original, const double* compressed) {

    double mse = 0.0;
    for (size_t i = 0; i < num_Elements_host; i++) {
        mse += std::pow(static_cast<double>(original[i]) - compressed[i], 2);
    }
    mse /= num_Elements_host;
    return mse;
}

double calculatePSNR( double* original, double* compressed, double maxValue) {
    double mse = calculateMSE(original, compressed);
    if (mse == 0) {
        return std::numeric_limits<double>::infinity(); // Perfect match
    }
    double psnr = -20.0*log10(sqrt(mse)/maxValue);
    return psnr;
}

std::vector<uint8_t> encodeTo3BitBitmap(const std::vector<int>& data) {
    std::vector<uint8_t> bitmap; // 存储结果的位图
    uint8_t currentByte = 0; // 当前字节
    int bitIndex = 0; // 当前位的索引

    for (int value : data) {
        if (value < 0 || value > 7) {
            
            std::cerr << "错误：输入值超出范围 (0-6)。"<<value << std::endl;
        return {};
        }

        // 将当前值的3位插入到当前字节
        currentByte |= (value << (bitIndex)); // 将值左移到正确的位置
        bitIndex += 3; // 增加3位

        // 检查是否需要写入字节
        if (bitIndex >= 8) { // 如果当前字节已填满（8位）
            bitmap.push_back(currentByte); // 将填满的字节加入位图
            bitIndex -= 8; // 减去已填满的位
            currentByte = (value >> (3 - bitIndex)); // 如果有剩余的位，存储到下一个字节的起始位置
        }
    }

    // 如果最后有未完成的字节，写入它
    if (bitIndex > 0) {
        bitmap.push_back(currentByte);
    }

    return bitmap;
}


template<typename T>
std::vector<T> get_diff_index(std::vector<T> index){
    std::vector<T> index_tmp;
    index_tmp.push_back(index[0]);
    for(int i = 1; i<index.size(); i++){
        index_tmp.push_back(index[i] - index[i-1]);
    }
    return index_tmp;
}

template<typename T>
size_t save_and_compress(const std::string& out_filename, const std::vector<T>& data, int compression_level = 3) {
    // 原始数据大小
    size_t data_size = data.size() * sizeof(T);
    // 申请压缩缓冲区
    size_t bound = ZSTD_compressBound(data_size);
    std::vector<char> compressed(bound);

    // 压缩
    size_t compressed_size = ZSTD_compress(compressed.data(), bound, data.data(), data_size, compression_level);
    if (ZSTD_isError(compressed_size)) {
        std::cerr << "ZSTD compression error: " << ZSTD_getErrorName(compressed_size) << std::endl;
        return 0;
    }

    // 写入压缩文件
    std::ofstream ofs(out_filename, std::ios::binary);
    ofs.write(compressed.data(), compressed_size);
    ofs.close();
    return compressed_size;
}

inline std::vector<uint8_t>
pack_nbits(const uint8_t* symbols, size_t n_symbols, unsigned nbits) {
    if (nbits == 0 || nbits > 16) throw std::invalid_argument("nbits must be 1..16");
    std::vector<uint8_t> out;
    out.reserve((n_symbols * nbits + 7) / 8);

    uint64_t bitbuf = 0; unsigned bitcnt = 0;
    const uint64_t mask = (1ULL << nbits) - 1ULL;
    for (size_t i = 0; i < n_symbols; ++i) {
        const uint64_t v = symbols[i];
        if (v > mask) throw std::out_of_range("symbol exceeds 0..N range");
        bitbuf |= (v & mask) << bitcnt;
        bitcnt += nbits;
        while (bitcnt >= 8) {
            out.push_back(static_cast<uint8_t>(bitbuf & 0xFF));
            bitbuf >>= 8; bitcnt -= 8;
        }
    }
    if (bitcnt) out.push_back(static_cast<uint8_t>(bitbuf & 0xFF));
    return out;
}

inline unsigned bits_for_max(unsigned N) {
    
    if (N == 0) return 1;
    return (unsigned)std::ceil(std::log2((double)N + 1.0));
}


inline std::vector<uint8_t> zstd_compress(const void* src, size_t src_sz, int level = 5) {
    size_t bound = ZSTD_compressBound(src_sz);
    std::vector<uint8_t> out(bound);
    size_t csz = ZSTD_compress(out.data(), bound, src, src_sz, level);
    if (ZSTD_isError(csz)) throw std::runtime_error(ZSTD_getErrorName(csz));
    out.resize(csz);
    return out;
}

inline std::vector<uint8_t> zstd_decompress_exact(const uint8_t* data, size_t compressed_size, size_t decompressed_size) {
    std::vector<uint8_t> out(decompressed_size);
    if (decompressed_size == 0) return out;
    size_t dsz = ZSTD_decompress(out.data(), decompressed_size, data, compressed_size);
    if (ZSTD_isError(dsz)) throw std::runtime_error(ZSTD_getErrorName(dsz));
    if (dsz != decompressed_size) throw std::runtime_error("ZSTD decompressed size mismatch");
    return out;
}

inline std::vector<uint8_t> zstd_compress(const uint8_t* data, size_t n, int level = 5) {
    return zstd_compress(static_cast<const void*>(data), n, level);
}


inline std::vector<uint8_t>
compress_symbols_0_N(const std::vector<uint8_t>& symbols,
                     unsigned N,               
                     int level = 5,
                     size_t* packed_bytes_out = nullptr)
{
    const unsigned nbits = bits_for_max(N);
    auto packed = pack_nbits(symbols.data(), symbols.size(), nbits);
    if (packed_bytes_out) *packed_bytes_out = packed.size();
    return zstd_compress(packed.data(), packed.size(), level);
}

std::vector<uint8_t> build_edit_class_map(const double* before,
                                          const double* after,
                                          const std::vector<uint8_t>& delta_counter) {
    std::vector<uint8_t> classes(num_Elements_host, 0);
    for (int i = 0; i < num_Elements_host; ++i) {
        if (before[i] == after[i]) {
            classes[i] = 0;
        } else if (delta_counter[i] == threshold_host) {
            classes[i] = 2;
        } else {
            classes[i] = 1;
        }
    }
    return classes;
}

std::string get_scratch_output_dir() {
    const char* scratch_env = std::getenv("SCRATCH");
    std::filesystem::path base = scratch_env ? std::string(scratch_env) : exactz_scratch();
    std::filesystem::path out_dir = base / "ExtreM_z_outputs";
    std::filesystem::create_directories(out_dir);
    return out_dir.string();
}

void save_edit_class_map(const std::string& out_path,
                         const std::vector<uint8_t>& classes,
                         const std::string& label) {
    saveArrayToBin<uint8_t>(classes.data(), classes.size(), out_path);

    size_t no_edit = 0, edited = 0, special = 0;
    for (uint8_t value : classes) {
        if (value == 0) no_edit++;
        else if (value == 1) edited++;
        else if (value == 2) special++;
    }

    std::cout << "[errormap] " << label
              << " path=" << out_path
              << ", no_edit=" << no_edit
              << ", edit=" << edited
              << ", lossless_or_special=" << special
              << std::endl;
}

struct LosslessEditExperimentStats {
    size_t total_points = 0;
    size_t edited_points = 0;
    size_t threshold_points = 0;
    size_t special_correction_exact_match_count = 0;
    size_t reconstruction_mismatch_count = 0;
    size_t bound_violation_count = 0;
    double max_abs_reconstruction_diff = 0.0;
    double mean_abs_special_correction = 0.0;
    double special_correction_codec_mae = 0.0;
    double special_correction_codec_max_abs = 0.0;
    size_t compressed_special_correction_bytes = 0;
};

struct BoundViolationStats {
    size_t total_points = 0;
    size_t violation_count = 0;
    double violation_probability = 0.0;
    double max_abs_error = 0.0;
    double max_excess_over_bound = 0.0;
};

BoundViolationStats compute_bound_violation_stats(
    const double* reference,
    const double* candidate,
    double bound_value) {
    BoundViolationStats stats;
    stats.total_points = num_Elements_host;
    if (!reference || !candidate || num_Elements_host <= 0) return stats;

    for (int i = 0; i < num_Elements_host; ++i) {
        double abs_err = std::abs(reference[i] - candidate[i]);
        stats.max_abs_error = std::max(stats.max_abs_error, abs_err);
        if (abs_err > bound_value) {
            stats.violation_count++;
            stats.max_excess_over_bound = std::max(stats.max_excess_over_bound, abs_err - bound_value);
        }
    }
    stats.violation_probability = static_cast<double>(stats.violation_count) / stats.total_points;
    return stats;
}

LosslessEditExperimentStats run_lossless_special_correction_experiment(
    const double* before,
    const double* after,
    const double* input_data,
    const std::vector<uint8_t>& delta_counter,
    const std::string& compressor_id,
    std::vector<double>* reconstructed_output = nullptr) {
    LosslessEditExperimentStats stats;
    stats.total_points = num_Elements_host;
    if (!before || !after || !input_data) return stats;
    if (reconstructed_output) {
        reconstructed_output->assign(num_Elements_host, 0.0);
    }

    std::vector<int> threshold_indices;
    std::vector<double> special_corrections;
    long double sum_abs_special_correction = 0.0;
    for (int i = 0; i < num_Elements_host; ++i) {
        if (before[i] != after[i]) {
            stats.edited_points++;
            if (delta_counter[i] == threshold_host) {
                threshold_indices.push_back(i);
                double correction = after[i] - before[i];
                special_corrections.push_back(correction);
                sum_abs_special_correction += std::abs(correction);
            }
        }
    }
    stats.threshold_points = threshold_indices.size();
    if (stats.threshold_points > 0) {
        stats.mean_abs_special_correction = static_cast<double>(sum_abs_special_correction / stats.threshold_points);
    }

    std::vector<double> decoded_special_corrections(stats.threshold_points, 0.0);
    if (!special_corrections.empty()) {
        if (compressor_id == "sz3" || compressor_id == "zfp" || compressor_id == "cuszp" ||
            compressor_id == "mgard" || compressor_id == "zfp_cuda" || compressor_id == "sperr") {
            SZ3::Config conf(static_cast<size_t>(special_corrections.size()));
            // conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
            conf.errorBoundMode = SZ3::EB_REL;
            conf.relErrorBound = er;
            size_t correction_cmp_size = 0;
            char* correction_cmp = SZ_compress(conf, special_corrections.data(), correction_cmp_size);
            stats.compressed_special_correction_bytes = correction_cmp_size;
            double* decoded_ptr = SZ_decompress<double>(conf, correction_cmp, correction_cmp_size);
            std::copy(decoded_ptr, decoded_ptr + special_corrections.size(), decoded_special_corrections.begin());
            delete[] correction_cmp;
            delete[] decoded_ptr;
        } else {
            std::vector<uint8_t> compressed = zstd_compress(
                static_cast<const void*>(special_corrections.data()),
                special_corrections.size() * sizeof(double),
                5);
            stats.compressed_special_correction_bytes = compressed.size();
            std::vector<uint8_t> decompressed = zstd_decompress_exact(
                compressed.data(),
                compressed.size(),
                special_corrections.size() * sizeof(double));
            std::memcpy(decoded_special_corrections.data(), decompressed.data(), decompressed.size());
        }

        long double codec_abs_sum = 0.0;
        for (size_t i = 0; i < special_corrections.size(); ++i) {
            double diff = std::abs(decoded_special_corrections[i] - special_corrections[i]);
            codec_abs_sum += diff;
            stats.special_correction_codec_max_abs = std::max(stats.special_correction_codec_max_abs, diff);
            if (diff == 0.0) {
                stats.special_correction_exact_match_count++;
            }
        }
        stats.special_correction_codec_mae = static_cast<double>(codec_abs_sum / special_corrections.size());
    }

    const double delta_host = host_bound / q_host;
    size_t threshold_cursor = 0;
    for (int i = 0; i < num_Elements_host; ++i) {
        double reconstructed = before[i];
        if (before[i] != after[i]) {
            if (delta_counter[i] == threshold_host) {
                reconstructed += decoded_special_corrections[threshold_cursor++];
            } else {
                for (int step = 0; step < static_cast<int>(delta_counter[i]); ++step) {
                    reconstructed -= delta_host;
                }
            }
        }
        if (reconstructed_output) {
            (*reconstructed_output)[i] = reconstructed;
        }

        double reconstruction_diff = std::abs(reconstructed - after[i]);
        stats.max_abs_reconstruction_diff = std::max(stats.max_abs_reconstruction_diff, reconstruction_diff);
        double tolerance = 1e-12 * std::max(1.0, std::abs(after[i]));
        if (reconstruction_diff > tolerance) {
            stats.reconstruction_mismatch_count++;
        }
        if (std::abs(input_data[i] - reconstructed) > host_bound + tolerance) {
            stats.bound_violation_count++;
        }
    }

    return stats;
}

void cost(int ite, std::string filename, double* decp_data, double* decp_data_copy, double* input_data, std::string compressor_id, std::vector<uint8_t> delta_counter, int q_study = 0, int experiment_mode = 0){
    std::vector<int> indexs(num_Elements_host, 0);
    std::vector<double> edits;
    std::vector<unsigned long long> exponents;
    std::vector<unsigned long long> mantissas;

    int cnt = 0;
    for (int i=0;i<num_Elements_host;i++){
        if (decp_data_copy[i]!=decp_data[i]){
            indexs[i] = delta_counter[i];
            if(indexs[i] == threshold_host){
                edits.push_back(input_data[i] - host_bound);
            }
            cnt++;
        }
    }
    
    
    std::vector<uint8_t> bitmap = {};

    
    std::string indexfilename = (exactz_scratch() + "/MSCz/data1")+filename+".bin";
    std::string editsfilename = (exactz_scratch() + "/MSCz/data_edits")+filename+".bin";
    std::string compressedindex = (exactz_scratch() + "/MSCz/data1")+filename+".bin.zst";
    std::string compressededits = (exactz_scratch() + "/MSCz/data_edits")+filename+".bin.zst";
    
    // Shockwave, 64, 64, 512
    std::vector<uint8_t> comp_deltas = compress_symbols_0_N(delta_counter, /*N=*/threshold_host, /*level=*/5);
    double ratio = double(cnt)/(num_Elements_host);
    std::cout<<cnt<<","<<ratio<<std::endl;

    std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(indexs.data()), indexs.size() * sizeof(int));
        file.close();
    } else {
        std::cerr << "cannot open file: " << filename << " ." << std::endl;
    }
    
    std::string command;
    command = "zstd -f " + indexfilename;
    std::cout << "Executing command: " << command << std::endl;
    int result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }

    std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
    if (file1.is_open()) {
        file1.write(reinterpret_cast<const char*>(edits.data()), edits.size()*sizeof(double));
        file1.close();
    } else {
        std::cerr << "cannot open file: " << filename << " ." << std::endl;
    }
    
    
    command = "zstd -f " + editsfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }

    std::uintmax_t compressed_editSize = std::filesystem::file_size(compressededits);
    std::uintmax_t compressed_indexSize = comp_deltas.size();
    std::uintmax_t original_editSize = std::filesystem::file_size(editsfilename);
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    std::uintmax_t compressed_dataSize = cmpSize;
    std::uintmax_t original_indexSize = std::filesystem::file_size(indexfilename);
    
    
    
   
    
    double overall_ratio = double(original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
   
    double bitRate = 64/overall_ratio; 

    double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);

    // std::ofstream outFile3("./stat_result/result_"+filename+"_"+compressor_id+"_detailed_additional_time.txt", std::ios::app);
    std::ofstream outFile3(get_scratch_output_dir() + "/new_single_GPU_result_" + filename + "_" + compressor_id + "_q_" + std::to_string(q_study) + ".txt", std::ios::app);
    
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return; // 返回错误码
    }

    outFile3 << std::to_string(host_bound)<<":" << std::endl;
    outFile3 << std::setprecision(17)<< "related_error: "<<er << std::endl;
    outFile3 << std::setprecision(17)<< "absolute_error: "<<host_bound << std::endl;
    outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
    outFile3 << "threshold: "<< threshold_host << std::endl;
    outFile3 <<std::setprecision(17)<< "CR: "<<double(original_dataSize)/compressed_dataSize << std::endl;
    outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
    outFile3 << std::setprecision(17)<<"BR: "<< 64/(double(original_dataSize)/compressed_dataSize) << std::endl;
    outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
    outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;
    outFile3 << std::setprecision(17)<<"ite: "<<ite << std::endl;

    BoundViolationStats after_bound_stats = compute_bound_violation_stats(input_data, decp_data, host_bound);
    outFile3 << "decp_after_bound_check:" << std::endl;
    outFile3 << "decp_after_violation_count: " << after_bound_stats.violation_count << std::endl;
    outFile3 << std::setprecision(17) << "decp_after_violation_probability: " << after_bound_stats.violation_probability << std::endl;
    outFile3 << std::setprecision(17) << "decp_after_max_abs_error: " << after_bound_stats.max_abs_error << std::endl;
    outFile3 << std::setprecision(17) << "decp_after_max_excess_over_bound: " << after_bound_stats.max_excess_over_bound << std::endl;
    std::cout << "[decp_after_bound_check] violation_count=" << after_bound_stats.violation_count
              << ", violation_probability=" << after_bound_stats.violation_probability
              << ", max_abs_error=" << after_bound_stats.max_abs_error
              << ", max_excess_over_bound=" << after_bound_stats.max_excess_over_bound
              << std::endl;


    
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
    outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
    if (experiment_mode == 1) {
        std::vector<double> reconstructed_field;
        LosslessEditExperimentStats experiment_stats =
            run_lossless_special_correction_experiment(decp_data_copy, decp_data, input_data, delta_counter, compressor_id, &reconstructed_field);
        double mismatch_probability = static_cast<double>(experiment_stats.reconstruction_mismatch_count) / experiment_stats.total_points;
        double violation_probability = static_cast<double>(experiment_stats.bound_violation_count) / experiment_stats.total_points;
        double lossy_special_ocr = 0.0;
        double lossy_special_br = 0.0;
        if (compressed_dataSize + compressed_indexSize + experiment_stats.compressed_special_correction_bytes > 0) {
            lossy_special_ocr = static_cast<double>(original_dataSize) /
                                static_cast<double>(compressed_dataSize + compressed_indexSize + experiment_stats.compressed_special_correction_bytes);
            lossy_special_br = 64.0 / lossy_special_ocr;
        }
        std::string reconstructed_bin = get_scratch_output_dir() + "/experiment_reconstructed_" + filename + "_" + compressor_id + "_" + std::to_string(host_bound) + "_q" + std::to_string(q_study) + ".bin";
        saveArrayToBin<double>(reconstructed_field.data(), reconstructed_field.size(), reconstructed_bin);
        outFile3 << "lossless_special_correction_experiment:" << std::endl;
        outFile3 << "threshold_points: " << experiment_stats.threshold_points << std::endl;
        outFile3 << std::setprecision(17) << "mean_abs_special_correction: " << experiment_stats.mean_abs_special_correction << std::endl;
        outFile3 << "compressed_special_correction_bytes: " << experiment_stats.compressed_special_correction_bytes << std::endl;
        outFile3 << std::setprecision(17) << "lossy_special_OCR: " << lossy_special_ocr << std::endl;
        outFile3 << std::setprecision(17) << "lossy_special_OBR: " << lossy_special_br << std::endl;
        outFile3 << "special_correction_exact_match_count: " << experiment_stats.special_correction_exact_match_count << std::endl;
        outFile3 << std::setprecision(17) << "special_correction_codec_mae: " << experiment_stats.special_correction_codec_mae << std::endl;
        outFile3 << std::setprecision(17) << "special_correction_codec_max_abs: " << experiment_stats.special_correction_codec_max_abs << std::endl;
        outFile3 << "reconstruction_mismatch_count: " << experiment_stats.reconstruction_mismatch_count << std::endl;
        outFile3 << std::setprecision(17) << "reconstruction_mismatch_probability: " << mismatch_probability << std::endl;
        outFile3 << "bound_violation_count: " << experiment_stats.bound_violation_count << std::endl;
        outFile3 << std::setprecision(17) << "bound_violation_probability: " << violation_probability << std::endl;
        outFile3 << std::setprecision(17) << "max_abs_reconstruction_diff: " << experiment_stats.max_abs_reconstruction_diff << std::endl;
        outFile3 << "experiment_reconstructed_bin: " << reconstructed_bin << std::endl;

        std::cout << "[lossless_special_correction_experiment] threshold_points=" << experiment_stats.threshold_points
                  << ", exact_match_count=" << experiment_stats.special_correction_exact_match_count
                  << ", codec_mae=" << experiment_stats.special_correction_codec_mae
                  << ", lossy_special_OCR=" << lossy_special_ocr
                  << ", reconstruction_mismatch=" << experiment_stats.reconstruction_mismatch_count
                  << ", mismatch_probability=" << mismatch_probability
                  << ", bound_violation_probability=" << violation_probability
                  << ", compressed_special_correction_bytes=" << experiment_stats.compressed_special_correction_bytes
                  << ", reconstructed_bin=" << reconstructed_bin
                  << std::endl;
    }
    outFile3 << "\n" << std::endl;

    
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;
    return;
}






void computeMax_saddle_table(){
    // 假设你原有的数据结构是：saddleTriplets_t[saddle * 46 + p]
    // 构建max到saddle的反向映射
    std::vector<std::vector<int>> max2saddles(nMax_host);
    
    for(int i = 0; i < nSaddle2_host; i++) {
        const int saddle = saddleTriplets_d[i * 46 + 45];
        int numMaxConnected = saddleTriplets_d[i * 46 + 44] - 1;
        for(int p = 0; p < numMaxConnected; p++) {
            const int max = max_index_d[saddleTriplets_d[i * 46 + p]];
            max2saddles[max].push_back(saddle);
        }
    }
    
    // 转换为offset形式，便于GPU快速访问
    std::vector<int> max_offsets(nMax_host + 1, 0);
    for(int i = 0; i < nMax_host; ++i) {
        max_offsets[i + 1] = max_offsets[i] + max2saddles[i].size();
    }
    std::vector<int> flattened_max2saddles(max_offsets[nMax_host]);
    for(int i = 0; i < nMax_host; ++i) {
        std::copy(max2saddles[i].begin(), max2saddles[i].end(),
                flattened_max2saddles.begin() + max_offsets[i]);
    }
    
    
    // offsets数组
    cudaMalloc(&d_max_offsets, (nMax_host + 1) * sizeof(int));
    // flattened数组
    cudaMalloc(&d_flattened_max2saddles, flattened_max2saddles.size() * sizeof(int));

    // 将max_offsets, flattened_max2saddles复制到GPU
    cudaMemcpy(d_max_offsets, max_offsets.data(),
           (nMax_host + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_flattened_max2saddles, flattened_max2saddles.data(),
    flattened_max2saddles.size() * sizeof(int), cudaMemcpyHostToDevice);

    // cudaMemcpyToSymbol(flattened_max2saddles, &d_flattened_max2saddles, sizeof(int*));
    // cudaMemcpyToSymbol(max_offsets, &d_max_offsets, sizeof(int*));

}

void computeMin_saddle_table(){

    std::vector<std::vector<int>> max2saddles(nMin_host);

    for(int i = 0; i < nSaddle1_host; i++) {
        const int saddle = saddle1Triplets_d[i * 46 + 45];
        int numMaxConnected = saddle1Triplets_d[i * 46 + 44];
        for(int p = 0; p < numMaxConnected; p++) {
            
            const int max = max_index_d[saddle1Triplets_d[i * 46 + p]];
           
            max2saddles[max].push_back(saddle);
        }
    }

    std::vector<int> max_offsets(nMin_host + 1, 0);
    for(int i = 0; i < nMin_host; ++i) {
        max_offsets[i + 1] = max_offsets[i] + max2saddles[i].size();
    }

    std::vector<int> flattened_min1saddles(max_offsets[nMin_host]);
    for(int i = 0; i < nMin_host; ++i) {
        // if(i == c){
        //     for(int i:max2saddles[i]) std::cout<<i<<std::endl;
        // }
        std::copy(max2saddles[i].begin(), max2saddles[i].end(), flattened_min1saddles.begin() + max_offsets[i]);
    }

    // offsets数组 
    // flattened数组 
    // 将max_offsets, flattened_max2saddles复制到GPU 
    // 转换为offset形式，便于GPU快速访问
    cudaMalloc(&d_min_offsets, (nMin_host + 1) * sizeof(int));
    cudaMalloc(&d_flattened_min1saddles, flattened_min1saddles.size() * sizeof(int));

    cudaMemcpy(d_min_offsets, max_offsets.data(), (nMin_host + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flattened_min1saddles, flattened_min1saddles.data(), flattened_min1saddles.size() * sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void findLargestSaddlePerMax(
    const int* __restrict__ d_max_offsets,
    const int* __restrict__ d_flattened_max2saddles
) {
    int max_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max_id >= nMax) return;

    int start = d_max_offsets[max_id];
    int end = d_max_offsets[max_id + 1];

    double max_val = -1e308;
    int best_saddle = -1;
    for (int i = start; i < end; ++i) {
        int saddle = d_flattened_max2saddles[i];
        double val = input_data[saddle];

        if (val > max_val || (val == max_val && saddle > best_saddle)){
            max_val = val;
            best_saddle = saddle;
        }
    }

    largestSaddlesForMax[max_id] = best_saddle;
}

__global__ void findSmallestSaddlePerMin(
    const int* __restrict__ d_min_offsets,
    const int* __restrict__ d_flattened_min1saddles
) {
    int min_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (min_id >= nMin) return;

    int start = d_min_offsets[min_id];
    int end = d_min_offsets[min_id + 1];

    double max_val = 1e308;
    int best_saddle = -1;
    for (int i = start; i < end; ++i) {
        int saddle = d_flattened_min1saddles[i];
        double val = input_data[saddle];

        if (val < max_val || (val == max_val && saddle < best_saddle)){
            max_val = val;
            best_saddle = saddle;
        }
    }

    smallestSaddlesForMin[min_id] = best_saddle;
}




int main(int argc, char** argv){
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] << " <file,nx,ny,nz> <relative_bound> <compressor> <sim> <q> <edit_type> <compression> <edit_mode> [zfp_cuda_cr] [q_study] [experiment_mode]\n";
        return 1;
    }
    exactz_prepare_work();
    std::cout<<LUT_SIZE<<std::endl;



    std::cout << std::fixed << std::setprecision(16);
    std::string dimension = argv[1];
    er = std::stod(argv[2]);
    compressor_id = argv[3];
    host_sim = std::stod(argv[4]);
    q_host = std::stoi(argv[5]);
    edit_type_host = std::stoi(argv[6]);
    int compression = std::stoi(argv[7]);
    edit_mode_host = std::stoi(argv[8]);
    double target_cr_zfp_cuda = 10.0;
    if (argc > 9) target_cr_zfp_cuda = std::stod(argv[9]);
    int q_study = 0;
    if (argc > 10) q_study = std::stoi(argv[10]);
    int experiment_mode = 0;
    if (argc > 11) experiment_mode = std::stoi(argv[11]);
    std::istringstream iss(dimension);
    char delimiter;
    

    // bool preserve_join_tree = true;
    
    if (std::getline(iss, file_path, ',')) {
        
        if (iss >> width_host >> delimiter && delimiter == ',' &&
            iss >> height_host >> delimiter && delimiter == ',' &&
            iss >> depth_host) {
            std::cout << "Filepath: " << file_path << std::endl;
            std::cout << "Width: " << width_host << std::endl;
            std::cout << "Height: " << height_host << std::endl;
            std::cout << "Depth: " << depth_host << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;
    }

    
    num_Elements_host = width_host*height_host*depth_host;

    
    double *input_data_host,  *decp_data_host, *decp_data_copy;
    std::cout<<file_path<<std::endl;
    decp_data_copy = new double[num_Elements_host];
    filename = extractFilename(file_path);
    cudaMalloc(&input_data_host, num_Elements_host * sizeof(double));
    cudaMalloc(&decp_data_host, num_Elements_host * sizeof(double));
    
    
    float elapsedTime = 0.0;
    cudaEvent_t startt, stop, start;
    cudaEventCreate(&startt);
    cudaEventCreate(&stop);
    cudaEventCreate(&start);
    cudaEventRecord(startt, 0);
    cudaEventRecord(start, 0);
    if(compressor_id == "sz3") getdata(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression);
    else if(compressor_id == "zfp") getdata_zfp(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression);
    else if(compressor_id == "zfp_cuda") getdata_zfp_cuda(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression, target_cr_zfp_cuda);
    else if(compressor_id == "mgard") getdata_mgard(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression);
    else if(compressor_id == "cuszp") getdata_cuszp(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression);
    else if(compressor_id == "tthresh") getdata_tthresh(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression);
    else if(compressor_id == "sperr") getdata_sperr(file_path, input_data_host, decp_data_host, decp_data_copy, er, host_bound, num_Elements_host, compression);
    double delta_host = host_bound / (q_host);
    cudaEventRecord(stop, 0);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // if (compressor_id != "mgard") compression_time = elapsedTime / 1000;
    threshold_host = q_host;
    checkCudaError(cudaMemcpyToSymbol(q, &q_host, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(delta, &delta_host, sizeof(double), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(threshold, &threshold_host, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(edit_type, &edit_type_host, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    
    
    int *vertex_type_host, *vertex_cells_host,
        *DS_M_host, *AS_M_host, *lowerStars_host,
        *upperStars_host,  *minimum_host, *maximum_host,
        *saddles1_host, *saddles2_host, *edits_host;
    uint8_t *delta_counter_host;
    int *vertex_typed = new int[3 * num_Elements_host];
    

    cudaMalloc(&vertex_type_host, 3 * num_Elements_host * sizeof(int));
    cudaMalloc(&delta_counter_host, num_Elements_host * sizeof(uint8_t));
    cudaMalloc(&edits_host, num_Elements_host * sizeof(int));
    
    cudaMalloc(&DS_M_host, num_Elements_host * sizeof(int));
    cudaMalloc(&AS_M_host, num_Elements_host * sizeof(int));
    
    
    cudaMalloc(&lowerStars_host, (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    cudaMalloc(&upperStars_host, (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    cudaMalloc(&minimum_host, num_Elements_host * sizeof(int));
    cudaMalloc(&saddles1_host, num_Elements_host * sizeof(int));
    cudaMalloc(&saddles2_host, num_Elements_host * sizeof(int));
    cudaMalloc(&maximum_host, num_Elements_host * sizeof(int));


    
    cudaMemset(vertex_type_host, -1,  3 * num_Elements_host * sizeof(int));
    cudaMemset(delta_counter_host, 0,  num_Elements_host * sizeof(uint8_t));
    cudaMemset(edits_host, 0,  num_Elements_host * sizeof(int));
    
    cudaMemset(DS_M_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(AS_M_host, -1,  num_Elements_host * sizeof(int));
    

   
    cudaMemset(lowerStars_host, -1,  (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    cudaMemset(upperStars_host, -1,  (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    
    cudaMemset(minimum_host, -1, num_Elements_host * sizeof(int));
    cudaMemset(saddles1_host, -1, num_Elements_host * sizeof(int));
    cudaMemset(saddles2_host, -1, num_Elements_host * sizeof(int));
    cudaMemset(maximum_host, -1, num_Elements_host * sizeof(int));

    cudaMemcpyToSymbol(vertex_type, &vertex_type_host, sizeof(int*));
    cudaMemcpyToSymbol(vertex_cells, &vertex_cells_host, sizeof(int*));
    
    cudaMemcpyToSymbol(delta_counter, &delta_counter_host, sizeof(uint8_t*));
    cudaMemcpyToSymbol(edits, &edits_host, sizeof(int*));
    cudaMemcpyToSymbol(DS_M, &DS_M_host, sizeof(int*));
    cudaMemcpyToSymbol(AS_M, &AS_M_host, sizeof(int*));

    cudaMemcpyToSymbol(lowerStars, &lowerStars_host, sizeof(int*));
    cudaMemcpyToSymbol(upperStars, &upperStars_host, sizeof(int*));

    
    cudaMemcpyToSymbol(minimum, &minimum_host, sizeof(int*));
    cudaMemcpyToSymbol(saddles1, &saddles1_host, sizeof(int*));
    cudaMemcpyToSymbol(saddles2, &saddles2_host, sizeof(int*));
    cudaMemcpyToSymbol(maximum, &maximum_host, sizeof(int*));
   
    


    cudaMemcpyToSymbol(width, &width_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(height, &height_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &depth_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maxNeighbors, &maxNeighbors_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_Elements, &num_Elements_host, sizeof(int), 0, cudaMemcpyHostToDevice);

    
    
    cudaMemcpyToSymbol(bound, &host_bound, sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(thr, &host_thre, sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(edit_mode, &edit_mode_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    
    std::cout<<host_thre<<std::endl;
    dim3 blockSize(256);
    dim3 gridSize((num_Elements_host + blockSize.x - 1) / blockSize.x);


    dim3 blockDim1(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridDim1((width_host + TILE_SIZE - 1) / TILE_SIZE,
                 (height_host + TILE_SIZE - 1) / TILE_SIZE,
                 (depth_host + TILE_SIZE - 1) / TILE_SIZE);
    generateLookupTable<<<1,1>>>();
    cudaDeviceSynchronize();
    
    loadLUTToGPU();
    cudaDeviceSynchronize();
    cudaEventRecord(startt, 0);
    
    cudaEventRecord(start, 0);
    classifyVertex_CUDA<<<gridSize, blockSize>>>(nullptr, nullptr, 0, 0, 0);
    cudaDeviceSynchronize(); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // cudaMemcpy(vertex_typed, vertex_type_host, 3 * num_Elements_host * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize(); 
    // std::cout<<"vertex classification: "<<elapsedTime/1000<<std::endl;


    // return 0;

    init_Manifold<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();


    ComputeDirection<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    ComputeDescendingManifold<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    ComputeAscendingManifold<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    
    ExtractCP<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    
    

    cudaMemcpyFromSymbol(&nSaddle2_host, nSaddle2, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSaddle1_host, nSaddle1, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nMax_host, nMax, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nMin_host, nMin, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    std::cout<<nSaddle2_host<< ", "<<nSaddle1_host<<", "<<nMax_host<<", "<<nMin_host<<std::endl;

    
    
    sort_indexs_by_data(saddles2_host, input_data_host, nSaddle2_host);
    cudaDeviceSynchronize();
    sort_indexs_by_data(saddles1_host, input_data_host, nSaddle1_host);
    cudaDeviceSynchronize();
    
    
    dim3 gridSize_max((nMax_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_min((nMin_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_1saddle((nSaddle1_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_2saddle((nSaddle2_host + blockSize.x - 1) / blockSize.x);
    
    
    // initialize saddletriples;
    int *saddleTriplets_host, *saddle1Triplets_host;
    int *saddlerank_host, *saddle1rank_host;
    cudaMalloc(&saddleTriplets_host, nSaddle2_host * 46 * sizeof(int));
    cudaMalloc(&saddle1Triplets_host, nSaddle1_host * 46 * sizeof(int));

    cudaMalloc(&saddlerank_host, nSaddle2_host * sizeof(int));
    cudaMalloc(&saddle1rank_host, nSaddle1_host * sizeof(int));

    cudaMemcpyToSymbol(saddleTriplets, &saddleTriplets_host, sizeof(int*));
    cudaMemcpyToSymbol(saddle1Triplets, &saddle1Triplets_host, sizeof(int*));
    
    findAscPaths<<<gridSize_2saddle, blockSize>>>(1);
    cudaDeviceSynchronize();
    findDescPaths<<<gridSize_1saddle, blockSize>>>(1);
    cudaDeviceSynchronize();
    
    
    saddleTriplets_d.resize(nSaddle2_host * 46);
    size_t bytes1 = (size_t)nSaddle2_host * 46 * sizeof(int);
    saddle1Triplets_d.resize(nSaddle2_host * 46);
    
    int* d_saddleTriplets_ptr = nullptr;
    cudaMemcpyFromSymbol(&d_saddleTriplets_ptr, saddleTriplets, sizeof(int*));
    cudaMemcpy(saddleTriplets_d.data(),
            d_saddleTriplets_ptr,
            bytes1,
            cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    size_t bytes = (size_t)nSaddle1_host * 46 * sizeof(int);
    saddle1Triplets_d.resize(nSaddle1_host * 46);
    int* d_saddle1Triplets_ptr = nullptr;
    cudaMemcpyFromSymbol(&d_saddle1Triplets_ptr, saddle1Triplets, sizeof(int*));
    cudaMemcpy(saddle1Triplets_d.data(),
            d_saddle1Triplets_ptr,
            bytes,
            cudaMemcpyDeviceToHost);
    checkCudaError(cudaDeviceSynchronize(), "before cloops");

    int host_count_f_max = 0;
    int host_count_f_min = 0;
    int host_count_f_saddle = 0;
    int host_count_f_dir = 0;
    int host_count_f_order_saddle = 0;
    int host_count_f_cases = 0;
    int ite = 0;
    std::cout<<"cloops started!"<<std::endl;
    c_loops(ite, gridSize, blockSize, host_count_f_max, host_count_f_min, host_count_f_saddle, host_count_f_dir, host_count_f_order_saddle, host_count_f_cases, delta_host);
    cudaDeviceSynchronize();
    std::cout<<"cloops ended!"<<std::endl;
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    std::cout<<"finally:"<<std::endl;

    cudaEventElapsedTime(&elapsedTime, startt, stop);
    std::cout<<"whole time:" <<elapsedTime/1000<<std::endl;
    additional_time = elapsedTime/1000;

    std::vector<double> decp_d;
    decp_d.resize(num_Elements_host);
    cudaMemcpy(decp_d.data(), decp_data_host, num_Elements_host * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<uint8_t> delta_counter_d;
    delta_counter_d.resize(num_Elements_host);
    cudaMemcpy(delta_counter_d.data(), delta_counter_host, num_Elements_host * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<double> input_data_d;
    input_data_d.resize(num_Elements_host);
    cudaMemcpy(input_data_d.data(), input_data_host, num_Elements_host * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    
    std::string errormap_bin = get_scratch_output_dir() + "/errormap_cuda_" + filename + "_" + compressor_id + "_" + std::to_string(host_bound) + "_q" + std::to_string(q_study) + ".bin";
    std::vector<uint8_t> errormap = build_edit_class_map(decp_data_copy, decp_d.data(), delta_counter_d);
    save_edit_class_map(errormap_bin, errormap, "ExtreM_cuda");

    cost(ite, filename, decp_d.data(), decp_data_copy, input_data_d.data(),compressor_id, delta_counter_d, q_study, experiment_mode);

    // saveArrayToBin(decp_d.data(), num_Elements_host, "fixed.bin");
    return 0;


}
