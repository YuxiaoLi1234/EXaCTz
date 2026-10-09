#include "release_paths.h"

// nvcc -DDETAILED_TIMING -ccbin CC -std=c++17 pExtreMz_all_order.cu -o pExtreMz_all_order -Xcompiler -fopenmp -lzstd -lzfp --extended-lambda
// salloc --nodes 1 --qos interactive --time 00:10:00 --constraint "gpu" --ntasks-per-node=1 --gpus-per-task=1 --account m4259
#include <iostream>
#include <fstream>
#include <vector>
#include "SZ3/api/sz.hpp"
#include <set>
#include <cfloat>
#include <bitset>
#include <filesystem>
#include <map>
#include <mpi.h>
#include <inttypes.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "zstd.h"
#include "./UnionFind.h"
#include <zfp.h>
#include <cub/cub.cuh>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <math_constants.h>
#include <ctime>
#include <cstring>
#define HALO 1

#define CUDA_CHECK(x) do {                                 \
  cudaError_t err = (x);                                   \
  if (err != cudaSuccess)                                  \
    throw std::runtime_error(cudaGetErrorString(err));     \
} while(0)


double print_gpu_mem(const char* tag){
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_GB = (double)free_byte / 1024.0 / 1024.0 / 1024.0;
    double total_GB = (double)total_byte / 1024.0 / 1024.0 / 1024.0;
    double used_GB = total_GB - free_GB;

    printf("[GPU MEM] %s: used = %.2f GB, free = %.2f GB, total = %.2f GB\n",
           tag, used_GB, free_GB, total_GB);
    fflush(stdout);
    return used_GB;
}



#define MAX_NEIGHBORS 14
#define MAX_V3 6 
#define NUM_LOOKUP 14 // 14 * 14 种 (v1, v2) 组合
#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE + 2)  // 这里为 10
#define SIZE (NUM_LOOKUP * MAX_V3 * 3 * sizeof(int))  // 计算数组大小

// P2P 通信配置参数（可根据实际情况调整）
// 对于 Cray Slingshot/OFI，建议 MAX_P2P_CONCURRENT = 2~4
// 对于 InfiniBand，可以尝试 8~16
#define MAX_P2P_CONCURRENT 4

std::string result_prefix;
std::vector<std::vector<float>> time_counter;

double start_time, end_time;
std::vector<std::map<std::string, double>> times;
std::vector<double> comm_time, comp_time;

std::string filename;

static long long* d_dummy_ll = nullptr;
static double*    d_dummy_d  = nullptr;
static int*       d_dummy_i  = nullptr;
static uint64_t*  d_dummy_u64 = nullptr;

static void ensure_dummy() {
    if (!d_dummy_ll)  cudaMalloc(&d_dummy_ll,  sizeof(long long));
    if (!d_dummy_d)   cudaMalloc(&d_dummy_d,   sizeof(double));
    if (!d_dummy_i)   cudaMalloc(&d_dummy_i,   sizeof(int));
    if (!d_dummy_u64) cudaMalloc(&d_dummy_u64, sizeof(uint64_t));
}

// ==================== 优化的 P2P 通信（分阶段，避免 OFI 队列溢出）====================

// 分阶段 P2P 版本的 Alltoall（严格控制并发数）
// Debug 标志：设置为 true 输出 MPI 调试信息
static bool g_mpi_debug = false;
static int g_mpi_debug_rank = 0;  // 只在 rank 0 输出

static void p2p_alltoall_int(const int* sendbuf, int* recvbuf, int count, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // 自己发给自己直接拷贝
    memcpy(&recvbuf[rank * count], &sendbuf[rank * count], count * sizeof(int));
    
    if (size == 1) return;
    
    // if (g_mpi_debug && rank == g_mpi_debug_rank) {
    //     printf("[DEBUG MPI] p2p_alltoall_int: rank=%d, size=%d, count=%d, total_sends=%d\n",
    //            rank, size, count, size - 1);
    //     fflush(stdout);
    // }
    
    // 分批发送，每批最多 MAX_CONCURRENT 个请求
    const int MAX_CONCURRENT = 16;  // 限制并发请求数，避免OFI队列溢出
    std::vector<MPI_Request> reqs;
    reqs.reserve(MAX_CONCURRENT * 2);
    
    int d = 1;
    while (d < size) {
        reqs.clear();
        int batch_start = d;
        int batch_count = 0;
        
        // 发起一批接收和发送
        while (d < size && batch_count < MAX_CONCURRENT) {
            int send_to = (rank + d) % size;
            int recv_from = (rank - d + size) % size;
            
            MPI_Request req_recv, req_send;
            MPI_Irecv(&recvbuf[recv_from * count], count, MPI_INT, recv_from, d, comm, &req_recv);
            MPI_Isend(&sendbuf[send_to * count], count, MPI_INT, send_to, d, comm, &req_send);
            reqs.push_back(req_recv);
            reqs.push_back(req_send);
            
            ++d;
            ++batch_count;
        }
        
        // if (g_mpi_debug && rank == g_mpi_debug_rank) {
        //     printf("[DEBUG MPI] p2p_alltoall_int: batch d=%d-%d, reqs=%zu, waiting...\n",
        //            batch_start, d - 1, reqs.size());
        //     fflush(stdout);
        // }
        
        // 等待这一批完成
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        
        // if (g_mpi_debug && rank == g_mpi_debug_rank) {
        //     printf("[DEBUG MPI] p2p_alltoall_int: batch d=%d-%d completed\n",
        //            batch_start, d - 1);
        //     fflush(stdout);
        // }
    }
}

// 真正的稀疏 P2P Alltoallv：只和有数据的邻居通信（不是 all-to-all）
template <typename T>
static void p2p_alltoallv_cuda_staged(
    const T* sendbuf, const int* sendcounts, const int* sdispls,
    T* recvbuf, const int* recvcounts, const int* rdispls,
    MPI_Datatype dtype, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    cudaDeviceSynchronize();
    
    // 自己发给自己
    if (sendbuf && recvbuf && sendcounts[rank] > 0) {
        cudaMemcpy((void*)&recvbuf[rdispls[rank]], (const void*)&sendbuf[sdispls[rank]], 
                   sendcounts[rank] * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    
    if (size == 1) return;
    
    // 收集需要通信的邻居（只有 sendcount > 0 或 recvcount > 0 的）
    std::vector<int> send_neighbors, recv_neighbors;
    for (int r = 0; r < size; ++r) {
        if (r != rank) {
            if (sendcounts[r] > 0) send_neighbors.push_back(r);
            if (recvcounts[r] > 0) recv_neighbors.push_back(r);
        }
    }
    
    if (g_mpi_debug && rank == g_mpi_debug_rank) {
        printf("[DEBUG MPI] p2p_alltoallv_cuda_staged: rank=%d, total_ranks=%d, "
               "send_neighbors=%zu, recv_neighbors=%zu\n", 
               rank, size, send_neighbors.size(), recv_neighbors.size());
        fflush(stdout);
    }
    
    std::vector<MPI_Request> reqs;
    reqs.reserve(send_neighbors.size() + recv_neighbors.size());
    
    // 先发起所有接收
    for (int peer : recv_neighbors) {
        int tag = rank < peer ? rank * 100000 + peer : peer * 100000 + rank;
        MPI_Request req;
        MPI_Irecv((void*)&recvbuf[rdispls[peer]], recvcounts[peer],
                  dtype, peer, tag, comm, &req);
        reqs.push_back(req);
    }
    
    // 再发起所有发送
    for (int peer : send_neighbors) {
        int tag = rank < peer ? rank * 100000 + peer : peer * 100000 + rank;
        MPI_Request req;
        MPI_Isend((const void*)&sendbuf[sdispls[peer]], sendcounts[peer],
                  dtype, peer, tag, comm, &req);
        reqs.push_back(req);
    }
    
    if (g_mpi_debug && rank == g_mpi_debug_rank) {
        printf("[DEBUG MPI] p2p_alltoallv_cuda_staged: total_requests=%zu, waiting...\n",
               reqs.size());
        cudaPointerAttributes attr;
        cudaError_t e = cudaPointerGetAttributes(&attr, sendbuf);
        printf("rank %d cudaPointerGetAttributes=%s, type=%d\n",
            rank, cudaGetErrorString(e), (int)attr.type);
        fflush(stdout);

    }
    
    // 等待所有请求完成
    if (!reqs.empty()) {
        

        int ret = MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        if (ret != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(ret, error_string, &length);
            printf("[ERROR MPI] p2p_alltoallv_cuda_staged: rank=%d, "
                   "MPI_Waitall failed: %s\n", rank, error_string);
            fflush(stdout);
        }
    }
    
    if (g_mpi_debug && rank == g_mpi_debug_rank) {
        printf("[DEBUG MPI] p2p_alltoallv_cuda_staged: completed\n");
        fflush(stdout);
    }
    
    cudaDeviceSynchronize();
}

template <typename T>
static void p2p_alltoallv_cuda_staged_throttled(
    const T* sendbuf, const int* sendcounts, const int* sdispls,
    T* recvbuf, const int* recvcounts, const int* rdispls,
    MPI_Datatype dtype, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    CUDA_CHECK(cudaDeviceSynchronize());
    ensure_dummy();

    // self copy
    if (sendcounts[rank] > 0) {
        CUDA_CHECK(cudaMemcpy((void*)&recvbuf[rdispls[rank]],
                              (const void*)&sendbuf[sdispls[rank]],
                              (size_t)sendcounts[rank] * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    }
    if (size == 1) return;

    constexpr int TAG = 0;
    for (int distance = 1; distance < size; ++distance) {
        const int send_to = (rank + distance) % size;
        const int recv_from = (rank - distance + size) % size;
        const T* send_ptr = sendcounts[send_to] > 0
            ? sendbuf + sdispls[send_to]
            : reinterpret_cast<const T*>(d_dummy_u64);
        T* recv_ptr = recvcounts[recv_from] > 0
            ? recvbuf + rdispls[recv_from]
            : reinterpret_cast<T*>(d_dummy_u64);

        int mpi_error = MPI_Sendrecv(
            const_cast<T*>(send_ptr), sendcounts[send_to], dtype, send_to, TAG,
            recv_ptr, recvcounts[recv_from], dtype, recv_from, TAG,
            comm, MPI_STATUS_IGNORE);
        if (mpi_error != MPI_SUCCESS) {
            fprintf(stderr,
                    "Rank %d: MPI_Sendrecv failed in throttled alltoallv at distance=%d\n",
                    rank, distance);
            MPI_Abort(comm, mpi_error);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}


const double INVALID_VAL = 1e30;
template <typename T> MPI_Datatype get_mpi_datatype();
template <> inline MPI_Datatype get_mpi_datatype<uint8_t>()  { return MPI_UINT8_T; }
template <> inline MPI_Datatype get_mpi_datatype<uint16_t>() { return MPI_UINT16_T; }
template <> inline MPI_Datatype get_mpi_datatype<uint32_t>() { return MPI_UINT32_T; }
template <> inline MPI_Datatype get_mpi_datatype<uint64_t>() { return MPI_UINT64_T; }
template <> inline MPI_Datatype get_mpi_datatype<int8_t>()   { return MPI_INT8_T; }
template <> inline MPI_Datatype get_mpi_datatype<int16_t>()  { return MPI_INT16_T; }
template <> inline MPI_Datatype get_mpi_datatype<int32_t>()  { return MPI_INT32_T; }
template <> inline MPI_Datatype get_mpi_datatype<int64_t>()  { return MPI_INT64_T; }
template <> inline MPI_Datatype get_mpi_datatype<float>()    { return MPI_FLOAT; }
template <> inline MPI_Datatype get_mpi_datatype<double>()   { return MPI_DOUBLE; }



// 
struct RecvBuffer {
    double* buffer;
    int dx, dy, dz;
    int dir_tag;
};

// ==================== 优化的通信计划结构 ====================
// 保存固定的通信模式，避免每次迭代重新计算
struct NeighborCommPlan {
    int total_send = 0;        // 我要向别人请求多少个值
    int total_recv = 0;        // 别人向我请求多少个值
    
    std::vector<int> sendcounts, senddispls;  // 我发给各 rank 的请求数
    std::vector<int> recvcounts, recvdispls;  // 各 rank 发给我的请求数
    
    // Device 端固定缓冲区（gid + slot，初始化后不变）
    uint64_t* d_my_gids = nullptr;      // [total_send] 我要请求的 gid
    int* d_my_slots = nullptr;          // [total_send] 我要请求的 slot
    uint64_t* d_their_gids = nullptr;   // [total_recv] 别人请求我的 gid
    int* d_their_slots = nullptr;       // [total_recv] 别人请求我的 slot
    
    // 每次迭代更新的缓冲区（只存 value）
    double* d_reply_vals = nullptr;     // [total_recv] 我回复给别人的 value
    double* d_received_vals = nullptr;  // [total_send] 别人回复给我的 value
    
    bool initialized = false;
};

inline void free_neighbor_comm_plan(NeighborCommPlan& plan) {
    if (plan.d_my_gids) cudaFree(plan.d_my_gids);
    if (plan.d_my_slots) cudaFree(plan.d_my_slots);
    if (plan.d_their_gids) cudaFree(plan.d_their_gids);
    if (plan.d_their_slots) cudaFree(plan.d_their_slots);
    if (plan.d_reply_vals) cudaFree(plan.d_reply_vals);
    if (plan.d_received_vals) cudaFree(plan.d_received_vals);
    plan = NeighborCommPlan{};
}

struct PeakCmp {
  const double*   val;
  const uint64_t* gid;
  const int*      owner;

  __host__ __device__
  bool operator()(int a, int b) const {
    double va = val[a], vb = val[b];
    if (va < vb) return true;
    if (va > vb) return false;

    uint64_t ga = gid[a], gb = gid[b];
    if (ga < gb) return true;
    if (ga > gb) return false;

    

    return false;
  }
};


static inline void CUDA_OK(cudaError_t e){ if(e!=cudaSuccess){ 
    printf("CUDA %s\n", cudaGetErrorString(e)); 
    fflush(stdout);
    abort(); } 
}


__device__ double *input_data, *decp_data, bound, *decp_data_copy_d, *decp_datac;
__device__ int width, height, depth, maxNeighbors, num_Elements, nSaddle2 = 0, nSaddle1 = 0, nMin = 0, nMax = 0, changed_flag = 0;
__device__ int *vertex_type,  *edits, *lowerStars, *upperStars,  *adjacency,
                *minimum, *saddles1, *saddles2, *maximum;    
__device__ uint8_t *delta_counter, *delta_counter_c;
int initialValue = 0, PX, PY, PZ, nSaddle2_host = 0, nMin_host = 0, nMax_host = 0, nSaddle1_host = 0;;
size_t num_Elements_host;
size_t width_host, height_host, depth_host;
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
__device__ int q, threshold, edit_type, global_width, global_height, global_depth, count_f_cases, progress_f_cases;
__device__ double delta;
int LUT[6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 6 * (1<<10) + (1 << 14)];

struct OffsetComparator {
    double* d_offset;
    __host__ __device__ OffsetComparator(double* offset) : d_offset(offset) {}

    __host__ __device__ bool operator()(int v, int u) const {
        if (v < 0 || u < 0) return false;
        return d_offset[v] > d_offset[u] || 
               ((fabs(d_offset[v] - d_offset[u]) == 0) && (v > u));
    }
};



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
std::vector<long long> saddleTriplets_d, saddle1Triplets_d, max_index_d;

template <typename T>
void getdata(const std::string &filename, T *&input_data_host, size_t data_size) {

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>((size_t)data_size * sizeof(T))) {
        std::cout<< "datasize is: "<< data_size<< ", datapoint size: "<< sizeof(T)<<std::endl;
        std::cout<<"file size: "<<size<<" calculated size: "<<static_cast<std::streamsize>((size_t)data_size * sizeof(T))<<std::endl;
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }


    input_data_host = new T[data_size];
    file.read(reinterpret_cast<char *>(input_data_host), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    minValue = *std::min_element(input_data_host, input_data_host + data_size);
    maxValue = *std::max_element(input_data_host, input_data_host + data_size);

    // cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(uint16_t), cudaMemcpyHostToDevice);    
    // std::cout << "Data read, compressed, and decompressed successfully." << std::endl;
}

__host__ __device__ size_t get_send_count(int dx, int dy, int dz, 
    size_t local_x, size_t local_y, size_t local_z) {
    int dsum = abs(dx) + abs(dy) + abs(dz);
    if (dsum == 1) {
        if (dx != 0) return 2 * local_y * local_z;
        if (dy != 0) return 2 * local_x * local_z;
        if (dz != 0) return 2 * local_x * local_y;
    } else if (dsum == 2) {
        if (dx == 0) return 2 * local_x;
        if (dy == 0) return 2 * local_y;
        if (dz == 0) return 2 * local_z;
    } else if (dsum == 3) {
        return 0;
    }
    return 0;
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




template <typename T>
__device__ void applyDeltaBuffer_local(uint8_t *delta_counter,
                                T *decp_data, T delta, T *input_data, 
                                T bound, int q, size_t tid, int *send_flag, uint8_t *changed_vertex_h) {
    const T old_value = decp_data[tid];
    if(fabs(input_data[tid]-(decp_data[tid] - delta)) < bound && delta_counter[tid] < q - 1){
        
        decp_data[tid] -= delta; 
        delta_counter[tid] += 1;
    }
    else{
        delta_counter[tid] = q;
        decp_data[tid] = input_data[tid] - bound;
    }
    if (!(decp_data[tid] < old_value)) return;

    atomicAdd(&progress_f_cases, 1);
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
    if(x <= 1 || x >= width -2 || y <= 1 || y >= height -2 || z <= 1 || z >= depth -2) *send_flag = 1;
}

__device__ bool islarger(const int v, const int u, const double *offset){
    return offset[v] > offset[u] || (offset[v] == offset[u] == 0 && v > u);
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

__device__ __forceinline__
bool local_id_from_gid(
        uint64_t gid,
        int NX, int NY,
        int offset_x,int offset_y,int offset_z,
        int local_x,int local_y,int local_z,
        int padded_x,int padded_y,
        int* out_lid)
{
    uint64_t gx_u = gid % (uint64_t)NX;
    uint64_t gy_u = (gid / (uint64_t)NX) % (uint64_t)NY;
    uint64_t gz_u = gid / ((uint64_t)NX * (uint64_t)NY);

    int gx = (int)gx_u, gy = (int)gy_u, gz = (int)gz_u;

    int lx = gx - offset_x;
    int ly = gy - offset_y;
    int lz = gz - offset_z;

    // owned 区域检查（强烈建议）
    if (lx < 0 || lx >= local_x ||
        ly < 0 || ly >= local_y ||
        lz < 0 || lz >= local_z) {
        return false;
    }

    int lid =
        (lz + HALO) * padded_y * padded_x +
        (ly + HALO) * padded_x +
        (lx + HALO);

    *out_lid = lid;
    return true;
}



__device__ uint64_t lid_to_gid(
    const int local_idx,
    int offset_x, int offset_y, int offset_z)
{
    if (local_idx >= num_Elements) return 0;

    int x = local_idx % width;
    int y = (local_idx / width) % height;
    int z = local_idx / ((size_t)width * height);

    int gx = x - HALO + offset_x;
    int gy = y - HALO + offset_y;
    int gz = z - HALO + offset_z;

    uint64_t gIdx =
        (uint64_t)gx +
        (uint64_t)gy * global_width +
        (uint64_t)gz * global_width * global_height;

    return gIdx;
}

__global__ void extractGlobalWrongOrder(
    // set0: saddle1
    int n1,
    const int*    __restrict__ ids1,        // saddles1 (device int*)
    const double* __restrict__ prev_val1,
    const double* __restrict__ next_val1,
    const int*    __restrict__ prev_rank1,
    const int*    __restrict__ next_rank1,
    const uint64_t* __restrict__ prev_idx1,
    const uint64_t* __restrict__ next_idx1,
    const uint8_t* __restrict__ need_prev1,   // NEW: mask for prev
    const uint8_t* __restrict__ need_next1,   // NEW: mask for next

    // set1: saddle2
    int n2,
    const int*    __restrict__ ids2,        // saddles2
    const double* __restrict__ prev_val2,
    const double* __restrict__ next_val2,
    const int*    __restrict__ prev_rank2,
    const int*    __restrict__ next_rank2,
    const uint64_t* __restrict__ prev_idx2,
    const uint64_t* __restrict__ next_idx2,
    const uint8_t* __restrict__ need_prev2,   // NEW
    const uint8_t* __restrict__ need_next2,   // NEW

    // set2: minimum
    int nMin,
    const int*    __restrict__ idsMin,      // minimum (device int*)
    const double* __restrict__ prev_valMin,
    const double* __restrict__ next_valMin,
    const int*    __restrict__ prev_rankMin,
    const int*    __restrict__ next_rankMin,
    const uint64_t* __restrict__ prev_idxMin,
    const uint64_t* __restrict__ next_idxMin,
    const uint8_t* __restrict__ need_prevMin, // NEW
    const uint8_t* __restrict__ need_nextMin, // NEW

    // set3: maximum
    int nMax,
    const int*    __restrict__ idsMax,      // maximum
    const double* __restrict__ prev_valMax,
    const double* __restrict__ next_valMax,
    const int*    __restrict__ prev_rankMax,
    const int*    __restrict__ next_rankMax,
    const uint64_t* __restrict__ prev_idxMax,
    const uint64_t* __restrict__ next_idxMax,
    const uint8_t* __restrict__ need_prevMax, // NEW
    const uint8_t* __restrict__ need_nextMax, // NEW

    int myrank,
    int* __restrict__ out_dest,     // capacity >= (n1+n2+nMin+nMax)  (按你当前：每个点最多发1条)
    uint64_t* __restrict__ out_idx,
    int* __restrict__ out_nreq,

    int offset_x,int offset_y,int offset_z,
    int local_x,int local_y,int local_z, int ite,
    int *send_flag, uint8_t *changed_vertex_h, 
    int checkFinal = 0
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n1 + n2 + nMin + nMax;
    if (tid >= N) return;

    // ---- choose set (0..3) and local index i ----
    int set_id, i;
    if (tid < n1) {
        set_id = 0; i = tid;
    } else if (tid < n1 + n2) {
        set_id = 1; i = tid - n1;
    } else if (tid < n1 + n2 + nMin) {
        set_id = 2; i = tid - (n1 + n2);
    } else {
        set_id = 3; i = tid - (n1 + n2 + nMin);
    }

    // ---- select pointers for this set ----
    const int* ids = nullptr;
    const double* prev_val = nullptr;
    const double* next_val = nullptr;
    const int* prev_rank = nullptr;
    const int* next_rank = nullptr;
    const uint64_t* prev_idx = nullptr;
    const uint64_t* next_idx = nullptr;
    const uint8_t* need_prev = nullptr;
    const uint8_t* need_next = nullptr;

    if (set_id == 0) {
        ids = ids1; prev_val = prev_val1; next_val = next_val1;
        prev_rank = prev_rank1; next_rank = next_rank1;
        prev_idx  = prev_idx1;  next_idx  = next_idx1;
        need_prev = need_prev1; need_next = need_next1;
    } else if (set_id == 1) {
        ids = ids2; prev_val = prev_val2; next_val = next_val2;
        prev_rank = prev_rank2; next_rank = next_rank2;
        prev_idx  = prev_idx2;  next_idx  = next_idx2;
        need_prev = need_prev2; need_next = need_next2;
    } else if (set_id == 2) {
        ids = idsMin; prev_val = prev_valMin; next_val = next_valMin;
        prev_rank = prev_rankMin; next_rank = next_rankMin;
        prev_idx  = prev_idxMin;  next_idx  = next_idxMin;
        need_prev = need_prevMin; need_next = need_nextMin;
    } else { // set_id == 3
        ids = idsMax; prev_val = prev_valMax; next_val = next_valMax;
        prev_rank = prev_rankMax; next_rank = next_rankMax;
        prev_idx  = prev_idxMax;  next_idx  = next_idxMax;
        need_prev = need_prevMax; need_next = need_nextMax;
    }

    int id = ids[i];                 // 原始 local id
    double value = decp_data[id];    // 你的全局 device 数据源
    uint64_t sgid = lid_to_gid(id, offset_x, offset_y, offset_z);

    int pr = prev_rank[i];
    int nr = next_rank[i];

    double pv = prev_val[i];
    double nv = next_val[i];
    uint64_t pi = prev_idx[i];
    uint64_t ni = next_idx[i];

    // --- prev-side violation: send to prev owner (remote only) ---
    // Only check if need_prev[i] == 1 (remote neighbor with close values)
    // ---- DEBUG need_prev pointer / bounds / values ----
    

    if (need_prev[i] == 1 && (value < pv || (value == pv && sgid < pi)) && pr != -1 && !isnan(pv)) {
        count_f_cases += 1;

        int lid;
        if (local_id_from_gid(pi, global_width, global_height,
                             offset_x, offset_y, offset_z,
                             local_x, local_y, local_z,
                             width, height, &lid)) {

            if (checkFinal == 0 && atomicCAS(&edits[lid], 0, 1) == 0 ) {
                // if(myrank == 0 && ite > 240 && lid == 93499405) printf("updateing local prev %d %.17f %.17f %llu %d \n", lid, decp_data[lid], pv, sgid, pr);
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta,
                                               input_data, bound, q, lid, send_flag, changed_vertex_h);
            }
        } else if(!isnan(pv)){
            // 远程：发给 pr，idx = pi
            int pos = atomicAdd(out_nreq, 1);
            // if(myrank == 0 && ite > 240) printf("updateing global prev %.17f %.17f %llu %llu %d %d\n", value, pv, sgid, pi, pr, i);
            // 9.02122945290378375 9.02122945339567295 284636478 363774829 3 
            // 9.02122801972739197 9.02122945339567295
            if (pos >= 2 * N) {
                printf("OOB out_nreq pos=%d max_req=%d i=%d pr=%d pi=%llu\n",
                    pos, 2 * N, i, pr, (unsigned long long)pi);
                return;
            }
            out_dest[pos] = pr;
            out_idx[pos]  = pi;
        }
    }
    else if (need_next[i] == 1 && (value > nv || (value == nv && sgid > ni)) && nr != -1 && !isnan(nv)) {
        count_f_cases += 1;
        if (checkFinal == 0 && atomicCAS(&edits[id], 0, 1) == 0) {
            // if(myrank == 0 && ite > 250)printf("updateing local next %d\n", id);
            applyDeltaBuffer_local<double>(delta_counter, decp_data, delta,
                                           input_data, bound, q, id, send_flag, changed_vertex_h);
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

__device__ bool areVectorsDifferent(int *dec_stars, int *o_stars, int lowerCount, int olowerCount){
    if(olowerCount != lowerCount){
        return true;
    }
    for (size_t j = 0; j < lowerCount; j++) {
        // if(i == 782) printf("%d %d\n", stars[(maxNeighbors+1) * i +j], dec_stars[(maxNeighbors+1) * i +j]);
        if (o_stars[j] != dec_stars[j]) {
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


__global__ void ExtractCP(int check = 0){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=num_Elements) return;


    int x = idx % width;
    int y = (idx / width) % height;
    int z = (idx / (width * height)) % depth;
    if (((x == 0 || x >= width - 1 || y == 0 || y >= height - 1 || z == 0 || z >= depth - 1) || input_data[idx] == INVALID_VAL) && check == 0){
        return;
    }
    int *minimum_t = minimum;
    int *saddles1_t = saddles1;
    int *saddles2_t = saddles2;
    int *maximum_t = maximum;
    int *nSaddle2_t = &nSaddle2;
    int *nSaddle1_t = &nSaddle1;
    int *nMax_t = &nMax;
    int *nMin_t = &nMin;
    size_t base = (size_t)idx * 3;
    int type = vertex_type[base];

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


__global__ void init_edits(uint8_t* changed_vertex_h, int type = 1){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    edits[i] = 0;
    changed_vertex_h[i] = 0;
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


__device__ void fix_saddle_local(int i, int *o_lowerStar_start, int o_size1, int *lowerStar, int lowerCount, int *send_flag, uint8_t *changed_vertex_h){
 
   
    int diff[MAX_NEIGHBORS];
    
    
    int count = findDifferences_local(lowerStar, lowerCount, o_lowerStar_start, o_size1, diff);

    for(int index = 0; index<count; index++){
        const int id = diff[index];
        // if(i == 220359) printf("%d %.17f %.17f %d %.17f %.17f\n", )
        if (atomicCAS(&edits[id], 0, 1) == 0){         
            applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, id, send_flag, changed_vertex_h);
        }
    }
    
    int diff1[MAX_NEIGHBORS];
    
    count = findDifferences_local(o_lowerStar_start, o_size1, lowerStar, lowerCount, diff1);
    if(count > 0){
        if (atomicCAS(&edits[i], 0, 1) == 0){
            applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, i, send_flag, changed_vertex_h);
        }
    }
    
    // }
    return;
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

void saveArrayToBin(const double* arr, size_t size, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(arr), size * sizeof(double));

    outFile.close();
}


__global__ void classifyVertex_CUDA(int rank, int *send_flag, int ite, uint8_t *changed_vertex_h, uint8_t *changed_vertex_copy, int type = 0, int local = 1, int check = 0, int checkFinal = 0, int final = 0) {
   
    double* heightMap = input_data;
    int* lowerStars_t = lowerStars;
    if(type == 1) heightMap = decp_data;
    
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(g_idx >= num_Elements) return;
   
    int x = g_idx % width;
    int y = (g_idx / width) % height;
    int z = (g_idx / (width * height)) % depth;
    if ((((x == 0 || x >= width - 1 || y == 0 || y >= height - 1 || z == 0 || z >= depth - 1) || heightMap[g_idx] == INVALID_VAL) && check == 0) 
    || (type == 1 && ite != 1 && changed_vertex_copy[g_idx] == 0)){
        return;
    }
    

    double data_value = heightMap[g_idx];
    int vertexId = g_idx;
    
    int lowerCount = 0, upperCount = 0;
    int lowerStar[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

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
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < num_Elements && newZ<depth && newZ>=0 && heightMap[r]!=INVALID_VAL) {
            
            double neighbor_value = heightMap[r];
            if (neighbor_value < data_value || (neighbor_value == data_value && r < g_idx)) {
                // if(type==0) lowerStars_t[vertexId*(maxNeighbors+1) + lowerCount] = r;
                binary[maxNeighbors - 1 - neighbor_size] = 0;
                lowerStar[lowerCount] = r;
                lowerCount++;
                
            } else if (neighbor_value > data_value || (neighbor_value == data_value && r > g_idx)) {
                binary[maxNeighbors - 1 - neighbor_size] = 1;
                upperCount++;
                
            }
            vertex_binary[13 - d] = 1;
            neighbor_size++;
            if(islarger_shared(r, largest_index, neighbor_value, largest_value)){
                // if(g_idx == 208446 && rank == 0 && type == 0) printf("%d's decp=%.17f, orignal-bound=%.17f; %d's decp=%.17f, orignal-bound=%.17f \n", r, decp_data[r], input_data[r] - bound, g_idx, decp_data[g_idx], input_data[g_idx] - bound);
                largest_index = r;
                largest_value = neighbor_value;
                
            }

            
            if(isless_shared(r, smallest_index, neighbor_value, smallest_value)){
                smallest_index = r;
                smallest_value = neighbor_value;
                
            }
        }
    }

    // if(type==0) lowerStars_t[vertexId*(maxNeighbors+1) + maxNeighbors] = lowerCount;
    
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
        size_t base = (size_t)g_idx * 3;
        vertex_type[base]     = LUT_result;
        vertex_type[base + 1] = smallest_index;
        vertex_type[base + 2] = largest_index;
        
        if (LUT_result == 2 || LUT_result == 3) {
            int pos = atomicAdd(&nSaddle2, 1);
        }

        if (LUT_result == 1 || LUT_result == 3) {
            int pos = atomicAdd(&nSaddle1, 1);
        }

        if (LUT_result == 4) {
            int pos = atomicAdd(&nMax, 1);
        }

        if (LUT_result == 0) {
            int pos = atomicAdd(&nMin, 1);
        }
    }
    
    else{
        size_t base = (size_t)g_idx * 3;
        int ori_type = vertex_type[base];
        if(LUT_result != ori_type){
            
                // maximum
                if((LUT_result==4 and ori_type!=4) or (LUT_result!=4 and ori_type==4)){
                    // printf("max \n");
                    // if(final==1) printf("max at %d\n", g_idx);
                    count_f_cases+=1;
                    if(checkFinal == 0){
                        if (ori_type != 4) {
                            if (atomicCAS(&edits[g_idx], 0, 1) == 0){
                                
                                applyDeltaBuffer_local<double>(delta_counter,
                                                decp_data, delta, input_data,
                                                bound, q, g_idx, send_flag, changed_vertex_h);
                            }
                            
                        } else {
                            if (atomicCAS(&edits[largest_index], 0, 1) == 0) {
                                applyDeltaBuffer_local<double>(delta_counter,
                                                decp_data, delta, input_data,
                                                bound, q, largest_index, send_flag, changed_vertex_h);
                            }
                            
                        }
                    }
                    
                } 
                if((LUT_result==0 and ori_type!=0) or (LUT_result!=0 and ori_type==0)){
                    // printf("min \n");
                    // if(final==1) printf("min at %d\n", g_idx);
                    count_f_cases+=1;
                    if(checkFinal == 0){
                        if (ori_type != 0) {
                            int original_smallest_index = vertex_type[base + 1];
                            if (atomicCAS(&edits[original_smallest_index], 0, 1) == 0) {
                                applyDeltaBuffer_local<double>(delta_counter,
                                                decp_data, delta, input_data,
                                                bound, q, original_smallest_index, send_flag, changed_vertex_h);
                            }
                        } else{
                            if (atomicCAS(&edits[g_idx], 0, 1) == 0) {
                                applyDeltaBuffer_local<double>(delta_counter,
                                                decp_data, delta, input_data,
                                                bound, q, g_idx, send_flag, changed_vertex_h);
                            }
                            
                        }
                    }
                }

                if((LUT_result==3 and ori_type!=3) or (LUT_result!=3 and ori_type==3) or (LUT_result==1 and ori_type!=1) or (LUT_result!=1 and ori_type==1) or 
                    (LUT_result==2 and ori_type!=2) or (LUT_result!=2 and ori_type==2)){
                    count_f_cases+=1;
                    // printf("saddle \n");
                    // if(final==1) printf("saddle at %d %d %d\n", g_idx, LUT_result, ori_type);
                    double ori_value = input_data[g_idx];
                    int lowerStar_o[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
                    int olowercount = 0;
                    for (int d = 0; d < maxNeighbors; d++) {
        
                        int dirX = directions[d * 3];     
                        int dirY = directions[d * 3 + 1]; 
                        int dirZ = directions[d * 3 + 2]; 
                        int newX = x + dirX;
                        int newY = y + dirY;
                        int newZ = z + dirZ;
                        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < num_Elements && newZ<depth && newZ>=0 && heightMap[r]!=INVALID_VAL) {
                            
                            double neighbor_value = input_data[r];
                            if (neighbor_value < ori_value || (neighbor_value == ori_value && r < g_idx)) {
                                lowerStar_o[olowercount] = r;
                                olowercount++;
                            }
                        }
                    }
                    if(checkFinal == 0) fix_saddle_local(g_idx, lowerStar_o, olowercount, lowerStar, lowerCount, send_flag, changed_vertex_h);
                }


        }

        else if(LUT_result==2 and ori_type==2 || LUT_result== 1 and ori_type==1 || LUT_result == 3 and ori_type==3){
                
            // if(areVectorsDifferent_local(g_idx, lowerStar, lowerCount)){
            double ori_value = input_data[g_idx];
            int lowerStar_o[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
            int olowercount = 0;
            for (int d = 0; d < maxNeighbors; d++) {

                int dirX = directions[d * 3];     
                int dirY = directions[d * 3 + 1]; 
                int dirZ = directions[d * 3 + 2]; 
                int newX = x + dirX;
                int newY = y + dirY;
                int newZ = z + dirZ;
                int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < num_Elements && newZ<depth && newZ>=0 && heightMap[r]!=INVALID_VAL) {
                    
                    double neighbor_value = input_data[r];
                    if (neighbor_value < ori_value || (neighbor_value == ori_value && r < g_idx)) {
                        lowerStar_o[olowercount] = r;
                        olowercount++;
                    }
                }
            }
            if(areVectorsDifferent(lowerStar, lowerStar_o, lowerCount, olowercount)){
                count_f_cases += 1;
                // printf("saddle local order \n");
                // if(final==1) printf("saddle local order at %d\n", g_idx);
                // if(checkFinal == 0) fix_saddle_local(g_idx, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors],send_flag, changed_vertex_h);
                if(checkFinal == 0) fix_saddle_local(g_idx, lowerStar_o, olowercount,  lowerStar, lowerCount, send_flag, changed_vertex_h);
            }
        }

        int original_largest_index  = vertex_type[base + 2];
        // 208447's decp=0.43427607621723580, orignal-bound=0.43427607621723580; 208446's decp=0.43356106076199175, orignal-bound=0.43356106076199175 
        // 212996's decp=0.44698272148209622, orignal-bound=0.44698272148209622; 208446's decp=0.43356106076199175, orignal-bound=0.43356106076199175
        if(largest_index != original_largest_index){
            // printf("largest_index \n");
            // if(g_idx == 208446 && rank == 0 ) printf("local order rank %d %d %d %d %.17f %.17f %.17f %.17f\n", g_idx, rank ,original_largest_index, largest_index, decp_data[original_largest_index], decp_data[largest_index], input_data[original_largest_index] - bound, input_data[largest_index] - bound);
            count_f_cases+=1;
            // if(final==1) printf("largest index at %d %d %d %d %.17f %.17f %.17f %.17f\n", g_idx, rank ,original_largest_index, largest_index, decp_data[original_largest_index], decp_data[largest_index], input_data[original_largest_index] - bound, input_data[largest_index] - bound);
            if(checkFinal == 0){
                if (atomicCAS(&edits[largest_index], 0, 1) == 0) {
                    applyDeltaBuffer_local<double>(delta_counter,
                                    decp_data, delta, input_data,
                                    bound, q, largest_index, send_flag, changed_vertex_h);
                }
            }
            
            
        }

        int original_smallest_index  = vertex_type[base + 1];
        
        if(smallest_index != original_smallest_index){
            
            count_f_cases+=1;
            // printf("smallest_index \n");
            // if(final==1) printf("smallest_index at %d\n", g_idx);
            if (atomicCAS(&edits[original_smallest_index], 0, 1) == 0 && checkFinal == 0) {
                applyDeltaBuffer_local<double>(delta_counter,
                                decp_data, delta, input_data,
                                bound, q, original_smallest_index, send_flag, changed_vertex_h);
            }
            
        }
        return;

    }
}

__global__ void count_dest_atomic(const int* dest, int n, int* counts){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) atomicAdd(&counts[dest[i]],1);
}

template <typename T>
__global__ void apply_recv_edits_kernel(
    const uint64_t* d_recv_idx, int recv_total, int myrank, int offset_x,int offset_y,int offset_z,
                          int local_x,int local_y,int local_z, int *send_flag, int ite, int rank, uint8_t* changed_vertex_h, int checkFinal = 0
){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= recv_total) return;

    uint64_t id = d_recv_idx[t];

    // printf("saddle global \n");
    int lid;
    if(local_id_from_gid(id, global_width, global_height, offset_x,offset_y,offset_z, local_x,local_y,local_z, width, height, &lid)){
        count_f_cases += 1;
        if (checkFinal == 0 && atomicCAS(&edits[lid], 0, 1) == 0) {
            // if(rank == 3 && ite > 250 && id == 363774829) printf("before apply_recv_edits_kernel %d %.17f\n", lid, decp_data[lid]);
            applyDeltaBuffer_local<T>(delta_counter, decp_data, delta, input_data, bound, q, lid, send_flag, changed_vertex_h);
            // if(rank == 3 && ite > 250 && id == 363774829) printf("after apply_recv_edits_kernel %d %.17f\n", lid, decp_data[lid]);
        }
    }   
    
    
}

static inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        fflush(stdout);
        std::abort();
    }
}


template <typename T>
void send_prev_violation_ids_and_apply(
    MPI_Comm comm,
    int rank,
    int size,

    // set0: saddle1
    int n1,
    dim3 blockSize,
    const int*     d_ids1,          // saddles1 device int*
    const double*  d_prev_val1, const double*  d_next_val1,
    const int*     d_prev_rank1, const int*     d_next_rank1,
    const uint64_t* d_prev_idx1, const uint64_t* d_next_idx1,
    const uint8_t* d_need_prev1, const uint8_t* d_need_next1,  // NEW

    // set1: saddle2
    int n2,
    const int*     d_ids2,          // saddles2
    const double*  d_prev_val2, const double*  d_next_val2,
    const int*     d_prev_rank2, const int*     d_next_rank2,
    const uint64_t* d_prev_idx2, const uint64_t* d_next_idx2,
    const uint8_t* d_need_prev2, const uint8_t* d_need_next2,  // NEW

    // set2: minimum
    int nMin,
    const int*     d_idsMin,        // minimum (device int*)
    const double*  d_prev_valMin, const double*  d_next_valMin,
    const int*     d_prev_rankMin, const int*     d_next_rankMin,
    const uint64_t* d_prev_idxMin, const uint64_t* d_next_idxMin,
    const uint8_t* d_need_prevMin, const uint8_t* d_need_nextMin,  // NEW

    // set3: maximum
    int nMax,
    const int*     d_idsMax,        // maximum (device int*)
    const double*  d_prev_valMax, const double*  d_next_valMax,
    const int*     d_prev_rankMax, const int*     d_next_rankMax,
    const uint64_t* d_prev_idxMax, const uint64_t* d_next_idxMax,
    const uint8_t* d_need_prevMax, const uint8_t* d_need_nextMax,  // NEW

    // debug/apply
    bool debug_print,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z,
    int ite, int *send_flag, uint8_t *changed_vertex_h, double global_start_time, double comm_end_time, double &total_cp_comm_time, double &comp_time_iter,
    int checkFinal = 0
    /* 这里继续加 apply_recv_edits_kernel<T> 需要的参数 */
){
    // 每个点最多发 1 条（prev-side violation 才远程 send）
    int max_req = n1 + n2 + nMin + nMax;

    int*      d_req_dest = nullptr;
    uint64_t* d_req_idx  = nullptr;
    int*      d_nreq     = nullptr;
    double comp_start_time = MPI_Wtime() - global_start_time;
    cudaCheck(cudaMalloc(&d_req_dest, (size_t)max_req * sizeof(int)),      "malloc d_req_dest");
    cudaCheck(cudaMalloc(&d_req_idx,  (size_t)max_req * sizeof(uint64_t)), "malloc d_req_idx");
    cudaCheck(cudaMalloc(&d_nreq, sizeof(int)),                            "malloc d_nreq");
    cudaCheck(cudaMemset(d_nreq, 0, sizeof(int)),                          "memset d_nreq");

    // ---- build request list (GPU) ----
    int threads = (int)blockSize.x;
    int blocks  = (max_req + threads - 1) / threads;

    if (max_req > 0) {
        extractGlobalWrongOrder<<<blocks, threads>>>(
            // saddle1
            n1, d_ids1,
            d_prev_val1, d_next_val1, d_prev_rank1, d_next_rank1, d_prev_idx1, d_next_idx1,
            d_need_prev1, d_need_next1,  // NEW

            // saddle2
            n2, d_ids2,
            d_prev_val2, d_next_val2, d_prev_rank2, d_next_rank2, d_prev_idx2, d_next_idx2,
            d_need_prev2, d_need_next2,  // NEW

            // minimum
            nMin, d_idsMin,
            d_prev_valMin, d_next_valMin, d_prev_rankMin, d_next_rankMin, d_prev_idxMin, d_next_idxMin,
            d_need_prevMin, d_need_nextMin,  // NEW

            // maximum
            nMax, d_idsMax,
            d_prev_valMax, d_next_valMax, d_prev_rankMax, d_next_rankMax, d_prev_idxMax, d_next_idxMax,
            d_need_prevMax, d_need_nextMax,  // NEW

            rank,
            d_req_dest, d_req_idx, d_nreq,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z,
            ite, send_flag, changed_vertex_h,
            checkFinal
        );
        cudaCheck(cudaGetLastError(), "launch extractGlobalWrongOrder4");
    }


    double comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
        
    int nreq = 0;
    cudaCheck(cudaMemcpy(&nreq, d_nreq, sizeof(int), cudaMemcpyDeviceToHost), "copy nreq D2H");
    cudaDeviceSynchronize();

    if (nreq > max_req) {
        printf("[rank %d] ERROR: nreq= %d > max_req=%d (increase buffer)\n", rank, nreq, max_req);
        fflush(stdout);
        MPI_Abort(comm, 1);
    }

    // ---- sort by dest (GPU) ----
    if (nreq > 0) {
        thrust::sort_by_key(
            thrust::device_pointer_cast(d_req_dest),
            thrust::device_pointer_cast(d_req_dest) + nreq,
            thrust::device_pointer_cast(d_req_idx)
        );
    }

    // ---- send_counts on GPU -> host ----
    int* d_send_counts = nullptr;
    cudaCheck(cudaMalloc(&d_send_counts, (size_t)size * sizeof(int)), "malloc d_send_counts");
    cudaCheck(cudaMemset(d_send_counts, 0, (size_t)size * sizeof(int)), "memset d_send_counts");

    if (nreq > 0) {
        count_dest_atomic<<<(nreq + 255) / 256, 256>>>(d_req_dest, nreq, d_send_counts);
        cudaCheck(cudaGetLastError(), "launch count_dest_atomic");
    }

    double comm_start_time = MPI_Wtime() - global_start_time;
    std::vector<int> h_send_counts(size, 0), h_recv_counts(size, 0);
    cudaCheck(cudaMemcpy(h_send_counts.data(), d_send_counts,
                         (size_t)size * sizeof(int), cudaMemcpyDeviceToHost),
              "copy send_counts D2H");
    cudaCheck(cudaDeviceSynchronize(), "sync after send_counts");

    int send_total = 0;
    for (int r=0; r<size; ++r) send_total += h_send_counts[r];
    if (send_total != nreq) {
        printf("[rank %d] ERROR: send_total=%d != nreq=%d\n", rank, send_total, nreq);
        fflush(stdout);
        MPI_Abort(comm, 1);
    }
    
    // 使用 P2P 版本的 Alltoall（避免 OFI 队列溢出）
    p2p_alltoall_int(h_send_counts.data(), h_recv_counts.data(), 1, comm);

    std::vector<int> h_sdispls(size, 0), h_rdispls(size, 0);
    for (int r = 1; r < size; ++r) {
        h_sdispls[r] = h_sdispls[r - 1] + h_send_counts[r - 1];
        h_rdispls[r] = h_rdispls[r - 1] + h_recv_counts[r - 1];
    }

    int recv_total = 0;
    for (int r = 0; r < size; ++r) recv_total += h_recv_counts[r];

    // ---- send/recv buffer ----
    uint64_t* d_recv_idx = nullptr;
    size_t recv_alloc_size = (recv_total > 0) ? recv_total : 1;
    cudaCheck(cudaMalloc(&d_recv_idx, recv_alloc_size * sizeof(uint64_t)), "malloc d_recv_idx");
    
    cudaCheck(cudaDeviceSynchronize(), "sync before P2P Alltoallv");
    
    // 使用分阶段 P2P 版本（适用于大规模 64+ GPU）
    p2p_alltoallv_cuda_staged_throttled<uint64_t>(
        d_req_idx, h_send_counts.data(), h_sdispls.data(),
        d_recv_idx, h_recv_counts.data(), h_rdispls.data(),
        MPI_UINT64_T, comm
    );
    
    cudaCheck(cudaDeviceSynchronize(), "sync after P2P Alltoallv");

    // ---- debug ----
    if (debug_print) {
    // if(true){
        printf("[rank %d] sent %d prev-violation requests (4 sets)\n", rank, nreq);
        printf("========== Rank %d received %d prev-violation requests (4 sets) ==========\n", rank, recv_total);
        // if (recv_total > 0) {
        //     std::vector<uint64_t> h_recv_idx(recv_total);
        //     cudaCheck(cudaMemcpy(h_recv_idx.data(), d_recv_idx,
        //                          (size_t)recv_total * sizeof(uint64_t),
        //                          cudaMemcpyDeviceToHost),
        //               "copy recv_idx D2H");
        //     for (int i = 0; i < recv_total; ++i) {
        //         printf("[rank %d] recv request %d : target_gid = %llu\n",
        //                rank, i, (unsigned long long)h_recv_idx[i]);
        //     }
        // }
        fflush(stdout);
    }

    comm_end_time = MPI_Wtime() - global_start_time;
    total_cp_comm_time = comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
    // ---- apply on GPU ----
    if (recv_total > 0) {
        int blocks_apply = (recv_total + 255) / 256;
        apply_recv_edits_kernel<T><<<blocks_apply, 256>>>(
            d_recv_idx, recv_total, rank,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z, send_flag, ite, rank, changed_vertex_h, 
            checkFinal
            /* 这里补上你 apply 需要的参数 */
        );
        cudaCheck(cudaGetLastError(), "launch apply_recv_edits_kernel");
    }

    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
        
    // ---- cleanup ----
    if (d_recv_idx)     cudaCheck(cudaFree(d_recv_idx),     "free d_recv_idx");
    if (d_send_counts)  cudaCheck(cudaFree(d_send_counts),  "free d_send_counts");
    if (d_req_dest)     cudaCheck(cudaFree(d_req_dest),     "free d_req_dest");
    if (d_req_idx)      cudaCheck(cudaFree(d_req_idx),      "free d_req_idx");
    if (d_nreq)         cudaCheck(cudaFree(d_nreq),         "free d_nreq");
}

template <typename T>
static void sparse_p2p_alltoallv_cudaaware(
    const T* sendbuf, const int* sendcounts, const int* sdispls,
    T* recvbuf, const int* recvcounts, const int* rdispls,
    MPI_Datatype dtype, MPI_Comm comm,
    int iter_tag_base = 0   // 传 0 或者传 ite*10 之类，避免不同阶段串线
){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // self copy（如果你需要）
    if (sendcounts[rank] > 0) {
        cudaMemcpy((void*)&recvbuf[rdispls[rank]],
                   (const void*)&sendbuf[sdispls[rank]],
                   (size_t)sendcounts[rank] * sizeof(T),
                   cudaMemcpyDeviceToDevice);
    }

    // 收发 peer 列表（非常 sparse 时这两个会很小）
    std::vector<int> recv_peers, send_peers;
    recv_peers.reserve(32);
    send_peers.reserve(32);

    for (int peer = 0; peer < size; ++peer) {
        if (peer == rank) continue;
        if (recvcounts[peer] > 0) recv_peers.push_back(peer);
        if (sendcounts[peer] > 0) send_peers.push_back(peer);
    }

    // 统一小 tag（关键：别用 pair 编码）
    const int TAG = 101 + iter_tag_base;

    std::vector<MPI_Request> reqs;
    reqs.reserve(recv_peers.size() + send_peers.size());

    // 先 Irecv（避免大家都先 Isend 抢资源）
    for (int peer : recv_peers) {
        MPI_Request r;
        MPI_Irecv((void*)&recvbuf[rdispls[peer]], recvcounts[peer],
                  dtype, peer, TAG, comm, &r);
        reqs.push_back(r);
    }

    // 再 Isend
    for (int peer : send_peers) {
        MPI_Request s;
        MPI_Isend((const void*)&sendbuf[sdispls[peer]], sendcounts[peer],
                  dtype, peer, TAG, comm, &s);
        reqs.push_back(s);
    }

    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }
}

static inline void cudaCheck_(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(e));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template <typename T>
static void p2p_alltoallv_host_staged_sparse(
    const T* d_sendbuf, const int* sendcounts, const int* sdispls,
    T*       d_recvbuf, const int* recvcounts, const int* rdispls,
    MPI_Datatype dtype, MPI_Comm comm,
    int tag_base = 7000000,          // 避免跟别的 P2P tag 冲突
    bool use_pinned = true           // pinned host buffer 更快更稳
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1) self-copy：device->device
    if (d_sendbuf && d_recvbuf && sendcounts[rank] > 0) {
        cudaCheck_(cudaMemcpy((void*)(d_recvbuf + rdispls[rank]),
                              (const void*)(d_sendbuf + sdispls[rank]),
                              (size_t)sendcounts[rank] * sizeof(T),
                              cudaMemcpyDeviceToDevice),
                   "self D2D memcpy");
    }

    if (size == 1) return;

    // 2) 收集稀疏 peer 列表
    std::vector<int> send_peers, recv_peers;
    send_peers.reserve(64);
    recv_peers.reserve(64);

    for (int r = 0; r < size; ++r) if (r != rank) {
        if (sendcounts[r] > 0) send_peers.push_back(r);
        if (recvcounts[r] > 0) recv_peers.push_back(r);
    }

    if (send_peers.empty() && recv_peers.empty()) return;

    // 3) 为每个 peer 分配 host staging buffer（按 peer 单独分配，简单稳）
    struct HostBuf {
        int peer;
        T*  h_ptr;
        int count;
    };
    std::vector<HostBuf> h_send, h_recv;
    h_send.reserve(send_peers.size());
    h_recv.reserve(recv_peers.size());

    auto host_alloc = [&](size_t bytes)->void* {
        void* p = nullptr;
        if (use_pinned) cudaCheck_(cudaHostAlloc(&p, bytes, cudaHostAllocPortable), "cudaHostAlloc");
        else {
            p = std::malloc(bytes);
            if (!p) {
                printf("[ERROR] malloc failed (%zu bytes)\n", bytes);
                fflush(stdout);
                MPI_Abort(comm, 1);
            }
        }
        return p;
    };
    auto host_free = [&](void* p) {
        if (!p) return;
        if (use_pinned) cudaCheck_(cudaFreeHost(p), "cudaFreeHost");
        else std::free(p);
    };

    // recv buffers
    for (int peer : recv_peers) {
        int c = recvcounts[peer];
        size_t bytes = (size_t)c * sizeof(T);
        T* h = (T*)host_alloc(std::max<size_t>(bytes, 1));
        h_recv.push_back({peer, h, c});
    }
    // send buffers
    for (int peer : send_peers) {
        int c = sendcounts[peer];
        size_t bytes = (size_t)c * sizeof(T);
        T* h = (T*)host_alloc(std::max<size_t>(bytes, 1));
        h_send.push_back({peer, h, c});
    }

    // 4) D2H：把每个 peer 的发送段从 device 拷到 host
    //    （你也可以用 stream + cudaMemcpyAsync 做 overlap；先给最稳同步版）
    for (auto &sb : h_send) {
        int peer = sb.peer;
        int c = sb.count;
        cudaCheck_(cudaMemcpy(sb.h_ptr,
                              (const void*)(d_sendbuf + sdispls[peer]),
                              (size_t)c * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "D2H pack per-peer");
    }
    cudaCheck_(cudaDeviceSynchronize(), "sync after D2H sends");

    // 5) P2P：先 Irecv(host) 再 Isend(host)
    std::vector<MPI_Request> reqs;
    reqs.reserve(h_recv.size() + h_send.size());

    auto pair_tag = [&](int a, int b)->int {
        int lo = std::min(a,b), hi = std::max(a,b);
        // 保持你原来的 pair tag 逻辑 + base
        return tag_base + lo * 100000 + hi;
    };

    // Irecv
    for (auto &rb : h_recv) {
        MPI_Request req;
        int tag = pair_tag(rank, rb.peer);
        MPI_Irecv((void*)rb.h_ptr, rb.count, dtype, rb.peer, tag, comm, &req);
        reqs.push_back(req);
    }
    // Isend
    for (auto &sb : h_send) {
        MPI_Request req;
        int tag = pair_tag(rank, sb.peer);
        MPI_Isend((const void*)sb.h_ptr, sb.count, dtype, sb.peer, tag, comm, &req);
        reqs.push_back(req);
    }

    if (!reqs.empty()) {
        int ret = MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        if (ret != MPI_SUCCESS) {
            char errstr[MPI_MAX_ERROR_STRING]; int len = 0;
            MPI_Error_string(ret, errstr, &len);
            printf("[MPI ERROR] rank %d Waitall failed: %s\n", rank, errstr);
            fflush(stdout);
            MPI_Abort(comm, 1);
        }
    }

    // 6) H2D：把每个 peer 收到的 host 段拷回 device 对应位置
    for (auto &rb : h_recv) {
        int peer = rb.peer;
        int c = rb.count;
        cudaCheck_(cudaMemcpy((void*)(d_recvbuf + rdispls[peer]),
                              (const void*)rb.h_ptr,
                              (size_t)c * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "H2D unpack per-peer");
    }
    cudaCheck_(cudaDeviceSynchronize(), "sync after H2D recvs");

    // 7) cleanup
    for (auto &sb : h_send) host_free(sb.h_ptr);
    for (auto &rb : h_recv) host_free(rb.h_ptr);
}

template <typename T>
void send_prev_violation_ids_and_apply_all2all(
    MPI_Comm comm,
    int rank,
    int size,

    // set0: saddle1
    int n1,
    dim3 blockSize,
    const int*     d_ids1,          // saddles1 device int*
    const double*  d_prev_val1, const double*  d_next_val1,
    const int*     d_prev_rank1, const int*     d_next_rank1,
    const uint64_t* d_prev_idx1, const uint64_t* d_next_idx1,
    const uint8_t* d_need_prev1, const uint8_t* d_need_next1,  // NEW

    // set1: saddle2
    int n2,
    const int*     d_ids2,          // saddles2
    const double*  d_prev_val2, const double*  d_next_val2,
    const int*     d_prev_rank2, const int*     d_next_rank2,
    const uint64_t* d_prev_idx2, const uint64_t* d_next_idx2,
    const uint8_t* d_need_prev2, const uint8_t* d_need_next2,  // NEW

    // set2: minimum
    int nMin,
    const int*     d_idsMin,        // minimum (device int*)
    const double*  d_prev_valMin, const double*  d_next_valMin,
    const int*     d_prev_rankMin, const int*     d_next_rankMin,
    const uint64_t* d_prev_idxMin, const uint64_t* d_next_idxMin,
    const uint8_t* d_need_prevMin, const uint8_t* d_need_nextMin,  // NEW

    // set3: maximum
    int nMax,
    const int*     d_idsMax,        // maximum (device int*)
    const double*  d_prev_valMax, const double*  d_next_valMax,
    const int*     d_prev_rankMax, const int*     d_next_rankMax,
    const uint64_t* d_prev_idxMax, const uint64_t* d_next_idxMax,
    const uint8_t* d_need_prevMax, const uint8_t* d_need_nextMax,  // NEW

    // debug/apply
    bool debug_print,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z,
    int ite, int *send_flag, uint8_t *changed_vertex_h, double global_start_time, double comm_end_time, double &total_cp_comm_time, double &comp_time_iter,
    int checkFinal = 0
    /* 这里继续加 apply_recv_edits_kernel<T> 需要的参数 */
){
    // 每个点最多发 1 条（prev-side violation 才远程 send）
    int max_req = n1 + n2 + nMin + nMax;

    int*      d_req_dest = nullptr;
    uint64_t* d_req_idx  = nullptr;
    int*      d_nreq     = nullptr;

    cudaCheck(cudaMalloc(&d_req_dest, (size_t)max_req * sizeof(int)),      "malloc d_req_dest");
    cudaCheck(cudaMalloc(&d_req_idx,  (size_t)max_req * sizeof(uint64_t)), "malloc d_req_idx");
    cudaCheck(cudaMalloc(&d_nreq, sizeof(int)),                            "malloc d_nreq");
    cudaCheck(cudaMemset(d_nreq, 0, sizeof(int)),                          "memset d_nreq");

    // ---- build request list (GPU) ----
    int threads = (int)blockSize.x;
    int blocks  = (max_req + threads - 1) / threads;
    double comp_start_time = MPI_Wtime() - global_start_time;
    
    if (max_req > 0) {
        extractGlobalWrongOrder<<<blocks, threads>>>(
            // saddle1
            n1, d_ids1,
            d_prev_val1, d_next_val1, d_prev_rank1, d_next_rank1, d_prev_idx1, d_next_idx1,
            d_need_prev1, d_need_next1,  // NEW

            // saddle2
            n2, d_ids2,
            d_prev_val2, d_next_val2, d_prev_rank2, d_next_rank2, d_prev_idx2, d_next_idx2,
            d_need_prev2, d_need_next2,  // NEW

            // minimum
            nMin, d_idsMin,
            d_prev_valMin, d_next_valMin, d_prev_rankMin, d_next_rankMin, d_prev_idxMin, d_next_idxMin,
            d_need_prevMin, d_need_nextMin,  // NEW

            // maximum
            nMax, d_idsMax,
            d_prev_valMax, d_next_valMax, d_prev_rankMax, d_next_rankMax, d_prev_idxMax, d_next_idxMax,
            d_need_prevMax, d_need_nextMax,  // NEW

            rank,
            d_req_dest, d_req_idx, d_nreq,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z,
            ite, send_flag, changed_vertex_h,
            checkFinal
        );
        cudaCheck(cudaGetLastError(), "launch extractGlobalWrongOrder4");
    }
    
    double comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
        
    int nreq = 0;
    cudaCheck(cudaMemcpy(&nreq, d_nreq, sizeof(int), cudaMemcpyDeviceToHost), "copy nreq D2H");
    cudaDeviceSynchronize();

    if (nreq > max_req) {
        printf("[rank %d] ERROR: nreq= %d > max_req=%d (increase buffer)\n", rank, nreq, max_req);
        fflush(stdout);
        MPI_Abort(comm, 1);
    }

    // ---- sort by dest (GPU) ----
    if (nreq > 0) {
        thrust::sort_by_key(
            thrust::device_pointer_cast(d_req_dest),
            thrust::device_pointer_cast(d_req_dest) + nreq,
            thrust::device_pointer_cast(d_req_idx)
        );
    }

    // ---- send_counts on GPU -> host ----
    int* d_send_counts = nullptr;
    cudaCheck(cudaMalloc(&d_send_counts, (size_t)size * sizeof(int)), "malloc d_send_counts");
    cudaCheck(cudaMemset(d_send_counts, 0, (size_t)size * sizeof(int)), "memset d_send_counts");

    if (nreq > 0) {
        count_dest_atomic<<<(nreq + 255) / 256, 256>>>(d_req_dest, nreq, d_send_counts);
        cudaCheck(cudaGetLastError(), "launch count_dest_atomic");
    }
    double comm_start_time = MPI_Wtime() - global_start_time;
    
    std::vector<int> h_send_counts(size, 0), h_recv_counts(size, 0);
    cudaCheck(cudaMemcpy(h_send_counts.data(), d_send_counts,
                         (size_t)size * sizeof(int), cudaMemcpyDeviceToHost),
              "copy send_counts D2H");
    cudaCheck(cudaDeviceSynchronize(), "sync after send_counts");

    int send_total = 0;
    for (int r=0; r<size; ++r) send_total += h_send_counts[r];
    if (send_total != nreq) {
        printf("[rank %d] ERROR: send_total=%d != nreq=%d\n", rank, send_total, nreq);
        fflush(stdout);
        MPI_Abort(comm, 1);
    }
    
    // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
    MPI_Barrier(comm);
    
    MPI_Alltoall(h_send_counts.data(), 1, MPI_INT,
                 h_recv_counts.data(), 1, MPI_INT, comm);


    std::vector<int> h_sdispls(size, 0), h_rdispls(size, 0);
    for (int r = 1; r < size; ++r) {
        h_sdispls[r] = h_sdispls[r - 1] + h_send_counts[r - 1];
        h_rdispls[r] = h_rdispls[r - 1] + h_recv_counts[r - 1];
    }

    int recv_total = 0;
    for (int r = 0; r < size; ++r) recv_total += h_recv_counts[r];

    // ---- send/recv buffer ----
    uint64_t* d_recv_idx = nullptr;
    size_t recv_alloc_size = (recv_total > 0) ? recv_total : 1;
    cudaCheck(cudaMalloc(&d_recv_idx, recv_alloc_size * sizeof(uint64_t)), "malloc d_recv_idx");
    
    cudaCheck(cudaDeviceSynchronize(), "sync before MPI_Alltoallv");
    
    // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
    MPI_Barrier(comm);
    
    // 确保 dummy buffer 存在
    ensure_dummy();
    
    // CUDA-aware MPI: 使用 dummy buffer 处理空的情况
    // uint64_t* send_ptr = (nreq > 0) ? d_req_idx : d_dummy_u64;
    // uint64_t* recv_ptr = (recv_total > 0) ? d_recv_idx : d_dummy_u64;

    p2p_alltoallv_host_staged_sparse<uint64_t>(
    /*d_sendbuf*/ (nreq > 0 ? d_req_idx : d_dummy_u64),
    /*sendcounts*/ h_send_counts.data(),
    /*sdispls*/    h_sdispls.data(),

    /*d_recvbuf*/  (recv_total > 0 ? d_recv_idx : d_dummy_u64),
    /*recvcounts*/ h_recv_counts.data(),
    /*rdispls*/    h_rdispls.data(),

    /*dtype*/      MPI_UINT64_T,
    /*comm*/       comm,
    /*tag_base*/   7100000,   // 随便选个不和别的通信冲突的 base
    /*use_pinned*/ true
    );
    // int total_send = 0;
    // int total_recv = 0;

    // for (int r = 0; r < size; ++r) {
    //     total_send += h_send_counts[r];
    //     total_recv += h_recv_counts[r];
    // }

    // // // 至少分配1个，避免空vector.data()==nullptr
    // std::vector<uint64_t> h_send(std::max(1, total_send));
    // std::vector<uint64_t> h_recv(std::max(1, total_recv));

    // // 2️⃣ D2H
    // if (total_send > 0) {
    //     cudaMemcpy(h_send.data(),
    //             d_req_idx,
    //             (size_t)total_send * sizeof(uint64_t),
    //             cudaMemcpyDeviceToHost);
    // }

    // // 3️⃣ Host Alltoallv
    // MPI_Alltoallv(
    //     h_send.data(),
    //     h_send_counts.data(), h_sdispls.data(), MPI_UINT64_T,
    //     h_recv.data(),
    //     h_recv_counts.data(), h_rdispls.data(), MPI_UINT64_T,
    //     comm
    // );

    // 4️⃣ H2D
    // if (total_recv > 0) {
    //     cudaMemcpy(d_recv_idx,
    //             h_recv.data(),
    //             (size_t)total_recv * sizeof(uint64_t),
    //             cudaMemcpyHostToDevice);
    // }

    
    cudaCheck(cudaDeviceSynchronize(), "sync after MPI_Alltoallv");
    

    comm_end_time = MPI_Wtime() - global_start_time;
    total_cp_comm_time = comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
        
    // ---- apply on GPU ----
    if (recv_total > 0) {
        int blocks_apply = (recv_total + 255) / 256;
        apply_recv_edits_kernel<T><<<blocks_apply, 256>>>(
            d_recv_idx, recv_total, rank,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z, send_flag, ite, rank, changed_vertex_h, 
            checkFinal
            /* 这里补上你 apply 需要的参数 */
        );
        cudaCheck(cudaGetLastError(), "launch apply_recv_edits_kernel");
    }
    MPI_Barrier(comm);
    
    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
        

    // ---- cleanup ----
    if (d_recv_idx)     cudaCheck(cudaFree(d_recv_idx),     "free d_recv_idx");
    if (d_send_counts)  cudaCheck(cudaFree(d_send_counts),  "free d_send_counts");
    if (d_req_dest)     cudaCheck(cudaFree(d_req_dest),     "free d_req_dest");
    if (d_req_idx)      cudaCheck(cudaFree(d_req_idx),      "free d_req_idx");
    if (d_nreq)         cudaCheck(cudaFree(d_nreq),         "free d_nreq");
}

template <typename T>
void send_prev_violation_ids_and_apply_all2all_cuda(
    MPI_Comm comm,
    int rank,
    int size,

    // set0: saddle1
    int n1,
    dim3 blockSize,
    const int*     d_ids1,          // saddles1 device int*
    const double*  d_prev_val1, const double*  d_next_val1,
    const int*     d_prev_rank1, const int*     d_next_rank1,
    const uint64_t* d_prev_idx1, const uint64_t* d_next_idx1,
    const uint8_t* d_need_prev1, const uint8_t* d_need_next1,  // NEW

    // set1: saddle2
    int n2,
    const int*     d_ids2,          // saddles2
    const double*  d_prev_val2, const double*  d_next_val2,
    const int*     d_prev_rank2, const int*     d_next_rank2,
    const uint64_t* d_prev_idx2, const uint64_t* d_next_idx2,
    const uint8_t* d_need_prev2, const uint8_t* d_need_next2,  // NEW

    // set2: minimum
    int nMin,
    const int*     d_idsMin,        // minimum (device int*)
    const double*  d_prev_valMin, const double*  d_next_valMin,
    const int*     d_prev_rankMin, const int*     d_next_rankMin,
    const uint64_t* d_prev_idxMin, const uint64_t* d_next_idxMin,
    const uint8_t* d_need_prevMin, const uint8_t* d_need_nextMin,  // NEW

    // set3: maximum
    int nMax,
    const int*     d_idsMax,        // maximum (device int*)
    const double*  d_prev_valMax, const double*  d_next_valMax,
    const int*     d_prev_rankMax, const int*     d_next_rankMax,
    const uint64_t* d_prev_idxMax, const uint64_t* d_next_idxMax,
    const uint8_t* d_need_prevMax, const uint8_t* d_need_nextMax,  // NEW

    // debug/apply
    bool debug_print,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z,
    int ite, int *send_flag, uint8_t *changed_vertex_h, double global_start_time, double comm_end_time, double &total_cp_comm_time, double &comp_time_iter,
    int checkFinal = 0
    /* 这里继续加 apply_recv_edits_kernel<T> 需要的参数 */
){
    // 每个点最多发 1 条（prev-side violation 才远程 send）
    int max_req = n1 + n2 + nMin + nMax;

    int*      d_req_dest = nullptr;
    uint64_t* d_req_idx  = nullptr;
    int*      d_nreq     = nullptr;

    cudaCheck(cudaMalloc(&d_req_dest, (size_t)max_req * sizeof(int)),      "malloc d_req_dest");
    cudaCheck(cudaMalloc(&d_req_idx,  (size_t)max_req * sizeof(uint64_t)), "malloc d_req_idx");
    cudaCheck(cudaMalloc(&d_nreq, sizeof(int)),                            "malloc d_nreq");
    cudaCheck(cudaMemset(d_nreq, 0, sizeof(int)),                          "memset d_nreq");

    // ---- build request list (GPU) ----
    int threads = (int)blockSize.x;
    int blocks  = (max_req + threads - 1) / threads;
    double comp_start_time = MPI_Wtime() - global_start_time;
    
    if (max_req > 0) {
        extractGlobalWrongOrder<<<blocks, threads>>>(
            // saddle1
            n1, d_ids1,
            d_prev_val1, d_next_val1, d_prev_rank1, d_next_rank1, d_prev_idx1, d_next_idx1,
            d_need_prev1, d_need_next1,  // NEW

            // saddle2
            n2, d_ids2,
            d_prev_val2, d_next_val2, d_prev_rank2, d_next_rank2, d_prev_idx2, d_next_idx2,
            d_need_prev2, d_need_next2,  // NEW

            // minimum
            nMin, d_idsMin,
            d_prev_valMin, d_next_valMin, d_prev_rankMin, d_next_rankMin, d_prev_idxMin, d_next_idxMin,
            d_need_prevMin, d_need_nextMin,  // NEW

            // maximum
            nMax, d_idsMax,
            d_prev_valMax, d_next_valMax, d_prev_rankMax, d_next_rankMax, d_prev_idxMax, d_next_idxMax,
            d_need_prevMax, d_need_nextMax,  // NEW

            rank,
            d_req_dest, d_req_idx, d_nreq,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z,
            ite, send_flag, changed_vertex_h,
            checkFinal
        );
        cudaCheck(cudaGetLastError(), "launch extractGlobalWrongOrder4");
    }
    
    double comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
        
    int nreq = 0;
    cudaCheck(cudaMemcpy(&nreq, d_nreq, sizeof(int), cudaMemcpyDeviceToHost), "copy nreq D2H");
    cudaDeviceSynchronize();

    if (nreq > max_req) {
        printf("[rank %d] ERROR: nreq= %d > max_req=%d (increase buffer)\n", rank, nreq, max_req);
        fflush(stdout);
        MPI_Abort(comm, 1);
    }

    // ---- sort by dest (GPU) ----
    if (nreq > 0) {
        thrust::sort_by_key(
            thrust::device_pointer_cast(d_req_dest),
            thrust::device_pointer_cast(d_req_dest) + nreq,
            thrust::device_pointer_cast(d_req_idx)
        );
    }

    // ---- send_counts on GPU -> host ----
    int* d_send_counts = nullptr;
    cudaCheck(cudaMalloc(&d_send_counts, (size_t)size * sizeof(int)), "malloc d_send_counts");
    cudaCheck(cudaMemset(d_send_counts, 0, (size_t)size * sizeof(int)), "memset d_send_counts");

    if (nreq > 0) {
        count_dest_atomic<<<(nreq + 255) / 256, 256>>>(d_req_dest, nreq, d_send_counts);
        cudaCheck(cudaGetLastError(), "launch count_dest_atomic");
    }
    double comm_start_time = MPI_Wtime() - global_start_time;
    
    std::vector<int> h_send_counts(size, 0), h_recv_counts(size, 0);
    cudaCheck(cudaMemcpy(h_send_counts.data(), d_send_counts,
                         (size_t)size * sizeof(int), cudaMemcpyDeviceToHost),
              "copy send_counts D2H");
    cudaCheck(cudaDeviceSynchronize(), "sync after send_counts");

    int send_total = 0;
    for (int r=0; r<size; ++r) send_total += h_send_counts[r];
    if (send_total != nreq) {
        printf("[rank %d] ERROR: send_total=%d != nreq=%d\n", rank, send_total, nreq);
        fflush(stdout);
        MPI_Abort(comm, 1);
    }
    
    // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
    MPI_Barrier(comm);
    MPI_Alltoall(h_send_counts.data(), 1, MPI_INT,
                 h_recv_counts.data(), 1, MPI_INT, comm);


    std::vector<int> h_sdispls(size, 0), h_rdispls(size, 0);
    for (int r = 1; r < size; ++r) {
        h_sdispls[r] = h_sdispls[r - 1] + h_send_counts[r - 1];
        h_rdispls[r] = h_rdispls[r - 1] + h_recv_counts[r - 1];
    }

    int recv_total = 0;
    for (int r = 0; r < size; ++r) recv_total += h_recv_counts[r];

    // ---- send/recv buffer ----
    uint64_t* d_recv_idx = nullptr;
    size_t recv_alloc_size = (recv_total > 0) ? recv_total : 1;
    cudaCheck(cudaMalloc(&d_recv_idx, recv_alloc_size * sizeof(uint64_t)), "malloc d_recv_idx");
    
    cudaCheck(cudaDeviceSynchronize(), "sync before MPI_Alltoallv");
    
    // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
    MPI_Barrier(comm);
    
    // 确保 dummy buffer 存在
    ensure_dummy();
    
    // CUDA-aware MPI: 使用 dummy buffer 处理空的情况
    uint64_t* send_ptr = (nreq > 0) ? d_req_idx : d_dummy_u64;
    uint64_t* recv_ptr = (recv_total > 0) ? d_recv_idx : d_dummy_u64;
    // MPI_Barrier(comm);
    // if (true) {
    // // if(true){
    //     printf("[rank %d] sent %d prev-violation requests (4 sets)\n", rank, nreq);
    //     printf("========== Rank %d received %d prev-violation requests (4 sets) ==========\n", rank, recv_total);
    //     // if (recv_total > 0) {
    //     //     std::vector<uint64_t> h_recv_idx(recv_total);
    //     //     cudaCheck(cudaMemcpy(h_recv_idx.data(), d_recv_idx,
    //     //                          (size_t)recv_total * sizeof(uint64_t),
    //     //                          cudaMemcpyDeviceToHost),
    //     //               "copy recv_idx D2H");
    //     //     for (int i = 0; i < recv_total; ++i) {
    //     //         printf("[rank %d] recv request %d : target_gid = %llu\n",
    //     //                rank, i, (unsigned long long)h_recv_idx[i]);
    //     //     }
    //     // }
    //     fflush(stdout);
    // }
    // 1️⃣ 计算总发送和接收数量
    MPI_Alltoallv(
        send_ptr,
        h_send_counts.data(), h_sdispls.data(), MPI_UINT64_T,
        recv_ptr,
        h_recv_counts.data(), h_rdispls.data(), MPI_UINT64_T,
        comm
    );

    
    cudaCheck(cudaDeviceSynchronize(), "sync after MPI_Alltoallv");
    

    comm_end_time = MPI_Wtime() - global_start_time;
    total_cp_comm_time = comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
        
    // ---- apply on GPU ----
    if (recv_total > 0) {
        int blocks_apply = (recv_total + 255) / 256;
        apply_recv_edits_kernel<T><<<blocks_apply, 256>>>(
            d_recv_idx, recv_total, rank,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z, send_flag, ite, rank, changed_vertex_h, 
            checkFinal
            /* 这里补上你 apply 需要的参数 */
        );
        cudaCheck(cudaGetLastError(), "launch apply_recv_edits_kernel");
    }
    MPI_Barrier(comm);
    
    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
        

    // ---- cleanup ----
    if (d_recv_idx)     cudaCheck(cudaFree(d_recv_idx),     "free d_recv_idx");
    if (d_send_counts)  cudaCheck(cudaFree(d_send_counts),  "free d_send_counts");
    if (d_req_dest)     cudaCheck(cudaFree(d_req_dest),     "free d_req_dest");
    if (d_req_idx)      cudaCheck(cudaFree(d_req_idx),      "free d_req_idx");
    if (d_nreq)         cudaCheck(cudaFree(d_nreq),         "free d_nreq");
}



// Given sorted positions -> write neighbor info back in ORIGINAL (gather) order [0..M)
static __global__ void compute_neighbors_scatter_kernel(
    int M,
    const int* perm_sorted,           // [M]
    const double*   all_val,          // [M]
    const int*      all_owner,        // [M]
    const uint64_t* all_gid,          // [M]  <-- 用 gid
    // outputs in original gather order [M]
    double* out_prev_val, int* out_prev_rank, uint64_t* out_prev_gid,
    double* out_next_val, int* out_next_rank, uint64_t* out_next_gid) {

  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= M) return;

  int orig = perm_sorted[pos];

  double pv = 0.0, nv = 0.0;
  int prank = -1, nrank = -1;
  uint64_t pgid = 0, ngid = 0;

  if (pos > 0) {
    int porig = perm_sorted[pos - 1];
    pv = all_val[porig];
    prank = all_owner[porig];
    pgid = all_gid[porig];
  }
  if (pos + 1 < M) {
    int norig = perm_sorted[pos + 1];
    nv = all_val[norig];
    nrank = all_owner[norig];
    ngid = all_gid[norig];
  }

  out_prev_val[orig]  = pv;
  out_prev_rank[orig] = prank;
  out_prev_gid[orig]  = pgid;

  out_next_val[orig]  = nv;
  out_next_rank[orig] = nrank;
  out_next_gid[orig]  = ngid;
}

static __global__ void gather_values_from_idx_kernel(const double* input_data,
                                                     const int* idx, // [m]
                                                     int m,
                                                     double* out_val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) out_val[i] = input_data[idx[i]];
}
// ---- main ----
__global__ void local_to_global_kernel(
    const int* local_idx,
    uint64_t* global_idx,
    int n,
    int offset_x, int offset_y, int offset_z
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int i = local_idx[tid];

    int x = i % width;
    int y = (i / width) % height;
    int z = i / ((size_t)width * height);

    int gx = x - HALO + offset_x;
    int gy = y - HALO + offset_y;
    int gz = z - HALO + offset_z;

    uint64_t gIdx =
        (uint64_t)gx +
        (uint64_t)gy * global_width +
        (uint64_t)gz * global_width * global_height;

    global_idx[tid] = gIdx;
}



void mpi_sort_neighbors_gpuaware(
    const double* d_input_data,
    const int*    d_maximum,     // local ids [nmax]
    int nmax,
    MPI_Comm comm,
    uint64_t** d_my_idx,         // OUT: device [nmax] global gid
    double**   d_prev_val, int** d_prev_rank, uint64_t** d_prev_idx,
    double**   d_next_val, int** d_next_rank, uint64_t** d_next_idx,
    int offset_x, int offset_y, int offset_z)
{
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int threads = 256;

  // -----------------------------
  // 0) ensure dummy (optional)
  // -----------------------------
  ensure_dummy();

  // -----------------------------
  // 1) allocate outputs (device)
  //    even if nmax==0 allocate 1 element
  // -----------------------------
  size_t alloc_n = (nmax > 0) ? (size_t)nmax : 1;

  CUDA_CHECK(cudaMalloc((void**)d_my_idx, alloc_n * sizeof(uint64_t)));

  CUDA_CHECK(cudaMalloc((void**)d_prev_val,  alloc_n * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)d_prev_rank, alloc_n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)d_prev_idx,  alloc_n * sizeof(uint64_t)));

  CUDA_CHECK(cudaMalloc((void**)d_next_val,  alloc_n * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)d_next_rank, alloc_n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)d_next_idx,  alloc_n * sizeof(uint64_t)));

  // -----------------------------
  // 2) build local peak gid + peak values
  // -----------------------------
  if (nmax > 0) {
    int blocks = (nmax + threads - 1) / threads;
    local_to_global_kernel<<<blocks, threads>>>(
        d_maximum, *d_my_idx, nmax,
        offset_x, offset_y, offset_z);
    CUDA_CHECK(cudaGetLastError());
  }

  double* d_peak_val = nullptr;
  CUDA_CHECK(cudaMalloc(&d_peak_val, alloc_n * sizeof(double)));

  if (nmax > 0) {
    int blocks = (nmax + threads - 1) / threads;
    gather_values_from_idx_kernel<<<blocks, threads>>>(
        d_input_data, d_maximum, nmax, d_peak_val);
    CUDA_CHECK(cudaGetLastError());
  }

  // ensure kernels done before MPI
  CUDA_CHECK(cudaDeviceSynchronize());

  // -----------------------------
  // 3) gather nmax -> counts/displs on root (HOST arrays)
  // -----------------------------
  std::vector<int> counts, displs;
  int M = 0;

  if (rank == 0) {
    counts.assign(size, 0);
    displs.assign(size, 0);
  }

  // IMPORTANT: non-root recvbuf must be nullptr
  // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
  MPI_Barrier(comm);
  
  MPI_Gather(&nmax, 1, MPI_INT,
             (rank == 0 ? counts.data() : nullptr), 1, MPI_INT,
             0, comm);

  if (rank == 0) {
    for (int r = 1; r < size; ++r) displs[r] = displs[r - 1] + counts[r - 1];
    M = std::accumulate(counts.begin(), counts.end(), 0);

  }

  const int* cnts_use   = (rank == 0 ? counts.data() : nullptr);
  const int* displs_use = (rank == 0 ? displs.data() : nullptr);

  // -----------------------------
  // 4) root alloc gather buffers (device, size M)
  // -----------------------------
  double*   d_all_val   = nullptr;
  uint64_t* d_all_gid   = nullptr;
  int*      d_all_owner = nullptr;

  double*   d_prev_val_all = nullptr;
  int*      d_prev_rank_all = nullptr;
  uint64_t* d_prev_gid_all  = nullptr;

  double*   d_next_val_all = nullptr;
  int*      d_next_rank_all = nullptr;
  uint64_t* d_next_gid_all  = nullptr;

  if (rank == 0 && M > 0) {
    CUDA_CHECK(cudaMalloc(&d_all_val,   (size_t)M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_gid,   (size_t)M * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_all_owner, (size_t)M * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_prev_val_all,  (size_t)M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prev_rank_all, (size_t)M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_gid_all,  (size_t)M * sizeof(uint64_t)));

    CUDA_CHECK(cudaMalloc(&d_next_val_all,  (size_t)M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_next_rank_all, (size_t)M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_gid_all,  (size_t)M * sizeof(uint64_t)));
  }

  // -----------------------------
  // 5) GPU-aware MPI Gatherv (device sendbuf, device recvbuf on root)
  //    non-root recvbuf must be nullptr per MPI standard
  // -----------------------------
  void* send_val_ptr = (void*)d_peak_val;
  void* send_gid_ptr = (void*)(*d_my_idx); // IMPORTANT: *d_my_idx is device pointer

  void* recv_val_ptr = (rank == 0 ? (void*)d_all_val : nullptr);
  void* recv_gid_ptr = (rank == 0 ? (void*)d_all_gid : nullptr);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
  MPI_Barrier(comm);
  
  MPI_Gatherv(send_val_ptr, nmax, MPI_DOUBLE,
              recv_val_ptr, cnts_use, displs_use, MPI_DOUBLE, 0, comm);

  MPI_Gatherv(send_gid_ptr, nmax, MPI_UINT64_T,
              recv_gid_ptr, cnts_use, displs_use, MPI_UINT64_T, 0, comm);

  // -----------------------------
  // 6) root: fill owner + sort + compute neighbors into *_all
  // -----------------------------
  if (rank == 0 && M > 0) {
    // fill owners by segment
    for (int r = 0; r < size; ++r) {
      int c = counts[r];
      if (c == 0) continue;
      auto begin = thrust::device_pointer_cast(d_all_owner) + displs[r];
      thrust::fill(begin, begin + c, r);
    }

    int* d_perm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_perm, (size_t)M * sizeof(int)));
    thrust::sequence(thrust::device_pointer_cast(d_perm),
                     thrust::device_pointer_cast(d_perm) + M, 0);

    PeakCmp cmp{d_all_val, d_all_gid, d_all_owner};
    thrust::sort(thrust::device_pointer_cast(d_perm),
                 thrust::device_pointer_cast(d_perm) + M,
                 cmp);

    int blocks = (M + threads - 1) / threads;
    compute_neighbors_scatter_kernel<<<blocks, threads>>>(
        M, d_perm,
        d_all_val, d_all_owner, d_all_gid,
        d_prev_val_all, d_prev_rank_all, d_prev_gid_all,
        d_next_val_all, d_next_rank_all, d_next_gid_all);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_perm));
  }

  // -----------------------------
  // 7) Scatterv back to each rank (device recvbuf on all ranks)
  //    sendbuf only meaningful on root; non-root sendbuf can be nullptr
  // -----------------------------
  void* send_prev_val_ptr  = (rank == 0 ? (void*)d_prev_val_all  : nullptr);
  void* send_prev_rank_ptr = (rank == 0 ? (void*)d_prev_rank_all : nullptr);
  void* send_prev_gid_ptr  = (rank == 0 ? (void*)d_prev_gid_all  : nullptr);

  void* send_next_val_ptr  = (rank == 0 ? (void*)d_next_val_all  : nullptr);
  void* send_next_rank_ptr = (rank == 0 ? (void*)d_next_rank_all : nullptr);
  void* send_next_gid_ptr  = (rank == 0 ? (void*)d_next_gid_all  : nullptr);

  void* recv_prev_val_ptr  = (void*)(*d_prev_val);
  void* recv_prev_rank_ptr = (void*)(*d_prev_rank);
  void* recv_prev_gid_ptr  = (void*)(*d_prev_idx);

  void* recv_next_val_ptr  = (void*)(*d_next_val);
  void* recv_next_rank_ptr = (void*)(*d_next_rank);
  void* recv_next_gid_ptr  = (void*)(*d_next_idx);

  const int* scnts_use   = (rank == 0 ? counts.data() : nullptr);
  const int* sdispls_use = (rank == 0 ? displs.data() : nullptr);

  // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
  MPI_Barrier(comm);
  
  MPI_Scatterv(send_prev_val_ptr,  scnts_use, sdispls_use, MPI_DOUBLE,
               recv_prev_val_ptr,  nmax,      MPI_DOUBLE, 0, comm);
  MPI_Scatterv(send_prev_rank_ptr, scnts_use, sdispls_use, MPI_INT,
               recv_prev_rank_ptr, nmax,      MPI_INT,    0, comm);
  MPI_Scatterv(send_prev_gid_ptr,  scnts_use, sdispls_use, MPI_UINT64_T,
               recv_prev_gid_ptr,  nmax,      MPI_UINT64_T, 0, comm);

  // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
  MPI_Barrier(comm);
  
  MPI_Scatterv(send_next_val_ptr,  scnts_use, sdispls_use, MPI_DOUBLE,
               recv_next_val_ptr,  nmax,      MPI_DOUBLE, 0, comm);
  MPI_Scatterv(send_next_rank_ptr, scnts_use, sdispls_use, MPI_INT,
               recv_next_rank_ptr, nmax,      MPI_INT,    0, comm);
  MPI_Scatterv(send_next_gid_ptr,  scnts_use, sdispls_use, MPI_UINT64_T,
               recv_next_gid_ptr,  nmax,      MPI_UINT64_T, 0, comm);

  // -----------------------------
  // 8) cleanup
  // -----------------------------
  if (rank == 0) {
    if (d_all_val)   CUDA_CHECK(cudaFree(d_all_val));
    if (d_all_gid)   CUDA_CHECK(cudaFree(d_all_gid));
    if (d_all_owner) CUDA_CHECK(cudaFree(d_all_owner));

    if (d_prev_val_all)  CUDA_CHECK(cudaFree(d_prev_val_all));
    if (d_prev_rank_all) CUDA_CHECK(cudaFree(d_prev_rank_all));
    if (d_prev_gid_all)  CUDA_CHECK(cudaFree(d_prev_gid_all));

    if (d_next_val_all)  CUDA_CHECK(cudaFree(d_next_val_all));
    if (d_next_rank_all) CUDA_CHECK(cudaFree(d_next_rank_all));
    if (d_next_gid_all)  CUDA_CHECK(cudaFree(d_next_gid_all));
  }

  if (d_peak_val) CUDA_CHECK(cudaFree(d_peak_val));
}



template <typename T>
__global__ void pack_send_buffer_two_layers_kernel(
    T* __restrict__ sendbuf,
    size_t local_x, size_t local_y, size_t local_z,
    int offset_x, int offset_y, int offset_z,
    int dx, int dy, int dz, int rank, int *count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

   
    size_t send_count = get_send_count(dx, dy, dz, local_x, local_y, local_z);
    size_t total_send_count = send_count;
    if (idx >= total_send_count) return;

    size_t i = idx / 2;       // Which face index (z/y/x combined)
    bool is_first_layer = (idx % 2 == 0);  // Select layer 0 or layer 1

    size_t x = 0, y = 0, z = 0;

    if (dx != 0 && dy == 0 && dz == 0) {
        z = HALO + i / local_y;
        y = HALO + i % local_y;
        x = (dx == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_x + (is_first_layer ? 0 : 1));
        
    } else if (dy != 0 && dx == 0 && dz == 0) {
        z = HALO + i / local_x;
        x = HALO + i % local_x;
        y = (dy == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_y + (is_first_layer ? 0 : 1));

    } else if (dz != 0 && dx == 0 && dy == 0) {
        y = HALO + i / local_x;
        x = HALO + i % local_x;
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 0 : 1));
    } else if (dz == 0 && dx != 0 && dy != 0) {
        x = dx == -1 ? (HALO - (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 0 : 1));
        y = dy == -1 ? (HALO - (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 0 : 1);
        z = HALO + i;
    }

    else if (dz != 0 && dx != 0 && dy == 0) {
        x = (dx == -1) ? (HALO - (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 0 : 1));
        y = HALO + i;
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 0 : 1));
    }

    else if (dz != 0 && dx == 0 && dy != 0) {
        x = HALO + i;
        y = (dy == -1) ? (HALO - (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 0 : 1);
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 0 : 1));
    }
    
    
    size_t global_idx = (z) * width * height + (y) * width + x;
    int global_x = x - HALO + offset_x;
    int global_y = y - HALO + offset_y;
    int global_z = z - HALO + offset_z;

    if (global_x < 0 || global_x >= global_width ||
        global_y < 0 || global_y >= global_height ||
        global_z < 0 || global_z >= global_depth) return;
    
    if(decp_data[global_idx] == INVALID_VAL) return;
    
    if(delta_counter[global_idx] != 0){
        sendbuf[idx] = delta_counter[global_idx];
    }
    
}

__global__ void init_counter(){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    delta_counter[i] = 0;
}

template <typename T>
__global__ void unpack_recv_buffer_two_layers_kernel(
    const T* __restrict__ recvbuf,
    size_t local_x, size_t local_y, size_t local_z,
    int dx, int dy, int dz, int rank, uint8_t* changed_vertex_h)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t total_recv_count = get_send_count(dx, dy, dz, local_x, local_y, local_z);
    if (idx >= total_recv_count || recvbuf[idx] == 0 ) return;
    
    size_t i = idx / 2;
    bool is_first_layer = (idx % 2 == 0);

    size_t x = 0, y = 0, z = 0;

    if (dx != 0 && dy == 0 && dz == 0) {
    
        z = HALO + i / local_y;
        y = HALO + i % local_y;
        x = (dx == -1) ? (0 + (is_first_layer ? 0 : 1))                  // left ghost
                       : (local_x + (is_first_layer ? 1 : 0));   // right ghost

    } else if (dy != 0 && dx == 0 && dz == 0) {
        z = HALO + i / local_x;
        x = HALO + i % local_x;
        y = (dy == -1) ? (0 + (is_first_layer ? 0 : 1))                  // front ghost
                       : (local_y + (is_first_layer ? 1 : 0));   // back ghost

    } else if (dz != 0 && dx == 0 && dy == 0) {
        y = HALO + i / local_x;
        x = HALO + i % local_x;
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1))                  // bottom ghost
                       : (local_z + (is_first_layer ? 1 : 0));   // top ghost
    }

    else if (dz == 0 && dx != 0 && dy != 0) {
        x = dx == -1 ? (0 + (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 1 : 0));
        y = dy == -1 ? (0 + (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 1 : 0);
        z = HALO + i;
    }

    else if (dz != 0 && dx != 0 && dy == 0) {
        x = (dx == -1) ? (0 + (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 1 : 0));
        y = HALO + i;
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 1 : 0));
    }

    else if (dz != 0 && dx == 0 && dy != 0) {
        x = HALO + i;
        y = (dy == -1) ? (0 + (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 1 : 0);
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 1 : 0));
    }
    

    size_t global_idx = z * height * width + y * width + x;
    T currentDelta = delta_counter[global_idx];
    T recvDelta = recvbuf[idx];
    
    if(decp_data[global_idx] != INVALID_VAL && currentDelta < recvDelta && currentDelta >= 0 && recvDelta >= 0){
        delta_counter[global_idx] = recvDelta;
        count_f_cases+=1;
        int x = global_idx % width;
        int y = (global_idx / width) % height;
        int z = (global_idx / (width * height)) % depth;
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
        changed_vertex_h[global_idx] = 1;
        if(recvDelta < 5 && fabs(input_data[global_idx] - (decp_data[global_idx] - delta * (recvDelta - currentDelta))) < bound) {
            decp_data[global_idx] -= delta * (recvDelta - currentDelta);
        }
        else {
            delta_counter[global_idx] = 5;
            decp_data[global_idx] = input_data[global_idx] - bound;
        }
    }
}

// Kernel: 打包稀疏数据（只打包标记为 1 的点）
template <typename T>
__global__ void pack_sparse_ghost_kernel(
    const int* compacted_indices,  // 压缩后的原始索引
    int n_sparse,
    int* out_indices,              // 输出：原始位置索引
    T* out_values,                 // 输出：delta_counter 值
    size_t local_x, size_t local_y, size_t local_z,
    int offset_x, int offset_y, int offset_z,
    int dx, int dy, int dz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_sparse) return;
    
    int idx = compacted_indices[t];  // 原始的 send buffer 索引
    
    size_t i = idx / 2;
    bool is_first_layer = (idx % 2 == 0);
    
    size_t x = 0, y = 0, z = 0;
    
    if (dx != 0 && dy == 0 && dz == 0) {
        z = HALO + i / local_y;
        y = HALO + i % local_y;
        x = (dx == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_x + (is_first_layer ? 0 : 1));
    } else if (dy != 0 && dx == 0 && dz == 0) {
        z = HALO + i / local_x;
        x = HALO + i % local_x;
        y = (dy == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_y + (is_first_layer ? 0 : 1));
    } else if (dz != 0 && dx == 0 && dy == 0) {
        y = HALO + i / local_x;
        x = HALO + i % local_x;
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) 
                       : (local_z + (is_first_layer ? 0 : 1));
    } else if (dz == 0 && dx != 0 && dy != 0) {
        x = dx == -1 ? (HALO - (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 0 : 1));
        y = dy == -1 ? (HALO - (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 0 : 1);
        z = HALO + i;
    } else if (dz != 0 && dx != 0 && dy == 0) {
        x = (dx == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_x + (is_first_layer ? 0 : 1));
        y = HALO + i;
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) 
                       : (local_z + (is_first_layer ? 0 : 1));
    } else if (dz != 0 && dx == 0 && dy != 0) {
        x = HALO + i;
        y = (dy == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : local_y + (is_first_layer ? 0 : 1);
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) 
                       : (local_z + (is_first_layer ? 0 : 1));
    }
    
    size_t global_idx = z * width * height + y * width + x;
    
    out_indices[t] = idx;  // 保存原始索引，用于接收端解包
    out_values[t] = delta_counter[global_idx];
}

// Kernel: 解包稀疏数据
template <typename T>
__global__ void unpack_sparse_ghost_kernel(
    const int* recv_indices,
    const T* recv_values,
    int n_sparse,
    size_t local_x, size_t local_y, size_t local_z,
    int dx, int dy, int dz,
    uint8_t* changed_vertex)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_sparse) return;
    
    int idx = recv_indices[t];
    T recvDelta = recv_values[t];
    
    if (recvDelta == 0) return;
    
    size_t i = idx / 2;
    bool is_first_layer = (idx % 2 == 0);
    
    size_t x = 0, y = 0, z = 0;
    
    if (dx != 0 && dy == 0 && dz == 0) {
        z = HALO + i / local_y;
        y = HALO + i % local_y;
        x = (dx == -1) ? (0 + (is_first_layer ? 0 : 1))
                       : (local_x + (is_first_layer ? 1 : 0));
    } else if (dy != 0 && dx == 0 && dz == 0) {
        z = HALO + i / local_x;
        x = HALO + i % local_x;
        y = (dy == -1) ? (0 + (is_first_layer ? 0 : 1))
                       : (local_y + (is_first_layer ? 1 : 0));
    } else if (dz != 0 && dx == 0 && dy == 0) {
        y = HALO + i / local_x;
        x = HALO + i % local_x;
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1))
                       : (local_z + (is_first_layer ? 1 : 0));
    } else if (dz == 0 && dx != 0 && dy != 0) {
        x = dx == -1 ? (0 + (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 1 : 0));
        y = dy == -1 ? (0 + (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 1 : 0);
        z = HALO + i;
    } else if (dz != 0 && dx != 0 && dy == 0) {
        x = (dx == -1) ? (0 + (is_first_layer ? 0 : 1))
                       : (local_x + (is_first_layer ? 1 : 0));
        y = HALO + i;
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1)) 
                       : (local_z + (is_first_layer ? 1 : 0));
    } else if (dz != 0 && dx == 0 && dy != 0) {
        x = HALO + i;
        y = (dy == -1) ? (0 + (is_first_layer ? 0 : 1))
                       : local_y + (is_first_layer ? 1 : 0);
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1)) 
                       : (local_z + (is_first_layer ? 1 : 0));
    }
    
    size_t global_idx = z * height * width + y * width + x;
    T currentDelta = delta_counter[global_idx];
    
    if (decp_data[global_idx] != INVALID_VAL && 
        currentDelta < recvDelta && 
        currentDelta >= 0 && recvDelta >= 0) {
        
        delta_counter[global_idx] = recvDelta;
        count_f_cases += 1;
        
        int lx = global_idx % width;
        int ly = (global_idx / width) % height;
        int lz = (global_idx / (width * height)) % depth;
        
        for (int d = 0; d < maxNeighbors; d++) {
            int dirX = directions[d * 3];
            int dirY = directions[d * 3 + 1];
            int dirZ = directions[d * 3 + 2];
            int newX = lx + dirX;
            int newY = ly + dirY;
            int newZ = lz + dirZ;
            int r = newX + newY * width + newZ * (height * width);
            if (newX >= 0 && newX < width && newY >= 0 && newY < height && 
                r >= 0 && r < num_Elements && newZ < depth && newZ >= 0) {
                changed_vertex[r] = 1;
            }
        }
        changed_vertex[global_idx] = 1;
        
        if (recvDelta < 5 && fabs(input_data[global_idx] - (decp_data[global_idx] - delta * (recvDelta - currentDelta))) < bound) {
            decp_data[global_idx] -= delta * (recvDelta - currentDelta);
        } else {
            delta_counter[global_idx] = 5;
            decp_data[global_idx] = input_data[global_idx] - bound;
        }
    }
}

// 优化的稀疏版本 Ghost Layer 交换函数（批量处理所有邻居）
template <typename T>
void exchange_ghost_layers_sparse(
    size_t padded_x, size_t padded_y, size_t padded_z,
    int offset_x, int offset_y, int offset_z,
    MPI_Comm cart_comm, int dims[3], int rank,
    int send_flag, int ite,
    size_t local_x, size_t local_y, size_t local_z,
    uint8_t* changed_vertex_h, double global_start_time, double &ghost_exchange_time, double &comp_time_iter)
{
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    
    dim3 blockSize(256);
    MPI_Datatype mpi_dtype_int = MPI_INT;
    MPI_Datatype mpi_dtype_T = get_mpi_datatype<T>();
    double comm_start_time, comm_end_time, comp_start_time, comp_end_time;
    // 预先收集所有有效邻居信息
    struct NeighborInfo {
        int nbr_rank;
        int dx, dy, dz;
        int dir_tag;
        size_t send_count;
        bool do_send, do_recv;
        int n_sparse_send, n_sparse_recv;
        int* d_send_indices;
        T* d_send_values;
        int* d_recv_indices;
        T* d_recv_values;
    };
    std::vector<NeighborInfo> neighbors;
    neighbors.reserve(26);
    comm_start_time = MPI_Wtime() - global_start_time;
    
    // ========== Phase 1: 收集邻居信息并交换 flags ==========
    for (int k = 0; k < maxNeighbors_host - 2; ++k) {
        int dx = directions_host[3 * k];
        int dy = directions_host[3 * k + 1];
        int dz = directions_host[3 * k + 2];
        if (dx == 0 && dy == 0 && dz == 0) continue;
        
        int nbr_coords[3] = {coords[0] + dx, coords[1] + dy, coords[2] + dz};
        bool neighbor_exists =
            (nbr_coords[0] >= 0 && nbr_coords[0] < dims[0] &&
             nbr_coords[1] >= 0 && nbr_coords[1] < dims[1] &&
             nbr_coords[2] >= 0 && nbr_coords[2] < dims[2]);
        
        if (!neighbor_exists) continue;
        
        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);
        
        NeighborInfo info;
        info.nbr_rank = nbr_rank;
        info.dx = dx; info.dy = dy; info.dz = dz;
        info.dir_tag = std::min(rank, nbr_rank) * 1000 + std::max(rank, nbr_rank);
        info.send_count = get_send_count(dx, dy, dz, local_x, local_y, local_z);
        info.d_send_indices = nullptr;
        info.d_send_values = nullptr;
        info.d_recv_indices = nullptr;
        info.d_recv_values = nullptr;
        info.n_sparse_send = 0;
        info.n_sparse_recv = 0;
        
        // 交换 send_flag
        int my_flag = send_flag;
        int nbr_flag = 0;
        MPI_Sendrecv(&my_flag, 1, MPI_INT, nbr_rank, info.dir_tag,
                     &nbr_flag, 1, MPI_INT, nbr_rank, info.dir_tag,
                     cart_comm, MPI_STATUS_IGNORE);
        
        info.do_send = (my_flag == 1);
        info.do_recv = (nbr_flag == 1);
        
        if (info.do_send || info.do_recv) {
            neighbors.push_back(info);
        }
    }
    
    if (neighbors.empty()) return;
    comm_end_time = MPI_Wtime() - global_start_time;
    ghost_exchange_time += comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
    // ========== Phase 2: 批量计算稀疏数据（所有邻居方向并行）==========
    // 使用简化的方法：直接用原始的非稀疏打包，然后用 thrust 筛选非零值
    for (auto& nbr : neighbors) {
        if (!nbr.do_send) continue;
        
        // 分配临时发送缓冲区（完整大小）
        T* d_temp_sendbuf = nullptr;
        cudaMalloc(&d_temp_sendbuf, nbr.send_count * sizeof(T));
        cudaMemset(d_temp_sendbuf, 0, nbr.send_count * sizeof(T));
        
        // 使用原始 kernel 打包所有数据
        int* count_dev = nullptr;
        cudaMalloc(&count_dev, sizeof(int));
        cudaMemset(count_dev, 0, sizeof(int));
        
        dim3 grid((nbr.send_count + blockSize.x - 1) / blockSize.x);
        pack_send_buffer_two_layers_kernel<T><<<grid, blockSize>>>(
            d_temp_sendbuf,
            local_x, local_y, local_z, offset_x, offset_y, offset_z,
            nbr.dx, nbr.dy, nbr.dz, rank, count_dev);
        
        cudaFree(count_dev);
        
        // 统计非零元素数量（使用 thrust）
        int n_nonzero = thrust::count_if(
            thrust::device_pointer_cast(d_temp_sendbuf),
            thrust::device_pointer_cast(d_temp_sendbuf) + nbr.send_count,
            [] __device__ (T val) { return val != 0; });
        
        nbr.n_sparse_send = n_nonzero;
        
        if (n_nonzero > 0) {
            // 分配稀疏缓冲区
            cudaMalloc(&nbr.d_send_indices, n_nonzero * sizeof(int));
            cudaMalloc(&nbr.d_send_values, n_nonzero * sizeof(T));
            
            // 使用 thrust 提取非零元素的索引和值
            thrust::counting_iterator<int> idx_begin(0);
            auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
                idx_begin,
                thrust::device_pointer_cast(d_temp_sendbuf)));
            auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(
                idx_begin + nbr.send_count,
                thrust::device_pointer_cast(d_temp_sendbuf) + nbr.send_count));
            
            auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_pointer_cast(nbr.d_send_indices),
                thrust::device_pointer_cast(nbr.d_send_values)));
            
            thrust::copy_if(zip_begin, zip_end, out_zip,
                [] __device__ (const thrust::tuple<int, T>& t) {
                    return thrust::get<1>(t) != 0;
                });
        }
        
        cudaFree(d_temp_sendbuf);
    }
    cudaDeviceSynchronize();
    
    // ========== Phase 3: 批量交换稀疏数据量 ==========
    std::vector<MPI_Request> count_requests;
    count_requests.reserve(neighbors.size() * 2);
    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
    comm_start_time = MPI_Wtime() - global_start_time;
    for (auto& nbr : neighbors) {
        MPI_Request req_send, req_recv;
        int tag_count = nbr.dir_tag + 20000;
        
        MPI_Isend(&nbr.n_sparse_send, 1, MPI_INT, nbr.nbr_rank, tag_count, cart_comm, &req_send);
        MPI_Irecv(&nbr.n_sparse_recv, 1, MPI_INT, nbr.nbr_rank, tag_count, cart_comm, &req_recv);
        count_requests.push_back(req_send);
        count_requests.push_back(req_recv);
    }
    MPI_Waitall(count_requests.size(), count_requests.data(), MPI_STATUSES_IGNORE);
    
    // ========== Phase 4: 分配接收缓冲区 ==========
    for (auto& nbr : neighbors) {
        if (nbr.do_recv && nbr.n_sparse_recv > 0) {
            cudaMalloc(&nbr.d_recv_indices, nbr.n_sparse_recv * sizeof(int));
            cudaMalloc(&nbr.d_recv_values, nbr.n_sparse_recv * sizeof(T));
        }
    }
    
    // ========== Phase 5: 批量发送/接收稀疏数据 ==========
    std::vector<MPI_Request> data_requests;
    data_requests.reserve(neighbors.size() * 4);
    
    for (auto& nbr : neighbors) {
        int tag_indices = nbr.dir_tag + 30000;
        int tag_values = nbr.dir_tag + 40000;
        
        if (nbr.do_send && nbr.n_sparse_send > 0) {
            MPI_Request req;
            MPI_Isend(nbr.d_send_indices, nbr.n_sparse_send, mpi_dtype_int, nbr.nbr_rank, tag_indices, cart_comm, &req);
            data_requests.push_back(req);
            MPI_Isend(nbr.d_send_values, nbr.n_sparse_send, mpi_dtype_T, nbr.nbr_rank, tag_values, cart_comm, &req);
            data_requests.push_back(req);
        }
        if (nbr.do_recv && nbr.n_sparse_recv > 0) {
            MPI_Request req;
            MPI_Irecv(nbr.d_recv_indices, nbr.n_sparse_recv, mpi_dtype_int, nbr.nbr_rank, tag_indices, cart_comm, &req);
            data_requests.push_back(req);
            MPI_Irecv(nbr.d_recv_values, nbr.n_sparse_recv, mpi_dtype_T, nbr.nbr_rank, tag_values, cart_comm, &req);
            data_requests.push_back(req);
        }
    }
    
    if (!data_requests.empty()) {
        MPI_Waitall(data_requests.size(), data_requests.data(), MPI_STATUSES_IGNORE);
    }
    cudaDeviceSynchronize();
    comm_end_time = MPI_Wtime() - global_start_time;
    ghost_exchange_time += comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
    // ========== Phase 6: 批量解包 ==========
    for (auto& nbr : neighbors) {
        if (nbr.do_recv && nbr.n_sparse_recv > 0) {
            dim3 grid_unpack((nbr.n_sparse_recv + blockSize.x - 1) / blockSize.x);
            unpack_sparse_ghost_kernel<T><<<grid_unpack, blockSize>>>(
                nbr.d_recv_indices, nbr.d_recv_values, nbr.n_sparse_recv,
                local_x, local_y, local_z, nbr.dx, nbr.dy, nbr.dz, changed_vertex_h);
        }
    }
    cudaDeviceSynchronize();
    
    // ========== Phase 7: 清理 ==========
    for (auto& nbr : neighbors) {
        if (nbr.d_send_indices) cudaFree(nbr.d_send_indices);
        if (nbr.d_send_values) cudaFree(nbr.d_send_values);
        if (nbr.d_recv_indices) cudaFree(nbr.d_recv_indices);
        if (nbr.d_recv_values) cudaFree(nbr.d_recv_values);
    }

    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
}



template <typename T>
struct NeighborExchangeInfo {
    int nbr_rank;
    int dx, dy, dz;
    int dir_tag;
    bool do_send;
    bool do_recv;
    T* sendbuf;
    T* recvbuf;
    size_t send_count;
    size_t recv_count;
};

template <typename T>
void exchange_ghost_layers_cuda_aware(size_t padded_x, size_t padded_y, size_t padded_z,
                                      int offset_x, int offset_y, int offset_z, 
                                      MPI_Comm cart_comm, int dims[3], int rank, int send_flag, 
                                      int ite, size_t local_x, size_t local_y, size_t local_z, uint8_t *changed_vertex_h,
                                      double global_start_time, double &ghost_exchange_time, double &comp_time_iter) {
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    dim3 blockSize(256);
    
    static int call_count = 0;
    call_count++;
    double comm_start_time, comm_end_time, comp_start_time, comp_end_time;
    if (g_mpi_debug && rank == g_mpi_debug_rank) {
        printf("[DEBUG MPI] exchange_ghost_layers_cuda_aware: call #%d, ite=%d, send_flag=%d\n",
               call_count, ite, send_flag);
        fflush(stdout);
    }
    
    MPI_Datatype mpi_dtype = get_mpi_datatype<T>();
    
    // ============ Phase 1: 收集所有邻居信息，准备buffer ============
    std::vector<NeighborExchangeInfo<T>> neighbors;
    
    for (int k = 0; k < maxNeighbors_host - 2; ++k) {
        int dx = directions_host[3 * k], dy = directions_host[3 * k + 1], dz = directions_host[3 * k + 2];
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int nbr_coords[3] = {coords[0] + dx, coords[1] + dy, coords[2] + dz};
        bool neighbor_exists =
            (nbr_coords[0] >= 0 && nbr_coords[0] < dims[0] &&
             nbr_coords[1] >= 0 && nbr_coords[1] < dims[1] &&
             nbr_coords[2] >= 0 && nbr_coords[2] < dims[2]);

        if (!neighbor_exists) continue;

        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);

        // Exchange send_flag with neighbor
        int my_flag = send_flag;
        int nbr_flag = 0;
        int dir_tag = std::min(rank, nbr_rank) * 1000 + std::max(rank, nbr_rank);
        
        MPI_Sendrecv(&my_flag, 1, MPI_INT, nbr_rank, dir_tag,
                     &nbr_flag, 1, MPI_INT, nbr_rank, dir_tag,
                     cart_comm, MPI_STATUS_IGNORE);
        
        bool do_send = (my_flag == 1);
        bool do_recv = (nbr_flag == 1);
         
        if (!do_send && !do_recv) continue;
        
        NeighborExchangeInfo<T> info;
        info.nbr_rank = nbr_rank;
        info.dx = dx; info.dy = dy; info.dz = dz;
        info.dir_tag = dir_tag;
        info.do_send = do_send;
        info.do_recv = do_recv;
        info.sendbuf = nullptr;
        info.recvbuf = nullptr;
        info.send_count = get_send_count(dx, dy, dz, local_x, local_y, local_z);
        info.recv_count = 0;
        comp_start_time = MPI_Wtime() - global_start_time;
        // 准备发送buffer
        if (do_send) {
            cudaMalloc(&info.sendbuf, info.send_count * sizeof(T));
            cudaMemset(info.sendbuf, 0, info.send_count * sizeof(T));

            dim3 grid((info.send_count + blockSize.x - 1) / blockSize.x);
            pack_send_buffer_two_layers_kernel<T><<<grid, blockSize>>>(
                info.sendbuf, 
                local_x, local_y, local_z, offset_x, offset_y, offset_z,
                dx, dy, dz, rank, nullptr);
        }
        comp_end_time = MPI_Wtime() - global_start_time;
        comp_time_iter += comp_end_time - comp_start_time;
        // Exchange count with neighbor
        int send_actual_count = do_send ? (int)info.send_count : 0;
        int recv_actual_count = 0;
        int tag_count = dir_tag + 10000;
        comm_start_time = MPI_Wtime() - global_start_time;
        MPI_Sendrecv(&send_actual_count, 1, MPI_INT, nbr_rank, tag_count,
                     &recv_actual_count, 1, MPI_INT, nbr_rank, tag_count,
                     cart_comm, MPI_STATUS_IGNORE);
        
        info.recv_count = recv_actual_count;
        
        // 准备接收buffer
        if (do_recv && recv_actual_count > 0) {
            cudaMalloc(&info.recvbuf, recv_actual_count * sizeof(T));
            cudaMemset(info.recvbuf, 0, recv_actual_count * sizeof(T));
        }
        
        comm_end_time = MPI_Wtime() - global_start_time;
        ghost_exchange_time += comm_end_time - comm_start_time;
        neighbors.push_back(info);
    }
    
    // 确保所有GPU kernel完成
    cudaDeviceSynchronize();
    comm_start_time = MPI_Wtime() - global_start_time;
    // 统计需要发送和接收的邻居数量
    int num_sends = 0, num_recvs = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (neighbors[i].do_send) num_sends++;
        if (neighbors[i].do_recv) num_recvs++;
    }
    
    if (g_mpi_debug && rank == g_mpi_debug_rank) {
        printf("[DEBUG MPI] exchange_ghost_layers_cuda_aware: %zu neighbors, %d sends, %d recvs\n",
               neighbors.size(), num_sends, num_recvs);
        fflush(stdout);
    }
    
    // ============ Phase 2: 逐个 Isend + Iprobe/Recv 交错，避免OFI队列溢出 ============
    
    std::vector<MPI_Request> send_reqs;
    send_reqs.reserve(num_sends);
    std::vector<bool> recv_done(neighbors.size(), false);
    
    int recvs_completed = 0;
    size_t send_idx = 0;
    
    // 找到下一个需要发送的邻居
    auto find_next_send = [&]() -> int {
        while (send_idx < neighbors.size()) {
            if (neighbors[send_idx].do_send && neighbors[send_idx].sendbuf != nullptr) {
                return send_idx++;
            }
            send_idx++;
        }
        return -1;
    };
    
    // 主循环：交替发送和接收
    while ((int)send_reqs.size() < num_sends || recvs_completed < num_recvs) {
        
        // 发起一个 Isend
        int si = find_next_send();
        if (si >= 0) {
            NeighborExchangeInfo<T>& info = neighbors[si];
            MPI_Request req;
            MPI_Isend(info.sendbuf, (int)info.send_count, mpi_dtype, 
                      info.nbr_rank, info.dir_tag, cart_comm, &req);
            send_reqs.push_back(req);
            
            if (g_mpi_debug && rank == g_mpi_debug_rank) {
                printf("[DEBUG MPI] Isend to rank %d, tag=%d, count=%zu (%zu/%d)\n",
                       info.nbr_rank, info.dir_tag, info.send_count,
                       send_reqs.size(), num_sends);
                fflush(stdout);
            }
        }
        
        // 多轮 Iprobe/Recv，给发送队列时间排空
        for (int probe_round = 0; probe_round < 3; ++probe_round) {
            for (size_t i = 0; i < neighbors.size(); ++i) {
                if (recv_done[i]) continue;
                
                NeighborExchangeInfo<T>& info = neighbors[i];
                if (!info.do_recv || info.recvbuf == nullptr) {
                    recv_done[i] = true;
                    continue;
                }
                
                int flag = 0;
                MPI_Status status;
                MPI_Iprobe(info.nbr_rank, info.dir_tag, cart_comm, &flag, &status);
                
                if (flag) {
                    MPI_Recv(info.recvbuf, (int)info.recv_count, mpi_dtype,
                             info.nbr_rank, info.dir_tag, cart_comm, MPI_STATUS_IGNORE);
                    recv_done[i] = true;
                    recvs_completed++;
                    
                    if (g_mpi_debug && rank == g_mpi_debug_rank) {
                        printf("[DEBUG MPI] Recv from rank %d, tag=%d, count=%zu (%d/%d)\n",
                               info.nbr_rank, info.dir_tag, info.recv_count, 
                               recvs_completed, num_recvs);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    
    // 等待所有发送完成
    if (!send_reqs.empty()) {
        MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    }
    
    if (g_mpi_debug && rank == g_mpi_debug_rank) {
        printf("[DEBUG MPI] exchange_ghost_layers_cuda_aware: all communication completed\n");
        fflush(stdout);
    }
    comm_end_time = MPI_Wtime() - global_start_time;
    ghost_exchange_time += comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
    // ============ Phase 3: Unpack received buffers ============
    for (size_t i = 0; i < neighbors.size(); ++i) {
        NeighborExchangeInfo<T>& info = neighbors[i];
        if (info.recvbuf != nullptr && info.recv_count > 0) {
            dim3 gridSize_unpack((info.recv_count + blockSize.x - 1) / blockSize.x);
            unpack_recv_buffer_two_layers_kernel<T><<<gridSize_unpack, blockSize>>>(
                info.recvbuf, 
                local_x, local_y, local_z, info.dx, info.dy, info.dz, rank, changed_vertex_h);
        }
    }
    cudaDeviceSynchronize();

    // ============ Phase 4: Free buffers ============
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (neighbors[i].sendbuf) cudaFree(neighbors[i].sendbuf);
        if (neighbors[i].recvbuf) cudaFree(neighbors[i].recvbuf);
    }
    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
}




__global__ void build_neighbor_value_requests(
    // saddle1
    int n1,
    const int* prev_rank1, const uint64_t* prev_idx1,
    const int* next_rank1, const uint64_t* next_idx1,
    const uint8_t* need_prev1, const uint8_t* need_next1,

    // saddle2
    int n2,
    const int* prev_rank2, const uint64_t* prev_idx2,
    const int* next_rank2, const uint64_t* next_idx2,
    const uint8_t* need_prev2, const uint8_t* need_next2,

    // minimum
    int nmin,
    const int* prev_rankMin, const uint64_t* prev_idxMin,
    const int* next_rankMin, const uint64_t* next_idxMin,
    const uint8_t* need_prevMin, const uint8_t* need_nextMin,

    // maximum
    int nmax,
    const int* prev_rankMax, const uint64_t* prev_idxMax,
    const int* next_rankMax, const uint64_t* next_idxMax,
    const uint8_t* need_prevMax, const uint8_t* need_nextMax,

    int myrank,
    int* req_dest, uint64_t* req_idx, int* req_slot,
    int* nreq,
    int max_req,
    int* overflow_flag)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nSaddle1 + nSaddle2 + nMin + nMax;
    if (tid >= N) return;

    int set_id = 0;
    int i = tid;

    if (tid < nSaddle1) {
        set_id = 0; i = tid;
    } else if (tid < nSaddle1 + nSaddle2) {
        set_id = 1; i = tid - nSaddle1;
    } else if (tid < nSaddle1 + nSaddle2 + nMin) {
        set_id = 2; i = tid - (nSaddle1 + nSaddle2);
    } else {
        set_id = 3; i = tid - (nSaddle1 + nSaddle2 + nMin);
    }

    auto push = [&](int dest, uint64_t idx, int is_next){
        if (dest < 0) return;               // 过滤非法 dest
        // 如果你还想过滤本地（可选）：if (dest == myrank) return;

        int p = atomicAdd(nreq, 1);
        if (p >= max_req) { atomicExch(overflow_flag, 1); return; }

        req_dest[p] = dest;
        req_idx[p]  = idx;
        req_slot[p] = (i << 3) | (set_id << 1) | (is_next & 1);
    };

    if (set_id == 0) {
        if (need_prev1[i]) push(prev_rank1[i], prev_idx1[i], 0);
        if (need_next1[i]) push(next_rank1[i], next_idx1[i], 1);
    } else if (set_id == 1) {
        if (need_prev2[i]) push(prev_rank2[i], prev_idx2[i], 0);
        if (need_next2[i]) push(next_rank2[i], next_idx2[i], 1);
    } else if (set_id == 2) {
        if (need_prevMin[i]) push(prev_rankMin[i], prev_idxMin[i], 0);
        if (need_nextMin[i]) push(next_rankMin[i], next_idxMin[i], 1);
    } else { // set_id == 3
        if (need_prevMax[i]) push(prev_rankMax[i], prev_idxMax[i], 0);
        if (need_nextMax[i]) push(next_rankMax[i], next_idxMax[i], 1);
    }
    
}



// ==================== 优化版本：只读 value 的 kernel ====================
__global__ void fulfill_values_only_kernel(
    const uint64_t* their_gids, int n,
    double* reply_vals,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z, int first = 1)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    
    int lid;
    bool ok = local_id_from_gid(
        their_gids[t], global_width, global_height,
        offset_x, offset_y, offset_z,
        local_x, local_y, local_z,
        width, height, &lid);
    
    if (ok) {
        reply_vals[t] = decp_data[lid];
    } else {
        reply_vals[t] = CUDART_NAN;
    }
}

// 用预存的 slot 写回 value
__global__ void scatter_values_only_kernel(
    const int* my_slots, const double* received_vals, int n,
    int n1, int n2, int nMin, int nMax,
    double* prev_val1, double* next_val1,
    double* prev_val2, double* next_val2,
    double* prev_valMin, double* next_valMin,
    double* prev_valMax, double* next_valMax, const uint64_t* prev_idx, const uint64_t* prev_idx1,
    const uint64_t* prev_idxMin, const uint64_t* prev_idxMax, int rank)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    
    int slot = my_slots[t];
    double v = received_vals[t];
    
    int is_next = slot & 1;
    int set_id = (slot >> 1) & 3;
    int i = slot >> 3;
    if(isnan(v)) return;
    if (set_id == 0) {
        if (i >= n1) return;
        // if(rank == 0 && !is_next && i == 202841 ) printf("scattered here: %llu %.17f\n", prev_idx[i], v);
        if (is_next) next_val1[i] = v;
        else         prev_val1[i] = v;
    } else if (set_id == 1) {
        if (i >= n2) return;
        // if(rank == 0 && !is_next && i == 202841 ) printf("scattered here: %llu %.17f\n", prev_idx1[i], v);
        if (is_next) next_val2[i] = v;
        else         prev_val2[i] = v;
    } else if (set_id == 2) {
        if (i >= nMin) return;
        // if(rank == 0 && !is_next && i == 202841 ) printf("scattered here: %llu %.17f\n", prev_idxMin[i], v);
        if (is_next) next_valMin[i] = v;
        else         prev_valMin[i] = v;
    } else {
        if (i >= nMax) return;
        // if(rank == 0 && !is_next && i == 202841 ) printf("scattered here: %llu %.17f\n", prev_idxMax[i], v);

        if (is_next) next_valMax[i] = v;
        else         prev_valMax[i] = v;
    }
}

// ==================== 初始化通信计划（只调用一次）====================
void init_neighbor_comm_plan(
    NeighborCommPlan& plan,
    MPI_Comm comm, int myrank, int size,

    // saddle1
    int nSaddle1_host,
    const int* d_prev_rank1, const uint64_t* d_prev_idx1,
    const int* d_next_rank1, const uint64_t* d_next_idx1,
    const uint8_t* need_prev1, const uint8_t* need_next1,

    // saddle2
    int nSaddle2_host,
    const int* d_prev_rank2, const uint64_t* d_prev_idx2,
    const int* d_next_rank2, const uint64_t* d_next_idx2,
    const uint8_t* need_prev2, const uint8_t* need_next2,

    // minimum
    int nMin_host,
    const int* d_prev_rankMin, const uint64_t* d_prev_idxMin,
    const int* d_next_rankMin, const uint64_t* d_next_idxMin,
    const uint8_t* need_prevMin, const uint8_t* need_nextMin,

    // maximum
    int nMax_host,
    const int* d_prev_rankMax, const uint64_t* d_prev_idxMax,
    const int* d_next_rankMax, const uint64_t* d_next_idxMax,
    const uint8_t* need_prevMax, const uint8_t* need_nextMax
){
    // 清理之前的
    free_neighbor_comm_plan(plan);
    
    int max_req = 2 * (nSaddle1_host + nSaddle2_host + nMin_host + nMax_host);
    
    int *d_req_dest = nullptr, *d_req_slot = nullptr, *d_nreq = nullptr;
    uint64_t *d_req_idx = nullptr;
    
    CUDA_OK(cudaMalloc(&d_req_dest, (size_t)max_req * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_req_idx,  (size_t)max_req * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_req_slot, (size_t)max_req * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_nreq, sizeof(int)));
    CUDA_OK(cudaMemset(d_nreq, 0, sizeof(int)));
    
    int threads = 256;
    int N = nSaddle1_host + nSaddle2_host + nMin_host + nMax_host;
    int blocks = (N + threads - 1) / threads;
    
    int* d_overflow = nullptr;
    CUDA_OK(cudaMalloc(&d_overflow, sizeof(int)));
    CUDA_OK(cudaMemset(d_overflow, 0, sizeof(int)));
    
    if (max_req > 0) {
        build_neighbor_value_requests<<<blocks, threads>>>(
            nSaddle1_host,
            d_prev_rank1, d_prev_idx1,
            d_next_rank1, d_next_idx1,
            need_prev1, need_next1,  // need_prev1, need_next1 not used in init
            
            nSaddle2_host,
            d_prev_rank2, d_prev_idx2,
            d_next_rank2, d_next_idx2,
            need_prev2, need_next2,
            
            nMin_host,
            d_prev_rankMin, d_prev_idxMin,
            d_next_rankMin, d_next_idxMin,
            need_prevMin, need_nextMin,
            
            nMax_host,
            d_prev_rankMax, d_prev_idxMax,
            d_next_rankMax, d_next_idxMax,
            need_prevMax, need_nextMax,
            
            myrank,
            d_req_dest, d_req_idx, d_req_slot, d_nreq, max_req, d_overflow
        );
        
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        
        int h_over = 0;
        CUDA_OK(cudaMemcpy(&h_over, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_over) {
            printf("[rank %d] ERROR: init_neighbor_comm_plan overflow\n", myrank);
            fflush(stdout);
            MPI_Abort(comm, 1);
        }
    }
    cudaFree(d_overflow);
    
    int nreq = 0;
    CUDA_OK(cudaMemcpy(&nreq, d_nreq, sizeof(int), cudaMemcpyDeviceToHost));
    plan.total_send = nreq;
    
    // 排序
    if (nreq > 0) {
        thrust::sort_by_key(
            thrust::device_pointer_cast(d_req_dest),
            thrust::device_pointer_cast(d_req_dest) + nreq,
            thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_pointer_cast(d_req_idx),
                thrust::device_pointer_cast(d_req_slot)))
        );
    }
    
    // 计算 send_counts
    int* d_send_counts = nullptr;
    CUDA_OK(cudaMalloc(&d_send_counts, (size_t)size * sizeof(int)));
    CUDA_OK(cudaMemset(d_send_counts, 0, (size_t)size * sizeof(int)));
    
    if (nreq > 0) {
        count_dest_atomic<<<(nreq + 255) / 256, 256>>>(d_req_dest, nreq, d_send_counts);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }
    
    plan.sendcounts.resize(size);
    plan.recvcounts.resize(size);
    plan.senddispls.resize(size);
    plan.recvdispls.resize(size);
    
    CUDA_OK(cudaMemcpy(plan.sendcounts.data(), d_send_counts, (size_t)size * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_send_counts);
    
    // 使用 P2P 版本的 Alltoall（避免 OFI 队列溢出）
    p2p_alltoall_int(plan.sendcounts.data(), plan.recvcounts.data(), 1, comm);
    
    plan.senddispls[0] = 0;
    plan.recvdispls[0] = 0;
    for (int r = 1; r < size; ++r) {
        plan.senddispls[r] = plan.senddispls[r-1] + plan.sendcounts[r-1];
        plan.recvdispls[r] = plan.recvdispls[r-1] + plan.recvcounts[r-1];
    }
    
    plan.total_recv = 0;
    for (int r = 0; r < size; ++r) plan.total_recv += plan.recvcounts[r];
    
    // 保存我的 gid 和 slot（固定不变）
    if (nreq > 0) {
        CUDA_OK(cudaMalloc(&plan.d_my_gids,  (size_t)nreq * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc(&plan.d_my_slots, (size_t)nreq * sizeof(int)));
        CUDA_OK(cudaMemcpy(plan.d_my_gids,  d_req_idx,  (size_t)nreq * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(plan.d_my_slots, d_req_slot, (size_t)nreq * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    // 交换 gid 和 slot（只做一次）
    if (plan.total_recv > 0) {
        CUDA_OK(cudaMalloc(&plan.d_their_gids,  (size_t)plan.total_recv * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc(&plan.d_their_slots, (size_t)plan.total_recv * sizeof(int)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
    
    ensure_dummy();
    
    // 使用分阶段 P2P 版本（适用于大规模 64+ GPU）
    p2p_alltoallv_cuda_staged<uint64_t>(
        d_req_idx, plan.sendcounts.data(), plan.senddispls.data(),
        plan.d_their_gids, plan.recvcounts.data(), plan.recvdispls.data(),
        MPI_UINT64_T, comm);
    
    p2p_alltoallv_cuda_staged<int>(
        d_req_slot, plan.sendcounts.data(), plan.senddispls.data(),
        plan.d_their_slots, plan.recvcounts.data(), plan.recvdispls.data(),
        MPI_INT, comm);
    
    // 分配 value 缓冲区
    if (plan.total_recv > 0) {
        CUDA_OK(cudaMalloc(&plan.d_reply_vals, (size_t)plan.total_recv * sizeof(double)));
    }
    if (nreq > 0) {
        CUDA_OK(cudaMalloc(&plan.d_received_vals, (size_t)nreq * sizeof(double)));
    }
    
    // 清理临时缓冲区
    cudaFree(d_req_dest);
    cudaFree(d_req_idx);
    cudaFree(d_req_slot);
    cudaFree(d_nreq);
    
    plan.initialized = true;
}

// ==================== 快速更新 value（每次迭代调用）====================
void update_neighbor_values_fast(
    NeighborCommPlan& plan,
    MPI_Comm comm,
    int rank, 
    int nSaddle1_host, int nSaddle2_host, int nMin_host, int nMax_host,
    double* d_prev_val1, double* d_next_val1,
    double* d_prev_val2, double* d_next_val2,
    double* d_prev_valMin, double* d_next_valMin,
    double* d_prev_valMax, double* d_next_valMax,
    const uint64_t* prev_idx, const uint64_t* prev_idx1,
    const uint64_t* prev_idxMin, const uint64_t* prev_idxMax,
    int offset_x, int offset_y, int offset_z, 
    int local_x, int local_y, int local_z, int first = 1)
{
    if (!plan.initialized) {
        fprintf(stderr, "ERROR: NeighborCommPlan not initialized!\n");
        fflush(stdout);
        return;
    }
    
    ensure_dummy();
    
    // 1) 填充回复值：从 decp_data 读取，并与 decp_datac 比较
    if (plan.total_recv > 0) {
        fulfill_values_only_kernel<<<(plan.total_recv + 255) / 256, 256>>>(
            plan.d_their_gids, plan.total_recv,
            plan.d_reply_vals,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z, first);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        fflush(stdout);
    }
    
    // 2) 只交换 value（使用分阶段 P2P 版本）
    // 注意：发送和接收的方向相反
    // CUDA-aware MPI 需要确保 GPU 操作完成
    cudaDeviceSynchronize();
    
    p2p_alltoallv_cuda_staged<double>(
        plan.d_reply_vals, plan.recvcounts.data(), plan.recvdispls.data(),
        plan.d_received_vals, plan.sendcounts.data(), plan.senddispls.data(),
        MPI_DOUBLE, comm);
    // 3) 用预存的 slot 写回 prev_val/next_val
    if (plan.total_send > 0) {
        scatter_values_only_kernel<<<(plan.total_send + 255) / 256, 256>>>(
            plan.d_my_slots, plan.d_received_vals, plan.total_send,
            nSaddle1_host, nSaddle2_host, nMin_host, nMax_host,
            d_prev_val1, d_next_val1,
            d_prev_val2, d_next_val2,
            d_prev_valMin, d_next_valMin,
            d_prev_valMax, d_next_valMax, prev_idx, prev_idx1, prev_idxMin, prev_idxMax, rank);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }
}




// ==================== 压缩版本：稀疏通信，只发送改变的值 ====================
struct CompactPackage {
    int* d_indices = nullptr;      // 压缩后的索引
    double* d_values = nullptr;     // 压缩后的值
    int count = 0;                  // 压缩后的数量
    
    ~CompactPackage() {
        if (d_indices) cudaFree(d_indices);
        if (d_values) cudaFree(d_values);
    }
};

// Kernel: 标记哪些位置需要发送
__global__ void mark_changed_flags_kernel(
    const uint64_t* their_gids, int n,
    int* flags,  // output: 1 if changed, 0 otherwise
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z, int rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int lid;
    bool ok = local_id_from_gid(
        their_gids[i], global_width, global_height,
        offset_x, offset_y, offset_z,
        local_x, local_y, local_z,
        width, height, &lid);
    flags[i] = (ok && delta_counter_c[lid] != delta_counter[lid] ) ? 1 : 0;
}

// Kernel: 填充压缩后的值
__global__ void fill_compacted_values_kernel(
    const uint64_t* their_gids,
    const int* compacted_indices,
    int n_compacted,
    double* compacted_values,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z, int rank)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_compacted) return;
    
    int original_idx = compacted_indices[t];
    
    int lid;
    
    bool ok = local_id_from_gid(
        their_gids[original_idx], global_width, global_height,
        offset_x, offset_y, offset_z,
        local_x, local_y, local_z,
        width, height, &lid);
    
    if (ok) compacted_values[t] = decp_data[lid];

}

// 压缩发送缓冲区
CompactPackage compact_send_buffer(
    const uint64_t* d_their_gids,
    int total_recv,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z, int rank)
{
    CompactPackage pkg;
    
    if (total_recv == 0) return pkg;
    
    // 1) 标记哪些位置需要发送
    int* d_flags = nullptr;
    cudaMalloc(&d_flags, total_recv * sizeof(int));
    
    int threads = 256;
    int blocks = (total_recv + threads - 1) / threads;
    mark_changed_flags_kernel<<<blocks, threads>>>(
        d_their_gids, total_recv, d_flags,
        offset_x, offset_y, offset_z, local_x, local_y, local_z, rank);
    cudaDeviceSynchronize();
    
    // 2) 使用 CUB 做 stream compaction
    int* d_compacted_indices = nullptr;
    int* d_num_selected = nullptr;
    cudaMalloc(&d_compacted_indices, total_recv * sizeof(int));
    cudaMalloc(&d_num_selected, sizeof(int));
    
    // 计算临时存储大小
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        thrust::counting_iterator<int>(0),
        d_flags,
        d_compacted_indices,
        d_num_selected,
        total_recv);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 执行压缩
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        thrust::counting_iterator<int>(0),
        d_flags,
        d_compacted_indices,
        d_num_selected,
        total_recv);
    
    cudaMemcpy(&pkg.count, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 3) 填充压缩后的值
    if (pkg.count > 0) {
        pkg.d_indices = d_compacted_indices;
        cudaMalloc(&pkg.d_values, pkg.count * sizeof(double));
        
        fill_compacted_values_kernel<<<(pkg.count + 255) / 256, 256>>>(
            d_their_gids, d_compacted_indices, pkg.count,
            pkg.d_values, offset_x, offset_y, offset_z, local_x, local_y, local_z, rank);
        cudaDeviceSynchronize();
    } else {
        cudaFree(d_compacted_indices);
    }
    
    // 清理临时内存
    cudaFree(d_flags);
    cudaFree(d_num_selected);
    cudaFree(d_temp_storage);
    
    return pkg;
}



__global__ void scatter_values_sparse_kernel(
    const int* all_slots,         // 完整的 my_slots 数组
    const int* sparse_indices,    // 稀疏索引（相对索引）
    const double* received_vals,  // 接收到的值
    int n,                        // 稀疏元素数量
    int n1, int n2, int nMin, int nMax,
    double* prev_val1, double* next_val1,
    double* prev_val2, double* next_val2,
    double* prev_valMin, double* next_valMin,
    double* prev_valMax, double* next_valMax,
    const int* recvdispls,        // 每个 rank 的接收偏移 (size+1)
    const int* senddispls,        // 我发给每个 rank 的起始位置 (用于计算绝对索引)
    int num_ranks)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    
    // 找出这个元素来自哪个 rank（二分搜索或线性搜索）
    int src_rank = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (t >= recvdispls[r] && (r == num_ranks - 1 || t < recvdispls[r + 1])) {
            src_rank = r;
            break;
        }
    }
    
    // 相对索引 -> 绝对索引
    int relative_idx = sparse_indices[t];
    int absolute_idx = senddispls[src_rank] + relative_idx;
    
    int slot = all_slots[absolute_idx];
    double v = received_vals[t];
    
    int is_next = slot & 1;
    int set_id = (slot >> 1) & 3;
    int i = slot >> 3;
    if(isnan(v)) return;
    if (set_id == 0) {
        if (i >= n1) return;
        if (is_next) next_val1[i] = v;
        else         prev_val1[i] = v;
    } else if (set_id == 1) {
        if (i >= n2) return;
        if (is_next) next_val2[i] = v;
        else         prev_val2[i] = v;
    } else if (set_id == 2) {
        if (i >= nMin) return;
        if (is_next) next_valMin[i] = v;
        else         prev_valMin[i] = v;
    } else {
        if (i >= nMax) return;
        if (is_next) next_valMax[i] = v;
        else         prev_valMax[i] = v;
    }
}

// 稀疏通信版本的更新函数
void update_neighbor_values_sparse(
    NeighborCommPlan& plan,
    MPI_Comm comm,
    int rank, int size,
    int nSaddle1_host, int nSaddle2_host, int nMin_host, int nMax_host,
    double* d_prev_val1, double* d_next_val1,
    double* d_prev_val2, double* d_next_val2,
    double* d_prev_valMin, double* d_next_valMin,
    double* d_prev_valMax, double* d_next_valMax,
    const uint64_t* prev_idx, const uint64_t* prev_idx1,
    const uint64_t* prev_idxMin, const uint64_t* prev_idxMax,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z, double comm_end_time, double &total_cp_comm_time, double global_start_time, double &comp_time_iter)
{
    // printf("[rank %d] entering update_neighbor_values_sparse\n", rank);
    // fflush(stdout);
    
    if (!plan.initialized) {
        fprintf(stderr, "ERROR: NeighborCommPlan not initialized!\n");
        return;
    }
    
    // 1) 压缩发送缓冲区
    auto send_pkg = compact_send_buffer(
        plan.d_their_gids, plan.total_recv,
        offset_x, offset_y, offset_z, local_x, local_y, local_z, rank);
    
    int total_send_sparse = send_pkg.count;
    int total_recv_original = plan.total_send;  // 我期望接收的总数（可能是稀疏的）
    
    // 2) 计算每个 rank 要发送的稀疏数据量
    // send_pkg.d_indices[i] 是在 d_their_gids 中的索引
    // 根据 recvdispls 可以确定这个索引对应哪个 rank 的请求
    std::vector<int> sendcounts_sparse(size, 0);
    std::vector<int> recvcounts_sparse(size, 0);
    double comm_start_time = MPI_Wtime() - global_start_time;
    if (send_pkg.count > 0) {
        // 统计每个 rank 的发送量
        int* h_indices = new int[send_pkg.count];
        cudaMemcpy(h_indices, send_pkg.d_indices, send_pkg.count * sizeof(int), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < send_pkg.count; i++) {
            int idx = h_indices[i];  // 在 d_their_gids 中的位置
            for (int r = 0; r < size; r++) {
                int start = plan.recvdispls[r];
                int end = start + plan.recvcounts[r];
                if (idx >= start && idx < end) {
                    sendcounts_sparse[r]++;
                    break;
                }
            }
        }
        delete[] h_indices;
    }
    
    // 使用 P2P 版本的 Alltoall（避免 OFI 队列溢出）
    p2p_alltoall_int(sendcounts_sparse.data(), recvcounts_sparse.data(), 1, comm);
    comm_end_time = MPI_Wtime() - global_start_time;
    total_cp_comm_time = comm_end_time - comm_start_time;
    double comp_start_time = MPI_Wtime() - global_start_time;
    // 计算接收的总量和偏移
    std::vector<int> senddispls_sparse(size, 0);
    std::vector<int> recvdispls_sparse(size, 0);
    int total_recv_sparse = 0;
    
    for (int r = 0; r < size; r++) {
        senddispls_sparse[r] = (r == 0) ? 0 : senddispls_sparse[r-1] + sendcounts_sparse[r-1];
        recvdispls_sparse[r] = (r == 0) ? 0 : recvdispls_sparse[r-1] + recvcounts_sparse[r-1];
        total_recv_sparse += recvcounts_sparse[r];
    }
    
    // printf("rank %d [SPARSE] send=%d, recv=%d (reduction: %.2f%%)\n",
    //        rank, total_send_sparse, total_recv_sparse,
    //        plan.total_recv > 0 ? (1.0 - (double)total_send_sparse / plan.total_recv) * 100.0 : 0.0);
    
    // 3) 重新组织发送缓冲区，按目标 rank 排序
    // 这样可以直接用于 MPI_Alltoallv
    int* d_send_indices_sorted = nullptr;
    double* d_send_values_sorted = nullptr;
    
    if (total_send_sparse > 0) {
        cudaMalloc(&d_send_indices_sorted, total_send_sparse * sizeof(int));
        cudaMalloc(&d_send_values_sorted, total_send_sparse * sizeof(double));
        
        // 复制并按 rank 排序（简化：在 CPU 上排序）
        if (send_pkg.count > 0) {
            int* h_indices = new int[send_pkg.count];
            double* h_values = new double[send_pkg.count];
            cudaMemcpy(h_indices, send_pkg.d_indices, send_pkg.count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_values, send_pkg.d_values, send_pkg.count * sizeof(double), cudaMemcpyDeviceToHost);
            
            // 按目标 rank 重新排列
            int* h_sorted_indices = new int[send_pkg.count];
            double* h_sorted_values = new double[send_pkg.count];
            int write_pos = 0;
            
            for (int r = 0; r < size; r++) {
                int start = plan.recvdispls[r];
                int end = start + plan.recvcounts[r];
                
                for (int i = 0; i < send_pkg.count; i++) {
                    if (h_indices[i] >= start && h_indices[i] < end) {
                        // 关键：将索引从"我的 d_their_gids 中的位置"
                        // 转换为"对方 d_my_gids 中的位置"
                        // 即：相对于该 rank 的起始位置的偏移
                        int relative_idx = h_indices[i] - start;
                        h_sorted_indices[write_pos] = relative_idx;
                        h_sorted_values[write_pos] = h_values[i];
                        write_pos++;
                    }
                }
            }
            
            cudaMemcpy(d_send_indices_sorted, h_sorted_indices, send_pkg.count * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_send_values_sorted, h_sorted_values, send_pkg.count * sizeof(double), cudaMemcpyHostToDevice);
            
            delete[] h_indices;
            delete[] h_values;
            delete[] h_sorted_indices;
            delete[] h_sorted_values;
        }
    }
    
    // 4) 分配接收缓冲区
    int* d_recv_indices = nullptr;
    double* d_recv_values = nullptr;
    
    if (total_recv_sparse > 0) {
        cudaMalloc(&d_recv_indices, total_recv_sparse * sizeof(int));
        cudaMalloc(&d_recv_values, total_recv_sparse * sizeof(double));
    }
    double comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
    
    // 5) 发送索引 - 使用分阶段 P2P 版本（适用于大规模 64+ GPU）
    // cuda version
    cudaDeviceSynchronize();
    comm_start_time = MPI_Wtime() - global_start_time;
    p2p_alltoallv_cuda_staged<int>(
        d_send_indices_sorted, sendcounts_sparse.data(), senddispls_sparse.data(),
        d_recv_indices, recvcounts_sparse.data(), recvdispls_sparse.data(),
        MPI_INT, comm);
    
    // 6) 发送值 - 使用分阶段 P2P 版本
    p2p_alltoallv_cuda_staged<double>(
        d_send_values_sorted, sendcounts_sparse.data(), senddispls_sparse.data(),
        d_recv_values, recvcounts_sparse.data(), recvdispls_sparse.data(),
        MPI_DOUBLE, comm);
    ensure_dummy();
    // cuda version

    // 5) 发送索引 - sparse host-staged P2P (non-CUDA-aware safe)
    // hostbuf version
    // cudaDeviceSynchronize();
    // comm_start_time = MPI_Wtime() - global_start_time;

    // p2p_alltoallv_host_staged_sparse<int>(
    //     /*d_sendbuf*/ d_send_indices_sorted,          // 可能为 nullptr（当 total_send_sparse==0）
    //     /*sendcounts*/ sendcounts_sparse.data(),
    //     /*sdispls*/    senddispls_sparse.data(),

    //     /*d_recvbuf*/  d_recv_indices,                // total_recv_sparse>0 时一定非空
    //     /*recvcounts*/ recvcounts_sparse.data(),
    //     /*rdispls*/    recvdispls_sparse.data(),

    //     /*dtype*/      MPI_INT,
    //     /*comm*/       comm,
    //     /*tag_base*/   7200000,
    //     /*use_pinned*/ true
    // );

    // // 6) 发送值 - sparse host-staged P2P (non-CUDA-aware safe)
    // p2p_alltoallv_host_staged_sparse<double>(
    //     /*d_sendbuf*/ d_send_values_sorted,
    //     /*sendcounts*/ sendcounts_sparse.data(),
    //     /*sdispls*/    senddispls_sparse.data(),

    //     /*d_recvbuf*/  d_recv_values,
    //     /*recvcounts*/ recvcounts_sparse.data(),
    //     /*rdispls*/    recvdispls_sparse.data(),

    //     /*dtype*/      MPI_DOUBLE,
    //     /*comm*/       comm,
    //     /*tag_base*/   7300000,   // 跟 indices 用不同 base，避免 tag 冲突更稳
    //     /*use_pinned*/ true
    // );

    comm_end_time = MPI_Wtime() - global_start_time;
    total_cp_comm_time = comm_end_time - comm_start_time;
    comp_start_time = MPI_Wtime() - global_start_time;
    // hostbuf version
    // CUDA-aware MPI: 使用 dummy buffer 处理空的情况
    
    // 7) 将接收到的稀疏数据写回（全 GPU，无需拷贝到 host）
    if (total_recv_sparse > 0) {
        // 将 displs 拷贝到 device（小数组，开销很小）
        int* d_recvdispls_sparse = nullptr;
        int* d_senddispls = nullptr;
        cudaMalloc(&d_recvdispls_sparse, (size + 1) * sizeof(int));
        cudaMalloc(&d_senddispls, size * sizeof(int));
        
        // 添加 size 作为最后一个元素，方便 kernel 做边界检查
        std::vector<int> recvdispls_with_end(size + 1);
        for (int r = 0; r < size; r++) recvdispls_with_end[r] = recvdispls_sparse[r];
        recvdispls_with_end[size] = total_recv_sparse;
        
        cudaMemcpy(d_recvdispls_sparse, recvdispls_with_end.data(), (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_senddispls, plan.senddispls.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        
        // 直接在 GPU 上做索引转换和 scatter
        scatter_values_sparse_kernel<<<(total_recv_sparse + 255) / 256, 256>>>(
            plan.d_my_slots, d_recv_indices, d_recv_values, total_recv_sparse,
            nSaddle1_host, nSaddle2_host, nMin_host, nMax_host,
            d_prev_val1, d_next_val1,
            d_prev_val2, d_next_val2,
            d_prev_valMin, d_next_valMin,
            d_prev_valMax, d_next_valMax,
            d_recvdispls_sparse, d_senddispls, size);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        
        cudaFree(d_recvdispls_sparse);
        cudaFree(d_senddispls);
    }
    
    // 清理
    if (d_send_indices_sorted) cudaFree(d_send_indices_sorted);
    if (d_send_values_sorted) cudaFree(d_send_values_sorted);
    if (d_recv_indices) cudaFree(d_recv_indices);
    if (d_recv_values) cudaFree(d_recv_values);
    comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;
}


void update_neighbor_values_sparse_alltoall(
    NeighborCommPlan& plan,
    MPI_Comm comm,
    int rank, int size,
    int nSaddle1_host, int nSaddle2_host, int nMin_host, int nMax_host,
    double* d_prev_val1, double* d_next_val1,
    double* d_prev_val2, double* d_next_val2,
    double* d_prev_valMin, double* d_next_valMin,
    double* d_prev_valMax, double* d_next_valMax,
    const uint64_t* prev_idx, const uint64_t* prev_idx1,
    const uint64_t* prev_idxMin, const uint64_t* prev_idxMax,
    int offset_x, int offset_y, int offset_z,
    int local_x, int local_y, int local_z, double comm_end_time, double &total_cp_comm_time, double global_start_time, double &comp_time_iter)
{
    // printf("[rank %d] entering update_neighbor_values_sparse\n", rank);
    // fflush(stdout);
    
    if (!plan.initialized) {
        fprintf(stderr, "ERROR: NeighborCommPlan not initialized!\n");
        return;
    }
    
    // 1) 压缩发送缓冲区
    auto send_pkg = compact_send_buffer(
        plan.d_their_gids, plan.total_recv,
        offset_x, offset_y, offset_z, local_x, local_y, local_z, rank);
    
    int total_send_sparse = send_pkg.count;
    int total_recv_original = plan.total_send;  // 我期望接收的总数（可能是稀疏的）
    
    // 2) 计算每个 rank 要发送的稀疏数据量
    // send_pkg.d_indices[i] 是在 d_their_gids 中的索引
    // 根据 recvdispls 可以确定这个索引对应哪个 rank 的请求
    std::vector<int> sendcounts_sparse(size, 0);
    std::vector<int> recvcounts_sparse(size, 0);
    double comm_start_time = MPI_Wtime() - global_start_time;
    
    if (send_pkg.count > 0) {
        // 统计每个 rank 的发送量
        int* h_indices = new int[send_pkg.count];
        cudaMemcpy(h_indices, send_pkg.d_indices, send_pkg.count * sizeof(int), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < send_pkg.count; i++) {
            int idx = h_indices[i];  // 在 d_their_gids 中的位置
            for (int r = 0; r < size; r++) {
                int start = plan.recvdispls[r];
                int end = start + plan.recvcounts[r];
                if (idx >= start && idx < end) {
                    sendcounts_sparse[r]++;
                    break;
                }
            }
        }
        delete[] h_indices;
    }
    
    // Barrier 确保所有进程同步，避免 OFI 发送队列溢出
    MPI_Barrier(comm);
    
    // 交换压缩后的数量信息
    MPI_Alltoall(sendcounts_sparse.data(), 1, MPI_INT,
                 recvcounts_sparse.data(), 1, MPI_INT, comm);
    
    // 计算接收的总量和偏移
    std::vector<int> senddispls_sparse(size, 0);
    std::vector<int> recvdispls_sparse(size, 0);
    int total_recv_sparse = 0;
    
    for (int r = 0; r < size; r++) {
        senddispls_sparse[r] = (r == 0) ? 0 : senddispls_sparse[r-1] + sendcounts_sparse[r-1];
        recvdispls_sparse[r] = (r == 0) ? 0 : recvdispls_sparse[r-1] + recvcounts_sparse[r-1];
        total_recv_sparse += recvcounts_sparse[r];
    }
    
    // printf("rank %d [SPARSE] send=%d, recv=%d (reduction: %.2f%%)\n",
    //        rank, total_send_sparse, total_recv_sparse,
    //        plan.total_recv > 0 ? (1.0 - (double)total_send_sparse / plan.total_recv) * 100.0 : 0.0);
    
    // 3) 重新组织发送缓冲区，按目标 rank 排序
    // 这样可以直接用于 MPI_Alltoallv
    int* d_send_indices_sorted = nullptr;
    double* d_send_values_sorted = nullptr;
    
    if (total_send_sparse > 0) {
        cudaMalloc(&d_send_indices_sorted, total_send_sparse * sizeof(int));
        cudaMalloc(&d_send_values_sorted, total_send_sparse * sizeof(double));
        
        // 复制并按 rank 排序（简化：在 CPU 上排序）
        if (send_pkg.count > 0) {
            int* h_indices = new int[send_pkg.count];
            double* h_values = new double[send_pkg.count];
            cudaMemcpy(h_indices, send_pkg.d_indices, send_pkg.count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_values, send_pkg.d_values, send_pkg.count * sizeof(double), cudaMemcpyDeviceToHost);
            
            // 按目标 rank 重新排列
            int* h_sorted_indices = new int[send_pkg.count];
            double* h_sorted_values = new double[send_pkg.count];
            int write_pos = 0;
            
            for (int r = 0; r < size; r++) {
                int start = plan.recvdispls[r];
                int end = start + plan.recvcounts[r];
                
                for (int i = 0; i < send_pkg.count; i++) {
                    if (h_indices[i] >= start && h_indices[i] < end) {
                        // 关键：将索引从"我的 d_their_gids 中的位置"
                        // 转换为"对方 d_my_gids 中的位置"
                        // 即：相对于该 rank 的起始位置的偏移
                        int relative_idx = h_indices[i] - start;
                        h_sorted_indices[write_pos] = relative_idx;
                        h_sorted_values[write_pos] = h_values[i];
                        write_pos++;
                    }
                }
            }
            
            cudaMemcpy(d_send_indices_sorted, h_sorted_indices, send_pkg.count * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_send_values_sorted, h_sorted_values, send_pkg.count * sizeof(double), cudaMemcpyHostToDevice);
            
            delete[] h_indices;
            delete[] h_values;
            delete[] h_sorted_indices;
            delete[] h_sorted_values;
        }
    }
    
    // 4) 分配接收缓冲区
    int* d_recv_indices = nullptr;
    double* d_recv_values = nullptr;
    
    if (total_recv_sparse > 0) {
        cudaMalloc(&d_recv_indices, total_recv_sparse * sizeof(int));
        cudaMalloc(&d_recv_values, total_recv_sparse * sizeof(double));
    }
    
    // 5) 发送索引 - CUDA-aware MPI，使用 dummy buffer 处理空的情况
    ensure_dummy();
    int* send_indices = (total_send_sparse > 0) ? d_send_indices_sorted : d_dummy_i;
    int* recv_indices = (total_recv_sparse > 0) ? d_recv_indices : d_dummy_i;
    
    cudaDeviceSynchronize();
    MPI_Barrier(comm);  // 避免 OFI 队列溢出
    MPI_Alltoallv(send_indices, sendcounts_sparse.data(), senddispls_sparse.data(), MPI_INT,
                  recv_indices, recvcounts_sparse.data(), recvdispls_sparse.data(), MPI_INT, comm);
    
    // 6) 发送值 - CUDA-aware MPI
    double* send_values = (total_send_sparse > 0) ? d_send_values_sorted : d_dummy_d;
    double* recv_values = (total_recv_sparse > 0) ? d_recv_values : d_dummy_d;
    
    cudaDeviceSynchronize();  // 确保上一个 Alltoallv 接收完成

    MPI_Alltoallv(send_values, sendcounts_sparse.data(), senddispls_sparse.data(), MPI_DOUBLE,
                  recv_values, recvcounts_sparse.data(), recvdispls_sparse.data(), MPI_DOUBLE, comm);


            
    comm_end_time = MPI_Wtime() - global_start_time;
    total_cp_comm_time = comm_end_time - comm_start_time;
    double comp_start_time = MPI_Wtime() - global_start_time;

    // 7) 将接收到的稀疏数据写回（全 GPU，无需拷贝到 host）
    if (total_recv_sparse > 0) {
        // 将 displs 拷贝到 device（小数组，开销很小）
        int* d_recvdispls_sparse = nullptr;
        int* d_senddispls = nullptr;
        cudaMalloc(&d_recvdispls_sparse, (size + 1) * sizeof(int));
        cudaMalloc(&d_senddispls, size * sizeof(int));
        
        // 添加 size 作为最后一个元素，方便 kernel 做边界检查
        std::vector<int> recvdispls_with_end(size + 1);
        for (int r = 0; r < size; r++) recvdispls_with_end[r] = recvdispls_sparse[r];
        recvdispls_with_end[size] = total_recv_sparse;
        
        cudaMemcpy(d_recvdispls_sparse, recvdispls_with_end.data(), (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_senddispls, plan.senddispls.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        
        // 直接在 GPU 上做索引转换和 scatter
        scatter_values_sparse_kernel<<<(total_recv_sparse + 255) / 256, 256>>>(
            plan.d_my_slots, d_recv_indices, d_recv_values, total_recv_sparse,
            nSaddle1_host, nSaddle2_host, nMin_host, nMax_host,
            d_prev_val1, d_next_val1,
            d_prev_val2, d_next_val2,
            d_prev_valMin, d_next_valMin,
            d_prev_valMax, d_next_valMax,
            d_recvdispls_sparse, d_senddispls, size);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        
        cudaFree(d_recvdispls_sparse);
        cudaFree(d_senddispls);
    }

    
    double comp_end_time = MPI_Wtime() - global_start_time;
    comp_time_iter += comp_end_time - comp_start_time;

    // if(rank == 0) std::cout<<rank<<" after cp ordering : "<<host_count_f_cases<<std::endl;

    // 清理
    if (d_send_indices_sorted) cudaFree(d_send_indices_sorted);
    if (d_send_values_sorted) cudaFree(d_send_values_sorted);
    if (d_recv_indices) cudaFree(d_recv_indices);
    if (d_recv_values) cudaFree(d_recv_values);
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


std::vector<uint8_t> zstd_compress(const void* src, size_t src_sz, int level);
inline std::vector<uint8_t> zstd_compress(const uint8_t* data, size_t n, int level = 5) {
    size_t bound = ZSTD_compressBound(n);
    std::vector<uint8_t> out(bound);
    size_t csz = ZSTD_compress(out.data(), bound, data, n, level);
    if (ZSTD_isError(csz)) throw std::runtime_error(ZSTD_getErrorName(csz));
    out.resize(csz);
    return out;
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

void cost(std::string filename, double* decp_data, double* decp_data_copy, double* input_data, std::string compressor_id, std::vector<uint8_t> delta_counter){
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

    
    std::string indexfilename = (exactz_scratch() + "/datasets/data1")+filename+".bin";
    std::string editsfilename = (exactz_scratch() + "/datasets/data_edits")+filename+".bin";
    std::string compressedindex = (exactz_scratch() + "/datasets/data1")+filename+".bin.zst";
    std::string compressededits = (exactz_scratch() + "/datasets/data_edits")+filename+".bin.zst";
    
    // Shockwave, 64, 64, 512
    std::vector<uint8_t> comp_deltas = compress_symbols_0_N(delta_counter, /*N=*/threshold_host, /*level=*/5);
    double ratio = double(cnt)/(num_Elements_host);
    // std::cout<<cnt<<","<<ratio<<std::endl;

    std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(indexs.data()), indexs.size());
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

    // double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    // double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);

    // std::ofstream outFile3("./stat_result/result_"+filename+"_"+compressor_id+"_detailed_additional_time.txt", std::ios::app);
    std::ofstream outFile3("../stat_result/result_"+filename+"_"+compressor_id+"_q.txt", std::ios::app);
    
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return; // 返回错误码
    }

    outFile3 << std::to_string(host_bound)<<":" << std::endl;
    outFile3 << std::setprecision(17)<< "absolute_error: "<<host_bound << std::endl;
    outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
    outFile3 << "threshold: "<< threshold_host << std::endl;
    outFile3 <<std::setprecision(17)<< "CR: "<<double(original_dataSize)/compressed_dataSize << std::endl;
    outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
    outFile3 << std::setprecision(17)<<"BR: "<< 64/(double(original_indexSize)/compressed_dataSize) << std::endl;
    // outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
    // outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;
    

    
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
    outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
    outFile3 << "\n" << std::endl;

    
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;
    return;
}


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}




std::string extractFilename(const std::string& path) {
    
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

    size_t dotPos = filename.find_last_of('.');
    std::string name = (dotPos == std::string::npos) ? filename : filename.substr(0, dotPos);

    return name;
}

__global__ void warmup_kernel(double* data, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0 + 1.0;
    }
}

template <typename T>
void compressLocalDataZFP(const std::string file_path, std::string cpfilename, const std::string filename,
    const T *input_data_host, T *&decp_data_host, size_t width_host, size_t height_host, size_t depth_host,
    double tolerance, std::string decpfilename, int rank, int processData = 0) 
{
    size_t data_size = width_host * height_host * depth_host;
    std::cout << "sub datasize: " << data_size << std::endl;

    // --- ZFP setup ---
    zfp_type type = (std::is_same<T, double>::value) ? zfp_type_double : zfp_type_float;
    zfp_field* field = zfp_field_3d(const_cast<T*>(input_data_host), type, width_host, height_host, depth_host);

    zfp_stream* zfp = zfp_stream_open(nullptr);
    zfp_stream_set_accuracy(zfp, tolerance);  // 绝对误差

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    if (!buffer) {
        std::cerr << "Failed to allocate buffer" << std::endl;
        return;
    }
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // --- Compression ---
    auto start = std::chrono::high_resolution_clock::now();
    size_t zfpsize = zfp_compress(zfp, field);
    auto end = std::chrono::high_resolution_clock::now();

    if (zfpsize == 0) {
        std::cerr << "Compression failed!" << std::endl;
        return;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double compression_time = double(duration.count()) / 1000.0;
    std::cout << "Compression time: " << compression_time << " s, compressed size = " << zfpsize << " bytes" << std::endl;

    // Save compressed data if needed
    if (processData != 0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);
        if (!outFile1) {
            std::cerr << "Failed to open file for writing compressed data." << std::endl;
        } else {
            outFile1.write(reinterpret_cast<const char*>(buffer), zfpsize);
            outFile1.close();
            std::cout << "Compressed data saved to " << cpfilename << std::endl;
        }
    }

    // --- Decompression ---
    decp_data_host = new T[data_size];
    zfp_field* field_decomp = zfp_field_3d(decp_data_host, type, width_host, height_host, depth_host);

    stream_rewind(stream);
    start = std::chrono::high_resolution_clock::now();
    if (!zfp_decompress(zfp, field_decomp)) {
        std::cerr << "Decompression failed!" << std::endl;
        return;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Decompression time: " << double(duration.count()) / 1000.0 << " s" << std::endl;

    // Save decompressed data if needed
    if (processData != 0) {
        std::ofstream outFile(decpfilename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Cannot open output file." << std::endl;
        } else {
            outFile.write(reinterpret_cast<const char*>(decp_data_host), (size_t)data_size * sizeof(T));
            outFile.close();
            std::cout << "Decompressed data saved to " << decpfilename << std::endl;
        }
    }

    // --- Compression Ratio ---
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / double(zfpsize);
    std::cout << "Data read, compressed, and decompressed successfully: CR = " << cr << std::endl;

    // --- Cleanup ---
    zfp_field_free(field);
    zfp_field_free(field_decomp);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
}

template <typename T>
void compressLocalData(const std::string file_path, std::string cpfilename, const std::string filename,  
    const T *input_data_host, T *&decp_data_host, size_t width_host, size_t height_host, size_t depth_host, 
    T &bound, std::string decpfilename, int rank, int processData = 0) {

    auto start = std::chrono::high_resolution_clock::now();
    SZ3::Config conf(static_cast<int>(depth_host), static_cast<int>(height_host),
    static_cast<int>(width_host));
    
    
    // conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = bound; 

    size_t data_size = width_host * height_host * depth_host;
    std::cout<<"sub datasize: "<< data_size <<std::endl;
    
    char *compressedData = SZ_compress(conf, input_data_host, cmpSize);

    std::cout<<"compression over: "<< cmpSize<<std::endl;

    decp_data_host = new T[data_size];
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    compression_time = double(duration.count())/1000;
    
    start = std::chrono::high_resolution_clock::now();
    SZ_decompress(conf, compressedData, cmpSize, decp_data_host);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<< "decompression time: "<< duration.count() <<std::endl;
    if(processData!=0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);

        if (!outFile1) {
            std::cerr << "Failed to open file for writing compressed data." << std::endl;
        } else {
            outFile1.write(compressedData, cmpSize);
            outFile1.close();
            std::cout << "Compressed data saved to compressed_output.sz" << std::endl;
        }
    }

    delete[] compressedData;


    std::ofstream outFile(decpfilename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open output file." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(decp_data_host), (size_t)data_size * sizeof(T));
    outFile.close();
    std::cout << "Decompressed data saved to "<< decpfilename << std::endl;
    
    
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / cmpSize;
    
    std::cout << "Data read, compressed, and decompressed successfully: "<<cr << std::endl;
}

template <typename T>
void read_subblock(const char* filename,
                   size_t offset_x, size_t offset_y, size_t offset_z,
                   size_t local_x, size_t local_y, size_t local_z,
                   size_t global_x, size_t global_y, size_t global_z,
                   T* local_data) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("fopen failed");
        exit(1);
    }
    
    size_t padded_x = local_x + 2 * HALO;
    size_t padded_y = local_y + 2 * HALO;

    for (size_t z = 0; z < local_z; ++z) {
        for (size_t y = 0; y < local_y; ++y) {
            size_t global_index = (offset_z + z) * global_y * global_x +
                                  (offset_y + y) * global_x +
                                  offset_x;
            long long byte_offset = (size_t)global_index * sizeof(T);

            if (fseek(f, byte_offset, SEEK_SET) != 0) {
                perror("fseek failed");
                exit(1);
            }

            size_t dst_index = (z + HALO) * padded_y * padded_x +
                               (y + HALO) * padded_x + HALO;

            size_t read_count = fread(&local_data[dst_index], sizeof(T), local_x, f);
            if (read_count != local_x) {
                fprintf(stderr, "fread failed at z=%zu y=%zu: read %zu\n", z, y, read_count);
                exit(1);
            }
        }
    }

    fclose(f);
}

template <typename T>
void exchange_ghost_layers(T* data, int padded_x, int padded_y, int padded_z,
                           int local_x, int local_y, int local_z,
                           MPI_Comm cart_comm, int dims[3], int rank) {
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    MPI_Request requests[200];  // large enough
    int req_idx = 0;
    std::vector<RecvBuffer> recv_buffers;
    MPI_Datatype dt = get_mpi_datatype<T>();
    for (int k = 0; k < maxNeighbors_host - 2; ++k) {
        int dx = directions_host[3*k];
        int dy = directions_host[3*k + 1];
        int dz = directions_host[3*k + 2];
        if (dx == 0 && dy == 0 && dz == 0) continue; // skip self

        int nbr_coords[3] = {coords[0] + dx, coords[1] + dy, coords[2] + dz};
        bool neighbor_exists = 
            (nbr_coords[0] >= 0 && nbr_coords[0] < dims[0] &&
             nbr_coords[1] >= 0 && nbr_coords[1] < dims[1] &&
             nbr_coords[2] >= 0 && nbr_coords[2] < dims[2]);

        // (1) Fill halo with INVALID_VAL if neighbor missing
        if (true) {
            if (dx != 0) {
                size_t x_face = (dx == -1) ? 0 : padded_x - 1;
                // #pragma omp parallel for collapse(2)
                for (size_t z = 0; z < padded_z; z++) {
                    for (size_t y = 0; y < padded_y; y++) {
                        data[z * padded_y * padded_x + y * padded_x + x_face] = INVALID_VAL;
                    }
                }
            }

            if (dy != 0) {
                size_t y_face = (dy == -1) ? 0 : padded_y - 1;
                // #pragma omp parallel for collapse(2)
                for (size_t z = 0; z < padded_z; z++) {
                    for (size_t x = 0; x < padded_x; x++) {
                        data[z * padded_y * padded_x + y_face * padded_x + x] = INVALID_VAL;
                    }
                }
            }

            if (dz != 0) {
                
                size_t z_face = (dz == -1) ? 0 : padded_z - 1;
                // #pragma omp parallel for collapse(2)
                for (size_t y = 0; y < padded_y; y++) {
                    for (size_t x = 0; x < padded_x; x++) {
                        data[z_face * padded_y * padded_x + y * padded_x + x] = INVALID_VAL;
                    }
                }
            }
            
            
            
        }
        if(!neighbor_exists) continue;

        // (2) Otherwise real exchange
        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);

        int dir_tag = std::min(rank, nbr_rank) * 1000 + std::max(rank, nbr_rank);
        int send_count = 0;
        T* send_ptr = nullptr;

        // Determine send buffer
        if (dx != 0 && dy == 0 && dz == 0) {
            // x-face
            send_count = local_y * local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int x_send = (dx == -1) ? HALO : HALO + local_x - 1;
            int idx = 0;
            for (int z = 0; z < local_z; z++)
            for (int y = 0; y < local_y; y++)
                send_ptr[idx++] = data[(z + HALO) * padded_y * padded_x + (y + HALO) * padded_x + x_send];
        }
        else if (dx == 0 && dy != 0 && dz == 0) {
            // y-face
            send_count = local_x * local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int y_send = (dy == -1) ? HALO : HALO + local_y - 1;
            int idx = 0;
            for (int z = 0; z < local_z; z++){
                for (int x = 0; x < local_x; x++){
                    send_ptr[idx++] = data[(z + HALO) * padded_y * padded_x + y_send * padded_x + (x + HALO)];
                }
            }
            
        }
        else if (dx == 0 && dy == 0 && dz != 0) {
            // z-face
            send_count = local_x * local_y;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int z_send = (dz == -1) ? HALO : HALO + local_z - 1;
          
            for (int y = 0; y < local_y; y++){
                for (int x = 0; x < local_x; x++){
                    send_ptr[x + y * local_x] = data[z_send * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO)];
                }
            }
            
        }
        else if (dx == 1 && dy == -1 && dz == 0) {
            
            
            send_count = local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));

            for (int z = HALO; z < HALO + local_z; z++){
                send_ptr[z-1] = data[z * padded_y * padded_x + (HALO) * padded_x + (HALO + local_x - 1)];
            }
        }
        else if (dx == -1 && dy == 1 && dz == 0) {
            
            send_count = local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int z = HALO; z < HALO + local_z; z++){
                send_ptr[idx++] = data[z * padded_y * padded_x + (HALO + local_y - 1) * padded_x + (HALO)];
            }
        }

        else if (dx == 0 && dy == -1 && dz == 1) {
            
            send_count = local_x;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int x = HALO; x < HALO + local_x; ++x)
                send_ptr[idx++] = data[(HALO + local_z - 1) * padded_y * padded_x + (HALO) * padded_x + x];
        }

        else if (dx == 0 && dy == 1 && dz == -1) {
            
            send_count = local_x;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int x = HALO; x < HALO + local_x; ++x)
                send_ptr[idx++] = data[HALO * padded_y * padded_x + (HALO + local_y - 1) * padded_x + x];
        }

        else if (dx == 1 && dy == 0 && dz == -1) {
            
            send_count = local_y;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int y = HALO; y < HALO + local_y; ++y)
                send_ptr[idx++] = data[HALO * padded_y * padded_x + y * padded_x + (HALO + local_x - 1)];
        }

        else if (dx == -1 && dy == 0 && dz == 1) {
            
            send_count = local_y;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int y = HALO; y < HALO + local_y; y++){
                send_ptr[idx++] = data[(HALO + local_z - 1) * padded_y * padded_x + y * padded_x + (HALO)];
            }
        }
        else continue;
        // Allocate recv buffer
        T* recv_ptr = (T*) malloc(send_count * sizeof(T));
        

        // Post send/recv
        
        MPI_Isend(send_ptr, send_count, dt, nbr_rank, dir_tag, cart_comm, &requests[req_idx++]);
        MPI_Irecv(recv_ptr, send_count, dt, nbr_rank, dir_tag, cart_comm, &requests[req_idx++]);
        recv_buffers.push_back({recv_ptr, dx, dy, dz, dir_tag});
        
    }

    MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

        // (3) Scatter received data into halo
    for (auto& rb : recv_buffers) {
        int idx = 0;
        
        if (rb.dx != 0 && rb.dy == 0 && rb.dz == 0) {
            // x-face
            int x_recv = (rb.dx == -1) ? 0 : padded_x - 1;
            for (int z = 0; z < local_z; z++){
                for (int y = 0; y < local_y; y++){
                    
                    data[(z + HALO) * padded_y * padded_x + (y + HALO) * padded_x + x_recv] = rb.buffer[y + z * local_y];
                }  
            }
            
        }
        else if (rb.dx == 0 && rb.dy != 0 && rb.dz == 0) {
            // y-face
            int y_recv = (rb.dy == -1) ? 0 : padded_y - 1;
            for (int z = 0; z < local_z; z++){
                for (int x = 0; x < local_x; x++){
                    
                    data[(z + HALO) * padded_y * padded_x + y_recv * padded_x + (x + HALO)] = rb.buffer[x + z * local_x];
                    
                }
            }

                
        }
        else if (rb.dx == 0 && rb.dy == 0 && rb.dz != 0) {
            // z-face
            int z_recv = (rb.dz == -1) ? 0 : padded_z - 1;
            for (int y = 0; y < local_y; y++){
                for (int x = 0; x < local_x; x++){
                    
                    data[z_recv * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO)] = rb.buffer[x + y * local_x];
                    
                }
            }
        }
        else if (rb.dx == 1 && rb.dy == -1 && rb.dz == 0) {
            // (x+, y-) corner
            int x_recv = padded_x - 1;
            int y_recv = 0;
            for (int z = 0; z < local_z; z++){
                data[(z + HALO) * padded_y * padded_x + y_recv * padded_x + x_recv] = rb.buffer[idx++];
            }
                
        }
        else if (rb.dx == -1 && rb.dy == 1 && rb.dz == 0) {
            // (x-, y+) corner
            int x_recv = 0;
            int y_recv = padded_y - 1;
            for (int z = 0; z < local_z; z++){
                data[(z + HALO) * padded_y * padded_x + y_recv * padded_x + x_recv] = rb.buffer[z];
            }
                
        }
        else if (rb.dx == 0 && rb.dy == -1 && rb.dz == 1) {
            // (y-, z+) corner
            int y_recv = 0;
            int z_recv = padded_z - 1;
            for (int x = 0; x < local_x; ++x){
                
                data[z_recv * padded_y * padded_x + y_recv * padded_x + (x + HALO)] = rb.buffer[x];
            }
                
        }
        else if (rb.dx == 0 && rb.dy == 1 && rb.dz == -1) {
            // (y+, z-) corner
            int y_recv = padded_y - 1;
            int z_recv = 0;
            for (int x = 0; x < local_x; ++x){
                
                data[z_recv * padded_y * padded_x + y_recv * padded_x + (x + HALO)] = rb.buffer[x];
            }
                
        }
        else if (rb.dx == 1 && rb.dy == 0 && rb.dz == -1) {
            // (x+, z-) corner
            int x_recv = padded_x - 1;
            int z_recv = 0;
            for (int y = 0; y < local_y; ++y){
                
                data[z_recv * padded_y * padded_x + (y + HALO) * padded_x + x_recv] = rb.buffer[y];
            }
        }
        else if (rb.dx == -1 && rb.dy == 0 && rb.dz == 1) {
            // (x-, z+) corner
            int x_recv = 0;
            int z_recv = padded_z - 1;
            for (int y = 0; y < local_y; ++y){
                
                data[z_recv * padded_y * padded_x + (y + HALO) * padded_x + x_recv] = rb.buffer[y];
            }
                
        }

        free(rb.buffer);
    }

}



__global__ void build_need_masks(
    int nSaddle,
    const int* my_idx,                 // d_my_idx
    const double* prev_val0,           // 初始 prev 值（来自排序那轮的 scatter）
    const double* next_val0,           // 初始 next 值
    const int* prev_rank,
    const int* next_rank,
    const uint64_t* prev_idx,
    const uint64_t* next_idx,
    int myrank,
    double two_er,
    uint8_t* need_prev,
    uint8_t* need_next, int* need_count, int offset_x, int offset_y, int offset_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nSaddle) return;

    double myv = input_data[my_idx[i]];

    uint64_t sgid = lid_to_gid(my_idx[i], offset_x, offset_y, offset_z);
    // 只考虑跨 rank 的边：本地边永远不需要 MPI
    int pr = prev_rank[i];
    int nr = next_rank[i];
    double d = fabs(prev_val0[i] - myv);
    // printf("%.17f %.17f %.17f %.17f %d\n", myv, prev_val0[i], d, two_er, i);
    if(pr != -1 && (d < two_er) ){
        need_prev[i] = 1;
        atomicAdd(need_count, 1);
    }
    else{
        need_prev[i] = 0;
    }
    
    d = fabs(next_val0[i] - myv);
    if(nr != -1 && (d < two_er) ){
        need_next[i] = 1;
        atomicAdd(need_count, 1);
    }
    else{
        need_next[i] = 0;
    }

}


__global__ void preserveCPOrder(const int *cpHost, int nCP, int *send_flag, uint8_t* changed_vertex_h, int checkFinal = 0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(nCP < 2) return;
    if(i>=nCP) return;
    int id = cpHost[i];
    double value = decp_data[id];
    if(i == 0){
        double nextValue = decp_data[cpHost[1]];
        if(isless_shared(cpHost[1], id, nextValue, value)){
            count_f_cases += 1;
            if (checkFinal == 0 &&atomicCAS(&edits[id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, id, send_flag, changed_vertex_h);
            }
        }
    }
    else if(i == nCP - 1){
        int prev_id = cpHost[i - 1];
        double prevValue = decp_data[prev_id];
        if(isless_shared(id, prev_id,  value, prevValue)){
            count_f_cases += 1;
            if (checkFinal == 0 && atomicCAS(&edits[prev_id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, prev_id, send_flag, changed_vertex_h);
            }
        }
    }
    else{
        int prev_id = cpHost[i - 1];
        int next_id = cpHost[i + 1];
        double nextValue = decp_data[next_id];
        double prevValue = decp_data[prev_id];
        if(isless_shared(next_id, id, nextValue, value)){
            count_f_cases += 1;
            if (checkFinal == 0 && atomicCAS(&edits[id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, id, send_flag, changed_vertex_h);
            }
        }

        if(isless_shared(id, prev_id,  value, prevValue)){
            count_f_cases += 1;
            if (checkFinal == 0 && atomicCAS(&edits[prev_id], 0, 1) == 0){         
                applyDeltaBuffer_local<double>(delta_counter, decp_data, delta, input_data, bound, q, prev_id, send_flag, changed_vertex_h);
            }
        }

    }
}


void c_loops(int rank, int size, MPI_Comm comm, int dims[3], size_t padded_x, 
            size_t padded_y, size_t padded_z, int offset_x, int offset_y, 
            int offset_z, dim3 gridSize, dim3 blockSize, int host_count_f_max, int host_count_f_min, int host_count_f_saddle, 
            int host_count_f_dir, int host_count_f_order_saddle, double delta, uint8_t *delta_counter_copy_host, int data_size,
            double* d_prev_val,  double* d_next_val, 
            const uint64_t* d_prev_idx, const int* d_prev_rank, 
            const int* d_next_rank, const uint64_t* d_next_idx, const uint8_t *d_need_prev, const uint8_t *d_need_next,
            double* d_prev_val1,  double* d_next_val1, const uint64_t* d_prev_idx1, const int* d_prev_rank1, 
            const int* d_next_rank1, const uint64_t* d_next_idx1, const uint8_t *d_need_prev1, const uint8_t *d_need_next1, 

            double* d_prev_valMin, double* d_next_valMin,
            const uint64_t* d_prev_idxMin, const int* d_prev_rankMin,
            const int* d_next_rankMin, const uint64_t* d_next_idxMin,
            const uint8_t* d_need_prevMin, const uint8_t* d_need_nextMin,

            // maximum neighbor info
            double* d_prev_valMax, double* d_next_valMax,
            const uint64_t* d_prev_idxMax, const int* d_prev_rankMax,
            const int* d_next_rankMax, const uint64_t* d_next_idxMax,
            const uint8_t* d_need_prevMax, const uint8_t* d_need_nextMax,
            int PX, int PY, int PZ, 
            double *decp_data_host, uint8_t *delta_counter_host,
            size_t local_x, size_t local_y, size_t local_z, int*saddles1_host, int *saddles2_host, int *minimum_host, int* maximum_host, uint8_t *changed_vertex_h){
    
    
    int K = maxNeighbors_host + 1;
    float c_loopt = 0.0;
    
    // Timing statistics for weak scaling analysis
    std::vector<double> time_cp_comm;        // CP communication (update_neighbor + send_prev_violation)
    std::vector<double> time_ghost_exchange;  // Ghost layer exchange
    std::vector<double> time_computation;     // Computation (kernels)
    int total_iterations = 0;                 // Total number of iterations (global_max_ite)
    double iter_start_time, iter_end_time;
    double comm_start_time, comm_end_time;
    double ghost_start_time, ghost_end_time;
    double comp_start_time, comp_end_time;
    
    // Gantt chart data: stores (iteration, event_type, start_time, end_time)
    std::vector<std::tuple<int, std::string, double, double>> gantt_events;

    dim3 blockDim1(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridDim1((padded_x + TILE_SIZE - 1) / TILE_SIZE,
                 (padded_y + TILE_SIZE - 1) / TILE_SIZE,
                 (padded_z + TILE_SIZE - 1) / TILE_SIZE);

    int send_flag;
    auto start = std::chrono::high_resolution_clock::now();
    init_edits<<<gridSize, blockSize>>>(changed_vertex_h);
    cudaDeviceSynchronize();
    int* d_send_flag;
    cudaMalloc(&d_send_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_send_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);  // ✅ 

    init_counter<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();

    uint8_t *changed_vertex_copy;
    cudaMalloc(&changed_vertex_copy, (size_t)data_size * sizeof(uint8_t));
    cudaMemset(changed_vertex_copy, 0,  (size_t)data_size * sizeof(uint8_t));

    cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(progress_f_cases, &initialValue, sizeof(int));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int host_count_f_cases = 1; 
    
    int ite = 0;
    dim3 gridSize_1saddle((nSaddle1_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_2saddle((nSaddle2_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_max((nMax_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_min((nMin_host + blockSize.x - 1) / blockSize.x);
    int global_count_f_cases = 1;

    // ==================== 初始化通信计划（只做一次）====================
    if(rank == 0) {
        printf("Rank %d: Initializing neighbor communication plan...\n", rank);
        printf("Rank %d: nSaddle1=%d, nSaddle2=%d, nMin=%d, nMax=%d\n", 
            rank, nSaddle1_host, nSaddle2_host, nMin_host, nMax_host);
        fflush(stdout);
    }
    
    NeighborCommPlan neighbor_plan;
    
    // Add error checking before init
    if (d_prev_rank1 == nullptr || d_prev_rank == nullptr || 
        d_prev_rankMin == nullptr || d_prev_rankMax == nullptr) {
        fprintf(stderr, "Rank %d: ERROR - NULL device pointers before init_neighbor_comm_plan\n", rank);
        MPI_Abort(comm, 1);
    }
    

    init_neighbor_comm_plan(
        neighbor_plan, comm, rank, size,
        nSaddle1_host, d_prev_rank1, d_prev_idx1, d_next_rank1, d_next_idx1, d_need_prev1, d_need_next1,
        nSaddle2_host, d_prev_rank, d_prev_idx, d_next_rank, d_next_idx, d_need_prev, d_need_next,
        nMin_host, d_prev_rankMin, d_prev_idxMin, d_next_rankMin, d_next_idxMin, d_need_prevMin, d_need_nextMin,
        nMax_host, d_prev_rankMax, d_prev_idxMax, d_next_rankMax, d_next_idxMax, d_need_prevMax, d_need_nextMax
    );

    
    // Verify neighbor_plan is initialized
    if (!neighbor_plan.initialized) {
        fprintf(stderr, "Rank %d: ERROR - neighbor_plan not initialized!\n", rank);
        MPI_Abort(comm, 1);
    }
    
    // 首次更新 value（使用快速版本）
    update_neighbor_values_fast(
        neighbor_plan, comm, rank, 
        nSaddle1_host, nSaddle2_host, nMin_host, nMax_host,
        d_prev_val1, d_next_val1,
        d_prev_val, d_next_val,
        d_prev_valMin, d_next_valMin,
        d_prev_valMax, d_next_valMax,  
        d_prev_idx, d_prev_idx1, d_prev_idxMin, d_prev_idxMax,
        offset_x, offset_y, offset_z,
        local_x, local_y, local_z, 0
    );

    if(rank == 0) std::cout<<"checked! at rank "<<rank <<std::endl;
    MPI_Barrier(comm);
    if (rank == 0) start_time = MPI_Wtime();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: before global c_loops: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    cudaMemcpy(delta_counter_copy_host, delta_counter_host, (size_t)data_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    MPI_Barrier(comm);
    double global_start_time = MPI_Wtime();
    double total_cp_comm_time, comp_time_iter;
    float elapsedTime;
    cudaEvent_t startt, stop;
    cudaEventCreate(&startt);
    cudaEventCreate(&stop);
    // cudaEventCreate(&start);
    while(global_count_f_cases > 0){
        double ghost_exchange_time = 0;
        std::vector<float> temp;
        // if(rank == 0) std::cout<<"global: "<< rank<<": "<<host_count_f_cases<<std::endl;
        int gantt_iter = ite;

#ifdef DETAILED_TIMING
        // 详细计时模式：记录每个阶段的时间
        MPI_Barrier(comm);
        // if(rank == 0){
        comp_start_time = MPI_Wtime() - global_start_time;
        // }
#endif

        int cnt = 0 ;
        while(host_count_f_cases > 0){
            // if(rank == 0) std::cout<<"local: "<< rank<<": "<<host_count_f_cases<<std::endl;
            cnt++;
            
            // 合并：init_edits 和 count_f_cases 重置不需要中间同步
            cudaMemcpy(changed_vertex_copy, changed_vertex_h, (size_t)data_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            init_edits<<<gridSize, blockSize>>>(changed_vertex_h);
            cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(progress_f_cases, &initialValue, sizeof(int));
            
            // classifyVertex 需要等前面完成
            cudaDeviceSynchronize();
            cudaEventRecord(startt, 0);

            classifyVertex_CUDA<<<gridSize, blockSize>>>(rank, d_send_flag, ite, changed_vertex_h, changed_vertex_copy, 1, 0);
            cudaDeviceSynchronize(); 
            cudaError_t err = cudaDeviceSynchronize(); 
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d: classifyvertex failed: %s\n", 
                        rank, cudaGetErrorString(err));
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, startt, stop);
            // cudaMemcpyFromSymbol(&host_count_f_cases, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
            // if(rank == 0) std::cout<<"local vertex: "<< rank<<": "<<host_count_f_cases<<std::endl;
            // if(rank == 0) std::cout<<"vertex classification at rank "<<rank<<"at ite "<< ite<<": "<<elapsedTime/1000<<std::endl;
            preserveCPOrder<<<gridSize_2saddle, blockSize>>>(saddles2_host, nSaddle2_host, d_send_flag, changed_vertex_h);
            preserveCPOrder<<<gridSize_1saddle, blockSize>>>(saddles1_host, nSaddle1_host, d_send_flag, changed_vertex_h);
            preserveCPOrder<<<gridSize_max, blockSize>>>(maximum_host, nMax_host, d_send_flag, changed_vertex_h);
            preserveCPOrder<<<gridSize_min, blockSize>>>(minimum_host, nMin_host, d_send_flag, changed_vertex_h);
            
            // 只在需要读取结果时同步
            err = cudaDeviceSynchronize(); 
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d: cudaGetDeviceCount failed: %s\n", 
                        rank, cudaGetErrorString(err));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            int host_violation_count = 0;
            int host_progress_count = 0;
            cudaMemcpyFromSymbol(&host_violation_count, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_progress_count, progress_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
            if (host_violation_count > 0 && host_progress_count == 0) {
                int stage_vertex = 0;
                int stage_saddle2 = 0;
                int stage_saddle1 = 0;
                int stage_max = 0;
                int stage_min = 0;

                cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
                classifyVertex_CUDA<<<gridSize, blockSize>>>(
                    rank, d_send_flag, ite, changed_vertex_h, changed_vertex_copy,
                    1, 0, 0, 1, 1);
                cudaDeviceSynchronize();
                cudaMemcpyFromSymbol(&stage_vertex, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);

                cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
                preserveCPOrder<<<gridSize_2saddle, blockSize>>>(
                    saddles2_host, nSaddle2_host, d_send_flag, changed_vertex_h, 1);
                cudaDeviceSynchronize();
                cudaMemcpyFromSymbol(&stage_saddle2, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);

                cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
                preserveCPOrder<<<gridSize_1saddle, blockSize>>>(
                    saddles1_host, nSaddle1_host, d_send_flag, changed_vertex_h, 1);
                cudaDeviceSynchronize();
                cudaMemcpyFromSymbol(&stage_saddle1, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);

                cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
                preserveCPOrder<<<gridSize_max, blockSize>>>(
                    maximum_host, nMax_host, d_send_flag, changed_vertex_h, 1);
                cudaDeviceSynchronize();
                cudaMemcpyFromSymbol(&stage_max, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);

                cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
                preserveCPOrder<<<gridSize_min, blockSize>>>(
                    minimum_host, nMin_host, d_send_flag, changed_vertex_h, 1);
                cudaDeviceSynchronize();
                cudaMemcpyFromSymbol(&stage_min, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);

                fprintf(stderr,
                        "[UNRESOLVED] rank=%d local_iter=%d violations=%d progress=0 "
                        "vertex=%d saddle2=%d saddle1=%d max=%d min=%d\n",
                        rank, cnt, host_violation_count,
                        stage_vertex, stage_saddle2, stage_saddle1, stage_max, stage_min);
                fflush(stderr);
                MPI_Abort(comm, 2);
            }
            host_count_f_cases = host_progress_count;
            ite++;
        }
        init_edits<<<gridSize, blockSize>>>(changed_vertex_h, 1);
        cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(progress_f_cases, &initialValue, sizeof(int));
        cudaDeviceSynchronize();
#ifdef DETAILED_TIMING
        MPI_Barrier(comm);
        // if(rank == 0){
        comp_end_time = MPI_Wtime() - global_start_time;
        comp_time_iter = comp_end_time - comp_start_time;
        // std::cout<<"at rank "<<rank<<" local comp at ite "<< ite <<" is : " <<comp_time_iter<<std::endl;
        comm_start_time = MPI_Wtime() - global_start_time;
        // }
        gantt_events.push_back(std::make_tuple(gantt_iter, "Computation", comp_start_time, comp_end_time));
#endif
        
        // if(rank == 0) std::cout<<rank<<" after local ordering : "<<host_count_f_cases<<std::endl;
        int global_max_ite = 0;
        MPI_Allreduce(&ite, &global_max_ite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        total_iterations = global_max_ite;  // Save for CSV output

        // ==================== 使用稀疏压缩版本更新 value（优化网络传输）====================

        update_neighbor_values_sparse_alltoall(
            neighbor_plan, comm, rank, size,
            nSaddle1_host, nSaddle2_host, nMin_host, nMax_host,
            d_prev_val1, d_next_val1,
            d_prev_val, d_next_val,
            d_prev_valMin, d_next_valMin,
            d_prev_valMax, d_next_valMax,
            d_prev_idx, d_prev_idx1, d_prev_idxMin, d_prev_idxMax,
            offset_x, offset_y, offset_z,
            local_x, local_y, local_z, comm_end_time, total_cp_comm_time, global_start_time, comp_time_iter
        );
        MPI_Barrier(comm);
        // if(rank == 0) std::cout<<rank<<" after first cp ordering : "<<host_count_f_cases<<std::endl;
        send_prev_violation_ids_and_apply_all2all_cuda<double>(
            comm, rank, size,
            nSaddle1_host,
            blockSize,
            saddles1_host,
            d_prev_val1, d_next_val1, 
            d_prev_rank1, d_next_rank1, 
            d_prev_idx1, d_next_idx1, 
            d_need_prev1, d_need_next1,  // ADD masks
            
            nSaddle2_host,
            saddles2_host,
            d_prev_val, d_next_val, 
            d_prev_rank, d_next_rank, 
            d_prev_idx, d_next_idx, 
            d_need_prev, d_need_next,  // ADD masks
            
            nMin_host,
            minimum_host,
            d_prev_valMin, d_next_valMin,
            d_prev_rankMin, d_next_rankMin,
            d_prev_idxMin, d_next_idxMin,
            d_need_prevMin, d_need_nextMin,  // ADD masks
            
            // maximum
            nMax_host,
            maximum_host,
            d_prev_valMax, d_next_valMax,
            d_prev_rankMax, d_next_rankMax,
            d_prev_idxMax, d_next_idxMax,
            d_need_prevMax, d_need_nextMax,  // ADD masks
            
            
            /*debug_print=*/false, 
            offset_x, offset_y,offset_z,
            local_x, local_y, local_z, ite, d_send_flag, changed_vertex_h, global_start_time, comm_end_time, total_cp_comm_time, comp_time_iter
            // 这里再加你 apply kernel 需要的参数
        );
        // MPI_Barrier(comm);
        
        MPI_Barrier(comm);
#ifdef DETAILED_TIMING
        if(rank == 0){
            // comm_end_time = MPI_Wtime() - global_start_time;
            // total_cp_comm_time = comm_end_time - comm_start_time;
            comm_start_time = MPI_Wtime() - global_start_time;
            // std::cout<<global_start_time<<std::endl;
        }
        gantt_events.push_back(std::make_tuple(gantt_iter, "CP_Comm", comm_start_time, comm_end_time));
        cudaMemcpy(&send_flag, d_send_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if(rank == 0) ghost_start_time = MPI_Wtime() - global_start_time;
        // if(rank == 0) std::cout<<rank<<" after cp ordering : "<<host_count_f_cases<<std::endl;
#endif
        // 使用原版本（稀疏版本开销太大）
        exchange_ghost_layers_sparse<uint8_t>(padded_x, padded_y, padded_z, offset_x, offset_y, offset_z, comm, dims, 
            rank, send_flag, ite, local_x, local_y, local_z, changed_vertex_h,
            global_start_time, ghost_exchange_time, comp_time_iter);
        double max_ghost, max_cp, max_comp;

        
        MPI_Allreduce(&ghost_exchange_time, &max_ghost, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&total_cp_comm_time, &max_cp, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&comp_time_iter, &max_comp, 1, MPI_DOUBLE, MPI_MAX, comm);
        #ifdef DETAILED_TIMING
        MPI_Barrier(comm);
        if(rank == 0){
            // std::cout<<global_start_time<<std::endl;
            ghost_end_time = MPI_Wtime() - global_start_time;
            
            // double ghost_exchange_time = ghost_end_time - ghost_start_time;
            // time_cp_comm.push_back(total_cp_comm_time);
            // time_ghost_exchange.push_back(ghost_exchange_time);
            // time_computation.push_back(comp_time_iter);
            time_cp_comm.push_back(max_cp);
            time_ghost_exchange.push_back(max_ghost);
            time_computation.push_back(max_comp);
        }
        
        gantt_events.push_back(std::make_tuple(gantt_iter, "Ghost_Exchange", ghost_start_time, ghost_end_time));
        #endif
        
        // 合并多个 memcpy 操作，只在最后同步一次
        int host_violation_count = 0;
        int host_progress_count = 0;
        cudaMemcpyFromSymbol(&host_violation_count, count_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_progress_count, progress_f_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        int global_violation_count = 0;
        int global_progress_count = 0;
        MPI_Allreduce(&host_violation_count, &global_violation_count, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&host_progress_count, &global_progress_count, 1, MPI_INT, MPI_SUM, comm);

        if (global_violation_count > 0 && global_progress_count == 0) {
            fprintf(stderr,
                    "[UNRESOLVED] rank=%d global_phase local_violations=%d "
                    "global_violations=%d global_progress=0\n",
                    rank, host_violation_count, global_violation_count);
            fflush(stderr);
            MPI_Abort(comm, 3);
        }
        host_count_f_cases = host_progress_count;
        cudaMemcpyToSymbol(count_f_cases, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(progress_f_cases, &initialValue, sizeof(int));
        err = cudaMemcpy(d_send_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        // if(rank == 0) std::cout<<rank<<" after ghost ordering : "<<host_count_f_cases<<std::endl;
        global_count_f_cases = global_progress_count;
        ite ++;
    }
    
    // === END-TO-END TIMING (always enabled) ===
    MPI_Barrier(comm);
    double global_end_time = MPI_Wtime();
    double end_to_end_time = global_end_time - global_start_time;
    
    // 获取所有 rank 中的最大时间
    double max_end_to_end_time;
    MPI_Reduce(&end_to_end_time, &max_end_to_end_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    if (rank == 0) {
        printf("\n=== End-to-End Timing ===\n");
        printf("Total iterations: %d\n", total_iterations);
        printf("End-to-end time:  %.6f s\n", max_end_to_end_time);
        printf("=========================\n\n");
    }
    
#ifdef DETAILED_TIMING
    // === Print detailed summary statistics ===
    if (rank == 0 && time_cp_comm.size() > 0) {
        double total_cp_comm = 0.0, total_ghost = 0.0, total_comp = 0.0;
        for (size_t i = 0; i < time_cp_comm.size(); i++) {
            total_cp_comm += time_cp_comm[i];
            total_ghost += time_ghost_exchange[i];
            total_comp += time_computation[i];
        }
        
        printf("\n=== Detailed Timing Summary (Rank 0) ===\n");
        printf("Total iterations: %zu\n", time_cp_comm.size());
        printf("CP Communication:    %.6f s (%.2f%%)\n", total_cp_comm, 
               total_cp_comm / (total_cp_comm + total_ghost + total_comp) * 100.0);
        printf("Ghost Exchange:      %.6f s (%.2f%%)\n", total_ghost,
               total_ghost / (total_cp_comm + total_ghost + total_comp) * 100.0);
        printf("Computation:         %.6f s (%.2f%%)\n", total_comp,
               total_comp / (total_cp_comm + total_ghost + total_comp) * 100.0);
        printf("Total time:          %.6f s\n", total_cp_comm + total_ghost + total_comp);
        printf("========================================\n\n");
        
        // Write timing data to file for plotting
        // Make filename unique per run using RUN_TAG env var, SLURM_JOB_ID, or timestamp
        std::string run_tag = "";
        const char* run_tag_env = std::getenv("RUN_TAG");
        if (run_tag_env && std::strlen(run_tag_env) > 0) {
            run_tag = std::string("_") + run_tag_env;
        } else if (const char* slurm = std::getenv("SLURM_JOB_ID")) {
            if (std::strlen(slurm) > 0) run_tag = std::string("_job") + slurm;
        } else {
            char timestr[32];
            time_t t = time(NULL);
            struct tm tm = *localtime(&t);
            snprintf(timestr, sizeof(timestr), "_%04d%02d%02d_%02d%02d%02d", tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
            run_tag = std::string(timestr);
        }
        std::string timingfilename = (exactz_scratch() + "/")+result_prefix+"/pExtreMz_all_order1_timing_breakdown_" + filename + "_" + std::to_string(size) + "rank_async" + run_tag + ".csv";
        FILE* fp = fopen(timingfilename.c_str(), "w");
        if (fp) {
            // Write metadata header with total iterations
            fprintf(fp, "# Total_Iterations=%d\n", total_iterations);
            fprintf(fp, "Iteration,CP_Communication,Ghost_Exchange,Computation\n");
            for (size_t i = 0; i < time_cp_comm.size(); i++) {
                fprintf(fp, "%zu,%.6f,%.6f,%.6f\n", i+1, 
                        time_cp_comm[i], time_ghost_exchange[i], time_computation[i]);
            }
            fclose(fp);
            printf("Timing data written to  %s (Total iterations: %d)\n", timingfilename.c_str(), total_iterations);
        }
    }
    
    // Write Gantt chart data to file - EVERY rank writes its own file
    if (gantt_events.size() > 0 && size == 16) {
        // Build run_tag for unique filenames (use RUN_TAG, SLURM_JOB_ID, or timestamp)
        std::string run_tag = "";
        const char* run_tag_env = std::getenv("RUN_TAG");
        if (run_tag_env && std::strlen(run_tag_env) > 0) {
            run_tag = std::string("_") + run_tag_env;
        } else if (const char* slurm = std::getenv("SLURM_JOB_ID")) {
            if (std::strlen(slurm) > 0) run_tag = std::string("_job") + slurm;
        } else {
            char timestr[32];
            time_t t = time(NULL);
            struct tm tm = *localtime(&t);
            snprintf(timestr, sizeof(timestr), "_%04d%02d%02d_%02d%02d%02d", tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
            run_tag = std::string(timestr);
        }
        char gantt_filename[200];
        std::string path =
            (exactz_scratch() + "/") +
            result_prefix +
            "/pExtreMz_all_order1_" +
            filename +
            "_gantt_data_rank" +
            std::to_string(rank) +
            "_" +
            std::to_string(size) +
            "_async" +
            run_tag +
            ".csv";

        sprintf(gantt_filename, "%s", path.c_str());
        FILE* gantt_fp = fopen(gantt_filename, "w");
        if (gantt_fp) {
            fprintf(gantt_fp, "# Rank=%d\n", rank);
            fprintf(gantt_fp, "Iteration,Event_Type,Start_Time,End_Time,Rank\n");
            for (const auto& event : gantt_events) {
                int iter = std::get<0>(event);
                std::string event_type = std::get<1>(event);
                double start_time = std::get<2>(event);
                double end_time = std::get<3>(event);
                fprintf(gantt_fp, "%d,%s,%.6f,%.6f,%d\n", 
                        iter, event_type.c_str(), start_time, end_time, rank);
            }
            fclose(gantt_fp);
            // printf("Rank %d: Gantt chart data written to %s (%zu events)\n", 
            //        rank, gantt_filename, gantt_events.size());
        }
    }
#endif  // DETAILED_TIMING

    // ==================== 清理通信计划 ====================
    free_neighbor_comm_plan(neighbor_plan);

    if(rank == 0) printf("iteration is: %d\n", ite);
    
}

template <typename T>
void compressLocalData_coreOnly(const std::string file_path,
    std::string cpfilename, const std::string filename,
    const T *input_data_host, T *&decp_data_host,
    size_t width_host, size_t height_host, size_t depth_host, // full size with ghost
                                            // ghost thickness
    T &bound, std::string decpfilename, int rank, int processData = 0)
{
    // === 1. 提取 core region ===
    size_t core_x = width_host  - 2 * HALO;
    size_t core_y = height_host - 2 * HALO;
    size_t core_z = depth_host  - 2 * HALO;
    size_t core_size = core_x * core_y * core_z;

    std::cout << "Full block = " << width_host << "×" << height_host << "×" << depth_host
              << ", core = " << core_x << "×" << core_y << "×" << core_z
              << ", core size = " << core_size << std::endl;

    T* core_buffer = new T[core_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            const T* src = input_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            T* dst = core_buffer + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }

    // === 2. SZ3 配置 ===
    auto start = std::chrono::high_resolution_clock::now();
    SZ3::Config conf(static_cast<int>(core_z), static_cast<int>(core_y), static_cast<int>(core_x));
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = bound;

    size_t cmpSize = 0;
    char *compressedData = SZ_compress(conf, core_buffer, cmpSize);

    auto end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Compression time: " << compression_time
              << " s, cmpSize = " << cmpSize << " bytes" << std::endl;

    if (processData != 0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);
        outFile1.write(compressedData, cmpSize);
        outFile1.close();
    }

    // === 3. 解压缩 core ===
    T* core_decomp = new T[core_size];
    start = std::chrono::high_resolution_clock::now();
    SZ_decompress(conf, compressedData, cmpSize, core_decomp);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Decompression time: "
              << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    // === 4. 把 core_decomp 写回 decp_data_host ===
    size_t full_size = width_host * height_host * depth_host;
    if (!decp_data_host) decp_data_host = new T[full_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            T* dst = decp_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            const T* src = core_decomp + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }

    // === 5. 保存解压结果 (可选) ===
    if (processData != 0) {
        std::ofstream outFile(decpfilename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(decp_data_host), full_size * sizeof(T));
        outFile.close();
    }

    // === 6. 压缩比 ===
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / cmpSize;
    std::cout << "CR (core-only compressed): " << cr << std::endl;

    // === Cleanup ===
    delete[] core_buffer;
    delete[] core_decomp;
    delete[] compressedData;
}

template <typename T>
void compressLocalDataZFP_coreOnly(const std::string file_path,
    std::string cpfilename, const std::string filename,
    const T *input_data_host, T *&decp_data_host,
    size_t width_host, size_t height_host, size_t depth_host,  // full size with ghost
    size_t core_x, size_t core_y, size_t core_z,               // core region size                                                // ghost layer size
    double tolerance, std::string decpfilename,
    int rank, int processData = 0)
{
    size_t full_size = width_host * height_host * depth_host;
    size_t core_size = core_x * core_y * core_z;
    std::cout << "Full block size: " << full_size << ", core size: " << core_size << std::endl;

    // --- 1. 提取 core region 到临时 buffer ---
    T* core_buffer = new T[core_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            const T* src = input_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            T* dst = core_buffer + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }

    // --- 2. 配置 ZFP ---
    zfp_type type = (std::is_same<T, double>::value) ? zfp_type_double : zfp_type_float;
    zfp_field* field = zfp_field_3d(core_buffer, type, core_x, core_y, core_z);

    zfp_stream* zfp = zfp_stream_open(nullptr);
    zfp_stream_set_accuracy(zfp, tolerance);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    if (!buffer) { std::cerr << "Failed to allocate buffer" << std::endl; return; }

    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // --- 3. 压缩 core ---
    auto start = std::chrono::high_resolution_clock::now();
    size_t zfpsize = zfp_compress(zfp, field);
    auto end = std::chrono::high_resolution_clock::now();
    if (zfpsize == 0) { std::cerr << "Compression failed!" << std::endl; return; }

    double compression_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Compression time: " << compression_time << " s, compressed size = " << zfpsize << " bytes" << std::endl;

    if (processData != 0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);
        outFile1.write(reinterpret_cast<const char*>(buffer), zfpsize);
        outFile1.close();
    }

    // --- 4. 解压缩 ---
    T* core_decomp = new T[core_size];
    zfp_field* field_decomp = zfp_field_3d(core_decomp, type, core_x, core_y, core_z);

    stream_rewind(stream);
    start = std::chrono::high_resolution_clock::now();
    zfp_decompress(zfp, field_decomp);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Decompression time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    // --- 5. 把解压缩结果填回 decp_data_host 的 core 区域 ---
    if (!decp_data_host) decp_data_host = new T[full_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            T* dst = decp_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            const T* src = core_decomp + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }

    // --- 6. 保存解压结果（可选） ---
    if (processData != 0) {
        std::ofstream outFile(decpfilename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(decp_data_host), full_size * sizeof(T));
        outFile.close();
    }

    // --- 7. CR ---
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / double(zfpsize);
    std::cout << "CR (core-only compressed): " << cr << std::endl;

    // --- Cleanup ---
    delete[] core_buffer;
    delete[] core_decomp;
    zfp_field_free(field);
    zfp_field_free(field_decomp);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);



    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 启用 MPI debug 输出（通过环境变量 MPI_DEBUG=1）
    if (getenv("MPI_DEBUG") && atoi(getenv("MPI_DEBUG")) > 0) {
        g_mpi_debug = true;
        // 可以通过 MPI_DEBUG_RANK 指定只在哪个 rank 输出
        if (getenv("MPI_DEBUG_RANK")) {
            g_mpi_debug_rank = atoi(getenv("MPI_DEBUG_RANK"));
        }
        if (rank == g_mpi_debug_rank) {
            printf("[DEBUG MPI] MPI debug enabled on rank %d, total ranks=%d\n", rank, size);
            printf("[DEBUG MPI] Environment variables for OFI tuning:\n");
            printf("[DEBUG MPI]   FI_CXI_DEFAULT_CQ_SIZE=%s\n", 
                   getenv("FI_CXI_DEFAULT_CQ_SIZE") ? getenv("FI_CXI_DEFAULT_CQ_SIZE") : "(not set)");
            printf("[DEBUG MPI]   FI_CXI_OFLOW_BUF_SIZE=%s\n", 
                   getenv("FI_CXI_OFLOW_BUF_SIZE") ? getenv("FI_CXI_OFLOW_BUF_SIZE") : "(not set)");
            printf("[DEBUG MPI]   FI_CXI_REQ_BUF_SIZE=%s\n", 
                   getenv("FI_CXI_REQ_BUF_SIZE") ? getenv("FI_CXI_REQ_BUF_SIZE") : "(not set)");
            printf("[DEBUG MPI]   MPICH_OFI_CQ_PROGRESS_INTERVAL=%s\n",
                   getenv("MPICH_OFI_CQ_PROGRESS_INTERVAL") ? getenv("MPICH_OFI_CQ_PROGRESS_INTERVAL") : "(not set)");
            fflush(stdout);
        }
    }

    // Get GPU device count
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaGetDeviceCount failed: %s\n", 
                rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Determine local rank (rank within node)
    int local_rank = -1;
    if (getenv("SLURM_LOCALID"))
        local_rank = atoi(getenv("SLURM_LOCALID"));
    else if (getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
        local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    else if (getenv("MV2_COMM_WORLD_LOCAL_RANK"))
        local_rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
    else {
        // Fallback: assume sequential ranks within node
        local_rank = rank % dev_count;
    }

    // Set GPU device with explicit visibility control
    if (dev_count > 0) {
        int dev = local_rank % dev_count;
        
        // First, reset device to clear any previous context
        cudaDeviceReset();
        
        // Set device with explicit flags
        err = cudaSetDevice(dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaSetDevice(%d) failed: %s\n", 
                    rank, dev, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Verify we got the right device
        int current_dev = -1;
        cudaGetDevice(&current_dev);
        if (current_dev != dev) {
            fprintf(stderr, "Rank %d: Device mismatch! Requested %d, got %d\n", 
                    rank, dev, current_dev);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // printf("Rank %d (local %d) using GPU %d/%d\n", rank, local_rank, dev, dev_count);
        fflush(stdout);
    } else {
        fprintf(stderr, "Rank %d: No GPUs found!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Critical: Synchronize CUDA before MPI barrier to avoid race conditions
    // printf("Rank %d: Synchronizing CUDA device before barrier\n", rank);
    fflush(stdout);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaDeviceSynchronize failed: %s\n", 
                rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Reset CUDA errors before MPI operations
    cudaGetLastError();
    
    // Synchronize to ensure all ranks have initialized their GPUs
    // printf("Rank %d: Entering MPI_Barrier (COMM_WORLD size=%d)\n", rank, size);
    // fflush(stdout);
    
    // int mpi_err = MPI_Barrier(MPI_COMM_WORLD);
    
    // printf("Rank %d: Exited MPI_Barrier with code %d\n", rank, mpi_err);
    // fflush(stdout);
    

    
    if (argc < 15) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: Not enough arguments. Got %d, need at least 15\n", argc);
            fprintf(stderr, "Usage: %s <file,width,height,depth> <bound> <compressor> <sim> <q> <edit_type> <compression> <PX> <PY> <PZ> <preserveFinal> <result_prefix> <decp_prefix> <compression_mode>\n", argv[0]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }

    
    
    std::string dimension = argv[1];
    host_bound = std::stod(argv[2]);
    
    
    std::string compressor_id = argv[3];
    host_sim = std::stod(argv[4]);
    q_host = std::stoi(argv[5]);
    edit_type_host = std::stoi(argv[6]);
    int compression = std::stoi(argv[7]);
    PX = std::stoi(argv[8]);
    PY = std::stoi(argv[9]);
    PZ = std::stoi(argv[10]); 
    int preserveFinal = std::stoi(argv[11]);
    exactz_prepare_work();
    result_prefix = argv[12];
    std::filesystem::create_directories(std::filesystem::path(exactz_scratch()) / result_prefix);
    std::string decp_prefix = argv[13];
    std::filesystem::create_directories(std::filesystem::path(exactz_scratch()) / "datasets/upsamples" / decp_prefix);
    int compression_mode = std::stoi(argv[14]);
    
    std::istringstream iss(dimension);
    char delimiter;
    
    
    if (std::getline(iss, file_path, ',')) {
        
        if (iss >> width_host >> delimiter && delimiter == ',' &&
            iss >> height_host >> delimiter && delimiter == ',' &&
            iss >> depth_host) {
            if(rank ==0){
                std::cout << "Filepath: " << file_path << std::endl;
                std::cout << "Width: " << width_host << std::endl;
                std::cout << "Height: " << height_host << std::endl;
                std::cout << "Depth: " << depth_host << std::endl;
            }
            
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;
    }
    filename = extractFilename(file_path);
    
    
    num_Elements_host = width_host*height_host*depth_host;
    size_t global_x = width_host, global_y = height_host, global_z = depth_host;
   
    int px = PX, py = PY, pz = PZ;
    int dims[3] = {px, py, pz};
    int periods[3] = {0, 0, 0};
    
    
    // Verify dimensions match number of ranks
    if (px * py * pz != size) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: PX * PY * PZ = %d * %d * %d = %d != total ranks = %d\n",
                    px, py, pz, px * py * pz, size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    
    int flag; int *ub;
    MPI_Comm_get_attr(cart_comm, MPI_TAG_UB, &ub, &flag);
    if (rank==0 && flag) printf("MPI_TAG_UB=%d\n", *ub);
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    
    int coords_x = coords[0], coords_y = coords[1], coords_z = coords[2];
    // // X dimension
    int base_local_x = global_x / px;
    int remainder_x = global_x % px;
    size_t local_x = base_local_x + (coords_x < remainder_x ? 1 : 0);
    size_t offset_x =static_cast<int>(coords_x) * base_local_x + std::min(coords_x, remainder_x);
    
    // Y dimension
    int base_local_y = global_y / py;
    int remainder_y = global_y % py;
    size_t local_y = base_local_y + (coords_y < remainder_y ? 1 : 0);
    size_t offset_y = static_cast<int>(coords_y) * base_local_y + std::min(coords_y, remainder_y);
    
    // Z dimension
    int base_local_z = global_z / pz;
    int remainder_z = global_z % pz;
    size_t local_z = base_local_z + (coords_z < remainder_z ? 1 : 0);
    size_t offset_z = static_cast<int>(coords_z) * base_local_z + std::min(coords_z, remainder_z);


    // std::cout<<offset_x<<", "<<offset_y <<", "<<offset_z<<std::endl;
    size_t padded_x = local_x + 2 * HALO;
    size_t padded_y = local_y + 2 * HALO;
    size_t padded_z = local_z + 2 * HALO;
    double *input_data_host,  *decp_data_host, *decp_data_copy;
    // std::cout<<file_path<<std::endl;
    

    int world_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int cart_rank = -1;
    if (cart_comm != MPI_COMM_NULL) MPI_Comm_rank(cart_comm, &cart_rank);

    int N = 1 << 20;  // 1M doubles
    double* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemset(d_data, 0, N * sizeof(double));

    // 1. GPU kernel warm-up
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    for (int i = 0; i < 10; i++) {
        warmup_kernel<<<grid, block>>>(d_data, N);
        cudaDeviceSynchronize();
    }

    // 2. MPI warm-up
    for (int i = 0; i < 10; i++) {
        double local_sum = rank + i;
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "Warm-up done" << std::endl;
    }

    // Use size_t to avoid integer overflow for large data sizes
    size_t data_size = (size_t)padded_x * padded_y * padded_z;
    int data_size_int = (int)data_size; // For APIs that require int
    
    cudaMalloc(&input_data_host, (size_t)data_size * sizeof(double));
    cudaMalloc(&decp_data_host, (size_t)data_size * sizeof(double));
    // cudaMalloc(&decp_data_copy, data_size * sizeof(double));
    
    // 分配 decp_datac 用于保存上一次迭代的值
    
    
    // decp_NYXlog_sz3_0.000000_512_512_1024
    
    std::string decpfilename = (exactz_scratch() + "/datasets/upsamples/")+decp_prefix+"/dec_"+filename+"_"+compressor_id+'_'+
    std::to_string(host_bound)+".bin";
    if(compression == 1) decpfilename = (exactz_scratch() + "/datasets/dec_")+filename+"_"+compressor_id+'_'+std::to_string(host_bound)+".bin";
    std::cout<<decpfilename<<std::endl;
    std::string cpfilename;

    float elapsedTime = 0.0;
    cudaEvent_t startt, stop, start;
    cudaEventCreate(&startt);
    cudaEventCreate(&stop);
    cudaEventCreate(&start);
    cudaEventRecord(startt, 0);

    double *input_data_total, *decp_data_total;
    int result = 0;
    std::string command;
    
      
    if(rank == 0 && compression == 1 && compression_mode == 0){
        double range  = host_bound;
        input_data_total = new double[num_Elements_host];
        getdata<double>(file_path, input_data_total, num_Elements_host);     
        std::ostringstream oss1;
        oss1 << std::scientific << std::setprecision(17) << host_bound;
        cpfilename = (exactz_scratch() + "/datasets/compressed_")+filename+"_"+std::to_string(host_bound)+".sz";
        if(compressor_id == "sz3") {
            compressLocalData<double>(file_path, cpfilename, filename, input_data_total, decp_data_total, width_host, height_host, depth_host, host_bound, decpfilename, rank, 0);
        }
        else if(compressor_id=="zfp"){
            auto start = std::chrono::high_resolution_clock::now();
            cpfilename = (exactz_scratch() + "/datasets/compressed_")+filename+"_"+std::to_string(host_bound)+".zfp";
            
            compressLocalDataZFP<double>(file_path, cpfilename, filename, input_data_total, decp_data_total, padded_x, padded_y, padded_z, host_bound, decpfilename, rank, 1);
        }

        else if(compressor_id=="tthresh"){
            cpfilename = (exactz_scratch() + "/datasets/compressed_")+filename+"_"+std::to_string(host_bound)+".raw";
            if(depth_host == 1) return 0;
            command = "tthresh -i " + file_path + " -t double -c " + cpfilename +" -o "+decpfilename + " -s " + std::to_string(width_host) +" "+std::to_string(height_host) +" " +std::to_string(depth_host) +" -e "+std::to_string(range);
            std::cout << "Executing command: " << command << std::endl;
        
            result = std::system(command.c_str());
            if (result == 0) {
                std::cout << "Compression successful." << std::endl;
            } else {
                std::cout << "Compression failed." << std::endl;
            }
        }
    }

    MPI_Bcast(&host_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(compressor_id == "sz3") cpfilename = (exactz_scratch() + "/datasets/compressed_")+filename+"_"+std::to_string(host_bound)+".sz";
    else if(compressor_id == "zfp") cpfilename = cpfilename = (exactz_scratch() + "/datasets/compressed_")+filename+"_"+std::to_string(host_bound)+".zfp";
    else cpfilename = (exactz_scratch() + "/datasets/compressed_")+filename+"_"+std::to_string(host_bound)+".raw";
    
    double delta_host = host_bound / (q_host);

    double dataStart = 0.0;

    cudaEventRecord(stop, 0);
    compression_time = elapsedTime/1000;
    threshold_host = q_host;
    checkCudaError(cudaMemcpyToSymbol(q, &q_host, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(global_width, &global_x, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(global_height, &global_y, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(global_depth, &global_z, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(delta, &delta_host, sizeof(double), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(threshold, &threshold_host, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(edit_type, &edit_type_host, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");

    cudaEventRecord(startt, 0);
    int *vertex_type_host, *upperStars_host,*lowerStars_host, *minimum_host, *maximum_host,
        *saddles1_host, *saddles2_host, *edits_host;
        
    uint8_t *delta_counter_host, *delta_counter_copy_host, *changed_vertex_h;
    

    cudaMalloc(&vertex_type_host, (size_t)3 * data_size * sizeof(int));
    cudaMalloc(&delta_counter_host, (size_t)data_size * sizeof(uint8_t));
    cudaMalloc(&delta_counter_copy_host, (size_t)data_size * sizeof(uint8_t));
    cudaMalloc(&edits_host, (size_t)data_size * sizeof(int));
    cudaMalloc(&changed_vertex_h, (size_t)data_size * sizeof(uint8_t));
    // cudaMalloc(&lowerStars_host, (size_t)(maxNeighbors_host+1) * data_size * sizeof(int));


    cudaMemset(vertex_type_host, -1,  (size_t)3 * data_size * sizeof(int));
    cudaMemset(delta_counter_host, 0,  (size_t)data_size * sizeof(uint8_t));
    cudaMemset(edits_host, 0,  (size_t)data_size * sizeof(int));
    cudaMemset(changed_vertex_h, 0,  (size_t)data_size * sizeof(uint8_t));
    // cudaMemset(lowerStars_host, -1,  (size_t)(maxNeighbors_host+1) * data_size * sizeof(int));
    

    cudaMemcpyToSymbol(vertex_type, &vertex_type_host, sizeof(int*));
    cudaMemcpyToSymbol(delta_counter, &delta_counter_host, sizeof(uint8_t*));
    cudaMemcpyToSymbol(edits, &edits_host, sizeof(int*));
    // cudaMemcpyToSymbol(lowerStars, &lowerStars_host, sizeof(int*));

    
   
    cudaMemcpyToSymbol(width, &padded_x, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(height, &padded_y, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &padded_z, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maxNeighbors, &maxNeighbors_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_Elements, &data_size_int, sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(bound, &host_bound, sizeof(double), 0, cudaMemcpyHostToDevice);

    generateLookupTable<<<1,1>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_input: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    double dataTime = 0.0;
    double start_time_t = 0.0;
    loadLUTToGPU();
    cudaDeviceSynchronize();
    MPI_Barrier(cart_comm);
    if(rank == 0) dataStart = MPI_Wtime();
    double *host_input = new double[data_size], *host_decp = new double[data_size], *host_decp_copy = new double[data_size];
    read_subblock<double>(file_path.c_str(),
                   offset_x, offset_y, offset_z,
                   local_x, local_y, local_z,
                   global_x, global_y, global_z,
                   host_input);
    exchange_ghost_layers<double>(host_input, padded_x, padded_y, padded_z, local_x, local_y, local_z, cart_comm, dims, rank);
    MPI_Barrier(cart_comm);
    
    if(compression_mode == 1 && compressor_id == "sz3") compressLocalData_coreOnly<double>(file_path, cpfilename, filename, host_input, host_decp, padded_x, padded_y, padded_z, host_bound, decpfilename, rank);
    else if(compression_mode == 1 && compressor_id == "zfp") compressLocalDataZFP_coreOnly<double>(file_path, cpfilename, filename, host_input, host_decp, padded_x, padded_y, padded_z, local_x, local_y, local_z, host_bound, decpfilename, rank);


    if(compression_mode == 0) read_subblock<double>(decpfilename.c_str(), offset_x, offset_y, offset_z, local_x, local_y, local_z, global_x, global_y, global_z, host_decp);
    MPI_Barrier(cart_comm);

    exchange_ghost_layers<double>(host_decp, padded_x, padded_y, padded_z, local_x, local_y, local_z, cart_comm, dims, rank);
    MPI_Barrier(cart_comm);

    if (rank == 0) {
        std::cout << "data loading done" << std::endl;
    }

    cudaMemcpy(input_data_host, host_input, (size_t)data_size * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(double*));
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_input: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    
    cudaMemcpy(decp_data_host, host_decp, (size_t)data_size * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(double*));
    
    // 初始化 decp_datac（复制 decp_data 的初始值）
    // cudaMemcpy(decp_datac_host, host_decp, data_size * sizeof(double), cudaMemcpyHostToDevice);
    // err = cudaMemcpyToSymbol(decp_datac, &decp_datac_host, sizeof(double*));
    err = cudaMemcpyToSymbol(delta_counter_c, &delta_counter_copy_host, sizeof(uint8_t*));

    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_decp: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }


    dim3 blockSize(256);
    dim3 gridSize((data_size + blockSize.x - 1) / blockSize.x);


    MPI_Barrier(MPI_COMM_WORLD);
    double dataEnd = 0.0;
    if(rank == 0){
        dataEnd = MPI_Wtime();
        dataTime =  dataEnd - dataStart;
        start_time_t = MPI_Wtime();
    }
    cudaEventRecord(start, 0);
    classifyVertex_CUDA<<<gridSize, blockSize>>>(rank, nullptr, 0, changed_vertex_h, nullptr, 0, 0);
    err = cudaDeviceSynchronize(); 
    if (err != cudaSuccess) {
        printf("Rank %d: classifyVertex_CUDA: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaDeviceSynchronize(); 

    cudaMemcpyFromSymbol(&nSaddle2_host, nSaddle2, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSaddle1_host, nSaddle1, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nMax_host, nMax, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nMin_host, nMin, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
    cudaMalloc(&minimum_host, nMin_host * sizeof(int));
    cudaMalloc(&saddles1_host, nSaddle1_host * sizeof(int));
    cudaMalloc(&saddles2_host, nSaddle2_host * sizeof(int));
    cudaMalloc(&maximum_host, nMax_host * sizeof(int));

    cudaMemset(minimum_host, -1, nMin_host * sizeof(int));
    cudaMemset(saddles1_host, -1, nSaddle1_host * sizeof(int));
    cudaMemset(saddles2_host, -1, nSaddle2_host * sizeof(int));
    cudaMemset(maximum_host, -1, nMax_host * sizeof(int));

    cudaMemcpyToSymbol(minimum, &minimum_host, sizeof(int*));
    cudaMemcpyToSymbol(saddles1, &saddles1_host, sizeof(int*));
    cudaMemcpyToSymbol(saddles2, &saddles2_host, sizeof(int*));
    cudaMemcpyToSymbol(maximum, &maximum_host, sizeof(int*));
    
    int zero = 0;
    cudaMemcpyToSymbol(nMin, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSaddle1, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nSaddle2, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nMax, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);

    
    cudaDeviceSynchronize();
    
    if(rank == 0) std::cout<<"vertex classification: "<<elapsedTime/1000<<std::endl;
    ExtractCP<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();    
   
    
   
    sort_indexs_by_data(saddles2_host, input_data_host, nSaddle2_host);
    cudaDeviceSynchronize();
    sort_indexs_by_data(saddles1_host, input_data_host, nSaddle1_host);
    cudaDeviceSynchronize();
    sort_indexs_by_data(maximum_host, input_data_host, nMax_host);
    cudaDeviceSynchronize();
    sort_indexs_by_data(minimum_host, input_data_host, nMin_host);
    err = cudaDeviceSynchronize();
    

    uint64_t *d_my_idx = nullptr, *d_prev_idx = nullptr, *d_prev_idx1 = nullptr;
    uint64_t *d_my_idx1 = nullptr, *d_next_idx = nullptr, *d_next_idx1 = nullptr;

    double *d_prev_val = nullptr, *d_next_val = nullptr;
    int *d_prev_rank = nullptr;
    int *d_next_rank = nullptr;

    double *d_prev_val1 = nullptr, *d_next_val1 = nullptr;
    int *d_prev_rank1 = nullptr;
    int *d_next_rank1 = nullptr;

    uint64_t *d_my_idx_min=nullptr, *d_prev_idx_min=nullptr, *d_next_idx_min=nullptr;
    uint64_t *d_my_idx_max=nullptr, *d_prev_idx_max=nullptr, *d_next_idx_max=nullptr;

    double *d_prev_val_min=nullptr, *d_next_val_min=nullptr;
    double *d_prev_val_max=nullptr, *d_next_val_max=nullptr;

    int *d_prev_rank_min=nullptr, *d_next_rank_min=nullptr;
    int *d_prev_rank_max=nullptr, *d_next_rank_max=nullptr;

    // 每次 mpi_sort 调用之间加 MPI_Barrier 让 OFI/libfabric 有时间清理发送队列
    // 错误 "Resource temporarily unavailable" 是因为发送队列溢出

    mpi_sort_neighbors_gpuaware(
        input_data_host,
        minimum_host,
        nMin_host,
        MPI_COMM_WORLD,
        &d_my_idx_min,
        &d_prev_val_min, &d_prev_rank_min, &d_prev_idx_min,
        &d_next_val_min, &d_next_rank_min, &d_next_idx_min,
        offset_x, offset_y, offset_z
    );
    CUDA_OK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);  // 等待所有进程完成

    mpi_sort_neighbors_gpuaware(
        input_data_host,
        maximum_host,
        nMax_host,
        MPI_COMM_WORLD,
        &d_my_idx_max,
        &d_prev_val_max, &d_prev_rank_max, &d_prev_idx_max,
        &d_next_val_max, &d_next_rank_max, &d_next_idx_max,
        offset_x, offset_y, offset_z
    );
    CUDA_OK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);  // 等待所有进程完成

    mpi_sort_neighbors_gpuaware(
        input_data_host, 
        saddles2_host,
        nSaddle2_host,
        MPI_COMM_WORLD,
        &d_my_idx,
        &d_prev_val, &d_prev_rank, &d_prev_idx, 
        &d_next_val, &d_next_rank, &d_next_idx,
        offset_x, offset_y, offset_z
    );
    CUDA_OK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);  // 等待所有进程完成

    mpi_sort_neighbors_gpuaware(
        input_data_host, 
        saddles1_host,
        nSaddle1_host,
        MPI_COMM_WORLD,
        &d_my_idx1,
        &d_prev_val1, &d_prev_rank1, &d_prev_idx1, 
        &d_next_val1, &d_next_rank1, &d_next_idx1,
        offset_x, offset_y, offset_z
    );
    CUDA_OK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);



    uint8_t* d_need_prev, *d_need_prev1, *d_need_next, *d_need_next1;
    uint8_t *d_need_prevMin=nullptr, *d_need_nextMin=nullptr;
    uint8_t *d_need_prevMax=nullptr, *d_need_nextMax=nullptr;
    cudaMalloc(&d_need_prev, nSaddle2_host*sizeof(uint8_t));
    cudaMalloc(&d_need_next, nSaddle2_host*sizeof(uint8_t));
    
    cudaMalloc(&d_need_prevMin, nMin_host*sizeof(uint8_t));
    cudaMalloc(&d_need_nextMin, nMin_host*sizeof(uint8_t));

    cudaMalloc(&d_need_prevMax, nMax_host*sizeof(uint8_t));
    cudaMalloc(&d_need_nextMax, nMax_host*sizeof(uint8_t));

    cudaMalloc(&d_need_prev1, nSaddle1_host*sizeof(uint8_t));
    cudaMalloc(&d_need_next1, nSaddle1_host*sizeof(uint8_t));

    // cudaMemset(d_need_prev,  1, nSaddle2_host * sizeof(uint8_t));
    // cudaMemset(d_need_next,  1, nSaddle2_host * sizeof(uint8_t));

    // cudaMemset(d_need_prev_min, 1, nMin_host * sizeof(uint8_t));
    // cudaMemset(d_need_next_min, 1, nMin_host * sizeof(uint8_t));

    // cudaMemset(d_need_prev_max,  1, nMax_host * sizeof(uint8_t));
    // cudaMemset(d_need_next_max,  1, nMax_host * sizeof(uint8_t));

    // cudaMemset(d_need_prev1, 1, nSaddle1_host * sizeof(uint8_t));
    // cudaMemset(d_need_next1, 1, nSaddle1_host * sizeof(uint8_t));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: d_need_prev malloc: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    int host_need_count = 0;
    int*      d_count     = nullptr;
    cudaCheck(cudaMalloc(&d_count, sizeof(int)),                            "malloc d_nreq");
    cudaCheck(cudaMemset(d_count, 0, sizeof(int)),                          "memset d_nreq");

    cudaDeviceSynchronize();
    build_need_masks<<<(nSaddle2_host+255)/256,256>>>(
        nSaddle2_host, saddles2_host,
        d_prev_val, d_next_val,
        d_prev_rank, d_next_rank,
        d_prev_idx, d_next_idx,
        rank, 2 * host_bound,
        d_need_prev, d_need_next, d_count, offset_x, offset_y, offset_z
    );
    cudaDeviceSynchronize();
    cudaMemcpy(&host_need_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemset(d_count, 0, sizeof(int));
    cudaDeviceSynchronize();
    std::cout<<"saddle2: "<< host_need_count<<", "<<nSaddle2_host<<std::endl;

    build_need_masks<<<(nSaddle1_host+255)/256,256>>>(
        nSaddle1_host, saddles1_host,
        d_prev_val1, d_next_val1,
        d_prev_rank1, d_next_rank1,
        d_prev_idx1, d_next_idx1,
        rank, 2 * host_bound,
        d_need_prev1, d_need_next1, d_count, offset_x, offset_y, offset_z
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&host_need_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemset(d_count, 0, sizeof(int));
    cudaDeviceSynchronize();
    std::cout<<"saddle1: "<< host_need_count<<", "<<nSaddle1_host<<std::endl;
    
    build_need_masks<<<(nMax_host+255)/256,256>>>(
        nMax_host, maximum_host,
        d_prev_val_max, d_next_val_max,
        d_prev_rank_max, d_next_rank_max,
        d_prev_idx_max, d_next_idx_max,
        rank, 2 * host_bound,
        d_need_prevMax, d_need_nextMax, d_count, offset_x, offset_y, offset_z
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&host_need_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemset(d_count, 0, sizeof(int));
    cudaDeviceSynchronize();
    std::cout<<"max: "<< host_need_count<<", "<<nMax_host<<std::endl;
    

    build_need_masks<<<(nMin_host+255)/256,256>>>(
        nMin_host, minimum_host,
        d_prev_val_min, d_next_val_min,
        d_prev_rank_min, d_next_rank_min,
        d_prev_idx_min, d_next_idx_min,
        rank, 2 * host_bound,
        d_need_prevMin, d_need_nextMin, d_count, offset_x, offset_y, offset_z
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&host_need_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemset(d_count, 0, sizeof(int));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: before cloops: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    std::cout<<"min: "<< host_need_count<<", "<<nMin_host<<std::endl;

    dim3 gridSize_max((nMax_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_min((nMin_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_1saddle((nSaddle1_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_2saddle((nSaddle2_host + blockSize.x - 1) / blockSize.x);

    int K = maxNeighbors_host + 1;
   
    MPI_Barrier(cart_comm);

    int host_count_f_max = 0;
    int host_count_f_min = 0;
    int host_count_f_saddle = 0;
    int host_count_f_dir = 0;
    int host_count_f_order_saddle = 0;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: before cloops: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    double WholeTime;
    double end2end;
    if(rank == 0) {
        end2end = MPI_Wtime();
    }
    if(rank == 0) std::cout<<"cloops begins"<<std::endl;
    fflush(stdout);
    c_loops(rank, size, cart_comm, dims, padded_x, padded_y, padded_z, offset_x, offset_y, offset_z, gridSize, blockSize, host_count_f_max, 
        host_count_f_min, host_count_f_saddle, host_count_f_dir, host_count_f_order_saddle, delta_host, delta_counter_copy_host, data_size,
        d_prev_val, d_next_val, d_prev_idx, d_prev_rank, 
        d_next_rank, d_next_idx, d_need_prev, d_need_next,
        d_prev_val1, d_next_val1, d_prev_idx1, d_prev_rank1, 
        d_next_rank1, d_next_idx1, d_need_prev1, d_need_next1, 
        d_prev_val_min, d_next_val_min,
        d_prev_idx_min, d_prev_rank_min,
        d_next_rank_min, d_next_idx_min,
        d_need_prevMin, d_need_nextMin,

        // maximum neighbor info
        d_prev_val_max, d_next_val_max,
        d_prev_idx_max, d_prev_rank_max,
        d_next_rank_max, d_next_idx_max,
        d_need_prevMax, d_need_nextMax,PX, PY, PZ, 
        decp_data_host, delta_counter_host, local_x, local_y, local_z, saddles1_host, saddles2_host, minimum_host, maximum_host, changed_vertex_h
    );
    cudaDeviceSynchronize();
    // std::cout<<"cloops ended!"<<std::endl;
    
    MPI_Barrier(cart_comm);
    if(rank == 0) {
        double end2end1 = MPI_Wtime();
        WholeTime = end2end1 - end2end;
        std::cout<<"whole time: "<< WholeTime <<"s "<<std::endl;
    }
    MPI_Finalize();
    return 0;

}
