// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_select.cuh>

#include "cub/test/test_util.h"

using namespace std;
using namespace cub;

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&time, start,stop); \
    t = time; \
}

#define NUM_SMS 24
// must be power of two
#define BLOCK_SIZE 256
// must be power of two
#define NUM_THREADS_PER_SM 2048
#define NUM_BLOCKS_PER_SM (NUM_THREADS_PER_SM / BLOCK_SIZE)
#define NUM_BLOCKS (NUM_SMS * NUM_BLOCKS_PER_SM)
#define SIZE ((1 << 22) * NUM_SMS)
#define NUM_THREADS (NUM_THREADS_PER_SM * NUM_SMS)
#define ITEMS_PER_THREAD (SIZE / NUM_THREADS)

// either less than BLOCK_SIZE or a multiple of BLOCK_SIZE
#ifndef COUNTS
#define COUNTS 8388608
// should be set to min(COUNTS, 8) ... only relevant if GLOBAL_ALL==0
#define LOCAL_COUNTS 8
// should be set to min(COUNTS, 8192) ... only relevant if GLOBAL_ALL==0
#define SHARED_COUNTS 8192
#endif

// arbitrary values that happen to provide the right cutoffs here
#define SHARED_MEM_BYTES 400000
#define GLOBAL_MEM_BYTES 7000000000

__global__ void truncateKeys(uint *keys) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    keys[index] = keys[index] % COUNTS;
}

__global__ void computeCountsGlobal(uint *keys, uint *counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd((uint *)&counts[keys[index]], 1);
}

__global__ void computeCountsLocal(uint *keys, uint *counts) {
    __shared__ uint local_counts[LOCAL_COUNTS * BLOCK_SIZE];
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int my_offset = threadIdx.x * COUNTS;
    for (int i = 0; i < COUNTS; i++) {
      local_counts[my_offset + i] = 0;
    }
    // TODO __syncthreads() needed?
    for (int i = index * ITEMS_PER_THREAD; i < (index + 1) * ITEMS_PER_THREAD; i++) {
      local_counts[my_offset + keys[i]]++;
    }
    for (int i = 0; i < COUNTS; i++) {
      counts[i * NUM_THREADS + index] = local_counts[my_offset + i];
    }
}

__global__ void computeCountsLocal2(uint *keys, uint *counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index * ITEMS_PER_THREAD; i < (index + 1) * ITEMS_PER_THREAD; i++) {
      counts[keys[i] * NUM_THREADS + index]++;
    }
}

__global__ void computeCountsShared(uint *keys, uint *counts) {
    __shared__ uint local_counts[SHARED_COUNTS];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int counts_per_block_thread = COUNTS / BLOCK_SIZE;

    if (COUNTS >= BLOCK_SIZE) {
      for (int i = threadIdx.x * counts_per_block_thread;
	i < (threadIdx.x + 1) * counts_per_block_thread; i++) {
        local_counts[i] = 0;
      }
    } else if (threadIdx.x == 0) {
       for (int i = 0; i < COUNTS; i++) {
        local_counts[i] = 0;
      }
    }
    __syncthreads();

    for (int i = index * ITEMS_PER_THREAD; i < (index + 1) * ITEMS_PER_THREAD; i++) {
      atomicAdd((uint *)&local_counts[keys[i]], 1);
    }
    __syncthreads();

    if (COUNTS >= BLOCK_SIZE) {
      for (int i = threadIdx.x * counts_per_block_thread;
	i < (threadIdx.x + 1) * counts_per_block_thread; i++) {
        counts[i * NUM_BLOCKS + blockIdx.x] = local_counts[i];
      }
    } else if (threadIdx.x == 0) {
       for (int i = 0; i < COUNTS; i++) {
        counts[i * NUM_BLOCKS + blockIdx.x] = local_counts[i];
      }
    }
}

__global__ void computeCountsShared2(uint *keys, uint *counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index * ITEMS_PER_THREAD; i < (index + 1) * ITEMS_PER_THREAD; i++) {
      atomicAdd((uint *)&counts[keys[i] * NUM_BLOCKS + blockIdx.x], 1);
    }
}

/*
typedef struct {
    float val;
    uint key;
} T;

/// Selection functor type
struct NonEmpty
{
    uint compare;

    __host__ __device__ __forceinline__
    NonEmpty(uint compare) : compare(compare) {};

    __host__ __device__ __forceinline__
    bool operator()(const T &a) const {
        return (a.key > compare);
    }
};

__global__ static void build_groupby_key(uint *key, float *val, T *ht, uint hsize, uint num_items) {
    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = offset; i < num_items; i += stride) {
        uint k = key[i];
        float v = val[i];

        uint hash = (k % hsize) + 1;
        T local;

        // Check at max the entire table
        for (int x = 0; x < hsize; x++) {
            local = ht[hash];
            if (local.key == k) {
                atomicAdd((float*)&ht[hash].val, v);
                break;
            }
            if (local.key == 0) {
                // We need CAS
                uint old =  atomicCAS((uint*)&ht[hash].key, 0, k);
                // local.key = k;
                if (old == 0 || old == k) {
                    atomicAdd((float*)&ht[hash].val, v);
                    break;
                }
            }

            hash = (hash + 1) % hsize;
        }
    }
}
*/

int main(int argc, char** argv)
{
    uint num_items           = SIZE;
    int num_trials          = 3;
    bool full_agg           = true;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("f", full_agg);
    args.GetCmdLineArgument("t", num_trials);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items>] "
            "[--f=<full agg>] "
            "[--t=<num trials>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint  *d_keys;
    uint  *d_counts_global = NULL;
    uint  *d_counts_shared = NULL;
    int  *d_shared_offsets = NULL;
    uint  *d_shared_final = NULL;
    uint  *d_counts_local = NULL;
    int  *d_local_offsets = NULL;
    uint  *d_local_final = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys, sizeof(uint) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_counts_global, sizeof(uint) * COUNTS));
    cudaMemset(d_counts_global, 0, sizeof(uint) * COUNTS);

    curandGenerator_t generator;
    int seed = 0;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    curandGenerate(generator, d_keys, num_items);
    truncateKeys<<<num_items/BLOCK_SIZE, BLOCK_SIZE>>>(d_keys);

    float time_global;
    TIME_FUNC((computeCountsGlobal<<<num_items/BLOCK_SIZE, BLOCK_SIZE>>>(d_keys, d_counts_global)), time_global);
    cudaMemset(d_counts_global, 0, sizeof(uint) * COUNTS);
    TIME_FUNC((computeCountsGlobal<<<num_items/BLOCK_SIZE, BLOCK_SIZE>>>(d_keys, d_counts_global)), time_global);
    cout << "\"time_global\":" << time_global << endl;
    uint *global_counts = new uint[COUNTS];
    CubDebugExit(cudaMemcpy(global_counts, d_counts_global, COUNTS * sizeof(uint), cudaMemcpyDeviceToHost));

    // SHARED
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_counts_shared, sizeof(uint) * COUNTS * NUM_BLOCKS));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_shared_offsets, sizeof(int) * (COUNTS + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_shared_final, sizeof(uint) * COUNTS));
    int *shared_offsets = new int[COUNTS + 1];
    for (int i = 0; i <= COUNTS; i++) {
      shared_offsets[i] = i * NUM_BLOCKS;
    }
    CubDebugExit(cudaMemcpy(d_shared_offsets, shared_offsets, sizeof(uint) * (COUNTS + 1), cudaMemcpyHostToDevice));

    float time_shared_first;
    if (COUNTS * sizeof(uint) * NUM_BLOCKS_PER_SM < SHARED_MEM_BYTES) {
      // warmup
      TIME_FUNC((computeCountsShared<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_shared)), time_shared_first);
      TIME_FUNC((computeCountsShared<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_shared)), time_shared_first);
    } else {
      TIME_FUNC((computeCountsShared2<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_shared)), time_shared_first);
      cudaMemset(d_counts_shared, 0, sizeof(uint) * COUNTS * NUM_BLOCKS);
      TIME_FUNC((computeCountsShared2<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_shared)), time_shared_first);
    }

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_counts_shared, d_shared_final, COUNTS,
      d_shared_offsets, d_shared_offsets + 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));

    float time_shared_second;
    TIME_FUNC(CubDebugExit(DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_counts_shared, d_shared_final, COUNTS,
      d_shared_offsets, d_shared_offsets + 1)), time_shared_second);
    cout << "\"time_shared\":" << (time_shared_first + time_shared_second) << " (" << time_shared_first << "," << time_shared_second << ")" << endl;

    uint *shared_counts = new uint[COUNTS];
    CubDebugExit(cudaMemcpy(shared_counts, d_shared_final, sizeof(uint) * COUNTS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < COUNTS; i++) {
      if (shared_counts[i] != global_counts[i]) {
        cout << "shared and global differ at " << i << " (" << shared_counts[i] << "," << global_counts[i] << ")" << endl;
        break;
      }
    }

    // LOCAL
    if (sizeof(uint) * COUNTS * NUM_THREADS < GLOBAL_MEM_BYTES) {
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_counts_local, sizeof(uint) * COUNTS * NUM_THREADS));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_local_offsets, sizeof(int) * (COUNTS + 1)));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_local_final, sizeof(uint) * COUNTS));
      int *local_offsets = new int[COUNTS + 1];
      for (int i = 0; i <= COUNTS; i++) {
        local_offsets[i] = i * NUM_THREADS;
      }
      CubDebugExit(cudaMemcpy(d_local_offsets, local_offsets, sizeof(uint) * (COUNTS + 1), cudaMemcpyHostToDevice));

      float time_local_first;
      if (COUNTS * sizeof(uint) * NUM_THREADS_PER_SM < SHARED_MEM_BYTES) {
        // warmup
        TIME_FUNC((computeCountsLocal<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_local)), time_local_first);
        TIME_FUNC((computeCountsLocal<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_local)), time_local_first);
      } else {
        TIME_FUNC((computeCountsLocal2<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_local)), time_local_first);
        cudaMemset(d_counts_local, 0, sizeof(uint) * COUNTS * NUM_THREADS);
        TIME_FUNC((computeCountsLocal2<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_keys, d_counts_local)), time_local_first);
      }

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      CubDebugExit(DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_counts_local, d_local_final, COUNTS,
        d_local_offsets, d_local_offsets + 1));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));

      float time_local_second;
      TIME_FUNC(CubDebugExit(DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_counts_local, d_local_final, COUNTS,
        d_local_offsets, d_local_offsets + 1)), time_local_second);
      cout << "\"time_local\":" << (time_local_first + time_local_second)  << " (" << time_local_first << "," << time_local_second << ")" << endl;

      uint *local_counts = new uint[COUNTS];
      CubDebugExit(cudaMemcpy(local_counts, d_local_final, sizeof(uint) * COUNTS, cudaMemcpyDeviceToHost));
      for (int i = 0; i < COUNTS; i++) {
        if (local_counts[i] != global_counts[i]) {
          cout << "local and global differ at " << i << " (" << local_counts[i] << "," << global_counts[i] << ")" << endl;
          break;
        }
      }
    }


    /*
    int hash_table_size = 65536;
    int     *d_num_selected_out   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int)));

    for (int i = 0; i < num_trials; i++) {
        // Full Aggregation.
        float time_full_agg;
        float *d_out;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(float)));

        // Allocate temporary storage
        void            *d_temp_storage = NULL;
        size_t          temp_storage_bytes = 0;

        CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_value, d_out, num_items));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));

        TIME_FUNC(CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
            d_value, d_out, num_items)), time_full_agg);

        float result;
        CubDebugExit(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
        if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

        // Key Aggregation
        float time_build_hash, time_compact;
        T* hash_table;
        T* d_out2;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&hash_table, hash_table_size * (sizeof(int) + sizeof(float))));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out2, hash_table_size * (sizeof(int) + sizeof(float))));

        cudaMemset(hash_table, 0, hash_table_size * (sizeof(int) + sizeof(float)));
        cudaDeviceSynchronize();

        TIME_FUNC((build_groupby_key<<<8192, 256>>>(d_keys, d_value, hash_table, hash_table_size, num_items)), time_build_hash);
        cudaDeviceSynchronize();

        // Allocate temporary storage
        void            *d_temp_storage2 = NULL;
        size_t          temp_storage_bytes2 = 0;
        NonEmpty select_op(0);

        CubDebugExit(DeviceSelect::If(d_temp_storage2, temp_storage_bytes2,
            hash_table, d_out2, d_num_selected_out, hash_table_size, select_op));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage2, temp_storage_bytes2));

        // Run
        TIME_FUNC(CubDebugExit(DeviceSelect::If(d_temp_storage2, temp_storage_bytes2,
            hash_table, d_out2, d_num_selected_out, hash_table_size, select_op)), time_compact);

        int num_results;
        CubDebugExit(cudaMemcpy(&num_results, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost));

        T* copy = new T[num_results];
        CubDebugExit(cudaMemcpy(copy, d_out2, sizeof(int) * 2 * num_results, cudaMemcpyDeviceToHost));
        float sum = 0;
        for (int i = 0; i < num_results; i++) {
            T j = copy[i];
            sum += j.val;
        }
        delete[] copy;

        cout<< "{"
            << "\"time_full_agg\":" << time_full_agg
            << "\"time_counts\":" << time_count
            << ",\"temp_storage_bytes\":" << temp_storage_bytes
            << ",\"result\":" << result
            << ",\"time_build_hash\":" << time_build_hash
            << ",\"time_compact\":" << time_compact
            << ",\"temp_storage_bytes2\":" << temp_storage_bytes2
            << ",\"num_results\":" << num_results
            << ",\"sum\":" << sum
            << "}" << endl;

        if (d_out2) CubDebugExit(g_allocator.DeviceFree(d_out2));
        if (hash_table) CubDebugExit(g_allocator.DeviceFree(hash_table));
        if (d_temp_storage2) CubDebugExit(g_allocator.DeviceFree(d_temp_storage2));
    }
    */

    if (d_keys) CubDebugExit(g_allocator.DeviceFree(d_keys));
    if (d_counts_global) CubDebugExit(g_allocator.DeviceFree(d_counts_global));
    if (d_counts_shared) CubDebugExit(g_allocator.DeviceFree(d_counts_shared));
    if (d_shared_offsets) CubDebugExit(g_allocator.DeviceFree(d_shared_offsets));
    if (d_shared_final) CubDebugExit(g_allocator.DeviceFree(d_shared_final));
    if (d_counts_local) CubDebugExit(g_allocator.DeviceFree(d_counts_local));
    if (d_local_offsets) CubDebugExit(g_allocator.DeviceFree(d_local_offsets));
    if (d_local_final) CubDebugExit(g_allocator.DeviceFree(d_local_final));

    return 0;
}
