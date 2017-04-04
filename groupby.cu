// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
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

__global__ void truncateKeys(uint *keys) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    keys[index] = (keys[index] % 25000) + 1;
}

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

int main(int argc, char** argv)
{
    uint num_items           = 1<<28;
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
    float *d_value;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys, sizeof(uint) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_value, sizeof(float) * num_items));

    curandGenerator_t generator;
    int seed = 0;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    curandGenerate(generator, d_keys, num_items);
    truncateKeys<<<num_items/256, 256>>>(d_keys);
    curandGenerateUniform(generator, d_value, num_items);

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

    if (d_keys) CubDebugExit(g_allocator.DeviceFree(d_keys));
    if (d_value) CubDebugExit(g_allocator.DeviceFree(d_value));
    if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d_num_selected_out));

    return 0;
}
