#include "rand_gpu.h"

// Cuda includes begin
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
// cuda includes end

#define NULL_HASH UINT_MAX

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

//----------------------------------------------------------------------------------------------------------------------

#define CUDA_CALL(x) {\
    if( (x) !=cudaSuccess) {\
    printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(x));\
    exit(0);\
    }\
    }

//---------------------------------------------------------------------------

#define CURAND_CALL(x) {\
    if((x)!=CURAND_STATUS_SUCCESS) {\
    printf("CURAND failure at %s:%d\n",__FILE__,__LINE__);\
    exit(0);\
    }\
    }

//----------------------------------------------------------------------------------------------------------------------

/**
 * Fill an array with random floats using the CURAND function.
 * \param devData The chunk of GPU memory you want to fill with floats within the range (0,1]
 * \param n The size of the chunk of data
 */

int Rand_GPU::randFloatsInternal(float *&devData, const size_t n)
{
    // The generator, used for random numbers
    curandGenerator_t gen;

    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    
    // Set seed to be the current time (note that calls close together will have same seed!)
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    // Cleanup
    CURAND_CALL(curandDestroyGenerator(gen));
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------------------------------------------------

/**
 * This function takes an stl vector by reference and fills it up with random numbers generated on the GPU
 * \param tgt The target vector to fill
 * \return EXIT_SUCCESS if everything went well
 */
int Rand_GPU::randFloats(std::vector<float>& tgt)
{
    int ret_val = EXIT_SUCCESS;
    // Create a device array using CUDA
    float *d_Rand_ptr;
    CUDA_CALL(cudaMalloc(&d_Rand_ptr, tgt.size() * sizeof(float)));

    // Fill the thrust vector using the randFloats() function
    randFloatsInternal(d_Rand_ptr, tgt.size());

    // Copy the data back to the input vector
    float *h_Rand_ptr = (float*) malloc(tgt.size() * sizeof(float));

    // Need to check if the malloc was successful
    if (h_Rand_ptr != NULL)
    {
        // Copy the memory to the local pointer
        CUDA_CALL(cudaMemcpy(h_Rand_ptr, d_Rand_ptr, sizeof(float) * tgt.size(), cudaMemcpyDeviceToHost));

        // Transfer this memory into the target structure
        std::copy(h_Rand_ptr, h_Rand_ptr + tgt.size(), tgt.begin());

        // Free up the local memory
        free(h_Rand_ptr);
    } else {
        // The memory allocation failed so this will ensure the exit is "graceful"
        ret_val = EXIT_FAILURE;
    }

    // Free up the gpu memory
    cudaFree(d_Rand_ptr);

    // Return success
    return ret_val;
}

int Rand_GPU::randFloats(float *&devData, const size_t n) {
    // The generator, used for random numbers
    curandGenerator_t gen;

    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set seed to be the current time (note that calls close together will have same seed!)
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    // Cleanup
    CURAND_CALL(curandDestroyGenerator(gen));
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void add(float *sum, float *A, float *B, size_t arrayLength)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < arrayLength)
    {
        sum[idx] = A[idx] + B[idx];
    }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void pointHash(unsigned int *hash,
                          const float *Px,
                          const float *Py,
                          const float *Pz,
                          const unsigned int N,
                          const unsigned int res) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
        // each dimension and the points are also in the same space.
        int gridPos[3];
        gridPos[0] = floor(Px[idx] * res);
        gridPos[1] = floor(Py[idx] * res);
        gridPos[2] = floor(Pz[idx] * res);

        // Test to see if all of the points are inside the grid
        bool isInside = true;
        unsigned int i;
        for (i=0; i<3; ++i)
            if ((gridPos[i] < 0) || (gridPos[i] > res)) {
                isInside = false;
            }

        // Write out the hash value if the point is within range [0,1], else write NULL_HASH
        if (isInside) {
            hash[idx] = gridPos[0] * res * res + gridPos[1] * res + gridPos[2];
        } else {
            hash[idx] = NULL_HASH;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void Rand_GPU::sort()
{
    unsigned int NUM_POINTS = 50;
    unsigned int GRID_RESOLUTION = 4;
    thrust::device_vector<float> d_Rand(NUM_POINTS*3);
    float * d_Rand_ptr = thrust::raw_pointer_cast(&d_Rand[0]);
    randFloats(d_Rand_ptr, NUM_POINTS * 3);


    thrust::device_vector<float> d_Px(d_Rand.begin(), d_Rand.begin()+NUM_POINTS);
    thrust::device_vector<float> d_Py(d_Rand.begin()+NUM_POINTS, d_Rand.begin()+2*NUM_POINTS);
    thrust::device_vector<float> d_Pz(d_Rand.begin()+2*NUM_POINTS, d_Rand.end());
    thrust::device_vector<unsigned int> d_hash(NUM_POINTS);

    unsigned int * d_hash_ptr = thrust::raw_pointer_cast(&d_hash[0]);
    float * d_Px_ptr = thrust::raw_pointer_cast(&d_Px[0]);
    float * d_Py_ptr = thrust::raw_pointer_cast(&d_Py[0]);
    float * d_Pz_ptr = thrust::raw_pointer_cast(&d_Pz[0]);
    unsigned int nThreads = 1024;
    unsigned int nBlocks = NUM_POINTS / nThreads + 1;

    pointHash<<<nBlocks, nThreads>>>(d_hash_ptr, d_Px_ptr, d_Py_ptr, d_Pz_ptr,
                                     NUM_POINTS,
                                     GRID_RESOLUTION);

    // Make sure all threads have wrapped up before completing the timings
    cudaThreadSynchronize();
    thrust::sort(thrust::device, d_hash.begin(), d_hash.end());

    auto vals = thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_Pz.begin() );
//    auto a = thrust::make_zip_iterator( vals );

    thrust::sort_by_key( thrust::device, d_hash.begin(), d_hash.end(), d_Px.begin());
}

//----------------------------------------------------------------------------------------------------------------------

void Pippo::print()
{
    printf("pippo here \n");

    host_A = (float*) malloc(N * sizeof(float));
    if (host_A == NULL) exit(0);

    for (unsigned int i = 0; i < N; ++i)
    {
        host_A[i] = float(i);
    }
    if (cudaMalloc(&dev_A, N * sizeof(float)) != cudaSuccess)
        exit(0);
    if (cudaMalloc(&dev_B, N * sizeof(float)) != cudaSuccess)
        exit(0);
    if (cudaMalloc(&dev_C, N * sizeof(float)) != cudaSuccess)
        exit(0);
    if (cudaMemcpy(dev_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        exit(0);
    if (cudaMemcpy(dev_B, host_A, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        exit(0);

    add<<<numBlocks, maxThreadsPerBlock>>>(dev_C, dev_A, dev_B, N);
    float *result =(float*) malloc(N * sizeof(float));
    if( cudaMemcpy(result, dev_C, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) exit(0);

    cudaThreadSynchronize();
    for (int i = 0; i < N; ++i)
    {
        printf("The result of vector %d is %d \n", i, result[i]);
    }
}

//----------------------------------------------------------------------------------------------------------------------
