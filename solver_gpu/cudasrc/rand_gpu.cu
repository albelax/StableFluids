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

#define DOUBLE 1
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

int Rand_GPU::randFloatsInternal(real *&devData, const size_t n)
{
  // The generator, used for random numbers
  curandGenerator_t gen;

  // Create pseudo-random number generator
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  // Set seed to be the current time (note that calls close together will have same seed!)
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

#if DOUBLE
    CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
#endif
#if FLOAT
    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
#endif
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
int Rand_GPU::randFloats(std::vector<real>& tgt)
{
  int ret_val = EXIT_SUCCESS;
  // Create a device array using CUDA
  real *d_Rand_ptr;
  CUDA_CALL(cudaMalloc(&d_Rand_ptr, tgt.size() * sizeof(real)));

  // Fill the thrust vector using the randFloats() function
  randFloatsInternal(d_Rand_ptr, tgt.size());

  // Copy the data back to the input vector
  real *h_Rand_ptr = (real*) malloc(tgt.size() * sizeof(real));

  // Need to check if the malloc was successful
  if (h_Rand_ptr != NULL)
  {
    // Copy the memory to the local pointer
    CUDA_CALL(cudaMemcpy(h_Rand_ptr, d_Rand_ptr, sizeof(real) * tgt.size(), cudaMemcpyDeviceToHost));

    // Transfer this memory into the target structure
    std::copy(h_Rand_ptr, h_Rand_ptr + tgt.size(), tgt.begin());

    // Free up the local memory
    free(h_Rand_ptr);
  }
  else
  {
    // The memory allocation failed so this will ensure the exit is "graceful"
    ret_val = EXIT_FAILURE;
  }

  // Free up the gpu memory
  cudaFree(d_Rand_ptr);

  // Return success
  return ret_val;
}

int Rand_GPU::randFloats(real *&devData, const size_t n)
{
  // The generator, used for random numbers
  curandGenerator_t gen;

  // Create pseudo-random number generator
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  // Set seed to be the current time (note that calls close together will have same seed!)
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

#if DOUBLE
    CURAND_CALL(curandGenerateUniformDouble(gen, devData, n));
#endif
#if FLOAT
    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
#endif
  // Cleanup
  CURAND_CALL(curandDestroyGenerator(gen));

  return EXIT_SUCCESS;
}
