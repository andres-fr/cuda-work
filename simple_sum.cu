// compile for the geforce GTX 970 with CUDA in home
// nvcc -arch=sm_52 -I/home/rodriguez/cuda8/samples/common/inc simple_sum.cu -o simple_sum && ./simple_sum


#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "cuda_runtime.h"
#include <helper_cuda.h>


////////////////////////////////////////////////////////////////////////////////////////////////////
/// KERNEL
////////////////////////////////////////////////////////////////////////////////////////////////////
// each grid can have a maximum of 65535**dim  blocks.
// a thread is a subtask (like a single addition). each block can contain up to 512 or 1024 threads
// (depends on the GPU). At best a multiple of 32. The if statement is needed to prevent multiple
// threads from accessing the same portion of memory (e.g. we have two blocks of 512 threads each
// but n_el is 515. In this case, without that condition, threads from 516 to 1024 could do some
// unexpected work).
static const int max_block_size = 1024; // max threads per block

__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el){
  // the thread Index for this instance of the kernel
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_el){ // is this if really necessary? how much does it delay?
    C[tid] = A[tid] + B[tid];
  }
}

// function which invokes the kernel
void sum(const float* A, const float* B, float* C, int n_el){
  int threadsPerBlock,blocksPerGrid;
  if (n_el<max_block_size){
    threadsPerBlock = n_el;
    blocksPerGrid   = 1;
  } else {
    threadsPerBlock = max_block_size;
    blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
  }
  kernel_sum<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, n_el);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
  // declare the vectors' number of elements and their size in bytes
  static const int n_el = 10000;
  static const size_t size = n_el * sizeof(float);
 
  // declare and allocate input vectors h_A and h_B in the host (CPU) memory
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);
  // declare and allocate device vectors in the device (GPU) memory
  float *d_A,*d_B,*d_C;
  checkCudaErrors(cudaMalloc(&d_A, size));
  checkCudaErrors(cudaMalloc(&d_B, size));
  checkCudaErrors(cudaMalloc(&d_C, size));

  // initialize input vectors
  for (int i=0; i<n_el; i++){
    h_A[i]=sin(i);
    h_B[i]=cos(i);
  }

  // copy CPU->GPU
  checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // call kernel
  sum(d_A, d_B, d_C, n_el);

  // copy GPU->CPU, and free GPU
  checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // compute the cumulative error
  double err=0;
  for (int i=0; i<n_el; i++) {
    double diff=double((h_A[i]+h_B[i])-h_C[i]);
    err+=diff*diff;
    // print results for manual checking.
    std::cout << "A+B: " << h_A[i]+h_B[i] << "\n";
    std::cout << "C: " << h_C[i] << "\n";
  }
  err=sqrt(err);
  std::cout << "err: " << err << "\n";

  // free CPU and return
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return cudaDeviceSynchronize();
}