// This program is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.

//     This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.

//     You should have received a copy of the GNU General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//     author: Andres FR (Goethe-UNI Frankfurt -- https://github.com/andres-fr)

// compile for the geforce GTX 970 (run normally like a c program)
// nvcc -arch=sm_52 -I /home/rodriguez/cuda8/samples/common/inc -L/home/rodriguez/cuda8 -lcudart -lcufft xcorr.cu -o xcorr && cuda-memcheck ./xcorr

#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <cufft.h>


#define MAX_BLOCK_SIZE 1024 // max threads per block. Depends on the gpu

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////
// each grid can have a maximum of 65535**dim  blocks.
// a thread is a subtask (like a single addition). each block can contain up to 512 or 1024 threads
// (depends on the GPU). At best a multiple of 32. The if statement is needed to prevent multiple
// threads from accessing the same portion of memory (e.g. we have two blocks of 512 threads each
// but n_el is 515. In this case, without that condition, threads from 516 to 1024 could do some
// unexpected work).

template<typename T>
__global__ void kernel_divide_real_by(cufftReal* a, const T div, const size_t n_el){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n_el){
    a[i] /= div;
  }
}

template<typename T>
void divide_real_by(cufftReal* a, const T div, const size_t n_el){
  int threadsPerBlock,blocksPerGrid;
  if (n_el<MAX_BLOCK_SIZE){
    threadsPerBlock = n_el;
    blocksPerGrid   = 1;
  } else {
    threadsPerBlock = MAX_BLOCK_SIZE;
    blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
  }
  kernel_divide_real_by<<<blocksPerGrid,threadsPerBlock>>>(a, div, n_el);
}

__global__ void kernel_complex_mul(const cufftComplex* a, const cufftComplex* b, cufftComplex* c,
                                   const size_t n_el){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n_el){                               // a+ib * c+id =
    c[i].x = a[i].x*b[i].x - a[i].y*b[i].y;    // ac+iad+ibc-bd = 
    c[i].y = a[i].x*b[i].y + a[i].y*b[i].x;    // ac-bd, i(ad+bc)
  }
}
__global__ void kernel_conjugate_mul(const cufftComplex* a, const cufftComplex* b, cufftComplex* c,
                                   const size_t n_el){
  int i = blockDim.x*blockIdx.x + threadIdx.x;  // conj(a) * b
  if (i < n_el){                                // a-ib * c+id =
    c[i].x = a[i].x*b[i].x + a[i].y*b[i].y;     // ac+iad-ibc+bd =
    c[i].y = a[i].x*b[i].y - a[i].y*b[i].x;     // ac+bd, i(ad-bc)
  }
}

cufftComplex* spectral_convolution(const cufftComplex* a, const cufftComplex* b, const size_t n_el,
                                   const bool cross_correlation=false){
  int threadsPerBlock,blocksPerGrid;
  if (n_el<MAX_BLOCK_SIZE){
    threadsPerBlock = n_el;
    blocksPerGrid   = 1;
  } else {
    threadsPerBlock = MAX_BLOCK_SIZE;
    blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
  }
  cufftComplex* result;
  checkCudaErrors(cudaMalloc((void**)&result, sizeof(cufftComplex)*n_el));
  if(cross_correlation){
    kernel_conjugate_mul<<<blocksPerGrid,threadsPerBlock>>>(b, a, result, n_el);
  } else{ // a proper convolution
    kernel_complex_mul<<<blocksPerGrid,threadsPerBlock>>>(a, b, result, n_el);
  }
  
  return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// HELPERS
////////////////////////////////////////////////////////////////////////////////////////////////////
int pow2ceil(double x){
  // this padding, for the FFT algo, is just for speed reasons (since power of 2 is fastest). Cyclic
  // convolution should be managed by the library itself (although I didn't check that).
  return pow(2, ceil(log2(x)));
}

template<typename T>
cufftComplex* realToComplex(const T arr_in, const size_t size){
  cufftComplex* result = new cufftComplex[size];
  for(unsigned int i=0; i<size; ++i){
    result[i].x = i;
  }
  return result;
}

template<typename T>
T copyToGPU(const T &arr_in, const size_t size, const size_t padding_size){
  size_t in_memsize = size*sizeof(T);
  size_t pad_memsize = padding_size*sizeof(T);
  T result;
  checkCudaErrors(cudaMalloc((void**)&result, in_memsize+pad_memsize));
  checkCudaErrors(cudaMemcpy(result, arr_in, in_memsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset (result+size, 0, pad_memsize)); // zero-pad signal
  return result;
}

class Real_1DCUFFT_Manager {
 private:
  cufftHandle forwardPlan;
  cufftHandle inversePlan;
  size_t paddedSize;
  size_t fftSize;
  size_t batchSize;
 public:
  // constructor
  Real_1DCUFFT_Manager(const size_t padded_size, const size_t batch_size=1, const cufftType_t mode=CUFFT_R2C){
    if (cufftPlan1d(&(forwardPlan), padded_size, CUFFT_R2C, batch_size) != CUFFT_SUCCESS){
      throw runtime_error("Real_1DCUFFT_Manager construction error: Forward plan creation failed");
    }
    if (cufftPlan1d(&(inversePlan), padded_size, CUFFT_C2R, batch_size) != CUFFT_SUCCESS){
      throw runtime_error("Real_1DCUFFT_Manager construction error: Inverse plan creation failed");
    }
    paddedSize = padded_size;
    fftSize = padded_size/2+1;
    batchSize = batch_size;
  }
  ~Real_1DCUFFT_Manager(){
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
  }
  cufftComplex* forwardFFT(cufftReal* inputSignal){
    cufftComplex* outputSignal;
    checkCudaErrors(cudaMalloc((void**)&outputSignal, sizeof(cufftComplex)*(fftSize)*(batchSize)));
    if (cufftExecR2C(forwardPlan, inputSignal, outputSignal) != CUFFT_SUCCESS){
      throw runtime_error("Real_1DCUFFT_Manager.forwardFFT() failed. Is input size same as paddedSize?");
    }
    return outputSignal;
  }
  cufftReal* inverseFFT(cufftComplex* inputSignal, bool normalize=true){
    cufftReal* outputSignal;
    checkCudaErrors(cudaMalloc((void**)&outputSignal, sizeof(cufftReal)*paddedSize*batchSize));
    if (cufftExecC2R(inversePlan, inputSignal, outputSignal) != CUFFT_SUCCESS){
      throw runtime_error("Real_1DCUFFT_Manager.inverseFFT() failed. Is input size=fftSize?");
    }
    if(normalize){divide_real_by(outputSignal, paddedSize, paddedSize);};
    return outputSignal;
  }
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
  size_t o_size = 32000;
  size_t padded_size = pow2ceil(o_size)*2;
  size_t fft_size = padded_size/2+1;
  size_t m1_size = 32000;
  size_t xcorr1_size = o_size+m1_size-1;

  // CREATE THE REAL SIGNALS (1,2,3,4...) 
  cufftReal* h_o = new cufftReal[o_size]; for(int i=0; i<o_size; ++i){h_o[i] = i+1;}
  cufftReal* h_m1 = new cufftReal[m1_size]; for(int i=0; i<m1_size; ++i){h_m1[i]=1;}

  // COPY THEM TO THE GPU
  // time series are NOT zero-padded
  cufftReal* reconstruction = copyToGPU(h_o, o_size, padded_size-o_size);
  cufftReal* d_m1 = copyToGPU(h_m1, m1_size, padded_size-m1_size);


  // MAKE FFTS: input has padded_size, output has fft_size
 
  Real_1DCUFFT_Manager fftManager(padded_size, 1);
  cufftComplex* reconstruction_spec = fftManager.forwardFFT(reconstruction);
  cufftComplex* m1_spec = fftManager.forwardFFT(d_m1);

  // MAKE CONVOLUTIONS
  cufftComplex* m1_spec_xcorr = spectral_convolution(reconstruction_spec, m1_spec, fft_size, true);

  // MAKE IFFTS
  cufftReal* d_xcorr1 = fftManager.inverseFFT(m1_spec_xcorr);

  // COPY XCORRS BACK TO HOST
  cufftReal* h_xcorr1 = new cufftReal[xcorr1_size];
  memset(h_xcorr1, 0, xcorr1_size*sizeof(cufftReal));
  checkCudaErrors(cudaMemcpy(h_xcorr1, d_xcorr1+padded_size-m1_size+1,
                             sizeof(cufftReal)*(m1_size-1), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_xcorr1+m1_size-1, d_xcorr1, sizeof(cufftReal)*o_size, cudaMemcpyDeviceToHost));

  // // PRINT RESULTS
  // cout << endl;
  // for(int i=0; i<xcorr1_size; ++i){
  //   printf("i=%dxcorr1=%f\n", i, h_xcorr1[i]);
  // }
  cout << "finished!" << endl;

  // FREE CPU
  delete[] h_o;
  delete[] h_m1;
  delete[] h_xcorr1;
  // FREE GPU
  cudaFree(reconstruction);
  cudaFree(d_m1);
  cudaFree(reconstruction_spec);
  cudaFree(m1_spec);
  cudaFree(m1_spec_xcorr);
  cudaFree(d_xcorr1);

  return cudaDeviceSynchronize();
}