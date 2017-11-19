/*
  OverlapSaveConvolver: A small C++11 single-file program that performs
  efficient 1D convolution and cross-correlation of two float arrays.
  Copyright (C) 2017 Andres Fernandez (https://github.com/andres-fr)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software Foundation,
  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
*/


// TODO/BUGS:
// - Signal arrays are aligned, but SIMDization wasn't explicitly benchmarked
// - Calling FloatSignal(arr, size) where arr is a fftwf_alloc_real array
//   instead of a regular float* array causes the system to freeze
// - Add unit testing with catch
// - Add and use proper benchmarking lib

// MISC:
// g++ -O3 -std=c++11 -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v ./test
// https://google.github.io/styleguide/cppguide.html


#define REAL 0
#define IMAG 1

// comment this line to deactivate OpenMP for loop parallelizations.
// the number is the minimum size that a 'for' loop needs to get sent to OMP (1=>always sent)
#define WITH_OPENMP_ABOVE 1


//
#include <string.h>
#include <math.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <initializer_list>
//
#include <fftw3.h>
#ifdef WITH_OPENMP_ABOVE
# include <omp.h>
#endif
//
#include "TStopwatch.h"
//
using namespace std;

#include <sstream>
#include <iterator>
#include <algorithm>





////////////////////////////////////////////////////////////////////////////////////////////////////
/// HELPERS
////////////////////////////////////////////////////////////////////////////////////////////////////

// template <class Iter>
// string iterable_to_string(Iter it, Iter end){
//   string result = "{";
//   stringstream sstr;
//   for(;next(it)!=end; ++it){
//     result += (to_string(*it)+", ");
//   }
//   return result + to_string(*it)+"}";
// }

// // if condition DOESN'T meet, an exception is raised
// template <class T, class Functor>
// void check_reduce(initializer_list<T> iterable, Functor reductor_predicate, const string message){
//   auto beg = iterable.begin();
//   auto end = iterable.end();
//   if(!accumulate(beg, end, *beg, reductor_predicate)){
//     throw length_error(message + iterable_to_string(beg, end));
//   }
// }

// void check_equal_lengths(initializer_list<size_t> lengths){
//   check_reduce(lengths, [&](const size_t a, const size_t b){return a*(a!=b);},
//                "ERROR [check_equal_lengths]: all sizes must be equal and are ");
// }


// check_equal_lengths({1234,1234,1234,1234});// , "ERROR [check_equal_lengths]: all sizes must be equal and are");
// check_equal_lengths({3,3,3,3,3});//, "ERROR [check_equal_lengths]: all sizes must be equal and are");
// check_equal_lengths({1,2,3,4}); //, "ERROR [check_equal_lengths]: all sizes must be equal and are");


size_t pow2_ceil(size_t x){return pow(2, ceil(log2(x)));}

// a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)
 void complex_mul(const fftwf_complex &a, const fftwf_complex &b, fftwf_complex &result){
  result[REAL] = a[REAL]*b[REAL] - a[IMAG]*b[IMAG];
  result[IMAG] = a[IMAG]*b[REAL] + a[REAL]*b[IMAG];
}

// a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
void conjugate_mul(const fftwf_complex &a, const fftwf_complex &b, fftwf_complex &result){
  result[REAL] = a[REAL]*b[REAL] + a[IMAG]*b[IMAG];
  result[IMAG] = a[IMAG]*b[REAL] - a[REAL]*b[IMAG];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is an abstract base class that provides some basic, type-independent functionality for
// any container that should behave as a signal. It is not intended to be instantiated directly.
template <class T>
class Signal {
protected:
  T* data_;
  size_t size_;
public:
  // Given a size and a reference to an array, it fills the array with <SIZE> zeros.
  // Therefore, **IT DELETES THE CONTENTS OF THE ARRAY**. It is intended to be passed a newly
  // allocated array by the classes that inherit from Signal, because it isn't an expensive
  // operation and avoids memory errors due to non-initialized values.
  explicit Signal(T* data, size_t size) : data_(data), size_(size){
    memset(data_, 0, sizeof(T)*size);
  }
  // The destructor is empty because this class didn't allocate the contained array
  virtual ~Signal(){}
  // getters
  size_t &getSize(){return size_;}
  const size_t &getSize() const{return size_;}
  T* getData(){return data_;}
  const T* getData() const{return data_;}
  // overloaded operators
  T &operator[](size_t idx){return data_[idx];}
  T &operator[](size_t idx) const {return data_[idx];}
  // basic print function. It may be overriden if, for example, the type <T> is a struct.
  void print(const string name="signal"){
    cout << endl;
    for(size_t i=0; i<size_; ++i){
      cout << name << "[" << i << "]\t=\t" << data_[i] << endl;
    }
  }
};

// This class is a Signal that works on aligned float arrays allocated by FFTW.
// It also overloads some further operators to do basic arithmetic
class FloatSignal : public Signal<float>{
public:
  // the basic constructor allocates an aligned, float array, which is zeroed by the superclass
  explicit FloatSignal(size_t size)
    : Signal(fftwf_alloc_real(size), size){}
  explicit FloatSignal(float* data, size_t size) : FloatSignal(size){
    memcpy(data_, data, sizeof(float)*size);
  }
  explicit FloatSignal(float* data, size_t size, size_t pad_bef, size_t pad_aft)
    : FloatSignal(size+pad_bef+pad_aft){
    memcpy(data_+pad_bef, data, sizeof(float)*size);
  }
  // the destructor frees the only resource allocated
  ~FloatSignal() {fftwf_free(data_);}
  // overloaded operators: TODO: benchmark OMPization
  void operator+=(const float x){for(size_t i=0; i<size_; ++i){data_[i] += x;}}
  void operator*=(const float x){for(size_t i=0; i<size_; ++i){data_[i] *= x;}}
  void operator/=(const float x){for(size_t i=0; i<size_; ++i){data_[i] /= x;}}
};

// This class is a Signal that works on aligned complex (float[2]) arrays allocated by FFTW.
// It also overloads some further operators to do basic arithmetic
class ComplexSignal : public Signal<fftwf_complex>{
public:
  // the basic constructor allocates an aligned, float[2] array, which is zeroed by the superclass
  explicit ComplexSignal(size_t size)
    : Signal(fftwf_alloc_complex(size), size){}
  ~ComplexSignal(){fftwf_free(data_);}
  // overloaded operators: TODO: benchmark OMPization
  void operator*=(const float x){
    for(size_t i=0; i<size_; ++i){
      data_[i][REAL] *= x;
      data_[i][IMAG] *= x;
    }
  }
  void operator+=(const float x){for(size_t i=0; i<size_; ++i){data_[i][REAL] += x;}}
  void operator+=(const fftwf_complex x){
    for(size_t i=0; i<size_; ++i){
      data_[i][REAL] += x[REAL];
      data_[i][IMAG] += x[IMAG];
    }
  }
  // override print method to show both fields of the complex number
  void print(const string name="signal"){
    for(size_t i=0; i<size_; ++i){
      printf("%s[%zu]\t=\t(%f, i%f)\n",name.c_str(),i,data_[i][REAL],data_[i][IMAG]);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This free function takes three complex signals a,b,c of the same size and computes the complex
// element-weise multiplication:   a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)   The computation
// loop isn't sent to OMP because this function itself is already expected to be called by multiple
// threads, and it would actually slow down the process.
// It throuws an exception if
void spectral_convolution(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  const size_t size_a = a.getSize();
  const size_t size_b = b.getSize();
  const size_t size_result = result.getSize();
  if(size_a!=size_b || size_a!=size_result){
    throw runtime_error(string("ERROR [spectral_convolution]: all sizes must be equal and are (")
                        + to_string(size_a) + ", " + to_string(size_b) + ", " +
                        to_string(size_result) + ")\n");
  }
  for(size_t i=0; i<size_a; ++i){
    // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
     result[i][REAL] = a[i][REAL]*b[i][REAL] - a[i][IMAG]*b[i][IMAG];
     result[i][IMAG] = a[i][IMAG]*b[i][REAL] + a[i][REAL]*b[i][IMAG];
  }
}

// This function behaves identically to spectral_convolution, but computes c=a*conj(b) instead
// of c=a*b:         a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
void spectral_correlation(const ComplexSignal &a, const ComplexSignal &b, ComplexSignal &result){
  const size_t size_a = a.getSize();
  const size_t size_b = b.getSize();
  const size_t size_result = result.getSize();
  if(size_a!=size_b || size_a!=size_result){
    throw runtime_error(string("ERROR [spectral_correlation]: all sizes must be equal and are (")
                        + to_string(size_a) + ", " + to_string(size_b) + ", " +
                        to_string(size_result) + ")\n");
  }
  for(size_t i=0; i<size_a; ++i){
    // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
     result[i][REAL] = a[i][REAL]*b[i][REAL] + a[i][IMAG]*b[i][IMAG];
     result[i][IMAG] = a[i][IMAG]*b[i][REAL] - a[i][REAL]*b[i][IMAG];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This class is a simple wrapper for the memory management of the fftw plans, plus a
// parameterless execute() method which is also a wrapper for FFTW's execute.
// It is not expected to be used directly: rather, to be extended by specific plans, for instance,
// if working with real, 1D signals, only 1D complex<->real plans are needed.
class FFT_Plan{
private:
  fftwf_plan plan_;
public:
  explicit FFT_Plan(fftwf_plan p): plan_(p){}
  virtual ~FFT_Plan(){fftwf_destroy_plan(plan_);}
  void execute(){fftwf_execute(plan_);}
};

// This forward plan (1D, R->C) is adequate to process 1D floats (real).
class FFT_ForwardPlan : public FFT_Plan{
public:
  // This constructor creates a real->complex plan that performs the FFT(real) and saves it into the
  // complex. As explained in the FFTW docs (http://www.fftw.org/#documentation), the size of
  // the complex has to be size(real)/2+1, so the constructor will throw a runtime error if
  // this condition doesn't hold. Since the signals and the superclass already have proper
  // destructors, no special memory management has to be done.
  explicit FFT_ForwardPlan(FloatSignal &fs, ComplexSignal &cs)
    : FFT_Plan(fftwf_plan_dft_r2c_1d(fs.getSize(), fs.getData(), cs.getData(), FFTW_ESTIMATE)){
    const size_t size_fs = fs.getSize();
    const size_t size_cs = cs.getSize();
    if(size_cs!=(size_fs/2+1)){
      throw runtime_error(string("ERROR [FFT_ForwardPlan]: size of ComplexSignal must equal size(FloatSignal)/2+1! sizes were (F,C): " + to_string(size_fs) + ", " + to_string(size_cs) + "\n"));
    }
  }
};

// This backward plan (1D, C->R) is adequate to process spectra of 1D floats (real).
class FFT_BackwardPlan : public FFT_Plan{
public:
  // This constructor creates a complex->real plan that performs the IFFT(complex) and saves it
  // complex. As explained in the FFTW docs (http://www.fftw.org/#documentation), the size of
  // the complex has to be size(real)/2+1, so the constructor will throw a runtime error if
  // this condition doesn't hold. Since the signals and the superclass already have proper
  // destructors, no special memory management has to be done.
  explicit FFT_BackwardPlan(ComplexSignal &cs, FloatSignal &fs)
    : FFT_Plan(fftwf_plan_dft_c2r_1d(fs.getSize(), cs.getData(), fs.getData(), FFTW_ESTIMATE)){
    const size_t size_fs = fs.getSize();
    const size_t size_cs = cs.getSize();
    if(size_cs!=(size_fs/2+1)){
      throw runtime_error(string("ERROR [FFT_BackardPlan]: size of ComplexSignal must equal size(FloatSignal)/2+1! sizes were (F,C): " + to_string(size_fs) + ", " + to_string(size_cs) + "\n"));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This function is a small script that calculates the FFT wisdom for all powers of two (since those
// are the only expected sizes to be used with the FFTs), and exports it to the given path. The
// wisdom is a brute-force search of the most efficient implementations for the FFTs: It takes a
// while to compute, but has to be done only once (per computer), and then it can be quickly loaded
// for faster FFT computation, as explained in the docs (http://www.fftw.org/#documentation).
// See also the docs for different flags. Note that using a wisdom file is optional.
void make_and_export_fftw_wisdom(const string path_out, const size_t min_2pow=0,
                        const size_t max_2pow=25, const unsigned flag=FFTW_PATIENT){
  for(size_t i=min_2pow; i<=max_2pow; ++i){
    size_t size = pow(2, i);
    FloatSignal fs(size);
    ComplexSignal cs(size/2+1);
    printf("creating forward and backward plans for size=2**%zu=%zu and flag %u...\n", i, size, flag);
    FFT_ForwardPlan fwd(fs, cs);
    FFT_BackwardPlan bwd(cs, fs);
  }
  fftwf_export_wisdom_to_filename(path_out.c_str());
}

// Given a path to a wisdom file generated with "make_and_export_fft_wisdom", reads and loads it
// into FFTW to perform faster FFT computations. Using a wisdom file is optional.
void import_fftw_wisdom(const string path_in, const bool throw_exception_if_fail=true){
  int result = fftwf_import_wisdom_from_filename(path_in.c_str());
  if(result!=0){
    cout << "[import_fftw_wisdom] succesfully imported " << path_in << endl;
  } else{
    string message = "[import_fftw_wisdom] ";
    message += "couldn't import wisdom! is this a path to a valid wisdom file? -->"+path_in+"<--\n";
    if(throw_exception_if_fail){throw runtime_error(string("ERROR: ") + message);}
    else{cout << "WARNING: " << message;}
  }
}





////////////////////////////////////////////////////////////////////////////////////////////////////
/// PERFORM CONVOLUTION/CORRELATION
////////////////////////////////////////////////////////////////////////////////////////////////////

// This class performs an efficient version of the spectral convolution/cross-correlation between
// two 1D float arrays, <SIGNAL> and <PATCH>, called overlap-save:
// http://www.comm.utoronto.ca/~dkundur/course_info/real-time-DSP/notes/8_Kundur_Overlap_Save_Add.pdf
// This algorithm requires that the length of <PATCH> is less or equal the length of <SIGNAL>.
class OverlapSaveConvolver {
private:
  // grab input lengths
  size_t signal_size;
  size_t patch_size;
  size_t xcorr_size;
  // make padded copies of the inputs and get chunk measurements
  FloatSignal padded_patch;
  size_t xcorr_chunksize;
  size_t xcorr_chunksize_complex;
  size_t xcorr_stride;
  ComplexSignal padded_patch_complex;
  // padded copy of the signal
  FloatSignal padded_signal;
  // the deconstructed signal
  vector<FloatSignal*> s_chunks;
  vector<ComplexSignal*> s_chunks_complex;
  // the corresponding xcorrs
  vector<FloatSignal*> xcorr_chunks;
  vector<ComplexSignal*> xcorr_chunks_complex;
  // the corresponding plans (plus the plan of the patch)
  vector<FFT_ForwardPlan*> forward_plans;
  vector<FFT_BackwardPlan*> backward_plans;
  //
  void _execute(const bool cross_correlate){
    auto operation = (cross_correlate)? spectral_correlation : spectral_convolution;
    // do ffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<forward_plans.size();i++){
      forward_plans.at(i)->execute();
    }
    // multiply spectra
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<xcorr_chunks.size();i++){
      operation(*s_chunks_complex.at(i), this->padded_patch_complex, *xcorr_chunks_complex.at(i));
    }
    // do iffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<xcorr_chunks.size();i++){
      backward_plans.at(i)->execute();
      *xcorr_chunks.at(i) /= this->xcorr_chunksize;
    }
  }

public:
  OverlapSaveConvolver(FloatSignal &signal, FloatSignal &patch, const string wisdomPath="")
    : signal_size(signal.getSize()),
      patch_size(patch.getSize()),
      xcorr_size(signal_size+patch_size-1),
      //
      padded_patch(patch.getData(), patch_size, 0, 2*pow2_ceil(patch_size)-patch_size),
      xcorr_chunksize(padded_patch.getSize()),
      xcorr_chunksize_complex(xcorr_chunksize/2+1),
      xcorr_stride(xcorr_chunksize-patch_size+1),
      padded_patch_complex(xcorr_chunksize_complex),
      //
      padded_signal(signal.getData(),signal_size,patch_size-1, xcorr_chunksize-(xcorr_size%xcorr_stride)){
      //
    if(signal_size<patch_size){
      throw length_error(string("ERROR [OverlapSaveConvolver]: ") +
                         "len(signal) can't be smaller than len(patch)\n");
    }

    if(!wisdomPath.empty()){import_fftw_wisdom(wisdomPath, false);}
    // chunk the signal into strides of same size as padded patch
    // and make complex counterparts too, as well as the corresponding xcorr signals
    for(size_t i=0; i<=padded_signal.getSize()-xcorr_chunksize; i+=xcorr_stride){
      s_chunks.push_back(new FloatSignal(&padded_signal[i], xcorr_chunksize));
      s_chunks_complex.push_back(new ComplexSignal(xcorr_chunksize_complex));
      xcorr_chunks.push_back(new FloatSignal(xcorr_chunksize));
      xcorr_chunks_complex.push_back(new ComplexSignal(xcorr_chunksize_complex));
    }
    // make one forward plan per signal chunk, and one for the patch
    // Also backward plans for the xcorr chunks
    forward_plans.push_back(new FFT_ForwardPlan(padded_patch, padded_patch_complex));
    for (size_t i =0; i<s_chunks.size();i++){
      forward_plans.push_back(new FFT_ForwardPlan(*s_chunks.at(i), *s_chunks_complex.at(i)));
      backward_plans.push_back(new FFT_BackwardPlan(*xcorr_chunks_complex.at(i), *xcorr_chunks.at(i)));
    }
  }
  void execute_conv(){_execute(false);}
  void execute_xcorr(){_execute(true);}
  // getting info from the convolfer
  void print_chunks(const string name="convolver"){
    for (size_t i =0; i<xcorr_chunks.size();i++){
      xcorr_chunks.at(i)->print(name+"_chunk_"+to_string(i));
    }
  }
  FloatSignal extract_result(){
    const size_t offset = patch_size-1;
    FloatSignal result(xcorr_size);
    float* result_arr = result.getData();
    // collapse all the xcorr chunks into result except for the first one
    for (size_t i=1; i<xcorr_chunks.size();i++){
      float* xc_arr = xcorr_chunks.at(i)->getData();
      memcpy(result_arr+(i*xcorr_stride-offset), xc_arr, sizeof(float)*xcorr_stride);
    }
    // collapse the first chunk into result: negative indexes go at the end
    float* xc_0 = xcorr_chunks.at(0)->getData();
    for(size_t i=0; i<offset; ++i){result[i+signal_size] = xc_0[i];}
    for(size_t i=offset; i<xcorr_chunksize-offset; ++i){result[i-offset] = xc_0[i];}
    return result;
  }
  ~OverlapSaveConvolver(){
    // clear vectors holding signals
    for (size_t i =0; i<s_chunks.size();i++){
      delete (s_chunks.at(i));
      delete (s_chunks_complex.at(i));
      delete (xcorr_chunks.at(i));
      delete (xcorr_chunks_complex.at(i));
    }
    s_chunks.clear();
    s_chunks_complex.clear();
    xcorr_chunks.clear();
    xcorr_chunks_complex.clear();
    // clear vector holding forward FFT plans
    for (size_t i =0; i<forward_plans.size();i++){
      delete (forward_plans.at(i));
    }
    forward_plans.clear();
    // clear vector holding backward FFT plans
    for (size_t i =0; i<backward_plans.size();i++){
      delete (backward_plans.at(i));
    }
    backward_plans.clear();
  }
};





////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN ROUTINE
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,  char** argv){
  const string wisdomPatient = "wisdom_real_dft_pow2_patient";

  // do this just once
  // make_and_export_fftw_wisdom(wisdomPatient, 0, 29, FFTW_PATIENT);

  size_t o_size = 10;//44100*10;
  float* o = new float[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 3;// 44100*1;
  float* m1 = new float[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=m1_size-i;}
  size_t xcorr_size = pow2_ceil(o_size+m1_size);

  FloatSignal s(o, o_size);
  FloatSignal p(m1, m1_size);

  s.print("signal");
  p.print("material");

  OverlapSaveConvolver x(s, p);
  x.execute_xcorr();
  x.extract_result().print("xcorr");

  x.execute_conv();
  x.extract_result().print("conv");


  TStopwatch timer1;
  for(int k=0; k<10; ++k){
    cout << "iter no "<< k << endl;
    x.execute_xcorr();
  }
  timer1.Stop();
  double t1 = timer1.RealTime()*1000;
  printf("\ttime of function 1 (ms): %f\n", timer1.RealTime()*1000);

  // x.extract_current_xcorr().print("extracted");

  //
  delete[] o;
  delete[] m1;
  return 0;
}


// TODO: conv/xcorr mechanism is buggy.
// change method names
// finish commenting convolver class
// make proper error system maybe inheriting?
// dont forget to check valgrind
//

// FURTHER DO:
// add unit test and benchmarking libraries
// write the class "optimizer": consider wether it can hold a collection of "convolvers", plus an extra
//  to-be-optimized signal, and work like that
// make sure that the optimizer allows for flexible signal picking and optimization heuristics, without losing performance
