// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v  ./test

// 3) run
// g++ -O3 -std=c++11 -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v ./test



#define REAL 0
#define IMAG 1
#define WITH_OPENMP_ABOVE 1 // minsize of a for loop that gets sent to OMP (has to be benchmarked!)


#include <string.h>
#include <math.h>
// #include <time.h>
#include <iostream>
#include <stdexcept>
#include <fftw3.h>

#include "catch.hpp"
#include "TStopwatch.h"
#include <fstream>
#include <cerrno>

// #include<vector>
// #include<assert.h>


 #ifdef WITH_OPENMP_ABOVE
 # include <omp.h>
 #endif

using namespace std;





////////////////////////////////////////////////////////////////////////////////////////////////////
/// HELPERS
////////////////////////////////////////////////////////////////////////////////////////////////////


string get_file_contents(const char *filename){
  ifstream in(filename, ios::in | ios::binary);
  if (in){
    std::string contents;
    in.seekg(0, ios::end);
    contents.resize(in.tellg());
    in.seekg(0, ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return(contents);
  }
  throw(errno);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t pow2_ceil(size_t x){
  return pow(2, ceil(log2(x)));
}

void complex_mul(const fftwf_complex &a, const fftwf_complex &b, fftwf_complex &result){
  // a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)
  result[REAL] = a[REAL]*b[REAL] - a[IMAG]*b[IMAG];
  result[IMAG] = a[IMAG]*b[REAL] + a[REAL]*b[IMAG];
}

void conjugate_mul(const fftwf_complex &a, const fftwf_complex &b, fftwf_complex &result){
  // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
  result[REAL] = a[REAL]*b[REAL] + a[IMAG]*b[IMAG];
  result[IMAG] = a[IMAG]*b[REAL] - a[REAL]*b[IMAG];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
class Signal {
protected:
  T* data;
  size_t size;
public:
  Signal(T* data, size_t size) : data(data), size(size){
    memset(data, 0, sizeof(T)*size);
  }
  virtual ~Signal(){}
  size_t getSize(){return this->size;}
  T* getData(){return this->data;}
  //
  T &operator[](size_t idx){return data[idx];}
  T &operator[](size_t idx) const {return data[idx];}
  //
  void print(const string name="signal"){
    cout << endl;
    for(size_t i=0; i<size; ++i){
      cout << name << "[" << i << "]\t=\t" << this->data[i] << endl;
    }
  }
};


class FloatSignal : public Signal<float>{
public:
  explicit FloatSignal(size_t size)
    : Signal(fftwf_alloc_real(size), size){}
  explicit FloatSignal(float* d, size_t size) : FloatSignal(size){
    memcpy(this->data, d, sizeof(float)*size);
  }
  explicit FloatSignal(float* d, size_t sz, size_t pad_bef, size_t pad_aft)
    : FloatSignal(sz+pad_bef+pad_aft){
    memcpy(this->data+pad_bef, d, sizeof(float)*sz);
  }
  ~FloatSignal() {fftwf_free(this->data);}
  //
  void operator+=(const float x){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<this->size; ++i){this->data[i] += x;}
  }
  void operator*=(const float x){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<this->size; ++i){this->data[i] *= x;}
  }
  void operator/=(const float x){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<this->size; ++i){this->data[i] /= x;}
  }
};

class ComplexSignal : public Signal<fftwf_complex>{
public:
  explicit ComplexSignal(size_t size)
    : Signal(fftwf_alloc_complex(size), size){}
  ~ComplexSignal(){fftwf_free(this->data);}
  void operator*=(const float x){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<this->size; ++i){
      this->data[i][REAL] *= x;
      this->data[i][IMAG] *= x;
    }
  }
  void operator+=(const float x){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<this->size; ++i){this->data[i][REAL] += x;}
  }
  void operator+=(const fftwf_complex x){
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
    for(size_t i=0; i<this->size; ++i){
      this->data[i][REAL] += x[REAL];
      this->data[i][IMAG] += x[IMAG];
    }
  }
  void print(const string name="signal"){
    for(size_t i=0; i<size; ++i){
      printf("%s[%zu]\t=\t(%f, i%f)\n", name.c_str(), i, this->data[i][REAL], this->data[i][IMAG]);
    }
  }
};


void spectral_correlation(ComplexSignal &a, ComplexSignal &b, ComplexSignal &result){
  size_t size_a = a.getSize();
  size_t size_b = b.getSize();
  if(size_a!=size_b){
    throw runtime_error(string("ERROR [spectral_convolution]: both sizes must be equal and are (")
                        + to_string(size_a) + ", " + to_string(size_b) + ")\n");
  }
  // #ifdef WITH_OPENMP_ABOVE
  // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
  // #endif
  for(size_t i=0; i<size_a; ++i){
    // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
     result[i][REAL] = a[i][REAL]*b[i][REAL] + a[i][IMAG]*b[i][IMAG];
     result[i][IMAG] = a[i][IMAG]*b[i][REAL] - a[i][REAL]*b[i][IMAG];
  }
}

void spectral_convolution(ComplexSignal &a, ComplexSignal &b, ComplexSignal &result){
  size_t size_a = a.getSize();
  size_t size_b = b.getSize();
  if(size_a!=size_b){
    throw runtime_error(string("ERROR [spectral_convolution]: both sizes must be equal and are (")
                        + to_string(size_a) + ", " + to_string(size_b) + ")\n");
  }
  // #ifdef WITH_OPENMP_ABOVE
  // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
  // #endif
  for(size_t i=0; i<size_a; ++i){
    // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
     result[i][REAL] = a[i][REAL]*b[i][REAL] - a[i][IMAG]*b[i][IMAG];
     result[i][IMAG] = a[i][IMAG]*b[i][REAL] + a[i][REAL]*b[i][IMAG];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class FFT_Plan{
private:
  fftwf_plan plan;
public:
  explicit FFT_Plan(fftwf_plan p): plan(p){}
  virtual ~FFT_Plan(){fftwf_destroy_plan(this->plan);}
  void execute(){fftwf_execute(this->plan);}
};

class FFT_ForwardPlan : public FFT_Plan{
public:
  explicit FFT_ForwardPlan(size_t size, FloatSignal &fs, ComplexSignal &cs)
    : FFT_Plan(fftwf_plan_dft_r2c_1d(size, fs.getData(), cs.getData(), FFTW_ESTIMATE)){}
};

class FFT_BackwardPlan : public FFT_Plan{
public:
  explicit FFT_BackwardPlan(size_t size, ComplexSignal &cs, FloatSignal &fs)
    : FFT_Plan(fftwf_plan_dft_c2r_1d(size, cs.getData(), fs.getData(), FFTW_ESTIMATE)){}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void make_and_export_fftw_wisdom(const string path_out, const size_t min_2pow=0,
                        const size_t max_2pow=25, const unsigned flag=FFTW_PATIENT){
  for(size_t i=min_2pow; i<=max_2pow; ++i){
    size_t size = pow(2, i);
    FloatSignal fs(size);
    ComplexSignal cs(size/2+1);
    printf("creating forward and backward plans for size=2**%zu=%zu and flag %u...\n", i, size, flag);
    FFT_ForwardPlan fwd(size, fs, cs);
    FFT_BackwardPlan bwd(size, cs, fs);
  }
  fftwf_export_wisdom_to_filename(path_out.c_str());
}

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

class SimpleXCORR {
public:
  //
  FloatSignal &s;
  ComplexSignal s_complex;
  //
  FloatSignal& p;
  ComplexSignal p_complex;
  //
  FloatSignal xcorr;
  ComplexSignal xcorr_complex;
  //
  FFT_ForwardPlan plan_forward_s;
  FFT_ForwardPlan plan_forward_p;
  FFT_BackwardPlan plan_backward_xcorr;
  ~SimpleXCORR(){}
  SimpleXCORR(FloatSignal &signal, FloatSignal &patch, const string wisdomPath="")
    : s(signal),
      s_complex(ComplexSignal(signal.getSize()/2+1)),
      p(patch),
      p_complex(ComplexSignal(patch.getSize()/2+1)),
      xcorr(FloatSignal(s.getSize())),
      xcorr_complex(xcorr.getSize()/2+1),
      plan_forward_s(signal.getSize(), signal, s_complex),
      plan_forward_p(p.getSize(), p, p_complex),
      plan_backward_xcorr(xcorr.getSize(), xcorr_complex, xcorr){
    if(!wisdomPath.empty()){
      import_fftw_wisdom(wisdomPath, false);
    }
  }
  void execute_xcorr(){
    // do ffts
    this->plan_forward_s.execute();
    this->plan_forward_p.execute();
    // multiply spectra
    spectral_correlation(s_complex, p_complex, xcorr_complex);
    // do ifft and normalize
    this->plan_backward_xcorr.execute();
    this->xcorr /= this->xcorr.getSize();
  }
};

// OVERLAP-SAVE METHOD:
// original mide W, patch mide M.
// 1. le metes al original M-1 por delante.
// 2. escoges L de manera q X=L+M-1 sea potencia de 2: X = pow2ceil(M)
// 3. padeas el patch de manera q mida X, es decir le metes L-1
// 4. calculas la FFT del patch (mide X)
// 5. calculas las FFT del original, con overlapping! es decir i=i+L
// 6. multiplicas cada uno de los chirmes, e inviertes el resultado
 // 7. de cada invertido, descartas los primeros (M-1), y concatenas todo


class OverlapSaveXCORR {
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
public:
  OverlapSaveXCORR(FloatSignal &signal, FloatSignal &patch, const string wisdomPath="")
    : signal_size(signal.getSize()),
      patch_size(patch.getSize()),
      xcorr_size(signal_size+patch_size-1),
      //
      padded_patch(patch.getData(), patch_size, 0, 2*pow2_ceil(patch_size)-patch_size),
      xcorr_chunksize(padded_patch.getSize()),
      xcorr_chunksize_complex(xcorr_size/2+1),
      xcorr_stride(xcorr_chunksize-patch_size+1),
      padded_patch_complex(xcorr_chunksize_complex),
      //
      padded_signal(signal.getData(),signal_size,patch_size-1, xcorr_chunksize-(xcorr_size%xcorr_stride)){
      //
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
    forward_plans.push_back(new FFT_ForwardPlan(xcorr_chunksize, padded_patch, padded_patch_complex));
    for (size_t i =0; i<s_chunks.size();i++){
      forward_plans.push_back(new FFT_ForwardPlan(xcorr_chunksize, *s_chunks.at(i), *s_chunks_complex.at(i)));
      backward_plans.push_back(new FFT_BackwardPlan(xcorr_chunksize, *xcorr_chunks_complex.at(i), *xcorr_chunks.at(i)));
    }
  }
  void execute_xcorr(){
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
      spectral_correlation(*s_chunks_complex.at(i), this->padded_patch_complex, *xcorr_chunks_complex.at(i));
    }
    // do iffts
    #ifdef WITH_OPENMP_ABOVE
    #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    #endif
    for (size_t i =0; i<xcorr_chunks.size();i++){
      backward_plans.at(i)->execute();
      *xcorr_chunks.at(i) /= this->xcorr_chunksize;
    }
    // for (size_t i =0; i<xcorr_chunks.size();i++){
    //   xcorr_chunks.at(i)->print(string("XCORR"+to_string(i)));
    // }
  }
  FloatSignal extract_current_xcorr(){
    const size_t offset = patch_size-1;
    FloatSignal result(xcorr_size);
    float* result_arr = result.getData();
    // collapse all the xcorr chunks into result except for the first one
    // NOT SURE IF PARALLELIZING THIS BECAUSE IT IS PURELY A MEMORY OPERATION
    // #ifdef WITH_OPENMP_ABOVE
    // #pragma omp parallel for schedule(static, WITH_OPENMP_ABOVE)
    // #endif
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
  ~OverlapSaveXCORR(){
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
  const string wisdomPatient = "wisdom_real_dft_pow2_patient"; // make_and_export_fftw_wisdom(wisdomPatient, 0, 29, FFTW_PATIENT);

  size_t o_size = 44100*10;
  float* o = new float[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 44100*1;
  float* m1 = new float[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=m1_size-i;}
  size_t xcorr_size = pow2_ceil(o_size+m1_size);

  FloatSignal s(o, o_size);
  FloatSignal p(m1, m1_size);

  OverlapSaveXCORR x(s, p);

  TStopwatch timer1;
  for(int k=0; k<1000; ++k){
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



// the optimizer inherits from the manager, but adds:
//   the xcorr field has to be adapted to hold the original at the beginning and be subtracted and such
//   the "picker" method which reads an ASCII config file (a cyclic sequence, a random dist, some heuristic...)
//   the "strategy" config, that tells, once having the xcorr, where to add the file
//   the "optimizer" method which following {picker_conf, strategy_conf, extra_parameters} performs the optimization and outputs list and wave (if some flag)
//
// add unit testing with catch
// BUG: calling FloatSignal(arr, size) where arr is a fftwf_alloc_real array causes the sys to freeze
// benchmark memset vs. multithreaded set USE A PROPER BENCHMARKING LIB
//


// TODO: now circular conv almost working, extra block seems to be missing.
