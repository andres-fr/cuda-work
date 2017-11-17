// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v  ./test

// 3) run
// g++ -O3 -std=c++11 -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v ./test

#include <string.h>
#include <math.h>
// #include <time.h>
#include <iostream>
#include <stdexcept>
#include <omp.h>

#include <fftw3.h>

#include "catch.hpp"
#include "TStopwatch.h"
// #include <fstream>
// #include <cerrno>

// #include<vector>
// #include<assert.h>

#define REAL 0
#define IMAG 1
#define OMP_MIN_VALUE 1024 // this has to be benchmarked

using namespace std;





////////////////////////////////////////////////////////////////////////////////////////////////////
/// HELPERS
////////////////////////////////////////////////////////////////////////////////////////////////////


// string get_file_contents(const char *filename)
// {
//   ifstream in(filename, ios::in | ios::binary);
//   if (in){
//     std::string contents;
//     in.seekg(0, ios::end);
//     contents.resize(in.tellg());
//     in.seekg(0, ios::beg);
//     in.read(&contents[0], contents.size());
//     in.close();
//     return(contents);
//   }
//   throw(errno);
// }


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
  explicit FloatSignal(float* d, size_t size, size_t pad_bef, size_t pad_aft)
    : FloatSignal(size+pad_bef+pad_aft){
    memcpy(this->data+pad_bef, d, sizeof(float)*size);
  }
  ~FloatSignal() {fftwf_free(this->data);}
  //
  void operator+=(const float x){
    #pragma omp parallel for schedule(static, OMP_MIN_VALUE)
    for(size_t i=0; i<this->size; ++i){this->data[i] += x;}
  }
  void operator*=(const float x){
    #pragma omp parallel for schedule(static, OMP_MIN_VALUE)
    for(size_t i=0; i<this->size; ++i){this->data[i] *= x;}
  }
  void operator/=(const float x){
    #pragma omp parallel for schedule(static, OMP_MIN_VALUE)
    for(size_t i=0; i<this->size; ++i){this->data[i] /= x;}
  }
};

class ComplexSignal : public Signal<fftwf_complex>{
public:
  explicit ComplexSignal(size_t size)
    : Signal(fftwf_alloc_complex(size), size){}
  ~ComplexSignal(){fftwf_free(this->data);}
  void operator*=(const float x){
    #pragma omp parallel for schedule(static, OMP_MIN_VALUE)
    for(size_t i=0; i<this->size; ++i){
      this->data[i][REAL] *= x;
      this->data[i][IMAG] *= x;
    }
  }
  void operator+=(const float x){
    #pragma omp parallel for schedule(static, OMP_MIN_VALUE)
    for(size_t i=0; i<this->size; ++i){this->data[i][REAL] += x;}
  }
  void operator+=(const fftwf_complex x){
    #pragma omp parallel for schedule(static, OMP_MIN_VALUE)
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
  SimpleXCORR(FloatSignal &signal, FloatSignal &patch)
    : s(signal),
      s_complex(ComplexSignal(signal.getSize()/2+1)),
      p(patch),
      p_complex(ComplexSignal(patch.getSize()/2+1)),
      xcorr(FloatSignal(s.getSize())),
      xcorr_complex(xcorr.getSize()/2+1),
      plan_forward_s(signal.getSize(), signal, s_complex),
      plan_forward_p(p.getSize(), p, p_complex),
      plan_backward_xcorr(xcorr.getSize(), xcorr_complex, xcorr){}
  void execute_xcorr(){
    this->plan_forward_s.execute();
    this->plan_forward_p.execute();
    const size_t N = this->s_complex.getSize();
    xcorr_complex[0][REAL] = s_complex[0][REAL]*p_complex[0][REAL];
    #pragma omp parallel for
    for(size_t i=1; i<N; ++i){
      conjugate_mul(s_complex[i], p_complex[i], xcorr_complex[i]);
    }
    this->plan_backward_xcorr.execute();
    this->xcorr /= this->xcorr.getSize();
  }
};


// class OverlapSaveXCORR : public SimpleXCORR{
// private:
//   OverlapSaveXCORR(FloatSignal &signal, FloatSignal &patch){

//   }
// }



////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN ROUTINE
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,  char** argv){


  size_t o_size = 16;
  float* o = new float[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 10;
  float* m1 = new float[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=1;}


  size_t xcorr_size = pow2_ceil(o_size+m1_size);

  FloatSignal s(o, o_size, 0, xcorr_size-o_size);
  FloatSignal p(m1, m1_size, 0, xcorr_size-m1_size);

  SimpleXCORR sxc(s, p);

  // for(int k=0; k<100; ++k){
  //   cout << "iter no "<< k << endl;
  sxc.execute_xcorr();
  // }
  sxc.xcorr.print("XCORR");

  // FloatSignal(size_t size, size_t padding_before=0, size_t padding_after= 0)

  // // FloatSignal s(o, o_size, 0, pow(2, ceil(log2(o_size)))*2-o_size);
  // // ComplexSignal c(11);
  // // FFT_ForwardPlan pp(20, s, c);

  // FloatSignal d(o, o_size);
  // ComplexSignal e(o_size/2+1);
  // FFT_ForwardPlan(o_size, d, e);
  // // FFT_ForwardPlan k(o_size, d, e);
  // // FFT_Plan pp(fftwf_plan_dft_r2c_1d(o_size, d.getData(), e.getData(), FFTW_ESTIMATE));
  // // fftwf_execute(p);
  // // fftwf_destroy_plan(p);




  delete[] o;
  delete[] m1;
  cout << "done!" << endl;
  return 0;

}


// TODO:
// overlap-save: http://www.comm.utoronto.ca/~dkundur/course_info/real-time-DSP/notes/8_Kundur_Overlap_Save_Add.pdf
// add support for multiple "materials": maybe a "fft_signal" class with
//    the array, its length, its padded len, the complex len, the complex array and the fft plan
// the fft manager holds a signal for the original and a vector<signal> for the materials

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



// OVERLAP-SAVE METHOD:
// original mide W, patch mide M.
// 1. le metes al original M-1 por delante.
// 2. escoges L de manera q X=L+M-1 sea potencia de 2
// 3. padeas el patch de manera q mida X, es decir le metes L-1
// 4. calculas la FFT del patch (mide X)
// 5. calculas las FFT del original, con overlapping! es decir i=i+L
// 6. multiplicas cada uno de los chirmes, e inviertes el resultado
// 7. de cada invertido, descartas los primeros (M-1), y concatenas todo
