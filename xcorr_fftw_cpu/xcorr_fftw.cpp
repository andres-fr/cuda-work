// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v  ./test

// 3) run??
// g++ -std=c++11 -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v ./test



#include <string.h>
#include <math.h>
// #include <time.h>
#include <iostream>
#include <stdexcept>
#include <omp.h>

#include <fftw3.h>

#include "catch.hpp"

// #include <fstream>
// #include <cerrno>

// #include<vector>
// #include<assert.h>

#define REAL 0
#define IMAG 1

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


float* allocate_padded_aligned_real(const size_t size, const size_t padding_before,
                                    const size_t padding_after){
  float* result = fftwf_alloc_real(size+padding_before+padding_after);
  if(padding_before>0){memset(result, 0, sizeof(float)*padding_before);}
  if(padding_after>0){memset(result+size, 0, sizeof(float)*padding_after);}
  return result;
}


fftwf_complex* allocate_padded_aligned_complex(const size_t size, const size_t padding_before,
                                               const size_t padding_after){
  fftwf_complex* result = fftwf_alloc_complex(size+padding_before+padding_after);
  if(padding_before>0){memset(result, 0, sizeof(fftwf_complex)*padding_before);}
  if(padding_after>0){memset(result+size, 0, sizeof(fftwf_complex)*padding_after);}
  return result;
}

void complex_mul(const fftwf_complex a, const fftwf_complex b, fftwf_complex &result){
  // a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)
  result[REAL] = a[REAL]*b[REAL] - a[IMAG]*b[IMAG];
  result[IMAG] = a[REAL]*b[IMAG] + a[IMAG]*b[REAL];
}

void conjugate_mul(const fftwf_complex a, const fftwf_complex b, fftwf_complex &result){
  // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
  result[REAL] = a[REAL]*b[REAL] + a[IMAG]*b[IMAG];
  result[IMAG] =  a[IMAG]*b[REAL] - a[REAL]*b[IMAG];
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
  T &operator[](size_t idx){return data[idx];}
  T &operator[](size_t idx) const {return data[idx];}
  // void divideBy(const float x){
  //   const size_t start = this->padding_before;
  //   const size_t end = start+this->size;
  //   #pragma omp parallel for
  //   for(size_t i=start; i<end; ++i){this->data[i] /= x;}
  // }
  void print(){
    T* x = this->data;
    for(size_t i=0; i<size; ++i){
      cout << "signal[" << i << "]\t=\t" << x[i] << endl;
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
  ~FloatSignal() {fftwf_free(this->data);}
};

class ComplexSignal : public Signal<fftwf_complex>{
public:
  explicit ComplexSignal(size_t size)
    : Signal(fftwf_alloc_complex(size), size){}
  ~ComplexSignal(){fftwf_free(this->data);}
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


class Real_XCORR_Manager {
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
  //
  string wisdom_path;
  bool with_wisdom;

  ~Real_XCORR_Manager(){}
  Real_XCORR_Manager(FloatSignal &signal, FloatSignal &patch, const string wisdom_path="")
    : s(signal),
      s_complex(ComplexSignal(signal.getSize()/2+1)),
      p(patch),
      p_complex(ComplexSignal(patch.getSize()/2+1)),
      xcorr(FloatSignal(s.getSize())),
      xcorr_complex(xcorr.getSize()/2+1),
      plan_forward_s(signal.getSize(), signal, s_complex),
      plan_forward_p(p.getSize(), p, p_complex),
      plan_backward_xcorr(xcorr.getSize(), xcorr_complex, xcorr),
      wisdom_path(wisdom_path),
      with_wisdom(!wisdom_path.empty())
  {
    if(this->with_wisdom && fftwf_import_wisdom_from_filename(wisdom_path.c_str())==0){
      cout << "FFTW import wisdom was unsuccessfull. Is this ->"
           <<  wisdom_path << "<- a path to a valid FFTW wisdom file?";
      this->with_wisdom = false;
    }
  }

  void execute_xcorr(){
    this->plan_forward_s.execute();
    this->plan_forward_p.execute();
    fftwf_complex* s = this->s_complex.getData();
    fftwf_complex* p = this->p_complex.getData();
    fftwf_complex* x = this->xcorr_complex.getData();
    const size_t N = this->s_complex.getSize();
    x[0][REAL] = s[0][REAL]*p[0][REAL];
    #pragma omp parallel for
    for(size_t i=1; i<N; ++i){
      conjugate_mul(s[i], p[i], x[i]);
    }
    this->plan_backward_xcorr.execute();
    if(this->with_wisdom){fftwf_export_wisdom_to_filename(this->wisdom_path.c_str());}
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN ROUTINE
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,  char** argv){
  size_t o_size = 32; // 44100*3;
  float* o = new float[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 4; // 44100*0.5;
  float* m1 = new float[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=1;}


  FloatSignal s(o, o_size);
  FloatSignal p(m1, m1_size);
  Real_XCORR_Manager manager(s, p);

  // for(int k=0; k<100; ++k){
  //   cout << "iter no "<< k << endl;
  //   manager.execute_xcorr();
  // }
  // xcorr_signal.print()

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

  // FloatSignal c(o_size);
  // FloatSignal d(o, o_size);
  // d.print();
  // float* f = fftwf_alloc_real(100);
  // FloatSignal e(f, 100);
  // fftwf_free(f);
