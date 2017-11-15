// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v  ./test

// 3) run??
// g++ -std=c++11 -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v ./test



#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include <omp.h>

#include <fftw3.h>

#include <fstream>
#include <cerrno>

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
  size_t size;
  T* data;
public:
  Signal(size_t size, T* data) : size(size), data(data){}
  virtual ~Signal(){}
  size_t getSize(){return this->size;}
  T* getData(){return this->data;}
  T &operator[](int idx){return data[idx];}
  T &operator[](int idx) const {return data[idx];}
  // void divideBy(const float x){
  //   const size_t start = this->padding_before;
  //   const size_t end = start+this->size;
  //   #pragma omp parallel for
  //   for(size_t i=start; i<end; ++i){this->data[i] /= x;}
  // }
  void print(){
    const float* x = this->data;
    for(size_t i=0; i<size; ++i){
      printf("signal[%zu]=%f", i, x[i]);
    }
  }
};


class FloatSignal : public Signal<float>{
private:
  size_t set_and_return_size_padded(size_t sz, size_t p_bef, size_t p_aft){
    this->size_padded = sz+p_bef+p_aft;
    return this->size_padded;
  }
public:
  size_t padding_before;
  size_t padding_after;
  size_t size_padded;
  explicit FloatSignal(size_t size, size_t padding_before=0, size_t padding_after= 0)
    : Signal(size, fftwf_alloc_real(set_and_return_size_padded(size, padding_before,padding_after))),
      padding_before(padding_before),
      padding_after(padding_after){}
  explicit FloatSignal(float* data, size_t size, size_t padding_before=0, size_t padding_after= 0)
    : FloatSignal(size, padding_before, padding_after){
    if(padding_before>0){memset(this->data, 0, sizeof(float)*padding_before);}
    if(data!=NULL){memcpy(this->data, data, sizeof(fftwf_complex)*size);}
    if(padding_after>0){memset(this->data+padding_before+size, 0, sizeof(float)*padding_after);}
  }
  size_t getSizePadded(){return this->size_padded;}
  ~FloatSignal(){fftwf_free(this->data);}
};


class ComplexSignal : public Signal<fftwf_complex>{
public:
  explicit ComplexSignal(size_t size)
    : Signal(size, fftwf_alloc_complex(size)){};
  explicit ComplexSignal(fftwf_complex* data, size_t size) :ComplexSignal(size){
    memcpy(this->data, data, sizeof(fftwf_complex)*size);
  }
  ~ComplexSignal(){fftwf_free(this->data);}
};

class FFT_Plan{
private:
  fftwf_plan plan;
public:
  explicit FFT_Plan(fftwf_plan p): plan(p){}
  ~FFT_Plan(){fftwf_destroy_plan(this->plan);}
  void execute(){fftwf_execute(this->plan);}
};

class FFT_ForwardPlan : public FFT_Plan{
public:
  explicit FFT_ForwardPlan(size_t size, FloatSignal fs, ComplexSignal cs)
    : FFT_Plan(fftwf_plan_dft_r2c_1d(size, fftwf_alloc_real(10000), fftwf_alloc_complex(5001), FFTW_ESTIMATE)){} //  // fftwf_plan_dft_r2c_1d(size, fs.getData(), cs.getData(), FFTW_ESTIMATE)
};

class FFT_BackwardPlan : public FFT_Plan{
public:
  explicit FFT_BackwardPlan(size_t size, ComplexSignal cs, FloatSignal fs)
    : FFT_Plan(fftwf_plan_dft_c2r_1d(size, cs.getData(), fs.getData(), FFTW_ESTIMATE)){}
};


class Real_XCORR_Manager {
public:
  //
  FloatSignal s_signal;
  ComplexSignal s_signal_complex;
  // //
  // FloatSignal p_signal;
  // ComplexSignal p_signal_complex;
  // //
  // FloatSignal xcorr_signal;
  // ComplexSignal xcorr_signal_complex;
  // //
  FFT_ForwardPlan plan_forward_s;
  // FFT_ForwardPlan plan_forward_p;
  // FFT_BackwardPlan plan_backward_xcorr;
  // //
  // string wisdom_path;
  // bool with_wisdom;

  Real_XCORR_Manager(float* signal, const size_t s_size,
                     float* patch, const size_t p_size,
                     const string wisdom_path="")
    : s_signal(signal, s_size, 0, pow(2, ceil(log2(s_size)))*2-s_size),
      s_signal_complex(s_signal.getSizePadded()/2+1),
      // p_signal(patch, p_size, 0, s_signal.getSizePadded()-p_size),
      // p_signal_complex(p_signal.getSizePadded()/2+1),
      // xcorr_signal(s_signal.getSizePadded()),
      // xcorr_signal_complex(s_signal_complex.getSize()),
      plan_forward_s(s_signal.getSizePadded(), s_signal, s_signal_complex)
      // plan_forward_p(p_signal.getSizePadded(), p_signal, p_signal_complex),
      // plan_backward_xcorr(xcorr_signal.getSizePadded(), xcorr_signal_complex, xcorr_signal),
      // wisdom_path(wisdom_path),
      // with_wisdom(!wisdom_path.empty())
  {
    // if(this->with_wisdom && fftwf_import_wisdom_from_filename(wisdom_path.c_str())==0){
    //   cout << "FFTW import wisdom was unsuccessfull. Is this ->"
    //        <<  wisdom_path << "<- a path to a valid FFTW wisdom file?";
    //   this->with_wisdom = false;
    // }
  }

  // void execute_xcorr(){
  //   this->plan_forward_s.execute();
  //   this->plan_forward_p.execute();
  //   fftwf_complex* s = this->s_signal_complex.getData();
  //   fftwf_complex* p = this->p_signal_complex.getData();
  //   fftwf_complex* x = this->xcorr_signal_complex.getData();
  //   const size_t N = this->s_signal_complex.getSize();
  //   x[0][REAL] = s[0][REAL]*p[0][REAL];
  //   #pragma omp parallel for
  //   for(size_t i=1; i<N; ++i){
  //     conjugate_mul(s[i], p[i], x[i]);
  //   }
  //   this->plan_backward_xcorr.execute();
  //   if(this->with_wisdom){fftwf_export_wisdom_to_filename(this->wisdom_path.c_str());}
  // }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN ROUTINE
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,  char** argv){
  size_t o_size = 20; // 44100*3;
  float* o = new float[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 3; // 44100*0.5;
  float* m1 = new float[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=1;}

  fftwf_plan p = fftwf_plan_dft_r2c_1d(20, fftwf_alloc_real(10000), fftwf_alloc_complex(5001), FFTW_ESTIMATE);

  // Real_XCORR_Manager manager(o, o_size, m1, m1_size);

  // for(int k=0; k<100; ++k){
  //   cout << "iter no "<< k << endl;
  //   manager.execute_xcorr();
  // }
  // manager.xcorr_signal.print();

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
