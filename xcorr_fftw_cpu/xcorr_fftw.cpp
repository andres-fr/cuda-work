// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v  ./test

// 3) run??
// g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && ./test




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


// class ComplexSignal{
// public:
//   float* data;
//   size_t size;
//   size_t size_padded;
//   size_t size_complex;
//   //
//   fftwf_complex* spectrum;
//   fftwf_plan plan;
//   FFT_Forward_Signal(size_t size){
//     this->size = size;
//     this->size_padded = pow(2, ceil(log2(size)))*2;
//     this->size_complex = this->size_padded/2+1;
//     this->data = allocate_padded_aligned_real(size, 0, this->size_padded-size);
//     //
//     this->spectrum = allocate_padded_aligned_complex(this->size_complex, 0, 0);
//     this->plan = fftwf_plan_dft_r2c_1d(this->size_padded, this->data,
//                                        this->spectrum, FFTW_ESTIMATE);
//   }
//   ~FFT_Forward_Signal(){
//     fftwf_free(this->data);
//     fftwf_free(this->spectrum);
//     fftwf_destroy_plan(this->plan);
//   }
// };

class FloatSignal{
public:
  float* data;
  size_t size;
  size_t padding_before;
  size_t padding_after;
  size_t size_padded;
  explicit FloatSignal(float* data, size_t size, size_t padding_before=0, size_t padding_after= 0)
    : size(size), padding_before(padding_before), padding_after(padding_after) {
    this->size_padded = size+padding_before+padding_after;
    this->data = fftwf_alloc_real(size+padding_before+padding_after);
    if(padding_before>0){memset(this->data, 0, sizeof(float)*padding_before);}
    if(data!=NULL){memcpy(this->data, data, sizeof(fftwf_complex)*size);}
    if(padding_after>0){memset(this->data+padding_before+size, 0, sizeof(float)*padding_after);}
  }
  explicit FloatSignal(size_t size, size_t padding_before=0, size_t padding_after= 0)
    : FloatSignal(NULL, size, padding_before, padding_after){}
  ~FloatSignal(){fftwf_free(this->data);}
  // deleted copy- and assignment constructors
  FloatSignal(const FloatSignal& copyFrom)=delete;
  FloatSignal& operator= (const FloatSignal& copyFrom)=delete;
  //
  void divideBy(const float x){
    const size_t start = this->padding_before;
    const size_t end = start+this->size;
    #pragma omp parallel for
    for(size_t i=start; i<end; ++i){this->data[i] /= x;}
  }
  void print(){
    const float* x = this->data;
    for(size_t i=0; i<size_padded; ++i){
      printf("signal[%zu]=%f", i, x[i]);
    }
  }
};

class ComplexSignal{
public:
  fftwf_complex* data;
  size_t size;
  explicit ComplexSignal(fftwf_complex* data, size_t size) : size(size) {
    this->data = fftwf_alloc_complex(size);
    if(data!=NULL){memcpy(this->data, data, sizeof(fftwf_complex)*size);}
  }
  explicit ComplexSignal(size_t size) : ComplexSignal(NULL, size){}
  ~ComplexSignal(){fftwf_free(this->data);}
  // deleted copy- and assignment constructors
  ComplexSignal(const ComplexSignal& copyFrom)=delete;
  ComplexSignal& operator= (const ComplexSignal& copyFrom)=delete;
};



class FFT_ForwardPlan{
private:
  fftwf_plan plan;
public:
  explicit FFT_ForwardPlan(const size_t size, FloatSignal fs, ComplexSignal cs,
                            const unsigned flags=FFTW_ESTIMATE){
    this->plan = fftwf_plan_dft_r2c_1d(size, fs.data, cs.data, flags);
  }
  ~FFT_ForwardPlan(){fftwf_destroy_plan(this->plan);}
  // deleted copy- and assignment constructors
  FFT_ForwardPlan(const FFT_ForwardPlan& copyFrom)=delete;
  FFT_ForwardPlan& operator= (const FFT_ForwardPlan& copyFrom)=delete;
  //
  void execute(){fftwf_execute(this->plan);}
};

class FFT_BackwardPlan{
private:
  fftwf_plan plan;
public:
  explicit FFT_BackwardPlan(const size_t size, ComplexSignal cs, FloatSignal fs,
                            const unsigned flags=FFTW_ESTIMATE){
    this->plan = fftwf_plan_dft_r2c_1d(size, cs.data, fs.data, flags);
  }
  ~FFT_BackwardPlan(){fftwf_destroy_plan(this->plan);}
  // deleted copy- and assignment constructors
  FFT_BackwardPlan(const FFT_BackwardPlan& copyFrom)=delete;
  FFT_BackwardPlan& operator= (const FFT_BackwardPlan& copyFrom)=delete;
  //
  void execute(){fftwf_execute(this->plan);}
};

  
class Real_XCORR_Manager {
public:
  //
  FloatSignal s_signal;
  ComplexSignal s_signal_complex;
  //
  FloatSignal p_signal;
  ComplexSignal p_signal_complex;
  //
  FloatSignal xcorr_signal;
  ComplexSignal xcorr_signal_complex;
  //
  FFT_BackwardPlan plan_forward_s;
  FFT_ForwardPlan plan_forward_p;
  FFT_BackwardPlan plan_backward_xcorr;

  Real_XCORR_Manager(float* signal, const size_t s_size,
                     float* patch, const size_t p_size,
                     const string wisdom_path="wisdomfile"){
    if(fftwf_import_wisdom_from_filename(wisdom_path.c_str())==0){
      cout << "FFTW import wisdom was unsuccessfull. Do you have a valid wisdom in "
           <<  wisdom_path << "?";
    }
    //
    size_t s_size_padded = pow(2, ceil(log2(s_size)))*2;
    size_t s_size_complex = s_size_padded/2+1;
    this->s_signal = FloatSignal(signal, s_size, 0, s_size_padded-s_size);
    this->s_signal_complex = ComplexSignal(s_size_complex);
    //
    size_t p_size_padded = s_size_padded;// pow(2, ceil(log2(p_size)))*2;
    size_t p_size_complex = p_size_padded/2+1;
    this->p_signal = FloatSignal(signal, p_size, 0, p_size_padded-p_size);
    this->p_signal_complex = ComplexSignal(p_size_complex);
    //
    this->xcorr_signal = FloatSignal(s_size_padded);
    this->xcorr_signal_complex = ComplexSignal(s_size_complex);
    // create the FFT plans
    this->plan_forward_s = FFT_ForwardPlan(s_size_padded, this->s_signal, this->s_signal_complex);
    this->plan_forward_p = FFT_ForwardPlan(p_size_padded, this->p_signal, this->p_signal_complex);
    this->plan_backward_xcorr = FFT_BackwardPlan(s_size_padded, this->xcorr_signal_complex,
                                                 this->xcorr_signal);
    // export updated wisdom
    fftwf_export_wisdom_to_filename(wisdom_path.c_str());
  }

  void execute_xcorr(){
    this->plan_forward_s.execute();
    this->plan_forward_p.execute();
    fftwf_complex* s = this->s_signal_complex.data;
    fftwf_complex* p = this->p_signal_complex.data;
    fftwf_complex* x = this->xcorr_signal_complex.data;
    const size_t N = this->s_signal_complex.size;
    x[0][REAL] = s[0][REAL]*p[0][REAL];
    #pragma omp parallel for
    for(size_t i=1; i<N; ++i){
      conjugate_mul(s[i], p[i], x[i]);
    }
    this->plan_backward_xcorr.execute();
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN ROUTINE
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,  char** argv){
  cout << "start" << endl;
  size_t o_size = 20; // 44100*3;
  float* o = new float[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 3; // 44100*0.5;
  float* m1 = new float[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=1;}

  Real_XCORR_Manager manager(o, o_size, m1, m1_size);

  for(int k=0; k<1000; ++k){
    cout << "iter no "<< k << endl;
    manager.execute_xcorr();
  }
  manager.xcorr_signal.print();

  delete[] o;
  delete[] m1;
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
