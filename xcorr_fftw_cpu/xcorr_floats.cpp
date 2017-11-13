// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3f -o test && valgrind --leak-check=full -v  ./test

// 3) run??
// g++ -Wall -Wextra xcorr_floats.cpp -fopenmp -lfftw3f -o test && ./test




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


float* allocate_padded_aligned_real(const size_t size, const size_t padding_size){
  float* result = fftwf_alloc_real(size+padding_size);
  if(padding_size>0){
    memset(result+size, 0, sizeof(float)*padding_size);
  }
  return result;
}

fftwf_complex* allocate_padded_aligned_complex(const size_t size, const size_t padding_size){
  fftwf_complex* result = fftwf_alloc_complex(size+padding_size);
  if(padding_size>0){
    memset(result+size, 0, sizeof(float)*padding_size);
  }
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


class FFT_Forward_Signal{
public:
  float* data;
  size_t size;
  size_t size_padded;
  size_t size_complex;
  //
  fftwf_complex* spectrum;
  fftwf_plan plan;
  FFT_Forward_Signal(size_t size){
    this->size = size;
    this->size_padded = pow(2, ceil(log2(size)))*2;
    this->size_complex = this->size_padded/2+1;
    this->data = allocate_padded_aligned_real(size, this->size_padded-size);
    //
    this->spectrum = allocate_padded_aligned_complex(this->size_complex, 0);
    this->plan = fftwf_plan_dft_r2c_1d(this->size_padded, this->data,
                                       this->spectrum, FFTW_ESTIMATE);
  }
  ~FFT_Forward_Signal(){
    fftwf_free(this->data);
    fftwf_free(this->spectrum);
    fftwf_destroy_plan(this->plan);
  }
};

class Real_XCORR_Manager {
private:
  //
  float* signal;
  size_t s_size;
  size_t s_size_padded;
  size_t s_size_complex;
  //
  float* patch;
  size_t p_size;
  size_t p_size_padded;
  size_t p_size_complex;
  //
  float* xcorr;
  //
  fftwf_complex* s_spectrum;
  fftwf_complex* p_spectrum;
  fftwf_complex* xcorr_spectrum;
  //
  fftwf_plan plan_forward_s;
  fftwf_plan plan_forward_p;
  fftwf_plan plan_backward_xcorr;
public:
  Real_XCORR_Manager(float* signal, const size_t s_size,
                     float* patch, const size_t p_size,
                     const string wisdom_path="wisdomfile"){
    if(fftwf_import_wisdom_from_filename(wisdom_path.c_str())==0){
      cout << "FFTW import wisdom was unsuccessfull. Do you have a valid wisdom in "
           <<  wisdom_path << "?";
    }
    // grab sizes, calculate padded size and its complex counterpart
    this->s_size = s_size;
    this->s_size_padded = pow(2, ceil(log2(s_size)))*2;
    this->s_size_complex = this->s_size_padded/2+1;
    this->p_size = p_size;
    this->p_size_padded = this->s_size_padded;// pow(2, ceil(log2(p_size)))*2;
    this->p_size_complex = this->p_size_padded/2+1;
    // allocate real signals (asume signal is longer than patch)
    this->signal = allocate_padded_aligned_real(s_size, this->s_size_padded-s_size);
    this->patch  = allocate_padded_aligned_real(p_size, this->p_size_padded-p_size);
    this->xcorr  = allocate_padded_aligned_real(s_size, this->s_size_padded-s_size);
    // copy to allocated signals
    // memcpy(this->signal, signal, sizeof(float)*s_size);
    // memcpy(this->patch, patch, sizeof(float)*p_size);
    #pragma omp parallel for
    for(size_t i=0; i<this->s_size; ++i){
      this->signal[i] = signal[i] /this->s_size_padded; // divide to re-normalize XCORR
    }
    #pragma omp parallel for
    for(size_t i=0; i<this->p_size; ++i){
      this->patch[i] = patch[i];//  /this->p_size_padded; // divide to re-normalize XCORR
    }

    // allocate complex signals
    this->s_spectrum = allocate_padded_aligned_complex(this->s_size_complex, 0);
    this->p_spectrum = allocate_padded_aligned_complex(this->p_size_complex, 0);
    this->xcorr_spectrum = allocate_padded_aligned_complex(this->s_size_complex, 0);
    // create the FFT plans
    this->plan_forward_s  = fftwf_plan_dft_r2c_1d(this->s_size_padded, this->signal,
                                                 this->s_spectrum, FFTW_ESTIMATE); // FFTW_ESTIMATE
    this->plan_forward_p  = fftwf_plan_dft_r2c_1d(this->p_size_padded, this->patch,
                                                 this->p_spectrum, FFTW_ESTIMATE); // FFTW_ESTIMATE
    this->plan_backward_xcorr = fftwf_plan_dft_c2r_1d(this->s_size_padded, this->xcorr_spectrum,
                                                      this->xcorr, FFTW_ESTIMATE);
    // export updated wisdom
    fftwf_export_wisdom_to_filename(wisdom_path.c_str());
  }
  ~Real_XCORR_Manager(){
    //
    fftwf_free(this->signal);
    fftwf_free(this->patch);
    fftwf_free(this->xcorr);
    fftwf_free(this->s_spectrum);
    fftwf_free(this->p_spectrum);
    fftwf_free(this->xcorr_spectrum);
    //
    fftwf_destroy_plan(this->plan_forward_s);
    fftwf_destroy_plan(this->plan_forward_p);
    fftwf_destroy_plan(this->plan_backward_xcorr);
  }
  void execute_xcorr(){
    const size_t sig_patch_ratio = this->s_size_padded / this->p_size_padded;
    fftwf_execute(this->plan_forward_s);
    fftwf_execute(this->plan_forward_p);
    this->xcorr_spectrum[0][REAL] = this->s_spectrum[0][REAL]*this->p_spectrum[0][REAL];
    #pragma omp parallel for
    for(size_t i=1; i<this->s_size_complex; ++i){
      if(i%sig_patch_ratio){ // the normal case, patch has no entry and therefore signal is taken directly
        this->xcorr_spectrum[i][REAL] = this->s_spectrum[i][REAL];
        this->xcorr_spectrum[i][IMAG] = this->s_spectrum[i][IMAG];
      }
      else{ // for the i%??==0 cases, patch has entry and gets multiplied
        conjugate_mul(this->s_spectrum[i], this->p_spectrum[i/sig_patch_ratio],
                      this->xcorr_spectrum[i]);
      }
    }
    fftwf_execute(this->plan_backward_xcorr);
  }
  void print_xcorr(){
    for(size_t i=0; i<this->s_size_padded; ++i){
      cout << this->xcorr[i] << endl;
    }
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
  manager.print_xcorr();

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
