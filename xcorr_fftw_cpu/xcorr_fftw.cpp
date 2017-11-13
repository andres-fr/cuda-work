// 1) configure your FFTW creating the "wisdom" (see f.e. https://www.systutorials.com/docs/linux/man/1-fftwf-wisdom/):
//    fftwf-wisdom -v -c -o wisdomfile DOESNT WORK! BAD FORMATTING? IS IT FFTW2?
//
// 2) compile and check run with valgrind (has "possibly lost" due to openMP, but it is ok)
//  g++ -Wall -Wextra xcorr_fftw.cpp -fopenmp -lfftw3 -o test && valgrind --leak-check=full -v  ./test

// 3) run??

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


double* allocate_padded_aligned_real(const size_t size, const size_t padding_size){
  double* result = fftw_alloc_real(size+padding_size);
  if(padding_size>0){
    memset(result+size, 0, sizeof(double)*padding_size);
  }
  return result;
}

fftw_complex* allocate_padded_aligned_complex(const size_t size, const size_t padding_size){
  fftw_complex* result = fftw_alloc_complex(size+padding_size);
  if(padding_size>0){
    memset(result+size, 0, sizeof(double)*padding_size);
  }
  return result;
}

void complex_mul(const fftw_complex a, const fftw_complex b, fftw_complex &result){
  // a+ib * c+id = ac+iad+ibc-bd = ac-bd + i(ad+bc)
  result[REAL] = a[REAL]*b[REAL] - a[IMAG]*b[IMAG];
  result[IMAG] = a[REAL]*b[IMAG] + a[IMAG]*b[REAL];
}

void conjugate_mul(const fftw_complex a, const fftw_complex b, fftw_complex &result){
  // a * conj(b) = a+ib * c-id = ac-iad+ibc+bd = ac+bd + i(bc-ad)
  result[REAL] = a[REAL]*b[REAL] + a[IMAG]*b[IMAG];
  result[IMAG] =  a[IMAG]*b[REAL] - a[REAL]*b[IMAG];
}

class Real_XCORR_Manager {
private:
  //
  double* signal;
  double* patch;
  size_t s_size;
  size_t p_size;
  //
  double* xcorr;
  size_t xcorr_size;
  size_t fft_size_real;
  size_t fft_size_complex;
  //
  fftw_complex* s_spectrum;
  fftw_complex* p_spectrum;
  fftw_complex* xcorr_spectrum;
  //
  fftw_plan plan_forward_s;
  fftw_plan plan_forward_p;
  fftw_plan plan_backward_s;
public:
  Real_XCORR_Manager(double* signal, const size_t s_size,
                     double* patch, const size_t p_size,
                     const string wisdom_path="wisdomfile"){
    if(fftw_import_wisdom_from_filename(wisdom_path.c_str())==0){
      throw invalid_argument("FFTW import wisdom was unsuccessfull. Do you have a valid wisdom in "+
                             wisdom_path+"?");
    }
    // grab sizes, calculate padded size and its complex counterpart
    this->s_size = s_size;
    this->p_size = p_size;
    this->fft_size_real = pow(2, ceil(log2(s_size)))*2;
    this->fft_size_complex = this->fft_size_real/2+1;
    // allocate real signals
    this->signal = allocate_padded_aligned_real(s_size, fft_size_real-s_size);
    this->patch  = allocate_padded_aligned_real(p_size, fft_size_real-p_size);
    this->xcorr  = allocate_padded_aligned_real(s_size, fft_size_real-s_size);
    // copy to allocated signals
    #pragma omp parallel for
    for(size_t i=0; i<s_size; ++i){
      this->signal[i] = signal[i]/fft_size_real; // divide to re-normalize XCORR
    }
    // memcpy(this->signal, signal, sizeof(double)*s_size);
    memcpy(this->patch, patch, sizeof(double)*p_size);
    // allocate complex signals
    this->s_spectrum = allocate_padded_aligned_complex(fft_size_complex, 0);
    this->p_spectrum = allocate_padded_aligned_complex(fft_size_complex, 0);
    this->xcorr_spectrum = allocate_padded_aligned_complex(fft_size_complex, 0);
    // create the FFT plans
    this->plan_forward_s  = fftw_plan_dft_r2c_1d(fft_size_real, this->signal,
                                                 this->s_spectrum, FFTW_ESTIMATE); // FFTW_ESTIMATE
    this->plan_forward_p  = fftw_plan_dft_r2c_1d(fft_size_real, this->patch,
                                                 this->p_spectrum, FFTW_ESTIMATE); // FFTW_ESTIMATE
    this->plan_backward_s = fftw_plan_dft_c2r_1d(fft_size_real, this->xcorr_spectrum,
                                                 this->xcorr, FFTW_ESTIMATE);
    // export updated wisdom
    fftw_export_wisdom_to_filename(wisdom_path.c_str());
  }
  ~Real_XCORR_Manager(){
    //
    fftw_free(this->signal);
    fftw_free(this->patch);
    fftw_free(this->xcorr);
    fftw_free(this->s_spectrum);
    fftw_free(this->p_spectrum);
    fftw_free(this->xcorr_spectrum);
    //
    fftw_destroy_plan(this->plan_forward_s);
    fftw_destroy_plan(this->plan_forward_p);
    fftw_destroy_plan(this->plan_backward_s);
  }
  void execute_xcorr(){
    fftw_execute(this->plan_forward_s);
    fftw_execute(this->plan_forward_p);
    #pragma omp parallel for
    for(size_t i=0; i<this->fft_size_complex; ++i){
      conjugate_mul(this->s_spectrum[i], this->p_spectrum[i], this->xcorr_spectrum[i]);
    }
    fftw_execute(this->plan_backward_s);
  }
  void print_xcorr(){
    for(size_t i=0; i<this->fft_size_real; ++i){
      cout << this->xcorr[i] << endl;
    }
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// MAIN ROUTINE
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,  char** argv){
  cout << "start" << endl;
  size_t o_size = 44100*3;
  double* o = new double[o_size];  for(size_t i=0; i<o_size; ++i){o[i] = i+1;}
  size_t m1_size = 44100*1;
  double* m1 = new double[m1_size]; for(size_t i=0; i<m1_size; ++i){m1[i]=1;}

  Real_XCORR_Manager manager(o, o_size, m1, m1_size);

  for(int k=0; k<100; ++k){
    cout << "iter no "<< k << endl;
    manager.execute_xcorr();
  }
  // manager.print_xcorr();

  delete[] o;
  delete[] m1;
  return 0;
}
