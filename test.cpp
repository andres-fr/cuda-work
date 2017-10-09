#include <math.h>
#include <iostream>
#include <string.h>

using namespace std;


int pow2ceil(double x){
  return pow(2, ceil(log2(x)));
}

template<typename T>
T copyToDev(const T arr_pointer, const size_t size, const size_t padding_size){
  int in_memsize = size*sizeof(T);
  int pad_memsize = padding_size*sizeof(T);
  T result = (T)malloc(in_memsize+pad_memsize);
  memcpy(result, arr_pointer, in_memsize);
  memset(result+size, 0, pad_memsize); // zero-pad
  return result;
}


template<typename T>
void divideBy(T* arr, const size_t size, const T &div){
  for(unsigned int i=0; i<size; ++i){
    arr[i] /= div;
  }
}

int main(){
  size_t size = 10;
  int* a = new int[size];
  float* b = new float[size];
  for(unsigned int i=0; i<size; ++i){
    a[i] = 10*i;
    b[i] = 10*i;
  }

  size_t out_size = pow2ceil(size);
  int* c = copyToDev(a, size, out_size-size);
  float* d = copyToDev(b, size, out_size-size);
  divideBy(c, size, 2);
  divideBy(d, size, 5.f);

  printf("SIZE(in/out): %zu, %zu\n", size, out_size);
  for(size_t i=0; i<out_size; ++i){
    cout << "i: " << i << "\tc: " << c[i] << "\td: " << d[i] << endl;
  }
}
