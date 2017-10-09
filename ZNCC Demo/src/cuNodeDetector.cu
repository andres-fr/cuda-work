//////////////////////////////////////////////////////////////////////////////////////////
//////////              Demo for real-time ZNCC                                   ////////
//////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2013  Institute of Imaging and Computer Vision, RWTH AACHEN      ////////
// Dorian Schneider (ds@lfb.rwth-aachen.de)                                       ////////
//////////////////////////////////////////////////////////////////////////////////////////
// Permission is hereby granted, free of charge, to any person obtaining          ////////
// a copy of this software and associated documentation files (the "Software"),   ////////
// to deal in the Software without restriction, including without limitation      ////////
// the rights to use, copy, modify, merge, publish, distribute, sublicense,       ////////
// and/or sell copies of the Software, and to permit persons to whom the          ////////
// Software is furnished to do so, subject to the following conditions:           ////////
// The above copyright notice and this permission notice shall be included        ////////
// in all copies or substantial portions of the Software.                         ////////
//                                                                                ////////
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS        ////////
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    ////////
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    ////////
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         ////////
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING        ////////
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER            ////////
// DEALINGS IN THE SOFTWARE.                                                      ////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "cuKernel.h"

#ifndef _DEBUG
	#define SHARED __shared__
#else
	#define SHARED
#endif

// ** * ********************************************************************* ** *
//                                CUDA KERNEL
// ** * ********************************************************************* ** *

__global__ void findMaxima(float* src, float* dst, int kernel, int w, int h) {
	 int id = threadIdx.x + blockIdx.x * blockDim.x;

     if(id < w*h){
		  SHARED float localmax;
		  SHARED int xfrom, xto;
		  SHARED int yfrom, yto;
		  SHARED int i,j, x, y;
		  SHARED int half;
		  SHARED int nid;

		 y    = id / w;
		 x    = id - y*w;
		 half = kernel/2;


		 // Clipping um die Ränder nicht zu überschreiten
		 xfrom = x - half < 0   ? 0   : x - half;
		 xto   = x + half > w-1 ? w-1 : x + half;

		 yfrom = y - half < 0   ? 0   : y - half;
		 yto   = y + half > h-1 ? h-1 : y + half;
		 
		 localmax = 0.0;

		 // Finden das Maximums unter dem Suchfenster
		 for( i = yfrom; i <= yto; i++) {
			  for( j = xfrom; j <= xto; j++) {

				  nid = i*w + j;				  
				  localmax = src[nid] > localmax ? src[nid] : localmax;
			  }
		 }

		 // ist Pixel Maximum unter dem Suchfesnter ?
		 dst[id] = (src[id] == localmax) ? src[id] : -1;
	 }
} // end findMaxima

__global__ void normalize (float* src, float* dst, float min, float max, int nPixels) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < nPixels){
		//dst[idx] = ((src[idx] + min) / (max + min) );
	}
} // end normalize

__global__ void denominator(float* dst, int* iimg, float* siimg, int w, int h, 
							int tw2l, int tw2r, int th2l, int th2r, float nPxlTempl, float tsigma, int nPixels ) {


	 ////////////////////////////////////////////////////
	 ///  Berechnet den Nennen gemäß Gl. 7 der Doku   ///
	 ///  mit Hilfe eines Integralbildes, dessen hier ///
	 ///  dessen Summen hier noch auf Templategröße   ///
	 ///  begrenzt werden müssen					  ///
	 ////////////////////////////////////////////////////
     int idx = threadIdx.x + blockIdx.x * blockDim.x;

     if(idx < nPixels){

		int y = idx / w ;
		int x = idx - y*w;

		float if2 = 0;
		float i2f = 0;

		int leftw  =  x-tw2l < 0   ? 0        : x-tw2l;  
		int rightw =  x+tw2r > w-1 ? w-1      : x+tw2r;
		int uph    =  y-th2l < 0   ? 0        : (y-th2l)*w;
		int downh  =  y+th2r > h-1 ? (h-1)*w  : (y+th2r)*w;

		int a, b, c, d;

		a = leftw  + uph;
		b = rightw + uph;
		c = leftw  + downh;
		d = rightw + downh;
		
		// Auf Template Größe begrenzen
		i2f = +  iimg[a] +   iimg[d]  -  iimg[b] -  iimg[c];
		if2 = + siimg[a] +  siimg[d] -  siimg[b] - siimg[c]; 
		
		i2f =  i2f*i2f; // ^2
		
		// vgl. Gl. (7) Doku Implementierung
		dst[idx] = ( ( sqrt( if2 -  i2f/nPxlTempl )*tsigma ) + 1.0e-10 );
		
	} // end if nPixels
} // end denominator


// ** * ********************************************************************* ** *
//                                C++ Interfaces
// ** * ********************************************************************* ** *
void cuDenominator( int nBlocks, int nThreads, float* dst, int* iimg,
					float* siimg, int w, int h, int tw, int th, int a, float tsigma ) {

	int th2l = floor(th/2.0);
	int th2r = ceil(th/2.0); 
	int tw2l = floor(tw/2.0);
	int tw2r = ceil(tw/2.0);

	denominator<<<nBlocks, nThreads>>>( dst, iimg, siimg, w, h, tw2l, tw2r, th2l, th2r, (float)a, tsigma, w*h);
}// end cuDenominator interface


void cuNormalize( int nBlocks, int nThreads,float* src, float* dst, float min, float max, int nPixels ) {
	normalize<<<nBlocks, nThreads>>>( src, dst, min, max, nPixels);
} // end cuNormalize Warpper

void cuNMS( int nBlocks, int nThreads, float* src, float* dst, int kernel, int w, int h ){
	findMaxima<<<nBlocks, nThreads>>>( src, dst, kernel, w, h);
} // end cuNMS Wrapper
