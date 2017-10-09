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
// INCLUDES CUDA
#include <cufft.h>

//INCLUDES SYSTEM
#include <vector>
#include <math.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <fstream>

//INCLUDES OPENCV
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

// CUDA KERNEL
#include "cuKernel.h"

using namespace cv;
using namespace std;

#ifdef _WINDLL
	#define EXPORT_DLL __declspec(dllexport)
#else
	#define EXPORT_DLL __declspec(dllimport)
#endif

/////////////////////////////////////////////
///  node Config Struct /////////////////////
/////////////////////////////////////////////
struct nodeConfig{
	int common_imageHeight;			// Height of the input image
	int common_imageWidth;		    // Width of the input image
	std::string imageLoadPath;		// Path to image
	std::string templateLoadPath;	// Path to template
	std::string imageSavePath;		// Path where image is saved
	bool showImage;					// Show image after processing

	double npd_resize;					// resize factor
	int npd_padih;						// padded image height
	int npd_padiw;						// padded image width
	int npd_tplh;						// template image height
	int npd_tplw;						// template image width
	int npd_strel;						// dilation element size
	int npd_offsets[2];					// x-y Offsets for node points
	double ya_minXcorr;
};

struct NodePoint {
		// Konstruktoren
		NodePoint(int x_, int y_) : xi(x_), yi(y_), idx(0), isSeed(0), nhLevel(0), found(0), x(0), y(0), xcorrVal(0.0){} 
		NodePoint(int x_, int y_, int idx_) : xi(x_), yi(y_), idx(idx_), isSeed(0), nhLevel(0), found(0), x(0), y(0), xcorrVal(0.0){} 
		NodePoint(int x_, int y_, int idx_, double xcor) : xi(x_), yi(y_), idx(idx_), isSeed(0), nhLevel(0), found(0), x(0), y(0), xcorrVal(xcor){} 
		NodePoint(): xi(0), yi(0), idx(0), isSeed(0), nhLevel(0), found(0), x(0), y(0), xcorrVal(0.0) {}

		int xi;	             // X-Koordinate im Bild
		int yi;				 // Y-Koordinate im Bild
		int idx;             // Index des Knotens in der Knotenliste
		int x;               // X-Koordinate des Knotens in der Fadenmatrix
		int y;               // Y-Koordinate des Knotens in der Fadenmatrix
		int nhLevel;         // Nutzung in der QueryPoint Liste, gibt an welcher Vek addiert werden muss
		bool isSeed;         // Ist der KP ein Saatpunkt ?
		bool found;          // Wurde der KP verarbeitet ?
		double xcorrVal;     // Korrelationswert 
};

class EXPORT_DLL NodeDetector {
	private:
		// ******************************************
		//      private Membervariablen
		// ******************************************
		vector<NodePoint> nodes;                // Gefundene Knotenpunkte
		cufftHandle fftPlan_FW, fftPlan_BW;		// Vor- & Rückwärts Plan für FFT
		Size dftSize, padSize;					// Größen für Padding
		Size iipad, normalSize;

		gpu::GpuMat tplTime;					// Template (Zeitbereich)
		gpu::GpuMat tplFFT;						// Template (Frequenzbereich, vergrößert)

		float tSigma;							// t_sigma Term, vgl. Punkt 1. Implementierung Doku

		gpu::GpuMat iimg;						// Integralbild
		gpu::GpuMat siimg;						// Quadriertes Integralbild

		gpu::GpuMat imgFFT,imgGPU, imgPAD;		// GPU Matrizen für das Nennerbild
		gpu::GpuMat numerator, denominator;		// Zähler- und Nennerbild
		gpu::GpuMat xcorrRes;					// Ergebnisbild
		gpu::GpuMat nodeMap;					// Bild mit den Knotenpunkten

		// ******************************************
		//      private Memberfunktionen
		// ******************************************
		void fft(gpu::GpuMat& src, gpu::GpuMat& dst, bool ifft = 0);
		void xcorr(const gpu::GpuMat &img);

		// ******************************************
		//      Helfer
		// ******************************************
		Mat  rotate180(const Mat& img);
		int  padding(gpu::GpuMat &src, gpu::GpuMat &dst);
		void swapQuadrants(gpu::GpuMat& img);
		void reset();

	public:	
		nodeConfig cfg;

		NodeDetector(const nodeConfig c, Mat tmpl);
		~NodeDetector();

		bool detect(const gpu::GpuMat &img);
		vector<NodePoint>& getNodes();

		void printNodePoints(Mat &img, double resize, double circleSize =1.0);
		void saveNodePointsCSV(string fileName);
};