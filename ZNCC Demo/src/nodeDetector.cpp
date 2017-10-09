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
#include "nodeDetector.h"

void showGpuImg(gpu::GpuMat img) {
	Mat showRes = img;
	namedWindow("",2);
	imshow("", showRes);
	waitKey(0);
}

void showImg(Mat img) {
	namedWindow("",2);
	imshow("", img);
	waitKey(0);
}

#include <sstream>
std::string int2Str(int number){
	std::stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	return ss.str();//return a string with the contents of the stream
}

NodeDetector::NodeDetector(const nodeConfig c, Mat tmpl) {

	// Konfiguration speichern
	this->cfg = c;

	this->cfg.common_imageWidth  = c.common_imageWidth*c.npd_resize;
	this->cfg.common_imageHeight = c.common_imageHeight*c.npd_resize;
	this->cfg.npd_padih      = c.npd_padih*c.npd_resize;
	this->cfg.npd_padiw      = c.npd_padiw*c.npd_resize;
	this->cfg.npd_tplh       = c.npd_tplh*c.npd_resize;
	this->cfg.npd_tplw       = c.npd_tplw*c.npd_resize;
	this->cfg.npd_strel      = c.npd_strel*c.npd_resize;

	// Std & Mittelwert des Templates
	Scalar tmean, tstd; 

	// Template verkleinern
	resize(tmpl,tmpl, Size(cfg.npd_tplw, cfg.npd_tplh)); 

	// In Float konvertieren
	tmpl.convertTo(tmpl, CV_32FC1);

	// Größe des Eingangsbildes
	this->normalSize = Size(this->cfg.common_imageWidth,   
							this->cfg.common_imageHeight
				           );

	// Absolute Größe der FFT Matrix
	this->dftSize = Size( this->cfg.npd_padiw/2 +1,   
						  this->cfg.npd_padih
				        );
	// Absolute Größe des gepaddeten Bildes
	this->padSize = Size(this->cfg.npd_padiw,		 
						 this->cfg.npd_padih
						);

	// Absolute Größe des gepaddeten Integralbildes
	this->iipad = Size(this->cfg.common_imageWidth + 1,
					   this->cfg.common_imageHeight + 1
					  );

	// Leere GPU Matrizen für DFT Template anlegen
	this->tplTime     = gpu::createContinuous(padSize, CV_32F);
	this->tplFFT      = gpu::createContinuous(this->dftSize, CV_32FC2 );

	// Leere GPU Matrizen für DFT Nenner anlegen
	this->imgFFT      = gpu::createContinuous(this->dftSize, CV_32FC2);
	this->imgGPU      = gpu::createContinuous(this->padSize, CV_32F);
	this->imgPAD      = gpu::createContinuous(this->padSize, CV_32F);
	this->numerator   = gpu::createContinuous(this->padSize, CV_32FC1);
	this->denominator = gpu::createContinuous(this->iipad, CV_32FC1);

	// Intergralbilder anlegen
	this->iimg        = gpu::createContinuous(this->iipad, CV_32SC1);
	this->siimg       = gpu::createContinuous(this->iipad, CV_32FC1);

	// Ergebnisbild
	this->xcorrRes    = gpu::createContinuous(this->normalSize, CV_32SC1);

	// Mittelwert & Stdv des Templates berechnen
	meanStdDev(tmpl,tmean, tstd);

	// Template 180° drehen (konj. komplex), Mittelwert abziehen,  padden
	padding(gpu::GpuMat(rotate180(tmpl) - tmean), this->tplTime);

	swapQuadrants(this->tplTime);

	// FFT Pläne anlegen
	cufftPlan2d(&this->fftPlan_FW, this->padSize.height, this->padSize.width, CUFFT_R2C);
	cufftPlan2d(&this->fftPlan_BW, this->padSize.height, this->padSize.width, CUFFT_C2R);
	
	// Mittelwertfreies Template in Frequenzbereich transformieren
	fft(this->tplTime, this->tplFFT);

	// t_sigma Term berechnen, vgl. Doku Implementierung Punkt 1
	this->tSigma = sqrt((float)(tmpl.cols*tmpl.rows))*tstd.val[0];


	// **********************************************
	//	 Testen ob die FFT funktioniert
	// **********************************************
		gpu::GpuMat diffImg;
		gpu::GpuMat dResult = gpu::createContinuous(padSize, CV_32F);
	
		fft(this->tplFFT, dResult, 1);
	
		gpu::multiply(dResult, Scalar::all(1.0f/this->padSize.area() ), dResult);
		gpu::absdiff(dResult, this->tplTime, diffImg);
		double totalDifference = gpu::sum(diffImg).val[0];
		assert(totalDifference < 10);
	// ********************************************** 

} // end constructor

void NodeDetector::saveNodePointsCSV(string fileName){
	ofstream outputFile;
	outputFile.open(fileName);

	string nodeString = "x-coordinate;y-coordinate\n";;
	for(int i = 0; i < this->nodes.size(); i++) {
		nodeString += int2Str(this->nodes[i].xi) + "," + int2Str(this->nodes[i].yi) +"\n";
	}

	outputFile << nodeString;

	outputFile.close();
}

bool NodeDetector::detect(const gpu::GpuMat &img){
	reset();

	// *********************************************
	//        Normierte Kreuzkorrelation
	// *********************************************
	xcorr(img);


	// *********************************************
	//  Dilatation & Vergleich: Non-Max-Suppression
	// *********************************************
	this->nodeMap = gpu::GpuMat(this->normalSize, CV_32F, this->xcorrRes.step1());

	int elems = this->xcorrRes.step1()*this->xcorrRes.rows;

	int cuNrBlocks   = std::ceil( elems / 512.0 );
	int cuNrThreads  = std::ceil( elems / (float)cuNrBlocks ); 	
		
	cuNMS( cuNrBlocks, 
		   cuNrThreads, 
		   this->xcorrRes.ptr<float>(), 
		   this->nodeMap.ptr<float>(), 
		   this->cfg.npd_strel, 
		   this->xcorrRes.step1(),
		   this->xcorrRes.rows
		  );
	
	
	// *********************************************
	//			Knotenpunkte extrahieren
	// *********************************************
	Mat hNodePoints = this->nodeMap;

	double iscale = 1.0/this->cfg.npd_resize;

	for(int i = 0; i < hNodePoints.rows; i++) {
		for(int j = 0; j < hNodePoints.cols; j++) {
			
			if(  hNodePoints.at<float>(i,j) >= this->cfg.ya_minXcorr &&
			     hNodePoints.at<float>(i,j) <= 1.0 
			  ) {

				NodePoint newNP( j*iscale + this->cfg.npd_offsets[1],
								 i*iscale + this->cfg.npd_offsets[0], 
								 this->nodes.size(),
								 (double)hNodePoints.at<float>(i,j)
							   ); 

				this->nodes.push_back( newNP );
			} // end if
		} // end for j
	} // end for i

    return false;

} // end fun detect


void NodeDetector::xcorr(const gpu::GpuMat &img) {
	gpu::GpuMat alignedImg;

	if( img.step1() % 32 != 0) {
		// falsche Speicherausrichtung
		// resize Funktion braucht 32 byte Alignment
		alignedImg = gpu::GpuMat(img.size(), CV_8U);
		img.copyTo(alignedImg);
	} else {
		alignedImg = img;
	}
	
	gpu::resize(alignedImg, this->imgGPU, this->normalSize );


	// *********************************************
	//        Integralbild berechnen
	// *********************************************
	gpu::integral(this->imgGPU, this->iimg, this->siimg);

	
	// *********************************************
	//        Zähler berechnen
	// *********************************************
	this->imgGPU.convertTo(this->imgGPU, CV_32F);
	padding(this->imgGPU, this->imgPAD);

	fft(this->imgPAD, this->imgFFT);

	#ifdef _DEBUG
		// Testen ob die FFT funktioniert
		gpu::GpuMat dResult = gpu::createContinuous(this->padSize, CV_32F);
		fft(imgFFT, dResult, 1);	
		gpu::GpuMat diffImg;
		gpu::multiply(dResult, Scalar::all(1.0f/this->padSize.area() ), dResult);
		gpu::absdiff(dResult, this->imgPAD, diffImg);

		double test = gpu::sum(diffImg).val[0];
		assert(gpu::sum(diffImg).val[0] < 255);
	#endif

	gpu::mulSpectrums(this->imgFFT, this->tplFFT, this->imgFFT,0);

	fft(this->imgFFT, this->numerator,1);

	gpu::multiply(this->numerator, Scalar::all(1.0f/this->padSize.area() ), this->numerator);

	// Padding-Ränder abschneiden, um zu registrieren
	int xStart = (this->padSize.width - this->normalSize.width)/2 ;
	int yStart = (this->padSize.height - this->normalSize.height)/2 ;

	gpu::GpuMat num = this->numerator(Rect(xStart,
							               yStart,
										   this->cfg.common_imageWidth, 
							               this->cfg.common_imageHeight  ));


	// *********************************************
	//      Nenner berechnen
	// *********************************************

	#ifdef _DEBUG
		assert(this->denominator.isContinuous());
	    assert(this->iimg.isContinuous());
		assert(this->siimg.isContinuous());
	#endif

	// CUDA Konfig anlegen
	int cuNrBlocks   = std::ceil( (float)this->denominator.size().area() / 512.0 );
	int cuNrThreads  = std::ceil( (float)this->denominator.size().area() / (float)cuNrBlocks );

	cuDenominator( cuNrBlocks,
				   cuNrThreads,
				   this->denominator.ptr<float>(), 
				   this->iimg.ptr<int>(), 
				   this->siimg.ptr<float>(), 
				   this->denominator.cols,
				   this->denominator.rows, 
				   this->cfg.npd_tplw, 
				   this->cfg.npd_tplh,
				   this->cfg.npd_tplh*this->cfg.npd_tplw,
				   this->tSigma
				);


	// Padding-Ränder vom Integralbild abschneiden
	gpu::GpuMat denom = this->denominator(Rect(1,1,this->cfg.common_imageWidth, this->cfg.common_imageHeight ));

	// *********************************************
	//      Gesamtergebnis bestimmen
	// *********************************************
	gpu::divide(num, denom, this->xcorrRes );

} // end xcorr

vector<NodePoint>& NodeDetector::getNodes(){
	return this->nodes;
} // end getNodes

void NodeDetector::fft(gpu::GpuMat& src, gpu::GpuMat& dst, bool ifft) {
	#ifdef _DEBUG
		assert(src.isContinuous());
		assert(dst.isContinuous());
	#endif

    if (ifft){ // rückwärts   
        cufftExecC2R(this->fftPlan_BW, src.ptr<cufftComplex>(), dst.ptr<cufftReal>());  
    } else{   // vorwärts
        cufftExecR2C( this->fftPlan_FW, src.ptr<cufftReal>(), dst.ptr<cufftComplex>());
    }	
} // end fft


int NodeDetector::padding(gpu::GpuMat &src, gpu::GpuMat &dst) {

	if ((this->padSize.height > src.rows) && (this->padSize.width > src.cols)) {
			int left, right, top, bottom, diff;

			top    = (this->padSize.height - src.rows)/2;
			bottom = this->padSize.height - src.rows - top;

			left   = (this->padSize.width - src.cols)/2;
			right  = this->padSize.width  - src.cols - left;

			gpu::copyMakeBorder( src, 
								 dst, 
								 top, 
								 bottom, 
								 left, 
								 right,
								 Scalar(0.0) 
							  );

	} else if ((this->padSize.height <= src.rows) && (this->padSize.width <= src.cols)) {
			(src(Rect(0, 0, this->cfg.npd_padiw, this->cfg.npd_padih))).copyTo(dst);
	} // end if
	
	#ifdef _DEBUG
		assert(dst.isContinuous());
	#endif

	return 0;
} // end padding

void NodeDetector::printNodePoints(Mat &img, double resize /*1.0*/, double circleSize/*=1.0*/) {
	#include "cMap_np.h"

	// Ggf. in Farbbild umwandeln
	if(img.channels() == 1) {
		cvtColor(img, img, CV_GRAY2RGB);
	}
	 
	 cv::Point a;
	 // Gefundene NP anzeigen
	 for(int i = 0; i < this->nodes.size(); i++) {	 
		a.x = (int)(nodes[i].xi*resize);
		a.y = (int)(nodes[i].yi*resize);
	
		int colorVal = min(255.0, abs(255.0*nodes[i].xcorrVal));
		Scalar colorCode = g_cMap_np[ (int)(colorVal)];

		int circlePerim = max(circleSize*resize, 1.0);

		cv::circle(img, a, circlePerim , colorCode, -1, CV_AA);	

		// Vom YM Modul aussortierte NP
	/*	if(nodes[i].xcorrVal < this->cfg.ym_minXcorr){
			cv::circle(img, a, circlePerim+2 , cv::Scalar( 0,0,0), circleSize, CV_AA);	
		}*/
	 } // end for i
	 
} // end print


void NodeDetector::swapQuadrants(gpu::GpuMat& img) {
	// Quadranten:
	//     2 | 1
	//     -----
	//     3 | 4

	Mat src = img;
	Mat dst = Mat(src.rows, src.cols, CV_32F, Scalar(0.0));

	int rowHalf = src.rows/2;
	int colHalf = src.cols/2;

	// Q2 -> Q4
	for(int i = 0; i < rowHalf; i++) {
		int swapi =  i + rowHalf;

		for(int j = 0; j < colHalf; j++) {
			int swapj =  j + colHalf;
			dst.at<float>(swapi,swapj) = src.at<float>(i,j);
		}
	}

	// Q3 -> Q1
	for(int i = rowHalf; i < src.rows; i++) {
		int swapi =  i - rowHalf;

		for(int j = 0; j < colHalf; j++) {
			int swapj =  j + colHalf;
			dst.at<float>(swapi,swapj) = src.at<float>(i,j);
		}
	}

	// Q1 -> Q3
	for(int i = 0; i < rowHalf; i++) {
		int swapi =  i + rowHalf;

		for(int j = colHalf; j < src.cols; j++) {
			int swapj =  j - colHalf;
			dst.at<float>(swapi,swapj) = src.at<float>(i,j);
		}
	}

	// Q4 -> Q2
	for(int i = rowHalf; i < src.rows; i++) {
		int swapi =  i - rowHalf;

		for(int j = colHalf; j < src.cols; j++) {
			int swapj =  j - colHalf;
			dst.at<float>(swapi,swapj) = src.at<float>(i,j);
		}
	}

	img = dst;
} // end swapQuadrants

Mat NodeDetector::rotate180(const Mat& src){
	Mat dst(src.rows, src.cols, CV_32F);
	
	for(int i = 0; i < src.rows; i++) {
		int miri =  src.rows - i -1;

		for(int j = 0; j < src.cols; j++) {

			int mirj =  src.cols - j -1;
			dst.at<float>(miri,mirj) = src.at<float>(i,j);
		}
	}
    return dst;
} // end rotate180

void NodeDetector::reset(){
	this->nodeMap.release();
	this->nodes.clear();
}

NodeDetector::~NodeDetector(){
}
