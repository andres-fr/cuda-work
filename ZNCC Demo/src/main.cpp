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

// CUSTOM INCLUDES
#include "nodeDetector.h"

nodeConfig readConfigIni(std::string iniPath);

int main(int argc, char **argv) {
	/////////////////////////////
	// Parse Commandline Parameter
	std::string configPath;
	if(argc>1)//check for parameter
		configPath = argv[1];
	else
		configPath = "config.ini";
	/////////////////////////////

	nodeConfig config = readConfigIni(configPath);

	// Bild, Template und Strukturelement laden:
	Mat img    = imread(config.imageLoadPath, CV_LOAD_IMAGE_GRAYSCALE);
	Mat tmp    = imread(config.templateLoadPath, CV_LOAD_IMAGE_GRAYSCALE);

	gpu::GpuMat gpuImg = gpu::createContinuous(img.size(), CV_8U);
	gpuImg = img;

	NodeDetector npFactory(config, tmp);

	npFactory.detect(gpuImg);

	npFactory.printNodePoints(img, 1.0, 4.0);
	

	namedWindow("",0);
	imshow("",img);
	waitKey();
	
}

nodeConfig readConfigIni(std::string iniPath){
  std::string line;
  std::string currentCat;
  std::string currentValName;
  std::string currentVal;
  std::ifstream myfile (iniPath);
  nodeConfig config;
  if (myfile.is_open())
  {
    while ( myfile.good() )
    {
      getline (myfile,line);
	  if(line.compare(0,1,"[")==0)
		  {
			currentCat = line.substr(1,line.length()-2);
		  }
	  else
		  {
		   int i = 0;
			while(line.compare(i,1,"=")!=0)
				{i++;}

				currentValName = line.substr(0,i);
				currentVal = line.substr(i+1,line.length()-(i+1));
				//// readout config
				if(currentCat.compare("general")==0)
					{
					if(currentValName.compare("showImage")==0)
						{
						if(currentVal.compare("true")==0)
							config.showImage = true;
						else
							config.showImage = false;
						}
					if(currentValName.compare("imageLoadPath")==0)
						{
						config.imageLoadPath = currentVal.substr(1,currentVal.length()-2);
						}
					if(currentValName.compare("imageSavePath")==0)
						{
						config.imageSavePath = currentVal.substr(1,currentVal.length()-2);
						}
					if(currentValName.compare("imageHeight")==0)
						{
						config.common_imageHeight = atoi(currentVal.c_str());
						}
					if(currentValName.compare("imageWidth")==0)
						{
						config.common_imageWidth = atoi(currentVal.c_str());
						}
					if(currentValName.compare("templateLoadPath")==0)
						{
						config.templateLoadPath = currentVal.substr(1,currentVal.length()-2);
						}
					}//end General
				if(currentCat.compare("node")==0)
					{
					if(currentValName.compare("resize")==0)
						{
						config.npd_resize = atof(currentVal.c_str());
						}
					if(currentValName.compare("padih")==0)
						{
						config.npd_padih = atoi(currentVal.c_str());
						}
					if(currentValName.compare("padiw")==0)
						{
						config.npd_padiw = atoi(currentVal.c_str());
						}
					if(currentValName.compare("tplh")==0)
						{
						config.npd_tplh = atoi(currentVal.c_str());
						}
					if(currentValName.compare("tplw")==0)
						{
						config.npd_tplw = atoi(currentVal.c_str());
						}
					if(currentValName.compare("strel")==0)
						{
						config.npd_strel = atoi(currentVal.c_str());
						}
					if(currentValName.compare("offsets0")==0)
						{
						config.npd_offsets[0] = atoi(currentVal.c_str());
						}
					if(currentValName.compare("offsets1")==0)
						{
						config.npd_offsets[1] = atoi(currentVal.c_str());
						}
					if(currentValName.compare("minCorrCoef")==0)
						{
						config.ya_minXcorr = atoi(currentVal.c_str());
						}
					
					}//end Clahe


		  }//end while
    }
    myfile.close();
  }//end while
  else std::cout << "Unable to open file"; 
	
  return config;
}