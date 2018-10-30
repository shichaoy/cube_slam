/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"

//using namespace cv;
namespace cv
{
namespace line_descriptor
{
Ptr<LSDDetector> LSDDetector::createLSDDetector()
{
  return Ptr<LSDDetector>( new LSDDetector() );
}

/* compute Gaussian pyramid of input image */
void LSDDetector::computeGaussianPyramid( const Mat& image, int numOctaves, int scale )
{
  /* clear class fields */
  gaussianPyrs.clear();

  /* insert input image into pyramid */
  cv::Mat currentMat = image.clone();
  //cv::GaussianBlur( currentMat, currentMat, cv::Size( 5, 5 ), 1 );
  gaussianPyrs.push_back( currentMat );

  /* fill Gaussian pyramid */
  for ( int pyrCounter = 1; pyrCounter < numOctaves; pyrCounter++ )
  {
    /* compute and store next image in pyramid and its size */
    pyrDown( currentMat, currentMat, Size( currentMat.cols / scale, currentMat.rows / scale ) );
    gaussianPyrs.push_back( currentMat );
  }
}

/* check lines' extremes [x1 y1 x2 y2] */
inline void checkLineExtremes( cv::Vec4f& extremes, cv::Size imageSize )
{

  if( extremes[0] < 0 )
    extremes[0] = 0;

  if( extremes[0] >= imageSize.width )
    extremes[0] = (float)imageSize.width - 1.0f;

  if( extremes[2] < 0 )
    extremes[2] = 0;

  if( extremes[2] >= imageSize.width )
    extremes[2] = (float)imageSize.width - 1.0f;

  if( extremes[1] < 0 )
    extremes[1] = 0;

  if( extremes[1] >= imageSize.height )
    extremes[1] = (float)imageSize.height - 1.0f;

  if( extremes[3] < 0 )
    extremes[3] = 0;

  if( extremes[3] >= imageSize.height )
    extremes[3] = (float)imageSize.height - 1.0f;
}

/* requires line detection (only one image) */
void LSDDetector::detect( const Mat& image, CV_OUT std::vector<KeyLine>& keylines, int scale, int numOctaves, LSDOptions& opts, const Mat& mask )
{
  std::vector<std::vector<KeyLine>> keyline_octaves;
  if (opts.save_octave_line == true)
  {    
    std::cout<<"LSD line detector reset save octave lines to False  to match output !!!"<<std::endl;;
    opts.save_octave_line = false;
  }
  if( mask.data != NULL && ( mask.size() != image.size() || mask.type() != CV_8UC1 ) )
    throw std::runtime_error( "Mask error while detecting lines: please check its dimensions and that data type is CV_8UC1" );

  else
    detectImpl( image, keylines, keyline_octaves, numOctaves, scale, opts, mask );
}

void LSDDetector::detect( const Mat& image, CV_OUT std::vector<std::vector<KeyLine>>& keyline_octaves, int scale, int numOctaves, 
			  LSDOptions& opts, const Mat& mask )
{
  std::vector<KeyLine> keylines;
  if (opts.save_octave_line == false)
  {
    std::cout<<"LSD line detector reset save octave lines to True to match output !!!"<<std::endl;
    opts.save_octave_line = true;
  }
  if( mask.data != NULL && ( mask.size() != image.size() || mask.type() != CV_8UC1 ) )
    throw std::runtime_error( "Mask error while detecting lines: please check its dimensions and that data type is CV_8UC1" );

  else
    detectImpl( image, keylines, keyline_octaves, numOctaves, scale, opts, mask );
}

/* requires line detection (more than one image) */
void LSDDetector::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves, 
			  LSDOptions& opts, const std::vector<Mat>& masks )
{
  /* detect lines from each image */
  for ( size_t counter = 0; counter < images.size(); counter++ )
  {
    std::vector<std::vector<KeyLine>> keyline_octaves;
    opts.save_octave_line = false;
    if( masks[counter].data != NULL && ( masks[counter].size() != images[counter].size() || masks[counter].type() != CV_8UC1 ) )
      throw std::runtime_error( "Masks error while detecting lines: please check their dimensions and that data types are CV_8UC1" );

    else
      detectImpl( images[counter], keylines[counter], keyline_octaves, numOctaves, scale, opts, masks[counter] );
  }
}

/* implementation of line detection */
void LSDDetector::detectImpl( const Mat& imageSrc, std::vector<KeyLine>& keylines, std::vector<std::vector<KeyLine>>& keyline_octaves,
			      int numOctaves, int scale, const LSDOptions& opts, const Mat& mask ) const
{
  cv::Mat image;
  if( imageSrc.channels() != 1 )
    cvtColor( imageSrc, image, COLOR_BGR2GRAY );
  else
    image = imageSrc.clone();

  /*check whether image depth is different from 0 */
  if( image.depth() != 0 )
    throw std::runtime_error( "Error, depth image!= 0" );

  /* create a pointer to self */
  LSDDetector *lsd = const_cast<LSDDetector*>( this );

  /* compute Gaussian pyramids */
  lsd->computeGaussianPyramid( image, numOctaves, scale );

  /* create an LSD extractor */
  cv::Ptr<cv::line_descriptor::LineSegmentDetector> ls = cv::line_descriptor::createLineSegmentDetector( cv::line_descriptor::LSD_REFINE_ADV );

  //LSDOptions opts
//   cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector( opts.refine,
//                                                                        opts.scale,
//                                                                        opts.sigma_scale,
//                                                                        opts.quant,
//                                                                        opts.ang_th,
//                                                                        opts.log_eps,
//                                                                        opts.density_th,
//                                                                        opts.n_bins);

  /* prepare a vector to host extracted segments */
  std::vector<std::vector<cv::Vec4f> > lines_lsd;    
  /* extract lines */
  for ( int i = 0; i < numOctaves; i++ )
  {
    std::vector<Vec4f> octave_lines;
    ls->detect( gaussianPyrs[i], octave_lines );
    lines_lsd.push_back( octave_lines );   // there is no correspondence between different layers
  }

  /* create keylines */
//   int class_counter = -1;
  if (opts.save_octave_line)
    if (keyline_octaves.size()>0)
      for ( int i = 0; i < numOctaves; i++ )
	  keyline_octaves[i].clear();  
  keyline_octaves.resize(numOctaves);
  std::vector<int> octave_class_coutner(lines_lsd.size(),-1);  
  
  
  float pre_boundary_thre=10; // LSD sometimes detect very close edge
  int img_width=imageSrc.cols;
  int img_height=imageSrc.rows;
  
  for ( int octaveIdx = 0; octaveIdx < (int) lines_lsd.size(); octaveIdx++ )
  {
    float octaveScale = pow( (float)scale, octaveIdx );
    for ( int k = 0; k < (int) lines_lsd[octaveIdx].size(); k++ )
    {
      KeyLine kl;
      cv::Vec4f extremes = lines_lsd[octaveIdx][k];

      /* check data validity */
      checkLineExtremes( extremes, gaussianPyrs[octaveIdx].size() );

      /* fill KeyLine's fields */
      kl.startPointX = extremes[0] * octaveScale;
      kl.startPointY = extremes[1] * octaveScale;
      kl.endPointX = extremes[2] * octaveScale;
      kl.endPointY = extremes[3] * octaveScale;
      
     // remove lines which are very close to boundaries
     if ( ((kl.startPointX<pre_boundary_thre) && (kl.endPointX<pre_boundary_thre)) || \
	((kl.startPointX>img_width-pre_boundary_thre) && (kl.endPointX>img_width-pre_boundary_thre)) || \
	((kl.startPointY<pre_boundary_thre) && (kl.endPointY<pre_boundary_thre)) || \
	((kl.startPointY>img_height-pre_boundary_thre) && (kl.endPointY>img_height-pre_boundary_thre)) )
	continue;
      
      
      kl.sPointInOctaveX = extremes[0];
      kl.sPointInOctaveY = extremes[1];
      kl.ePointInOctaveX = extremes[2];
      kl.ePointInOctaveY = extremes[3];
      kl.lineLength = (float) sqrt( pow( extremes[0] - extremes[2], 2 ) + pow( extremes[1] - extremes[3], 2 ) );

      /* compute number of pixels covered by line */
      LineIterator li( gaussianPyrs[octaveIdx], Point2f( extremes[0], extremes[1] ), Point2f( extremes[2], extremes[3] ) );
      kl.numOfPixels = li.count;

      kl.angle = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
      kl.class_id = ++(octave_class_coutner[octaveIdx]);  // absolute increase
      kl.octave = octaveIdx;
      kl.size = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
      kl.response = kl.lineLength / max( gaussianPyrs[octaveIdx].cols, gaussianPyrs[octaveIdx].rows );
      kl.pt = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

      if (opts.save_octave_line)
	  keyline_octaves[kl.octave].push_back(kl);
      else
	  keylines.push_back( kl );
    }
  }

  /* delete undesired KeyLines, according to input mask */  
    if( !mask.empty() )
    {
      if (!opts.save_octave_line)
      {
	for ( size_t keyCounter = 0; keyCounter < keylines.size(); keyCounter++ )
	{
	  KeyLine kl = keylines[keyCounter];
	  if( mask.at<uchar>( (int) kl.startPointY, (int) kl.startPointX ) == 0 && mask.at<uchar>( (int) kl.endPointY, (int) kl.endPointX ) == 0 )
	  {
	    keylines.erase( keylines.begin() + keyCounter );
	    keyCounter--;
	  }
	}
      }
      else
      {
	for (int oct=0;oct<numOctaves;oct++)
	  for ( size_t keyCounter = 0; keyCounter < keyline_octaves[oct].size(); keyCounter++ )
	  {
	    KeyLine kl = keyline_octaves[oct][keyCounter];
	    if( mask.at<uchar>( (int) kl.startPointY, (int) kl.startPointX ) == 0 && mask.at<uchar>( (int) kl.endPointY, (int) kl.endPointX ) == 0 )
	    {
	      keyline_octaves[oct].erase( keyline_octaves[oct].begin() + keyCounter );
	      keyCounter--;
	    }
	  }	
      }
    }
}


}
}

