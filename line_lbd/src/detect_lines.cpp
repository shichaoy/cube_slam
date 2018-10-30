/*
 * line_detection interface
 * Copyright Shichao Yang,2018, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu
 *
 */

#include <line_lbd/line_descriptor.hpp>

// #include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <line_lbd/line_lbd_allclass.h>

#include <ros/ros.h>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
      /* get parameters from comand line */
      if(argc<2){
      std::cout<<"Provide an image name"<<endl;
      return -1;
      }
 
      ros::init(argc, argv, "detect lines");
      ros::NodeHandle nh; 
 
      std::string image_path(argv[1]);
            
      cv::Mat raw_img = imread( image_path, 1 );
      if( raw_img.data == NULL )
      {
      std::cout << "Error, image could not be loaded. Please, check its path \n"<<image_path << std::endl;
      return -1;
      }
      
      
      std::string save_folder;
      bool use_LSD_algorithm;
      bool save_to_imgs;
      bool save_to_txts;
      nh.param<std::string>("save_folder", save_folder, "$(find line_lbd)/data");
      nh.param<bool>("use_LSD_algorithm",use_LSD_algorithm,true);
      nh.param<bool>("save_to_imgs",save_to_imgs,false);
      nh.param<bool>("save_to_txts",save_to_txts,false);
      
            
      int numOfOctave_ = 1;
      float Octave_ratio = 2.0;  

      line_lbd_detect* line_lbd_ptr = new line_lbd_detect(numOfOctave_,Octave_ratio); 
      line_lbd_ptr->use_LSD = use_LSD_algorithm;
      line_lbd_ptr->line_length_thres = 15;  // remove short edges
      
      
      // using my line detector class, could select LSD or edline.
      cv::Mat out_edges;
      std::vector< KeyLine> keylines_raw,keylines_out;
      line_lbd_ptr->detect_raw_lines(raw_img,keylines_raw);
      line_lbd_ptr->filter_lines(keylines_raw,keylines_out);  // remove short lines
      
      // show image
      if( raw_img.channels() == 1 )
        cvtColor( raw_img, raw_img, COLOR_GRAY2BGR );
      cv::Mat raw_img_cp;
      drawKeylines(raw_img, keylines_out, raw_img_cp, cv::Scalar( 0, 150, 0 ),2); // B G R
      imshow( "Line detector", raw_img_cp );
      waitKey();
      
      if (save_to_imgs)
      {
        std::string img_save_name = save_folder+"saved_edges.jpg";
        cv::imwrite(img_save_name,raw_img_cp);
      }
      
      if (save_to_txts)
      {
        std::string txt_save_name = save_folder+"saved_edges.txt";
        ofstream resultsFile;
        resultsFile.open(txt_save_name);
        for (int j=0;j<keylines_out.size();j++)
        {
          resultsFile <<keylines_out[j].startPointX <<"\t" <<keylines_out[j].startPointY  <<"\t"
                  <<keylines_out[j].endPointX   <<"\t" <<keylines_out[j].endPointY    <<endl;
        }
        resultsFile.close();
      }
  
}
