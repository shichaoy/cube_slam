#include <line_lbd/line_descriptor.hpp>
#include <line_lbd/line_descriptor/descriptor.hpp>
#include "line_lbd/line_lbd_allclass.h"

// #include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <ctime>
#include <math.h>

#include <unordered_set>
#include <set>
#include <cassert>

#define MATCHES_DIST_THRESHOLD 25 // this might need to change with image size
#define PI 3.14159265

using namespace cv;
using namespace cv::line_descriptor;
using namespace std;

/* change vector of keylines  to cvMat (float). */
void keylines_to_mat(const std::vector<KeyLine> &keylines_src, cv::Mat &linesmat_out, float scale)
{
  linesmat_out.create(keylines_src.size(), 4, CV_32FC1); // CV_32SC1
  for (int j = 0; j < keylines_src.size(); j++)
  {
    linesmat_out.at<float>(j, 0) = keylines_src[j].startPointX * scale;
    linesmat_out.at<float>(j, 1) = keylines_src[j].startPointY * scale;
    linesmat_out.at<float>(j, 2) = keylines_src[j].endPointX * scale;
    linesmat_out.at<float>(j, 3) = keylines_src[j].endPointY * scale;
  }
}

// only know the start and end point in detected octaves, octave scale is eg: 2^N, fill other information,
// length_threshold, oct_image_size and temp_img all in detected octaves.
bool fill_line_information(KeyLine *kl, float octaveScale, const Size &oct_image_size, float octave_length_thre,
                           const cv::Mat &oct_temp_img)
{
  float octave_delta_x = kl->ePointInOctaveX - kl->sPointInOctaveX;
  float octave_delta_y = kl->ePointInOctaveY - kl->sPointInOctaveY;
  kl->lineLength = sqrt(octave_delta_x * octave_delta_x + octave_delta_y * octave_delta_y);

  if (kl->lineLength < octave_length_thre)
    return false;

  kl->startPointX = kl->sPointInOctaveX * octaveScale;
  kl->startPointY = kl->sPointInOctaveY * octaveScale;
  kl->endPointX = kl->ePointInOctaveX * octaveScale;
  kl->endPointY = kl->ePointInOctaveY * octaveScale;
  kl->pt = Point2f((kl->endPointX + kl->startPointX) / 2, (kl->endPointY + kl->startPointY) / 2);

  kl->angle = atan2(octave_delta_y, octave_delta_x);
  kl->size = fabs(octave_delta_x * octave_delta_y) * octaveScale * octaveScale;
  kl->response = kl->lineLength / (float)std::max(oct_image_size.width, oct_image_size.height);

  /* compute number of pixels covered by line, required by computing descriptors */
  LineIterator li(oct_temp_img, Point2f(kl->sPointInOctaveX, kl->sPointInOctaveY), Point2f(kl->ePointInOctaveX, kl->ePointInOctaveY));
  kl->numOfPixels = li.count;
  return true;
}

// lines mat is in the detected octaves
// length_threshold is in original space
void mat_to_keylines(const cv::Mat &linesmat_src, std::vector<KeyLine> &keylines_out,
                     int raw_img_width, int raw_img_height, float raw_length_threshold, float close_boundary_threshold, float scaling,
                     int octave_id, float each_octave_scale)
{
  // TODO have a look at opencv_contrib, or LSDDetector end.
  keylines_out.clear();
  float octave_scale = pow(each_octave_scale, octave_id);

  float pre_boundary_thre = close_boundary_threshold / octave_scale; // LSD sometimes detect very close edge to the boundary, not using them
  float octave_length_thre = raw_length_threshold / octave_scale;

  int oct_img_width = raw_img_width / octave_scale;
  int oct_img_height = raw_img_height / octave_scale;
  Size imgsize = Size(oct_img_width, oct_img_height);
  cv::Mat temp_img(imgsize, CV_8UC1, Scalar(0));

  int line_ind = -1;
  for (int j = 0; j < linesmat_src.rows; j++)
  {
    KeyLine kl;
    kl.sPointInOctaveX = linesmat_src.at<float>(j, 0) * scaling; // scaling might be used for downsampling
    kl.sPointInOctaveY = linesmat_src.at<float>(j, 1) * scaling;
    kl.ePointInOctaveX = linesmat_src.at<float>(j, 2) * scaling;
    kl.ePointInOctaveY = linesmat_src.at<float>(j, 3) * scaling;

    // remove lines which are very close to boundaries
    if (((kl.startPointX < pre_boundary_thre) && (kl.endPointX < pre_boundary_thre)) ||
        ((kl.startPointX > oct_img_width - pre_boundary_thre) && (kl.endPointX > oct_img_width - pre_boundary_thre)) ||
        ((kl.startPointY < pre_boundary_thre) && (kl.endPointY < pre_boundary_thre)) ||
        ((kl.startPointY > oct_img_height - pre_boundary_thre) && (kl.endPointY > oct_img_height - pre_boundary_thre)))
      continue;

    if (fill_line_information(&kl, octave_scale, imgsize, octave_length_thre, temp_img))
    {
      keylines_out.push_back(kl);
      line_ind++;
      kl.class_id = line_ind; //TODO must start 0 to N, continuously
      kl.octave = octave_id;
    }
  }
}

line_lbd_detect::line_lbd_detect(int numoctaves, float octaveratio) : numoctaves_(numoctaves), octaveratio_(octaveratio)
{
  BinaryDescriptor::Params line_params;
  line_params.numOfOctave_ = numoctaves_;
  line_params.Octave_ratio = octaveratio_;

  lbd = BinaryDescriptor::createBinaryDescriptor(line_params);
  bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  lsd = LSDDetector::createLSDDetector();

  use_LSD = false;
  line_length_thres = 50;
}

void line_lbd_detect::detect_raw_lines(const cv::Mat &gray_img, std::vector<KeyLine> &keylines_out)
{
  if (use_LSD)
  {
    LSDDetector::LSDOptions opts; // lsd parameters  see Stvo-PL  I actually didn't use it.
    opts.refine = 0;
    opts.scale = 0;
    opts.sigma_scale = 0;
    opts.quant = 0;
    opts.ang_th = 0;
    opts.log_eps = 0;
    opts.density_th = 0;
    opts.n_bins = 0;
    opts.min_length = 0;

    lsd->detect(gray_img, keylines_out, octaveratio_, numoctaves_, opts); // seems different from my LSD.  already removed boundary lines
                                                                          //       std::cout<<"lsd edge size "<<keylines_out.size()<<std::endl;
  }
  else
  {
    cv::Mat mask1 = Mat::ones(gray_img.size(), CV_8UC1);
    lbd->detect(gray_img, keylines_out, mask1);
  }
}

void line_lbd_detect::detect_raw_lines(const cv::Mat &gray_img, std::vector<std::vector<KeyLine>> &keyline_octaves)
{
  if (use_LSD)
  {
    LSDDetector::LSDOptions opts; // lsd parameters  see Stvo-PL  I actually didn't use it.
    opts.refine = 0;
    opts.scale = 0;
    opts.sigma_scale = 0;
    opts.quant = 0;
    opts.ang_th = 0;
    opts.log_eps = 0;
    opts.density_th = 0;
    opts.n_bins = 0;
    opts.min_length = 0;

    lsd->detect(gray_img, keyline_octaves, octaveratio_, numoctaves_, opts); // seems different from my LSD.  already removed boundary lines
                                                                             //       std::cout<<"lsd edge size "<<keylines_out.size()<<std::endl;
  }
  else
  {
    cv::Mat mask1 = Mat::ones(gray_img.size(), CV_8UC1);
    lbd->detect(gray_img, keyline_octaves, mask1);
  }
}

void line_lbd_detect::detect_raw_lines(const Mat &gray_img, cv::Mat &lines_mat, bool downsample_img)
{
  cv::Mat gray_img2;
  if (downsample_img)
    cv::resize(gray_img, gray_img2, cv::Size(), 0.5, 0.5); //downsample raw image
  else
    gray_img2 = gray_img; // shallow copy, not deep

  std::vector<KeyLine> lbd_octave;
  detect_raw_lines(gray_img2, lbd_octave);
  if (downsample_img)
    keylines_to_mat(lbd_octave, lines_mat, 2);
  else
    keylines_to_mat(lbd_octave, lines_mat, 1);
}

void line_lbd_detect::get_line_descriptors(const cv::Mat &gray_img, const cv::Mat &linesmat_src, cv::Mat &line_descrips)
{
  std::vector<KeyLine> keylines;
  //     std::cout<<"input line mat   "<<linesmat_src<<std::endl;
  mat_to_keylines(linesmat_src, keylines, gray_img.cols, gray_img.rows);
  lbd->compute(gray_img, keylines, line_descrips);
  //     std::cout<<"matline to keylines:   "<<linesmat_src.rows<<"   "<<keylines.size()<<"   "<<line_descrips.rows<<std::endl;
}

void line_lbd_detect::filter_lines(std::vector<KeyLine> &keylines_in, std::vector<KeyLine> &keylines_out)
{
  keylines_out.clear();
  for (int i = 0; i < (int)keylines_in.size(); i++) // keep octave 0, and remove short lines
    if (keylines_in[i].octave == 0)
      if (keylines_in[i].lineLength > line_length_thres)
        keylines_out.push_back(keylines_in[i]);
}

void line_lbd_detect::detect_filter_lines(const cv::Mat &gray_img, std::vector<KeyLine> &keylines_out)
{
  std::vector<KeyLine> keylines_raw;
  detect_raw_lines(gray_img, keylines_raw);
  filter_lines(keylines_raw, keylines_out);
}

void line_lbd_detect::detect_filter_lines(const cv::Mat &gray_img, cv::Mat &linesmat_out)
{
  std::vector<KeyLine> keylines;
  detect_filter_lines(gray_img, keylines);
  keylines_to_mat(keylines, linesmat_out, 1);
}

/* detect edges and compute edge descriptors */
void line_lbd_detect::detect_descrip_lines(const cv::Mat &gray_img, cv::Mat &lines_mat, Mat &line_descrips)
{
  /* compute lines and descriptors together using EDlines detector */
  std::vector<KeyLine> keylines_raw;
  cv::Mat descrip_raw;

  detect_raw_lines(gray_img, keylines_raw);

  lbd->compute(gray_img, keylines_raw, descrip_raw);

  /* select keylines_raw from first octave and their descriptors */
  std::vector<KeyLine> lbd_octave; // keylines_raw

  line_descrips = cv::Mat().clone();
  for (int i = 0; i < (int)keylines_raw.size(); i++)
  {
    if (keylines_raw[i].octave == 0)
    {
      lbd_octave.push_back(keylines_raw[i]);
      line_descrips.push_back(descrip_raw.row(i));
    }
  }
  keylines_to_mat(lbd_octave, lines_mat);
}

void line_lbd_detect::detect_descrip_lines(const cv::Mat &gray_img, std::vector<KeyLine> &keylines_out, cv::Mat &line_descrips)
{
  std::vector<KeyLine> keylines_raw;
  cv::Mat descrip_raw;

  detect_raw_lines(gray_img, keylines_raw);

  lbd->compute(gray_img, keylines_raw, descrip_raw); //remove some lines before compute descriptprs to save time.  but the keylines index must be continuous.
                                                     //   std::cout<<"marker 1.1  finish detect raw lines"<<std::endl;

  /* select keylines_raw from first octave and their descriptors */
  keylines_out.clear();
  line_descrips = cv::Mat().clone();
  for (int i = 0; i < (int)keylines_raw.size(); i++)
    if (keylines_raw[i].octave == 0) // only first octave?
      if (keylines_raw[i].lineLength > line_length_thres)
      {
        keylines_out.push_back(keylines_raw[i]);
        line_descrips.push_back(descrip_raw.row(i));
      }
}

//change angle from [-pi,pi] to [-pi/2,pi/2]
float normalize_to_PI(float angle)
{
  if (angle > PI / 2)
    return angle - PI; // # change to -90 ~90
  else if (angle < -PI / 2)
    return angle + PI;
  else
    return angle;
}

void line_lbd_detect::detect_descrip_lines_octaves(const cv::Mat &gray_img, std::vector<std::vector<KeyLine>> &keylines_out,
                                                   std::vector<cv::Mat> &line_descrips)
{
  //   std::cout<<"marker 1.0  start detect raw lines"<<std::endl;
  cv::Mat image;
  if (gray_img.channels() != 1)
    cvtColor(gray_img, image, COLOR_BGR2GRAY);
  else
    image = gray_img;

  std::clock_t start = std::clock();
  std::vector<KeyLine> keylines;
  detect_raw_lines(image, keylines);
  double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  //     std::cout<<"marker line_lbd_detect  finish detect raw lines   "<<duration<<std::endl;

  //     for (int i=0;i<keylines.size();i++)
  // 	std::cout<<"index "<<keylines[i].class_id<<"  "<<keylines[i].octave <<std::endl;

  std::clock_t start2 = std::clock();
  // coudl also remove some lines before compute descriptprs to save time.  but the keylines index must be continuous to compute descriptors
  cv::Mat descrip_raw;
  lbd->compute(image, keylines, descrip_raw);

  double duration2 = (std::clock() - start2) / (double)CLOCKS_PER_SEC;
  //     std::cout<<"marker line_lbd_detect  compute descriptors  "<<duration2<<std::endl;

  keylines_out.resize(numoctaves_);
  line_descrips.resize(numoctaves_);
  for (int level = 0; level < numoctaves_; level++)
  {
    keylines_out[level].clear();
    line_descrips[level] = cv::Mat();
  }
  // addtention to class id;
  for (int i = 0; i < (int)keylines.size(); i++)
  {
    KeyLine kl = keylines[i];
    float octaveScale = pow((float)octaveratio_, kl.octave);
    // 	std::cout<<"kl.lineLength "<<kl.angle<<std::endl;
    if (kl.lineLength * octaveScale > line_length_thres)
    {
      if (kl.startPointX > kl.endPointX) // lines start x must smaller than end x
      {
        std::swap(kl.startPointX, kl.endPointX);
        std::swap(kl.startPointY, kl.endPointY);
        std::swap(kl.sPointInOctaveX, kl.ePointInOctaveX);
        std::swap(kl.sPointInOctaveY, kl.ePointInOctaveY);
        kl.angle = normalize_to_PI(kl.angle);
      }
      kl.class_id = keylines_out[kl.octave].size();
      keylines_out[kl.octave].push_back(kl);
      line_descrips[kl.octave].push_back(descrip_raw.row(i));
    }
  }
}

void line_lbd_detect::match_line_descrip(const cv::Mat &descrips_query, const cv::Mat &descrips_train,
                                         vector<DMatch> &good_matches, float matching_dist_thres)
{
  //     std::cout<<"query size "<<descrips_query.rows<<"  "<<descrips_query.cols<<std::endl;
  //     std::cout<<"train size "<<descrips_train.rows<<"  "<<descrips_train.cols<<std::endl;

  /* require match */
  std::vector<DMatch> matches;
  bdm->match(descrips_query, descrips_train, matches);

  /* select best matches */
  good_matches.clear();
  for (int i = 0; i < (int)matches.size(); i++)
  {
    if (matches[i].distance < matching_dist_thres)
      good_matches.push_back(matches[i]);
  }
}
