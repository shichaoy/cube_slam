#include "detect_3d_cuboid/matrix_utils.h"

// std c
#include <math.h>
#include <stdio.h>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>

using namespace Eigen;

template <class T>
Eigen::Quaternion<T> zyx_euler_to_quat(const T &roll, const T &pitch, const T &yaw)
{
    T sy = sin(yaw * 0.5);
    T cy = cos(yaw * 0.5);
    T sp = sin(pitch * 0.5);
    T cp = cos(pitch * 0.5);
    T sr = sin(roll * 0.5);
    T cr = cos(roll * 0.5);
    T w = cr * cp * cy + sr * sp * sy;
    T x = sr * cp * cy - cr * sp * sy;
    T y = cr * sp * cy + sr * cp * sy;
    T z = cr * cp * sy - sr * sp * cy;
    return Eigen::Quaternion<T>(w, x, y, z);
}
template Eigen::Quaterniond zyx_euler_to_quat<double>(const double &, const double &, const double &);
template Eigen::Quaternionf zyx_euler_to_quat<float>(const float &, const float &, const float &);

template <class T>
void quat_to_euler_zyx(const Eigen::Quaternion<T> &q, T &roll, T &pitch, T &yaw)
{
    T qw = q.w();
    T qx = q.x();
    T qy = q.y();
    T qz = q.z();

    roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    pitch = asin(2 * (qw * qy - qz * qx));
    yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}
template void quat_to_euler_zyx<double>(const Eigen::Quaterniond &, double &, double &, double &);
template void quat_to_euler_zyx<float>(const Eigen::Quaternionf &, float &, float &, float &);

template <class T>
void rot_to_euler_zyx(const Eigen::Matrix<T, 3, 3> &R, T &roll, T &pitch, T &yaw)
{
    pitch = asin(-R(2, 0));

    if (abs(pitch - M_PI / 2) < 1.0e-3)
    {
        roll = 0.0;
        yaw = atan2(R(1, 2) - R(0, 1), R(0, 2) + R(1, 1)) + roll;
    }
    else if (abs(pitch + M_PI / 2) < 1.0e-3)
    {
        roll = 0.0;
        yaw = atan2(R(1, 2) - R(0, 1), R(0, 2) + R(1, 1)) - roll;
    }
    else
    {
        roll = atan2(R(2, 1), R(2, 2));
        yaw = atan2(R(1, 0), R(0, 0));
    }
}
template void rot_to_euler_zyx<double>(const Matrix3d &, double &, double &, double &);
template void rot_to_euler_zyx<float>(const Matrix3f &, float &, float &, float &);

template <class T>
Eigen::Matrix<T, 3, 3> euler_zyx_to_rot(const T &roll, const T &pitch, const T &yaw)
{
    T cp = cos(pitch);
    T sp = sin(pitch);
    T sr = sin(roll);
    T cr = cos(roll);
    T sy = sin(yaw);
    T cy = cos(yaw);

    Eigen::Matrix<T, 3, 3> R;
    R << cp * cy, (sr * sp * cy) - (cr * sy), (cr * sp * cy) + (sr * sy),
        cp * sy, (sr * sp * sy) + (cr * cy), (cr * sp * sy) - (sr * cy),
        -sp, sr * cp, cr * cp;
    return R;
}
template Matrix3d euler_zyx_to_rot<double>(const double &, const double &, const double &);
template Matrix3f euler_zyx_to_rot<float>(const float &, const float &, const float &);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_homo_out;
    int raw_rows = pts_in.rows();
    int raw_cols = pts_in.cols();

    pts_homo_out.resize(raw_rows + 1, raw_cols);
    pts_homo_out << pts_in,
        Matrix<T, 1, Dynamic>::Ones(raw_cols);
    return pts_homo_out;
}
template MatrixXd real_to_homo_coord<double>(const MatrixXd &);
template MatrixXf real_to_homo_coord<float>(const MatrixXf &);

template <class T>
void real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_out)
{
    int raw_rows = pts_in.rows();
    int raw_cols = pts_in.cols();

    pts_homo_out.resize(raw_rows + 1, raw_cols);
    pts_homo_out << pts_in,
        Matrix<T, 1, Dynamic>::Ones(raw_cols);
}
template void real_to_homo_coord<double>(const MatrixXd &, MatrixXd &);
template void real_to_homo_coord<float>(const MatrixXf &, MatrixXf &);

template <class T> // though vector can be casted into matrix, to make output clear to be vector, it is better to define a new function.
Eigen::Matrix<T, Eigen::Dynamic, 1> real_to_homo_coord_vec(const Eigen::Matrix<T, Eigen::Dynamic, 1> &pts_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> pts_homo_out;
    int raw_rows = pts_in.rows();
    ;

    pts_homo_out.resize(raw_rows + 1);
    pts_homo_out << pts_in,
        1;
    return pts_homo_out;
}
template VectorXd real_to_homo_coord_vec<double>(const VectorXd &);
template VectorXf real_to_homo_coord_vec<float>(const VectorXf &);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_out(pts_homo_in.rows() - 1, pts_homo_in.cols());
    for (int i = 0; i < pts_homo_in.rows() - 1; i++)
        pts_out.row(i) = pts_homo_in.row(i).array() / pts_homo_in.bottomRows(1).array(); //replicate needs actual number, cannot be M or N

    return pts_out;
}
template MatrixXd homo_to_real_coord<double>(const MatrixXd &);
template MatrixXf homo_to_real_coord<float>(const MatrixXf &);

template <class T>
void homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_out)
{
    pts_out.resize(pts_homo_in.rows() - 1, pts_homo_in.cols());
    for (int i = 0; i < pts_homo_in.rows() - 1; i++)
        pts_out.row(i) = pts_homo_in.row(i).array() / pts_homo_in.bottomRows(1).array(); //replicate needs actual number, cannot be M or N
}
template void homo_to_real_coord<double>(const MatrixXd &, MatrixXd &);
template void homo_to_real_coord<float>(const MatrixXf &, MatrixXf &);

template <class T> // though vector can be casted into matrix, to make output clear to be vector, it is better to define a new function.
Eigen::Matrix<T, Eigen::Dynamic, 1> homo_to_real_coord_vec(const Eigen::Matrix<T, Eigen::Dynamic, 1> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> pt_out;
    if (pts_homo_in.rows() == 4)
        pt_out = pts_homo_in.head(3) / pts_homo_in(3);
    else if (pts_homo_in.rows() == 3)
        pt_out = pts_homo_in.head(2) / pts_homo_in(2);

    return pt_out;
}
template VectorXd homo_to_real_coord_vec<double>(const VectorXd &);
template VectorXf homo_to_real_coord_vec<float>(const VectorXf &);

void fast_RemoveRow(MatrixXd &matrix, int rowToRemove, int &total_line_number)
{
    matrix.row(rowToRemove) = matrix.row(total_line_number - 1);
    total_line_number--;
}

void vert_stack_m(const MatrixXd &a_in, const MatrixXd &b_in, MatrixXd &combined_out)
{
    assert(a_in.cols() == b_in.cols());
    combined_out.resize(a_in.rows() + b_in.rows(), a_in.cols());
    combined_out << a_in,
        b_in;
}

void vert_stack_m_self(MatrixXf &a_in, const MatrixXf &b_in)
{
    assert(a_in.cols() == b_in.cols());
    MatrixXf combined_out(a_in.rows() + b_in.rows(), a_in.cols());
    combined_out << a_in,
        b_in;
    a_in = combined_out; //TODO why not use convervative resize?
}

// make sure column size is given. no checks here. row will be adjusted automatically. if more cols given, will be zero.
template <class T>
bool read_all_number_txt(const std::string txt_file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &read_number_mat)
{
    if (!std::ifstream(txt_file_name))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    std::ifstream filetxt(txt_file_name.c_str());
    int row_counter = 0;
    std::string line;
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);

    while (getline(filetxt, line))
    {
        T t;
        if (!line.empty())
        {
            std::stringstream ss(line);
            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
            }
            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();

    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows

    return true;
}
template bool read_all_number_txt(const std::string, MatrixXd &);
template bool read_all_number_txt(const std::string, MatrixXi &);

bool read_obj_detection_txt(const std::string txt_file_name, Eigen::MatrixXd &read_number_mat, std::vector<std::string> &all_strings)
{
    if (!std::ifstream(txt_file_name))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    all_strings.clear();
    std::ifstream filetxt(txt_file_name.c_str());
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);
    int row_counter = 0;
    std::string line;
    while (getline(filetxt, line))
    {
        double t;
        if (!line.empty())
        {
            std::stringstream ss(line);
            std::string classname;
            ss >> classname;
            all_strings.push_back(classname);

            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
            }
            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();
    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows
}

bool read_obj_detection2_txt(const std::string txt_file_name, Eigen::MatrixXd &read_number_mat, std::vector<std::string> &all_strings)
{
    if (!std::ifstream(txt_file_name))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    all_strings.clear();
    std::ifstream filetxt(txt_file_name.c_str());
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);
    int row_counter = 0;
    std::string line;
    while (getline(filetxt, line))
    {
        double t;
        if (!line.empty())
        {
            std::stringstream ss(line);

            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
                if (colu > read_number_mat.cols() - 1)
                    break;
            }

            std::string classname;
            ss >> classname;
            all_strings.push_back(classname);

            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();
    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows
}

void sort_indexes(const Eigen::VectorXd &vec, std::vector<int> &idx, int top_k)
{
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&vec](int i1, int i2) { return vec(i1) < vec(i2); });
}

void sort_indexes(const Eigen::VectorXd &vec, std::vector<int> &idx)
{
    sort(idx.begin(), idx.end(), [&vec](int i1, int i2) { return vec(i1) < vec(i2); });
}

template <class T>
T normalize_to_pi(T angle)
{
    if (angle > M_PI / 2)
        return angle - M_PI; // # change to -90 ~90
    else if (angle < -M_PI / 2)
        return angle + M_PI;
    else
        return angle;
}
template double normalize_to_pi(double);

template <class T>
void print_vector(const std::vector<T> &vec)
{
    for (size_t i = 0; i < vec.size(); i++)
        std::cout << vec[i] << "  ";
    std::cout << std::endl;
}
template void print_vector(const std::vector<double> &);
template void print_vector(const std::vector<float> &);
template void print_vector(const std::vector<int> &);

template <class T>
void linespace(T starting, T ending, T step, std::vector<T> &res)
{
    res.reserve((ending - starting) / step + 2);
    while (starting <= ending)
    {
        res.push_back(starting);
        starting += step; // TODO could recode to better handle rounding errors
        if (res.size() > 1000)
        {
            std::cout << "Linespace too large size!!!!" << std::endl;
            break;
        }
    }
}
template void linespace(int, int, int, std::vector<int> &);
template void linespace(double, double, double, std::vector<double> &);
