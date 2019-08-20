#pragma once

#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "detect_3d_cuboid/matrix_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm> // std::swap

typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 3, 8> Matrix38d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace g2o
{

class cuboid
{
public:
	SE3Quat pose;		// 6 dof for object, object to world by default
	Vector3d scale; // [length, width, height]  half!

	cuboid()
	{
		pose = SE3Quat();
		scale.setZero();
	}

	// xyz roll pitch yaw half_scale
	inline void fromMinimalVector(const Vector9d &v)
	{
		Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3), v(4), v(5));
		pose = SE3Quat(posequat, v.head<3>());
		scale = v.tail<3>();
	}

	// xyz quaternion, half_scale
	inline void fromVector(const Vector10d &v)
	{
		pose.fromVector(v.head<7>());
		scale = v.tail<3>();
	}

	inline const Vector3d &translation() const { return pose.translation(); }
	inline void setTranslation(const Vector3d &t_) { pose.setTranslation(t_); }
	inline void setRotation(const Quaterniond &r_) { pose.setRotation(r_); }
	inline void setRotation(const Matrix3d &R) { pose.setRotation(Quaterniond(R)); }
	inline void setScale(const Vector3d &scale_) { scale = scale_; }

	// apply update to current cuboid. exponential map
	cuboid exp_update(const Vector9d &update)
	{
		cuboid res;
		res.pose = this->pose * SE3Quat::exp(update.head<6>()); // NOTE bug before. switch position
		res.scale = this->scale + update.tail<3>();
		return res;
	}

	// actual error between two cuboids.
	Vector9d cube_log_error(const cuboid &newone) const
	{
		Vector9d res;
		SE3Quat pose_diff = newone.pose.inverse() * this->pose;
		res.head<6>() = pose_diff.log(); //treat as se3 log error. could also just use yaw error
		res.tail<3>() = this->scale - newone.scale;
		return res;
	}

	// function called by g2o.
	Vector9d min_log_error(const cuboid &newone, bool print_details = false) const
	{
		bool whether_rotate_cubes = true; // whether rotate cube to find smallest error
		if (!whether_rotate_cubes)
			return cube_log_error(newone);

		// NOTE rotating cuboid... since we cannot determine the front face consistenly, different front faces indicate different yaw, scale representation.
		// need to rotate all 360 degrees (global cube might be quite different from local cube)
		// this requires the sequential object insertion. In this case, object yaw practically should not change much. If we observe a jump, we can use code
		// here to adjust the yaw.
		Vector4d rotate_errors_norm;
		Vector4d rotate_angles(-1, 0, 1, 2); // rotate -90 0 90 180
		Eigen::Matrix<double, 9, 4> rotate_errors;
		for (int i = 0; i < rotate_errors_norm.rows(); i++)
		{
			cuboid rotated_cuboid = newone.rotate_cuboid(rotate_angles(i) * M_PI / 2.0); // rotate new cuboids
			Vector9d cuboid_error = this->cube_log_error(rotated_cuboid);
			rotate_errors_norm(i) = cuboid_error.norm();
			rotate_errors.col(i) = cuboid_error;
		}
		int min_label;
		rotate_errors_norm.minCoeff(&min_label);
		if (print_details)
			if (min_label != 1)
				std::cout << "Rotate cube   " << min_label << std::endl;
		return rotate_errors.col(min_label);
	}

	// change front face by rotate along current body z axis. another way of representing cuboid. representing same cuboid (IOU always 1)
	cuboid rotate_cuboid(double yaw_angle) const // to deal with different front surface of cuboids
	{
		cuboid res;
		SE3Quat rot(Eigen::Quaterniond(cos(yaw_angle * 0.5), 0, 0, sin(yaw_angle * 0.5)), Vector3d(0, 0, 0)); // change yaw to rotation.
		res.pose = this->pose * rot;
		res.scale = this->scale;
		if ((yaw_angle == M_PI / 2.0) || (yaw_angle == -M_PI / 2.0) || (yaw_angle == 3 * M_PI / 2.0))
			std::swap(res.scale(0), res.scale(1));

		return res;
	}

	// transform a local cuboid to global cuboid  Twc is camera pose. from camera to world
	cuboid transform_from(const SE3Quat &Twc) const
	{
		cuboid res;
		res.pose = Twc * this->pose;
		res.scale = this->scale;
		return res;
	}

	// transform a global cuboid to local cuboid  Twc is camera pose. from camera to world
	cuboid transform_to(const SE3Quat &Twc) const
	{
		cuboid res;
		res.pose = Twc.inverse() * this->pose;
		res.scale = this->scale;
		return res;
	}

	// xyz roll pitch yaw half_scale
	inline Vector9d toMinimalVector() const
	{
		Vector9d v;
		v.head<6>() = pose.toXYZPRYVector();
		v.tail<3>() = scale;
		return v;
	}

	// xyz quaternion, half_scale
	inline Vector10d toVector() const
	{
		Vector10d v;
		v.head<7>() = pose.toVector();
		v.tail<3>() = scale;
		return v;
	}

	Matrix4d similarityTransform() const
	{
		Matrix4d res = pose.to_homogeneous_matrix();
		Matrix3d scale_mat = scale.asDiagonal();
		res.topLeftCorner<3, 3>() = res.topLeftCorner<3, 3>() * scale_mat;
		return res;
	}

	// 8 corners 3*8 matrix, each row is x y z
	Matrix3Xd compute3D_BoxCorner() const
	{
		Matrix3Xd corners_body;
		corners_body.resize(3, 8);
		corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
				1, -1, -1, 1, 1, -1, -1, 1,
				-1, -1, -1, -1, 1, 1, 1, 1;
		Matrix3Xd corners_world = homo_to_real_coord<double>(similarityTransform() * real_to_homo_coord<double>(corners_body));
		return corners_world;
	}

	// get rectangles after projection  [topleft, bottomright]
	Vector4d projectOntoImageRect(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
	{
		Matrix3Xd corners_3d_world = compute3D_BoxCorner();
		Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib * homo_to_real_coord<double>(campose_cw.to_homogeneous_matrix() * real_to_homo_coord<double>(corners_3d_world)));
		Vector2d bottomright = corner_2d.rowwise().maxCoeff(); // x y
		Vector2d topleft = corner_2d.rowwise().minCoeff();
		return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
	}

	// get rectangles after projection  [center, width, height]
	Vector4d projectOntoImageBbox(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
	{
		Vector4d rect_project = projectOntoImageRect(campose_cw, Kalib); // top_left, bottom_right  x1 y1 x2 y2
		Vector2d rect_center = (rect_project.tail<2>() + rect_project.head<2>()) / 2;
		Vector2d widthheight = rect_project.tail<2>() - rect_project.head<2>();
		return Vector4d(rect_center(0), rect_center(1), widthheight(0), widthheight(1));
	}
};

class VertexCuboid : public BaseVertex<9, cuboid> // NOTE  this vertex stores object pose to world
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	VertexCuboid(){};

	virtual void setToOriginImpl() { _estimate = cuboid(); }

	virtual void oplusImpl(const double *update_)
	{
		Eigen::Map<const Vector9d> update(update_);
		setEstimate(_estimate.exp_update(update));
	}

	virtual bool read(std::istream &is)
	{
		Vector9d est;
		for (int i = 0; i < 9; i++)
			is >> est[i];
		cuboid Onecube;
		Onecube.fromMinimalVector(est);
		setEstimate(Onecube);
		return true;
	}

	virtual bool write(std::ostream &os) const
	{
		Vector9d lv = _estimate.toMinimalVector();
		for (int i = 0; i < lv.rows(); i++)
		{
			os << lv[i] << " ";
		}
		return os.good();
	}
};

// camera -object 3D error
class EdgeSE3Cuboid : public BaseBinaryEdge<9, cuboid, VertexSE3Expmap, VertexCuboid>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EdgeSE3Cuboid(){};

	virtual bool read(std::istream &is)
	{
		return true;
	};

	virtual bool write(std::ostream &os) const
	{
		return os.good();
	};

	void computeError()
	{
		const VertexSE3Expmap *SE3Vertex = static_cast<const VertexSE3Expmap *>(_vertices[0]); //  world to camera pose
		const VertexCuboid *cuboidVertex = static_cast<const VertexCuboid *>(_vertices[1]);		 //  object pose to world

		SE3Quat cam_pose_Twc = SE3Vertex->estimate().inverse();
		cuboid global_cube = cuboidVertex->estimate();
		cuboid esti_global_cube = _measurement.transform_from(cam_pose_Twc);
		_error = global_cube.min_log_error(esti_global_cube);
	}
};

// camera -object 2D projection error, rectangle difference, could also change to iou
class EdgeSE3CuboidProj : public BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, VertexCuboid>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EdgeSE3CuboidProj(){};

	virtual bool read(std::istream &is)
	{
		return true;
	};

	virtual bool write(std::ostream &os) const
	{
		return os.good();
	};

	void computeError()
	{
		const VertexSE3Expmap *SE3Vertex = static_cast<const VertexSE3Expmap *>(_vertices[0]); //  world to camera pose
		const VertexCuboid *cuboidVertex = static_cast<const VertexCuboid *>(_vertices[1]);		 //  object pose to world

		SE3Quat cam_pose_Tcw = SE3Vertex->estimate();
		cuboid global_cube = cuboidVertex->estimate();

		Vector4d rect_project = global_cube.projectOntoImageBbox(cam_pose_Tcw, Kalib); // center, width, height

		_error = rect_project - _measurement;
	}
	Matrix3d Kalib;
};

} // namespace g2o