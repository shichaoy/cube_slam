// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_SE3QUAT_H_
#define G2O_SE3QUAT_H_

#include "se3_ops.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {
  using namespace Eigen;

  typedef Matrix<double, 6, 1> Vector6d;
  typedef Matrix<double, 7, 1> Vector7d;

  class SE3Quat {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;


    protected:

      Quaterniond _r;
      Vector3d _t;


    public:
      SE3Quat(){
        _r.setIdentity();
        _t.setZero();
      }

      SE3Quat(const Matrix3d& R, const Vector3d& t):_r(Quaterniond(R)),_t(t){ 
        normalizeRotation();
      }

      SE3Quat(const Quaterniond& q, const Vector3d& t):_r(q),_t(t){
        normalizeRotation();
      }
      
      // x y z qx qy qz qw
      SE3Quat(const Vector7d& v){
	   fromVector(v);
	   normalizeRotation();
      }
      
      /**
       * templaized constructor which allows v to be an arbitrary Eigen Vector type, e.g., Vector6d or Map<Vector6d>
       */
      template <typename Derived>
        explicit SE3Quat(const MatrixBase<Derived>& v)
        {
          assert((v.size() == 6 || v.size() == 7) && "Vector dimension does not match");
          if (v.size() == 6) {
            for (int i=0; i<3; i++){
              _t[i]=v[i];
              _r.coeffs()(i)=v[i+3];
            }
            _r.w() = 0.; // recover the positive w
            if (_r.norm()>1.){
              _r.normalize();
            } else {
              double w2=1.-_r.squaredNorm();
              _r.w()= (w2<0.) ? 0. : sqrt(w2);
            }
          }
          else if (v.size() == 7) {
            int idx = 0;
            for (int i=0; i<3; ++i, ++idx)
              _t(i) = v(idx);
            for (int i=0; i<4; ++i, ++idx)
              _r.coeffs()(i) = v(idx);
            normalizeRotation();
          }
        }

      inline const Vector3d& translation() const {return _t;}

      inline void setTranslation(const Vector3d& t_) {_t = t_;}

      inline const Quaterniond& rotation() const {return _r;}

      void setRotation(const Quaterniond& r_) {_r=r_;}

      inline SE3Quat operator* (const SE3Quat& tr2) const{
        SE3Quat result(*this);
        result._t += _r*tr2._t;
        result._r*=tr2._r;
        result.normalizeRotation();
        return result;
      }

      inline SE3Quat& operator*= (const SE3Quat& tr2){
        _t+=_r*tr2._t;
        _r*=tr2._r;
        normalizeRotation();
        return *this;
      }

      inline Vector3d operator* (const Vector3d& v) const {
        return _t+_r*v;
      }

      inline SE3Quat inverse() const{
        SE3Quat ret;
        ret._r=_r.conjugate();
        ret._t=ret._r*(_t*-1.);
        return ret;
      }

      inline double operator [](int i) const {
        assert(i<7);
        if (i<3)
          return _t[i];
        return _r.coeffs()[i-3];
      }

      // x y z qx qy qz qw
      inline Vector7d toVector() const{
        Vector7d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        v[6]=_r.w();
        return v;
      }

      // x y z qx qy qz qw
      inline void fromVector(const Vector7d& v){
        _r=Quaterniond(v[6], v[3], v[4], v[5]);
        _t=Vector3d(v[0], v[1], v[2]);
      }
      // x y z qx qy qz
      inline Vector6d toMinimalVector() const{
        Vector6d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        return v;
      }
      // x y z qx qy qz
      inline void fromMinimalVector(const Vector6d& v){
        double w = 1.-v[3]*v[3]-v[4]*v[4]-v[5]*v[5];
        if (w>0){
          _r=Quaterniond(sqrt(w), v[3], v[4], v[5]);
        } else {
          _r=Quaterniond(0, -v[3], -v[4], -v[5]);
        }
        _t=Vector3d(v[0], v[1], v[2]);
      }      
      
      // copied from my quat to euler
      void quat_to_euler_zyx_infuc(const Eigen::Quaterniond q, double& roll, double& pitch, double& yaw) const
      {
	    const double qw = q.w();
	    const double qx = q.x();
	    const double qy = q.y();
	    const double qz = q.z();
	    
	    roll = atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy));
	    pitch = asin(2*(qw*qy-qz*qx));
	    yaw = atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz));
      }
      
      inline Vector6d toXYZPRYVector() const{
	double yaw, pitch, roll;
	quat_to_euler_zyx_infuc(_r, roll, pitch, yaw);
        Vector6d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=roll;
        v[4]=pitch;
        v[5]=yaw;
        return v;
      }
            
      Eigen::Quaterniond zyx_euler_to_quat(const double &roll, const double &pitch, const double &yaw)
      {
	    double sy = sin(yaw*0.5);
	    double cy = cos(yaw*0.5);
	    double sp = sin(pitch*0.5);
	    double cp = cos(pitch*0.5);
	    double sr = sin(roll*0.5);
	    double cr = cos(roll*0.5);
	    double w = cr*cp*cy + sr*sp*sy;
	    double x = sr*cp*cy - cr*sp*sy;
	    double y = cr*sp*cy + sr*cp*sy;
	    double z = cr*cp*sy - sr*sp*cy;
	    return Eigen::Quaterniond(w,x,y,z);
      }
      inline void fromXYZPRYVector(const Vector6d& v){
	_r = zyx_euler_to_quat(v[3], v[4], v[5]);
	_t=Vector3d(v[0], v[1], v[2]);
      }

      
      Vector6d log() const {
        Vector6d res;
        Matrix3d _R = _r.toRotationMatrix();
        double d =  0.5*(_R(0,0)+_R(1,1)+_R(2,2)-1);
        Vector3d omega;
        Vector3d upsilon;

	
        Vector3d dR = deltaR(_R);
        Matrix3d V_inv;

        if (d>0.99999)
        {
	  
          omega=0.5*dR;
          Matrix3d Omega = skew(omega);
          V_inv = Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);
        }
        else
        {
          double theta = acos(d);
          omega = theta/(2*sqrt(1-d*d))*dR;
          Matrix3d Omega = skew(omega);
          V_inv = ( Matrix3d::Identity() - 0.5*Omega
              + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
        }

        upsilon = V_inv*_t;
        for (int i=0; i<3;i++){
          res[i]=omega[i];
        }
        for (int i=0; i<3;i++){
          res[i+3]=upsilon[i];
        }

        return res;

      }

      Vector3d map(const Vector3d & xyz) const
      {
        return _r*xyz + _t;
      }
      
      //  angle, translation
      static SE3Quat exp(const Vector6d & update)  // doesn't multiply into current...
      {
        Vector3d omega;
        for (int i=0; i<3; i++)
          omega[i]=update[i];
        Vector3d upsilon;
        for (int i=0; i<3; i++)
          upsilon[i]=update[i+3];

        double theta = omega.norm();
        Matrix3d Omega = skew(omega);

        Matrix3d R;
        Matrix3d V;
	Quaterniond q;

        if (theta<0.00001)
        {
          //TODO: CHECK WHETHER THIS IS CORRECT!!!
          R = (Matrix3d::Identity() + Omega + Omega*Omega);// must add Omega... though it is small
	  
// 	  double theta_sq=theta*theta;double theta_po4=theta_sq*theta_sq; // from LSD sophus
// 	  q.w() = 1-0.5*theta_sq + (1.0/384.0)*theta_po4; 
// 	  q.vec() = (0.5-1.0/48.0*theta_sq + 1.0/3840.0*theta_po4)*omega;
// 	  q.normalize();
	  
          V = R;
        }
        else
        {
          Matrix3d Omega2 = Omega*Omega;

          R = (Matrix3d::Identity()
              + sin(theta)/theta *Omega
              + (1-cos(theta))/(theta*theta)*Omega2);
	  
// 	  q.vec() = omega/theta*sin(theta/2); // from LSD sophus
// 	  q.w() = cos(theta/2);
// 	  q.normalize();
	  
          V = (Matrix3d::Identity()
              + (1-cos(theta))/(theta*theta)*Omega
              + (theta-sin(theta))/(pow(theta,3))*Omega2);
        }
//         if (q.w()<1)
// 	    std::cout<<"Quaterniond(R) "<<Quaterniond(R).coeffs().transpose()<<"  "<<q.coeffs().transpose()<<std::endl;
//         return SE3Quat(q,V*upsilon);
	  return SE3Quat(Quaterniond(R),V*upsilon);
      }

      Matrix<double, 6, 6> adj() const
      {
        Matrix3d R = _r.toRotationMatrix();
        Matrix<double, 6, 6> res;
        res.block(0,0,3,3) = R;
        res.block(3,3,3,3) = R;
        res.block(3,0,3,3) = skew(_t)*R;
        res.block(0,3,3,3) = Matrix3d::Zero(3,3);
        return res;
      }

      Matrix<double,4,4> to_homogeneous_matrix() const
      {
        Matrix<double,4,4> homogeneous_matrix;
        homogeneous_matrix.setIdentity();
        homogeneous_matrix.block(0,0,3,3) = _r.toRotationMatrix();
        homogeneous_matrix.col(3).head(3) = translation();

        return homogeneous_matrix;
      }

      void normalizeRotation(){
        if (_r.w()<0){
          _r.coeffs() *= -1;
        }
        _r.normalize();
      }

      /**
       * cast SE3Quat into an Eigen::Isometry3d
       */
      operator Eigen::Isometry3d() const
      {
        Eigen::Isometry3d result = (Eigen::Isometry3d) rotation();
        result.translation() = translation();
        return result;
      }
  };

  inline std::ostream& operator <<(std::ostream& out_str, const SE3Quat& se3)
  {
    out_str << se3.to_homogeneous_matrix()  << std::endl;
    return out_str;
  }

} // end namespace

#endif
