#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //measurement matrix for laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement matrix for radar
  Hj_ << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 1;

  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1;

  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1,0,1,0,
             0,1,0,1,
             0,0,1,0,
             0,0,0,1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /*
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];

      double c = cos(phi);
      double s = sin(phi);

      double px = rho*c;
      double py = rho*s;

      double vx = 0;
      double vy = 0;

      ekf_.x_ << px, py, vx, vy;
      ekf_.P_ << MatrixXd::Identity(4, 4);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      std::cout << measurement_pack.raw_measurements_ << std::endl;

      double px = measurement_pack.raw_measurements_[0];
      double py = measurement_pack.raw_measurements_[1];
      double vx = 0;
      double vy = 0;

      ekf_.x_ << px, py, vx, vy;

      ekf_.P_ << 3*MatrixXd::Identity(4, 4);
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ << 1, 0, dt,  0,
             0, 1,  0, dt,
             0, 0,  1,  0,
             0, 0,  0,  1;

  double dt_2 = dt*dt;
  double dt_3 = dt_2*dt;
  double dt_4 = dt_3*dt;

  double noise_ax = 9;
  double noise_ay = 9;

  ekf_.Q_ = MatrixXd(4, 4);

  ekf_.Q_ << 0.25*dt_4*noise_ax,           0,          0.5*dt_3*noise_ax,         0,
                   0,             0.25*dt_4*noise_ay,          0,           0.5*dt_3*noise_ay,
             0.5*dt_3*noise_ax,            0,          dt_2*noise_ax,             0,
                   0,             0.5*dt_3*noise_ay,           0,           dt_2*noise_ay;
  
  ekf_.Predict();

  /*****************************************************************************
   *  Update
  ****************************************************************************/


  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);


  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
