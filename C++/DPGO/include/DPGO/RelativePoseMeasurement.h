#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <DPGO/DPGO_types.h>

namespace DPGO {

/** A simple struct that contains the elements of a relative pose measurement */
struct RelativePoseMeasurement {
  /** 0-based index of first pose */
  Index i;

  /** 0-based index of second pose */
  Index j;

  /** Rotational measurement */
  Matrix R;

  /** Translational measurement */
  Vector t;

  /** Rotational measurement precision */
  Scalar kappa;

  /** Translational measurement precision */
  Scalar tau;

  /** Simple default constructor; does nothing */
  RelativePoseMeasurement() {}

  /** Basic constructor */
  RelativePoseMeasurement(Index ii, Index jj, const Matrix &relative_rotation,
                          const Vector &relative_translation,
                          Scalar rotational_precision,
                          Scalar translational_precision)
      : i(ii),
        j(jj),
        R(relative_rotation),
        t(relative_translation),
        kappa(rotational_precision),
        tau(translational_precision) {}

  /** A utility function for streaming Nodes to cout */
  inline friend std::ostream &operator<<(
      std::ostream &os, const RelativePoseMeasurement &measurement) {
    os << "i: (" << measurement.i.node << ", " << measurement.i.pose << ")"
       << std::endl;
    os << "j: (" << measurement.j.node << ", " << measurement.j.pose << ")"
       << std::endl;
    os << "R: " << std::endl << measurement.R << std::endl;
    os << "t: " << std::endl << measurement.t << std::endl;
    os << "Kappa: " << measurement.kappa << std::endl;
    os << "Tau: " << measurement.tau << std::endl;

    return os;
  }
};

/** Typedef for a vector of RelativePoseMeasurements */
typedef std::vector<DPGO::RelativePoseMeasurement> measurements_t;
}  // namespace DPGO
