#pragma once

#include <DPGO/DPGO_types.h>
#include <DPGO/RelativePoseMeasurement.h>

#include <PCM/PCM.hpp>

#include <memory>

namespace DPGO {
class PCM {
 public:
  struct Options {
    Scalar tolerance = 0.2;
    bool weighted = false;

    Options() {}
  };

  //===========================================================================
  // alpha: node alpha
  // beta: node beta
  // inter_measurements: inter-node measurements between node alpha and beta
  // num_n, num_s, index: information about the index of matrix X
  // X: intra-node poses of node alpha and beta
  //===========================================================================
  PCM() {
    exact_solver_ = std::make_shared<::PCM::PattabiramanMaxCliqueSolverExact>();
    heruistic_solver_ =
        std::make_shared<::PCM::PattabiramanMaxCliqueSolverHeuristic>();
  }

  Scalar tolerance() const { return opts_.tolerance; }

  bool weighted() const { return opts_.weighted; }

  const Eigen::MatrixXi& adjancecy_matrix() const { return adjancecy_matrix_; }

  const measurements_t& measurements() const { return measurements_; }

  const std::vector<bool>& results() const { return results_; }

  int update(
      int alpha, int beta, const measurements_t& inter_measurements,
      const Eigen::Matrix<int, 2, 1>& num_n,
      const Eigen::Matrix<int, 2, 1>& num_s,
      const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>>& index,
      const Matrix& X, const Options& opts = Options());

  const std::vector<bool>& solveExact() const;

  const std::vector<bool>& solveHeuristic() const;

  int reset() {
    opts_ = Options();
    num_m_ = 0;
    measurements_.clear();
    adjancecy_matrix_.resize(0, 0);
    results_.clear();

    return 0;
  }

 protected:
  Options opts_;
  int num_m_;
  Eigen::MatrixXi adjancecy_matrix_;
  measurements_t measurements_;

  mutable std::vector<bool> results_;

  std::shared_ptr<::PCM::PattabiramanMaxCliqueSolverExact> exact_solver_;
  std::shared_ptr<::PCM::PattabiramanMaxCliqueSolverHeuristic>
      heruistic_solver_;
};
}  // namespace DPGO
