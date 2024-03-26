#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <DPGO/DPGOProblem.h>
#include <DPGO/DPGO_types.h>
#include <DPGO/RelativePoseMeasurement.h>

namespace DPGO {
class DPGOStar {
 public:
  DPGOStar() {}
  DPGOStar(int num_nodes, const std::string& filename,
        const Options& options = Options());
  DPGOStar(const std::vector<measurements_t>& measurements,
        const Options& options = Options());
  int initialize(const Matrix& X) const;

  int iterate() const;
  int communicate() const;
  int update() const;

  std::vector<std::shared_ptr<const DPGOProblem>> problems() const {
    return problems_;
  }

  std::vector<DPGOResult> const& results() const { return results_; }

  Options const& options() const { return options_; }

  int set_options(Options const& options) {
    options_ = options;
    return 0;
  }

  int num_nodes() const { return num_nodes_; }

  const SparseMatrix& M() const { return M_; }

  const SparseMatrix& B0() const { return B0_; }

  const SparseMatrix& B1() const { return B1_; }

  Scalar fobj() const { return fobj_; }

  const Matrix& X() const { return Xk_; }

  int evaluate_f(const Matrix& X, Scalar& fobj) const;

  int evaluate_grad(const Matrix& X, Matrix& grad) const;

 private:
  int initialize_n(int node, const Matrix& X) const;
  int communicate_n(int node) const;
  int update_n(int node) const;
  int amm_pgo_n(int node) const;
  int mm_pgo_n(int node) const;
  int pm_pgo_n(int node) const;

 protected:
  int num_nodes_;
  int num_poses_;
  int d_;
  std::vector<std::map<int, int>> g_index_;
  Options options_;

  measurements_t intra_measurements_;
  measurements_t inter_measurements_;
  std::vector<std::shared_ptr<const DPGOProblem>> problems_;

  SparseMatrix M_;
  SparseMatrix B0_, B1_;
  int num_measurements_[2];

  SOdProduct SP_;

  mutable Matrix Xk_, Xkh_, Xkp_;
  mutable Scalar fobj_, F_;

  mutable std::vector<DPGOResult> results_;
};
}  // namespace DPGO
