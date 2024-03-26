#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <DChordal/DChordal_types.h>
#include <DChordal/DChordal_utils.h>
#include <DChordal/RelativePoseMeasurement.h>
#include <DChordal/SOdProduct.h>

namespace DChordal {
class DChordalProblem {
 protected:
  /** Index of node */
  int node_;

  /** Regualrized constant for matrix G*/
  Scalar reg_G_;

  /** Number of poses */
  Eigen::Matrix<int, 2, 1> n_ = Eigen::Matrix<int, 2, 1>::Zero();
  /** Offsets */
  Eigen::Matrix<int, 2, 1> s_ = Eigen::Matrix<int, 2, 1>::Zero();
  /** Number of measurements */
  Eigen::Matrix<int, 2, 1> m_ = Eigen::Matrix<int, 2, 1>::Zero();

  measurements_t intra_measurements_;
  measurements_t inter_measurements_;

  /** Dimension of PGO */
  int d_ = 0;
  int p_ = 0;

  mutable SparseMatrix G_, D_, H_, S_;
  mutable Matrix g_;
  mutable SparseMatrix B_;
  mutable SparseMatrix mB_[2];
  mutable Matrix b_;
  mutable Matrix mb_[2];

  /** Loss */
  Loss loss_;
  /** The parameter for non-trivial loss kernels */
  Scalar loss_reg_, sqrt_loss_reg_, squared_loss_reg_;

  /** Index */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> index_;

  /** Sent */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> sent_;

  /** Received */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> recv_;

  /** Cholesky */
  mutable SparseCholeskyFactorization LG_;

  mutable bool ready_;

 public:
  /// CONSTRUCTORS AND MUTATORS
  /** Basic constructor.
   * - node is the index of the node
   * - measurements is a vector of inter-node relative pose measurements
   *   defining the  pose-graph SLAM problem to be solved
   * - reg_G is a positive number added to G
   */
  DChordalProblem(int node, const measurements_t& measurements,
                  Scalar reg_G = 1e-5, Loss loss = Loss::None,
                  Scalar loss_reg = 1.0);

  /// ACCESSORS

  /** Returns the node index */
  int node() const { return node_; }

  /** Returns the index */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> const& index() const {
    return index_;
  }

  /** Returns the sent */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> const& sent() const {
    return sent_;
  }

  /** Returns the recv */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> const& recv() const {
    return recv_;
  }

  /** Returns the the number of poses */
  const Eigen::Matrix<int, 2, 1>& n() const { return n_; }

  /** Returns the offsets */
  const Eigen::Matrix<int, 2, 1>& s() const { return s_; }

  /** Returns the the number of measurements */
  const Eigen::Matrix<int, 2, 1>& m() const { return m_; }

  /** Returns the dimension of SE(d) */
  int d() const { return d_; }

  /** Returns the rows for each rotation/translation */
  int p() const { return p_; }

  const SparseMatrix& G() const { return G_; }
  const SparseMatrix& S() const { return S_; }
  const SparseMatrix& D() const { return D_; }
  const SparseMatrix& H() const { return H_; }
  const SparseMatrix& B() const { return B_; }
  const Matrix& b() const { return b_; }

  int check() const {
    assert(ready_ == true);

    if (!ready_) {
      LOG(ERROR) << "The problem is not setup." << std::endl;

      return -1;
    }

    return 0;
  }

  virtual int reset() const;

  virtual int solve(const Matrix& X, const Matrix& g, Matrix& Xn) const = 0;

  virtual int evaluate_Df(const Matrix& X, Matrix& Df) const {
    if (check()) return -1;

    if (loss_ == Loss::None) {
      Df = g_;
      Df.noalias() += H_ * X;
    } else {
      assert(0);
    }

    return 0;
  }

  virtual int evaluate_g(const Matrix& X, Matrix& g) const {
    if (check()) return -1;

    if (loss_ == Loss::None) {
      g = g_;
      g.noalias() += S_ * X;
    } else {
      assert(0);
    }

    return 0;
  }

  virtual int iterate(const Matrix& X, Matrix& g, Matrix& Xn) const {
    evaluate_g(X, g);
    solve(X, g, Xn);

    return 0;
  }
};

class DChordalProblem_R : public DChordalProblem {
 public:
  DChordalProblem_R(int node, const measurements_t& measurements,
                    Scalar reg_G = 1e-5, Loss loss = Loss::None,
                    Scalar loss_reg = 1.0);

  virtual int evaluate_Df(const Matrix& X, Matrix& Df) const override {
    if (check()) return -1;

    if (node_ == 0) {
      if (loss_ == Loss::None) {
        Df = g_;
        Df.noalias() += H_.bottomRows((n_[0] - 1) * d_) * X;
      } else {
        assert(0);
      }
    } else {
      if (loss_ == Loss::None) {
        Df = g_;
        Df.noalias() += H_ * X;
      } else {
        assert(0);
      }
    }

    return 0;
  }

  virtual int evaluate_g(const Matrix& X, Matrix& g) const override {
    if (check()) return -1;

    if (node_ == 0) {
      if (loss_ == Loss::None) {
        g = g_;
        g.noalias() += S_.bottomRows((n_[0] - 1) * d_) * X;
      } else {
        assert(0);
      }
    } else {
      if (loss_ == Loss::None) {
        g = g_;
        g.noalias() += S_ * X;
      } else {
        assert(0);
      }
    }

    return 0;
  }

  virtual int solve(const Matrix& R, const Matrix& g,
                    Matrix& Rn) const override {
    if (check()) return -1;

    if (node_ == 0) {
      Rn.setZero(n_[0] * d_, d_);
      Rn.topRows(d_) = R.topRows(d_);
      Rn.bottomRows((n_[0] - 1) * d_) =
          -LG_.solve(g);
    } else {
      Rn = -LG_.solve(g);
    }

    return 0;
  }

  int setup() const;
};

class DChordalProblem_t : public DChordalProblem {
 public:
  DChordalProblem_t(int node, const measurements_t& measurements,
                    Scalar reg_G = 1e-5, Loss loss = Loss::None,
                    Scalar loss_reg = 1.0);

  virtual int solve(const Matrix& t, const Matrix& g,
                    Matrix& tn) const override {
    if (check()) return -1;

    tn = -LG_.solve(g);

    return 0;
  }

  int setup(const Matrix& R) const;

  const Matrix& R() const { return R_; }

 protected:
  mutable Matrix R_;
};
}  // namespace DChordal
