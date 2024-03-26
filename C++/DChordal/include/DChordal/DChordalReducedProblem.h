#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <DChordal/DChordal_types.h>
#include <DChordal/DChordal_utils.h>
#include <DChordal/RelativePoseMeasurement.h>
#include <DChordal/SOdProduct.h>

namespace DChordal {
class DChordalReducedProblem {
 protected:
  /** Index of node */
  int node_;

  /** Regualrized constant for matrix G*/
  Scalar reg_G_;

  /** Number of neighbours */
  int nn_ = 0;

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

  mutable Matrix G_, D_, H_, S_;
  mutable Matrix g_;
  mutable SparseMatrix B_;
  mutable Matrix b_;

  /** Loss */
  Loss loss_;
  /** The parameter for non-trivial loss kernels */
  Scalar loss_reg_, sqrt_loss_reg_, squared_loss_reg_;

  /** Index */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> index_;
  std::map<int, int> n_index_;

  /** Sent */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> sent_;
  std::map<int, int> n_sent_;

  /** Received */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> recv_;
  std::map<int, int> n_recv_;

  mutable bool ready_;

 public:
  /// CONSTRUCTORS AND MUTATORS
  /** Basic constructor.
   * - node is the index of the node
   * - measurements is a vector of inter-node relative pose measurements
   *   defining the  pose-graph SLAM problem to be solved
   * - reg_G is a positive number added to G
   */
  DChordalReducedProblem(int node, const measurements_t& measurements,
                         Scalar reg_G = 1e-5, const Loss& loss = Loss::None,
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

  /** Returns the index */
  std::map<int, int> const& n_index() const { return n_index_; }

  /** Returns the sent */
  std::map<int, int> const& n_sent() const { return n_sent_; }

  /** Returns the recv */
  std::map<int, int> const& n_recv() const { return n_recv_; }

  /** Loss types */
  Loss loss() const { return loss_; }

  /** Returns the the number of neighbors */
  int nn() const { return nn_; }

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

  virtual int reset() const;

  const Matrix& G() const { return G_; }
  const Matrix& S() const { return S_; }
  const Matrix& D() const { return D_; }
  const Matrix& H() const { return H_; }
  const SparseMatrix& B() const { return B_; }
  const Matrix& b() const { return b_; }
  const Matrix& g() const { return g_; }

  int check() const {
    assert(ready_ == true);

    if (!ready_) {
      LOG(ERROR) << "The problem is not setup." << std::endl;

      return -1;
    }

    return 0;
  }

  virtual int solve(const Matrix& X, const Matrix& g, Matrix& Xn) const = 0;
  // virtual int jacobi(const Matrix& X, const Matrix& Df, Matrix& Xn) const =
  // 0;

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
    if (check()) return -1;

    evaluate_g(X, g);
    solve(X, g, Xn);

    return 0;
  }

  const measurements_t& intra_measurements() const { return intra_measurements_; }
};

class DChordalReducedProblem_R : public DChordalReducedProblem {
 public:
  DChordalReducedProblem_R(int node, const measurements_t& measurements,
                           Scalar reg_G = 1e-5, const Loss& loss = Loss::None,
                           Scalar loss_reg = 1.0);

  int setup(const Matrix& X) const;

  virtual int reset() const override {
    DChordalReducedProblem::reset();
    R_.resize(0, 0);

    return 0;
  }

  virtual int solve(const Matrix& R, const Matrix& g,
                    Matrix& Rn) const override {
    if (check()) return -1;

    Rn.noalias() = -Ginv_ * g;
    return 0;
  }

  // virtual int jacobi(const Matrix& R, const Matrix& Df,
  // Matrix& Rn) const override {
  // if (check()) return -1;

  // Rn = R;
  // Rn.noalias() -= 0.5 * Dinv_ * Df;

  // return 0;
  //}

  const Matrix& R() const { return R_; };

 protected:
  mutable Matrix Ginv_, Dinv_;
  mutable Matrix R_;
};

class DChordalReducedProblem_t : public DChordalReducedProblem {
 public:
  DChordalReducedProblem_t(int node, const measurements_t& measurements,
                           Scalar reg_G = 1e-5, const Loss& loss = Loss::None,
                           Scalar loss_reg = 1.0);

  int setup(const Matrix& X, const Matrix& nR_) const;

  virtual int reset() const override {
    DChordalReducedProblem::reset();

    X_.resize(0, 0);
    nR_.resize(0, 0);

    return 0;
  }

  virtual int solve(const Matrix& t, const Matrix& g,
                    Matrix& tn) const override {
    tn = -g / G_(0, 0);

    return 0;
  }

  template <typename Rotation, typename Translation>
  int recover_translations(const Eigen::MatrixBase<Rotation>& R,
                           Eigen::MatrixBase<Translation>& t) const {
    assert(R.cols() == d_ && R.rows() == d_ * n_[0]);

    Matrix temp = P_ * R;
    t.noalias() = -LL_.solve(temp);
    t.rowwise() -= t.row(0);

    return 0;
  }

  // virtual int jacobi(const Matrix& t, const Matrix& Df,
  // Matrix& tn) const override {
  // tn = t;
  // tn -= 0.95 * Df / D_(0, 0);

  // return 0;
  //}

 protected:
  SparseCholeskyFactorization LL_;
  SparseMatrix L_, P_;

  mutable Matrix X_;
  mutable Matrix nR_;
};
}  // namespace DChordal
