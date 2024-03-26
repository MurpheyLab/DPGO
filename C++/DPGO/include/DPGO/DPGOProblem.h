#pragma once

#define SIMPLE 1

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/RelativePoseMeasurement.h>
#include <DPGO/SOdProduct.h>

namespace DPGO {
class DPGOProblem {
 protected:
  /** Max and min rescale */
  static constexpr Scalar max_rescale_ = 1.0;
  static constexpr Scalar min_rescale_ = 0.01;

  /** Index of node */
  int node_;

  /** Regualrized constant for matrix G*/
  Scalar reg_G_;

  /** Number of poses */
  Eigen::Matrix<int, 2, 1> n_ = Eigen::Matrix<int, 2, 1>::Zero();

  /** Offsets */
  Eigen::Matrix<int, 2, 1> s_ = Eigen::Matrix<int, 2, 1>::Zero();

  /** Size */
  Eigen::Matrix<int, 2, 1> size_ = Eigen::Matrix<int, 2, 1>::Zero();

  /** Number of measurements */
  Eigen::Matrix<int, 2, 1> m_ = Eigen::Matrix<int, 2, 1>::Zero();

  /** measurements */
  measurements_t measurements_;

  /** intra-node measurements */
  measurements_t intra_measurements_;

  /** inter-node measurements */
  measurements_t inter_measurements_;

  /** Dimension of PGO */
  int d_ = 0;

  SparseMatrix M_;
  std::array<std::array<SparseMatrix, 2>, 2> mM_;
  SparseMatrix B_, mB_[2];
  SparseMatrix S_;
  SparseMatrix P0_;
  SparseMatrix H_;
  SparseMatrix P_;
  SparseMatrix U_;
  SparseMatrix T_;

  SparseMatrix D0_;
  SparseMatrix Q0_;

  DiagonalMatrix DiagT0_;
  SparseMatrix N0_;
  SparseMatrix V0_;

  SparseMatrix E_;
  SparseMatrix F_;

  std::array<std::array<int, 2>, 2> outer_index_G_;
  std::array<std::array<std::vector<int>, 2>, 2> inner_index_G_;
  std::array<std::array<int, 2>, 2> inner_nnz_G_;

  std::array<std::array<int, 2>, 2> outer_index_mG_;
  std::array<std::array<std::vector<int>, 2>, 2> inner_index_mG_;
  std::array<std::array<int, 2>, 2> inner_nnz_mG_;

  std::array<std::array<int, 2>, 2> outer_index_D_;
  std::array<std::array<std::vector<int>, 2>, 2> inner_index_D_;
  std::array<std::array<int, 2>, 2> inner_nnz_D_;

  std::array<std::array<int, 2>, 2> outer_index_Q_;
  std::array<std::array<std::vector<int>, 2>, 2> inner_index_Q_;
  std::array<std::array<int, 2>, 2> inner_nnz_Q_;

  int outer_index_T_;
  std::vector<int> inner_index_T_;
  int inner_nnz_T_;

  int outer_index_N_;
  std::vector<int> inner_index_N_;
  int inner_nnz_N_;

  int outer_index_V_;
  std::vector<int> inner_index_V_;
  int inner_nnz_V_;

  SparseMatrix G_;
  std::array<std::array<SparseMatrix, 2>, 2> mG_;
  SparseMatrix D_;
  SparseMatrix Q_;
  SparseMatrix V_;
  SparseMatrix N_;
  DiagonalMatrix DiagT_;
  SparseMatrix ScaleN_;

  mutable Vector SquaredErrNorm_;
  mutable Vector DiagReScale_;

  mutable Eigen::Map<Vector> Gval_ = Eigen::Map<Vector>(nullptr, 0);
  mutable Eigen::Map<Vector> mGval_[2][2] = {
      {Eigen::Map<Vector>(nullptr, 0), Eigen::Map<Vector>(nullptr, 0)},
      {Eigen::Map<Vector>(nullptr, 0), Eigen::Map<Vector>(nullptr, 0)}};
  mutable Eigen::Map<Vector> Dval_ = Eigen::Map<Vector>(nullptr, 0);
  mutable Eigen::Map<Vector> Qval_ = Eigen::Map<Vector>(nullptr, 0);
  mutable Eigen::Map<Vector> Tval_ = Eigen::Map<Vector>(nullptr, 0);
  mutable Eigen::Map<Vector> Nval_ = Eigen::Map<Vector>(nullptr, 0);
  mutable Eigen::Map<Vector> Vval_ = Eigen::Map<Vector>(nullptr, 0);
  mutable Eigen::Map<Vector> ScaleNval_ = Eigen::Map<Vector>(nullptr, 0);

  /** Loss */
  Loss loss_;
  /** The parameter for non-trivial loss kernels */
  Scalar loss_reg_, sqrt_loss_reg_, squared_loss_reg_;

  /** An Eigen sparse linear solver that encodes the Cholesky factor L used
   * in the computation of the orthogonal projection function */
  mutable SparseCholeskyFactorization L_;

  /** The preconditioning strategy to use when running the Riemannian
   * trust-region algorithm */
  Preconditioner preconditioner_;

  /** Rescale */
  Rescale rescale_;

  /** Diagonal Jacobi preconditioner */
  DiagonalMatrix Jacobi_precon_;

  /** Incomplete Cholesky Preconditioner */
  IncompleteCholeskyFactorization* iChol_precon_ = nullptr;

  /** Tikhonov-regularized Cholesky Preconditioner */
  SparseCholeskyFactorization reg_Chol_precon_;

  /** Upper-bound on the admissible condition number of the regularized
   * approximate Hessian matrix used for Cholesky preconditioner */
  Scalar reg_Chol_precon_max_cond_;

  /** SO(d)^n */
  SOdProduct SP_;

  /** Index */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> index_;

  /** Sent */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> sent_;

  /** Received */
  std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> recv_;

 public:
  // The iterate for DPGO
  struct PGOIterate {
    int iter;

    // X = [t^a R^a]' contains X^a
    Matrix X;

    // Z = [t^a R^a t^b R^b ....]' contains X^a and its neighbor poses X^b, ...
    Matrix Z;

    // coefficients used for Nesterov accelerated gradient method
    Scalar a, b, c;
  };

 public:
  /// CONSTRUCTORS AND MUTATORS

  /** Default constructor; doesn't actually do anything */
  DPGOProblem() {}

  /** Basic constructor.  Here
   * - node is the index of the node
   * - measurements is a vector of relative pose measurements defining the
   *      pose-graph SLAM problem to be solved.
   * - reg_G is a positive number added to G
   * - loss is an enum type specifying the loss type (trivial, Huber and German
   *   loss).
   *  - preconditioner is an enum type specifying the preconditioning strategy
   *      to employ
   */
  DPGOProblem(int node, const measurements_t& measurements, Scalar reg_G = 1e-5,
              const Loss& loss = Loss::None,
              const Preconditioner& preconditioner =
                  Preconditioner::RegularizedCholesky,
              Scalar reg_chol_precon_max_cond = 1e6,
              Rescale rescale = Rescale::Dynamic, Scalar loss_reg = 1.0);

  /// ACCESSORS

  /** Returns the node index */
  int node() const { return node_; }

  /** Returns the preconditioning strategy */
  Preconditioner preconditioner() const { return preconditioner_; }

  /** Returns the SO(d)^n */
  SOdProduct const& SOd() const { return SP_; }

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

  /** Rescale of the problem */
  Rescale rescale() const { return rescale_; }

  /** Returns the maximum admissible condition number for the regularized
   * Cholesky preconditioner */
  Scalar regularized_Cholesky_preconditioner_max_condition() const {
    return reg_Chol_precon_max_cond_;
  }

  /** Returns the regularized constant for G */
  Scalar regularized_G() const { return reg_G_; }

  /** Loss types */
  Loss loss() const { return loss_; }

  /** Returns the number of poses */
  const Eigen::Matrix<int, 2, 1>& n() const { return n_; }

  /** Returns the offsets */
  const Eigen::Matrix<int, 2, 1>& s() const { return s_; }

  /** Returns the number of measurements */
  const Eigen::Matrix<int, 2, 1>& m() const { return m_; }

  /** Returns the dimension of SE(d)---must be 2 or 3 */
  int d() const { return d_; }

  const measurements_t& intra_measurements() const {
    return intra_measurements_;
  }

  const measurements_t& inter_measurements() const {
    return inter_measurements_;
  }

  const SparseMatrix& G() const { return G_; }
  const SparseMatrix& B() const { return B_; }
  const SparseMatrix& D() const { return D_; }
  const SparseMatrix& S() const { return S_; }
  const SparseMatrix& Q() const { return Q_; }
  const SparseMatrix& P() const { return P_; }
  const SparseMatrix& P0() const { return P0_; }
  const SparseMatrix& H() const { return H_; }
  const DiagonalMatrix& T() const { return DiagT_; }
  const SparseMatrix& N() const { return N_; }
  const SparseMatrix& U() const { return U_; }
  const SparseMatrix& V() const { return V_; }

  /** Recover the translation given the rotation R and Euclidean graident g*/
  template <typename Rotation, typename Translation>
  int recover_translations(const Eigen::MatrixBase<Rotation>& R,
                           const Matrix& g,
                           Eigen::MatrixBase<Translation>& t) const {
    assert(g.cols() == d_ && g.rows() == (d_ + 1) * n_[0]);
    assert(R.cols() == d_ && R.rows() == d_ * n_[0]);

    if (g.cols() != d_ || g.rows() != (d_ + 1) * n_[0] || R.cols() != d_ ||
        R.rows() != d_ * n_[0]) {
      LOG(ERROR) << "Inconsistent inputs." << std::endl;

      return -1;
    }

    Matrix temp = g.topRows(n_[0]);
    temp.noalias() += mG_[0][1] * R;
    t.noalias() = -L_.solve(temp);

    return 0;
  }

  /// Riemannian Geometric Concepts
  /** Metric */
  double metric(const Matrix& V1, const Matrix& V2) const {
    return (V1.transpose() * V2).trace();
  }

  /** Given a matrix Y in the domain D of the DPGO optimization problem and a
   * tangent vector Ydot in T_D(Y), this function returns the point Yplus in D
   * obtained by retracting along Ydot */
  int retract(const Matrix& Y, const Matrix& Ydot, const Matrix& g,
              Matrix& Yplus) const;

  /** project Ydot onto the tangent space of Y */
  int full_tangent_space_projection(const Matrix& Y, const Matrix& Ydot,
                                    Matrix& gradF_Y) const;

  /** project Ydot onto the tangent space of Y */
  int reduced_tangent_space_projection(const Matrix& Y, const Matrix& Ydot,
                                       Matrix& gradF_Y) const;

  // Function G(X|Z) = 0.5 * <G*X, X> + <g, X> + f
  /** Given a matrix X, this function computes and returns G(X|Z) */
  int evaluate_G(const Matrix& Y, const Matrix& g, Scalar f, Scalar& G) const;

  // Function G(X|Z) = 0.5 * <G*(X-X0), X-X0> + <Dfobj, X-X0> + fobj
  /** Given a matrix X, this function computes and returns G(X|Z) */
  int evaluate_G(const Matrix& Y, const Matrix& Y0, const Matrix& Dfobj,
                 Scalar fobj, Scalar& G) const;

  /** Given X and Z, this function computes the simlified gradient g in G(X|Z)
   * and inter-node errors*/
  int evaluate_g_and_f0(const Matrix& Z, Matrix& g, Scalar& f0, Matrix& Dfobj,
                        Scalar& fobj, Matrix& DfobjE, Scalar& fobjE) const;

  int evaluate_none_g_and_f0(const Matrix& Z, Matrix& g, Scalar& f0) const;

  int evaluate_g_and_f0_rescale(const Matrix& Z, Matrix& g, Scalar& f0,
                                Matrix& Dfobj, Scalar& fobj, Matrix& DfobjE,
                                Scalar& fobjE, int& rescale_count,
                                int max_rescale_count = 25) const;

  int evaluate_g_and_f(const Matrix& Z, const Matrix& Z0, Scalar G,
                       const Matrix& DfobjE0, Scalar fobjE0, Matrix& g,
                       Scalar& f, Matrix& Dfobj, Scalar& fobj, Matrix& DfobjE,
                       Scalar& fobjE) const;

  int evaluate_none_g_and_f(const Matrix& Z, const Matrix& Z0, Scalar G,
                            Matrix& g, Scalar& f, Scalar& fobj) const;

  int evaluate_g_and_f_rescale(const Matrix& Z, const Matrix& Z0, Scalar G,
                               const Matrix& DfobjE0, Scalar fobjE0, Matrix& g,
                               Scalar& f, Matrix& Dfobj, Scalar& fobj,
                               Matrix& DfobjE, Scalar& fobjE,
                               int& rescale_count,
                               int max_rescale_count = 25) const;

  int evaluate_g_and_Df(const Matrix& Z, Matrix& g, Matrix& Df) const;

  /** Given a matrix Y, this function computes and returns nabla F(Y), the
   * full *Euclidean* gradient of F at Y. */
  int full_Euclidean_gradient_G(const Matrix& Y, Matrix const& g,
                                Matrix& nablaF_Y) const {
    assert(Y.rows() == (d_ + 1) * n_[0]);
    assert(Y.cols() == d_);
    assert(g.rows() == Y.rows());
    assert(g.cols() == d_);

    nablaF_Y = g;
    nablaF_Y.noalias() += G_ * Y;

    return 0;
  }

  /** Given a matrix Y in the domain D of the DPGO optimization problem, this
   * function computes and returns the full *Euclidean* and *Riemannian*
   * gradient of F at Y */
  int full_Riemannian_gradient_G(const Matrix& Y, const Matrix& g,
                                 Matrix& nablaF_Y, Matrix& gradF_Y) const {
    full_Euclidean_gradient_G(Y, g, nablaF_Y);
    return full_tangent_space_projection(Y, nablaF_Y, gradF_Y);
  }

  /** Given a matrix Y, this function computes and returns nabla F(Y), the
   * reduced *Euclidean* gradient of F at Y w.r.t. the rotation. */
  int reduced_Euclidean_gradient_G(const Matrix& Y, Matrix const& g,
                                   Matrix& nablaF_Y) const {
    assert(Y.rows() == (d_ + 1) * n_[0]);
    assert(Y.cols() == d_);
    assert(g.rows() == Y.rows());
    assert(g.cols() == d_);

    // A simplified formulation is used and we *only* need to compute the
    // Eulidean gradient w.r.t. the rotation
    nablaF_Y = g.bottomRows(d_ * n_[0]);
    nablaF_Y.noalias() += G_.bottomRows(d_ * n_[0]) * Y;

    return 0;
  }

  /** Given a matrix Y in the domain D of the DPGO optimization problem, this
   * function computes and returns the reduced *Riemannian* gradient of F at Y
   */
  int reduced_Riemannian_gradient_G(const Matrix& Y, const Matrix& nablaF_Y,
                                    Matrix& gradF_Y) const {
    assert(Y.rows() == (d_ + 1) * n_[0]);
    assert(Y.cols() == d_);
    assert(nablaF_Y.rows() == d_ * n_[0]);
    assert(nablaF_Y.cols() == d_);

    return reduced_tangent_space_projection(Y, nablaF_Y, gradF_Y);
  }

  /** Given a matrix Y in the domain D of the DPGO optimization problem, this
   * function computes and returns the reduced *Euclidean* and *Riemannian*
   * gradient of F at Y */
  int reduced_Riemannian_gradient_G(const Matrix& Y, const Matrix& g,
                                    Matrix& nablaF_Y, Matrix& gradF_Y) const {
    reduced_Euclidean_gradient_G(Y, g, nablaF_Y);
    return reduced_tangent_space_projection(Y, nablaF_Y, gradF_Y);
  }

  /** Given a matrix Y in the domain D of the DPGO optimization problem, the
   * *Euclidean* gradient nablaF_Y of F at Y, and a tangent vector dotY in
   * T_D(Y), the tangent space of the domain of the optimization problem at Y,
   * this function computes and returns Hess F(Y)[dotY], the action of the
   * Riemannian Hessian on dotY */
  int reduced_Riemannian_Hessian_vector_product(const Matrix& Y,
                                                const Matrix& nablaF_Y,
                                                const Matrix& Ydot,
                                                Matrix& Hess) const;

  int reduced_Riemannian_Hessian_vector_product(const Matrix& Y,
                                                const Matrix& Ydot,
                                                const Matrix& g,
                                                Matrix& nablaF_Y,
                                                Matrix& Hess) const {
    reduced_Euclidean_gradient_G(Y, g, nablaF_Y);

    return reduced_Riemannian_Hessian_vector_product(Y, nablaF_Y, Ydot, Hess);
  }

  Matrix precondition(const Matrix& Y, const Matrix& Ydot) const;

  /** Intermediate proximal method */
  int proximal(const Matrix& Z, const Matrix& DF, Matrix& X) const;

 protected:
  /** Inter-node error and its derivative*/
  int evaluate_E(const Matrix& Z, Vector& DiagReScale, Matrix& DfobjE,
                 Scalar& fobjE) const;

  /** Given Z, this function computes g in G(X|Z) */
  int evaluate_g(const Matrix& Z, Matrix& g) const;

  /** Give Z, this function computes the Euclidean gradient at Z*/
  int evaluate_Df(const Matrix& Z, const Matrix& g, Matrix& Df) const;

  /** Update diagonal regularizer */
  int update_diag_reg(const Vector& SquaredErrNorm);

  /** Rescale the majorimization minimization methods */
  int rescale(const Vector& DiagReScale) const;

  /** Update the quadratic matrix */
  int update_quadratic_mat(const Vector& DiagReScale) const;
};
}  // namespace DPGO
