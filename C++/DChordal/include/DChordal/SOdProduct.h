#pragma once

#include <random>  // For sampling random points on the manifold

#include <Eigen/Dense>

#include <DChordal/DChordal_types.h>

namespace DChordal {
class SOdProduct {
 private:
  // Number of vectors in each orthonormal d-frame
  int d_;

  // Number of copies of SO(d)^n in the product
  int n_;

 public:
  /// CONSTRUCTORS AND MUTATORS

  // Default constructor -- sets all dimensions to 0
  SOdProduct() {}

  SOdProduct(int d, int n) : d_(d), n_(n) {}

  void set_d(int d) { d_ = d; }
  void set_n(int n) { n_ = n; }

  /// ACCESSORS
  unsigned int get_d() const { return d_; }
  unsigned int get_n() const { return n_; }

  /// GEOMETRY

  /** Given a generic matrix A in R^{kn x p}, this function computes the
   * projection of A onto R (closest point in the Frobenius norm sense).  */
  template <typename Euclidean, typename Manifold>
  int project(const Eigen::MatrixBase<Euclidean> &A,
              Eigen::MatrixBase<Manifold> &R) const {
    assert(A.rows() == n_ * d_);
    assert(A.cols() == d_);
    assert(R.rows() == A.rows());
    assert(R.cols() == A.cols());

#pragma omp parallel for
    for (int i = 0; i < n_; ++i) {
      // Compute the (thin) SVD of the ith block of A
      Eigen::JacobiSVD<Matrix> svd(A.block(i * d_, 0, d_, d_),
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);

      Scalar detU = svd.matrixU().determinant();
      Scalar detV = svd.matrixV().determinant();

      if (detU * detV > 0) {
        R.block(i * d_, 0, d_, d_).noalias() =
            svd.matrixU() * svd.matrixV().transpose();
      } else {
        Matrix Uprime = svd.matrixU();
        Uprime.col(Uprime.cols() - 1) *= -1;
        R.block(i * d_, 0, d_, d_).noalias() =
            Uprime * svd.matrixV().transpose();
      }
    }

    return 0;
  }

  /** Helper function -- this computes and returns the product
   *
   *  P = A * SymBlockDiag(B^T * C)
   *
   * where A, B, and C are kn x p matrices.   */
  template <typename Manifold, typename Other, typename Euclidean,
            typename Riemannian>
  int SymBlockDiagProduct(const Eigen::MatrixBase<Manifold> &A,
                          const Eigen::MatrixBase<Other> &B,
                          const Eigen::MatrixBase<Euclidean> &C,
                          Eigen::MatrixBase<Riemannian> &P) const {
    assert(A.rows() == n_ * d_);
    assert(A.cols() == d_);
    assert(B.rows() == A.rows());
    assert(B.cols() == A.cols());
    assert(C.rows() == A.rows());
    assert(C.cols() == A.cols());
    assert(P.rows() == A.rows());
    assert(P.cols() == A.cols());

#pragma omp parallel for
    for (int i = 0; i < n_; ++i) {
      // Compute block product Bi' * Ci
      Matrix G =
          C.middleRows(i * d_, d_) * B.middleRows(i * d_, d_).transpose();
      // Symmetrize this block
      Matrix S = .5 * (G + G.transpose());
      // Compute Ai * S and set corresponding block of R
      P.middleRows(i * d_, d_).noalias() = S * A.middleRows(i * d_, d_);
    }

    return 0;
  }

  /** Given an element Y in M and a matrix V in T_X(R^{kn x p}) (that is, a (p
   * x kn)-dimensional matrix V considered as an element of the tangent space to
   * the *entire* ambient Euclidean space at X), this function computes and
   * returns the projection of V onto T_X(M), the tangent space of M at X. */
  template <typename Manifold, typename Euclidean, typename Riemannian>
  int Proj(const Eigen::MatrixBase<Manifold> &Y,
           const Eigen::MatrixBase<Euclidean> &V,
           Eigen::MatrixBase<Riemannian> &P) const {
    SymBlockDiagProduct(Y, Y, V, P);
    P = V - P;

    return 0;
  }

  /** Given an element Y in M and a tangent vector V in T_Y(M), this function
   * computes the retraction along V at Y using the QR-based retraction
   * specified in eq. (4.8) of Absil et al.'s  "Optimization Algorithms on
   * Matrix Manifolds").
   */
  template <typename Manifold, typename Tangent, typename Other>
  int retract(const Eigen::MatrixBase<Manifold> &Y,
              const Eigen::MatrixBase<Tangent> &V,
              Eigen::MatrixBase<Other> &R) const {
    R = Y + V;
    return project(R, R);
  }

  /** Sample a random point on M.  */
  Matrix random_sample() const {
    Matrix R(d_ * n_, d_);

    R.setRandom();
    project(R, R);

    return R;
  }
};

}  // namespace DChordal
