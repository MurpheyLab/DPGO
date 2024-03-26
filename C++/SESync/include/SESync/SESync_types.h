/** A set of typedefs describing the types of matrices and factorizations that
 * will be used in the SE-Sync algorithm.
 *
 * Copyright (C) 2016 - 2018 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Optimization/Riemannian/TNT.h"

namespace SESync {

/** Some useful typedefs for the SE-Sync library */
typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

/** We use row-major storage order to take advantage of fast (sparse-matrix) *
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

/** The specific formulation of special Euclidean synchronization problem to
 * solve */
enum class Formulation {
  /** Construct and solve the simplified version of the special Euclidean
   * synchronization problem obtained by analytically eliminating the
   *  translational states from the estimation (cf. Problem 4 in the SE-Sync
   * tech report).
   */
  Simplified,

  /** Construct and solve the formulation of the special Euclidean
   * synchronization problem that explicitly estimates both rotational and
   * translational states (cf. Problem 2 in the SE-Sync tech report).
   */
  Explicit,

  /** Construct and solve the rotation synchronization (rotation averaging)
     problem determined by the rotational data (ignoring all translations) */
  SOSync
};

/** The type of factorization to use when computing the action of the orthogonal
 * projection operator Pi when solving the Simplified form of the special
 * Euclidean synchronization problem */
enum class ProjectionFactorization { Cholesky, QR };

/** The set of available preconditioning strategies to use in the Riemannian
 * Trust Region when solving this problem */
enum class Preconditioner { None, Jacobi, RegularizedCholesky };

/** The strategy to use for constructing an initial iterate */
enum class Initialization { Chordal, Random };

/** A typedef for a user-definable function that can be used to
 * instrument/monitor the performance of the internal Riemannian
 * truncated-Newton trust-region optimization algorithm as it runs (see the
 * header file Optimization/Riemannian/TNT.h for details). */
typedef Optimization::Riemannian::TNTUserFunction<Matrix, Matrix, Scalar, Matrix>
    SESyncTNTUserFunction;

} // namespace SESync
