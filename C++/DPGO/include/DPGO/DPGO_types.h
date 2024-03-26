#pragma once

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <utility>

#include <Optimization/Riemannian/TNT.h>

namespace DPGO {
/** Some useful typedefs for the SE-Sync library */
typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> Array;
typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> Array1X;

/** We use row-major storage order to take advantage of fast (sparse-matrix) *
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

/** The type of the sparse Cholesky factorization to use in the computation of
 * the orthogonal projection operation */
typedef Eigen::CholmodDecomposition<SparseMatrix> SparseCholeskyFactorization;

/** The type of the incomplete Cholesky decomposition we will use for
 * preconditioning the conjugate gradient iterations in the RTR method */
typedef Eigen::IncompleteCholesky<Scalar> IncompleteCholeskyFactorization;

/** The set of available preconditioning strategies to use in the Riemannian
 * Trust Region when solving this problem */
enum class Preconditioner {
  None,
  Jacobi,
  IncompleteCholesky,
  RegularizedCholesky
};

/** How to update the quadratic approximations for robust kernels */
enum class Rescale {
  Static,
  Dynamic
};

struct Index {
  int node = 0;  // node index in the graph
  int pose = 0;  // pose index in the node

  Index() : node(0), pose(0) {}
  Index(int node, int pose) : node(node), pose(pose) {}
};

/** A typedef for a user-definable function that can be used to
 * instrument/monitor the performance of the internal Riemannian
 * truncated-Newton trust-region optimization algorithm as it runs (see the
 * header file Optimization/Smooth/TNT.h for details). */
typedef Optimization::Riemannian::TNTUserFunction<Matrix, Matrix, Scalar,
                                                  Matrix>
    TNTUserFunction;

typedef Optimization::Riemannian::TNTResult<Matrix, Scalar> TNTResults;

/** The loss kernels to formulate inter-node cost*/
enum class Loss { None, Huber, GemanMcClure, Welsch };

/** The set of available accelerated schemes */
enum class Scheme {
  /** The majorization minimization method */
  MM,
  /** The accelerated majorization minimization method */
  AMM,
};

// Solver Options
struct Options {
  //======================================================================
  // DPGO PARAMETERS
  //======================================================================
  //------------------------------------------------------
  // GENERAL SETUP
  //------------------------------------------------------
  /** The accelerated scheme for DPGO */
  Scheme scheme = Scheme::AMM;

  /** Regularizer for majorization minimization matrix */
  Scalar regularizer = 1e-10;

  /** acceptable gradient norm over objective value */
  Scalar accepted_delta = 5e-4;

  /** Whether to print output as the algorithm runs */
  bool verbose = false;

  //------------------------------------------------------
  // NESTEROV PARAMETERS
  //------------------------------------------------------
  /** The convex combination parameter for soft and hard adaptive restart */
  Scalar eta[2] = {5e-4, 2.5e-2};

  /** The control parameter for adaptive restart */
  Scalar psi = 1e-10;

  /** The control parameter for selecting X^{k+1/2} and X^{k+1} */
  Scalar phi = 1e-6;

  /** The maximum adaptive soft restart hits */
  int max_soft_restart_hits[2] = {10, 25};

  /** The period to count oscillations */
  int oscillation_cnt_period = 15;

  /** The maximum number of oscillations */
  int max_oscillations = 12;

  /** The loss kernel */
  Loss loss = Loss::None;

  //------------------------------------------------------
  // NESTEROV PARAMETERS
  //------------------------------------------------------
  /** The loss parameter (only used for non-trivial losses) */
  Scalar loss_reg = 1.0;

  /** If dynamically updating the quadratic matrices for robust kernels*/
  Rescale rescale = Rescale::Dynamic;

  /** The max rescale counts for robust loss kernels */
  int max_rescale_count = 5;

  //------------------------------------------------------
  // Riemannian OPTIMIZATION STOPPING CRITERIA
  //------------------------------------------------------
  /** Stopping tolerance for the norm of the Riemannian gradient */
  Scalar grad_norm_tol = 5e-3;

  /** Stopping criterion based upon the relative decrease in function value */
  Scalar rel_func_decrease_tol = 1e-6;

  /** Stopping criterion based upon the norm of an accepted update step */
  Scalar stepsize_tol = 1e-4;

  /** Maximum permitted number of (outer) iterations of the Riemannian
   * trust-region method */
  int max_iterations = 10;

  /** Maximum permitted number of (outer) accepted iterations of the Riemannian
   * trust-region method */
  int max_iterations_accepted = 1;

  /** The preconditioning strategy to use in the Riemannian trust-region
50 algorithm*/
  Preconditioner preconditioner = Preconditioner::RegularizedCholesky;

  /** Maximum admissible condition number for the regularized Cholesky
   * preconditioner */
  Scalar reg_Cholesky_precon_max_condition_number = 1e6;

  /** Stopping tolerance for the norm of the preconditioned Riemannian gradient
   */
  Scalar preconditioned_grad_norm_tol = 1e-4;

  /** Maximum number of inner (truncated conjugate-gradient) iterations to
   * perform per out iteration */
  int max_tCG_iterations = 10000;

  /// These next two parameters define the stopping criteria for the truncated
  /// preconditioned conjugate-gradient solver running in the inner loop --
  /// they control the tradeoff between the quality of the returned
  /// trust-region update step (as a minimizer of the local quadratic model
  /// computed at each iterate) and the computational expense needed to generate
  /// that update step.  You probably don't need to modify these unless you
  /// really know what you're doing.

  /** Gradient tolerance for the truncated preconditioned conjugate gradient
   * solver: stop if ||g|| < kappa * ||g_0||.  This parameter should be in the
   * range (0,1). */
  Scalar STPCG_kappa = 0.05;

  /** Gradient tolerance based upon a fractional-power reduction in the norm of
   * the gradient: stop if ||g|| < ||kappa||^{1+ theta}.  This value should be
   * positive, and controls the asymptotic convergence rate of the
   * truncated-Newton trust-region solver: specifically, for theta > 0, the TNT
   * algorithm converges q-superlinearly with order (1+theta). */
  Scalar STPCG_theta = 0.9;

  /** Maximum elapsed computation time (in seconds) */
  double max_computation_time = std::numeric_limits<double>::max();

  /** An optional user-supplied function that can be used to instrument/monitor
   * the performance of the internal Riemannian truncated-Newton trust-region
   * optimization algorithm as it runs. */
  std::optional<TNTUserFunction> user_function;

  /** If this value is true, the DPGO algorithm will log and return the
    * entire sequence of iterates generated by the Riemannian Staircase */
  bool log_iterates = false;

};

/** This struct contains the output of the SESync algorithm */
struct DPGOResult {
  bool updated = true;

  /** The estimate of x = [t^a R^a t^b R^b .....]^T */
  Matrix Xk;

  /** The estimate of x^a = [t^a R^a]^T */
  Matrix Xak;

  /** The intermediate estimate of x^a = [t^a R^a]^T */
  Matrix Xakh;

  /** The norm of the Riemannian gradient at x */
  Scalar gradFnorm;

  /** The objective value w.r.t. inter-node objective values */
  Scalar fobjE;

  /** The Euclidean gradient w.r.t. inter-node objective values */
  Matrix DfobjE;

  /** The current convex combination of objective values used in Nesterov's
   * method */
  Scalar Fk[2];

  /** The current objective value of G(X|Z) */
  Scalar Gk;

  // Function G(X|Z) = 0.5 * <G*X, X> + <g, X> + f
  /** The estimate of xhat at each iteration */
  std::vector<Matrix> X;

  /** The objective value at xhat in SE(d)^n for each iteration */
  std::vector<Scalar> fobj;

  /** Euclidean gradient for each iteration */
  std::vector<Matrix> Dfobj;

  /** g at each iteration */
  std::vector<Matrix> g;

  /** f at each iteration */
  std::vector<Scalar> f;

  /** The objective value of G(X|Z) */
  std::vector<Scalar> G;

  // Used for Nesterov's method
  /** gamma at each iteration for Nesterov's method */
  Scalar gamma;

  /** s at each iteration for Nesterov's method */
  std::vector<Scalar> s;

  /** The convex combination of objective values used in Nesterov's method */
  std::vector<Scalar> F[2];

  /** adaptive soft restart counts */
  int soft_restart_hits[2] = {0, 0};

  /** oscillation */
  std::vector<int> oscillations;

  /** The number of oscillations in the last 15 runs */
  int num_oscillations = 0;

  /* Riemannian gradient with respect to X */
  Matrix gradF;

  /** The total number of iterations */
  int iters = 0;

  /** The count of rescaling for robust loss kernels */
  int rescale_count = 0;

  /** The elapsed computation time of each iteration */
  std::vector<double> elapsed_optimization_times;

  /** A vector containing the sequence of function values obtained during the
   * Riemannian optimization at each iteration */
  std::vector<std::vector<Scalar>> function_values;

  /** A vector containing the sequence of norms of the Riemannian gradients
   * obtained during the optimization at each iteration */
  std::vector<std::vector<Scalar>> gradient_norms;

  /** A vector containing the sequence of (# Hessian-vector product operations)
   * carried out during the optimization at each iteration */
  std::vector<std::vector<int>> Hessian_vector_products;

  /** A vector containing the sequence of elapsed times in the optimization at
   * each level of the Riemannian Staircase at which the corresponding function
   * values and gradients were obtained */
  std::vector<std::vector<double>> optimization_times;

  void clear() {
    iters = 0;
    rescale_count = 0;
    updated = true;
    X.clear();
    g.clear();
    fobj.clear();
    Dfobj.clear();
    f.clear();
    s.clear();
    soft_restart_hits[0] = 0;
    soft_restart_hits[1] = 0;
    num_oscillations = 0;
    F[0].clear();
    F[1].clear();
    G.clear();
    optimization_times.clear();
    oscillations.clear();
    function_values.clear();
    gradient_norms.clear();
    Hessian_vector_products.clear();
    elapsed_optimization_times.clear();
  }
};

}  // namespace DPGO

