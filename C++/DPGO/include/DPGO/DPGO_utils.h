#pragma once

#include <glog/logging.h>

#include <memory>
#include <set>
#include <string>

#include <Eigen/Sparse>

#include <DPGO/DPGO_types.h>
#include <DPGO/RelativePoseMeasurement.h>
#include <DPGO/internal/project_to_SOd.h>

namespace DPGO {
// *****************************************************************************
// Generate a dataset of distributed PGO
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// filename: filename of g2o
// -----------------------------------------------------------------------------
// num_nodes: num_of_nodes
// -----------------------------------------------------------------------------
// measurements: measurements for each node
// -----------------------------------------------------------------------------
// num_poses: the number of poses
// -----------------------------------------------------------------------------
int read_g2o_file(int a, const std::string &filename, int &num_poses,
                  measurements_t &measurements);

// *****************************************************************************
// Generate a dataset of distributed PGO
// *****************************************************************************
// filename: filename of g2o
// -----------------------------------------------------------------------------
// num_nodes: num of nodes
// -----------------------------------------------------------------------------
// num_poses: num of poses
// -----------------------------------------------------------------------------
// measurements: measurements for each node
// -----------------------------------------------------------------------------
// intra_measurements: all the intra-node measurements
// -----------------------------------------------------------------------------
// inter_measurements: all the inter-node measurements
// -----------------------------------------------------------------------------
// g_index: g_index[a][i] returns the global index of g_i^a
// -----------------------------------------------------------------------------
int read_g2o(const std::string &filename, int num_nodes, int &num_poses,
             std::vector<measurements_t> &measurements,
             std::vector<std::map<int, int>> &g_index);

int read_g2o(const std::string &filename, int num_nodes, int &num_poses,
             std::vector<measurements_t> &measurements,
             measurements_t &intra_measurements,
             measurements_t &inter_measurements,
             std::vector<std::map<int, int>> &g_index);

// *****************************************************************************
// Generate a dataset of distributed PGO for a specific node
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// filename: filename of g2o
// -----------------------------------------------------------------------------
// num_nodes: num_of_nodes
// -----------------------------------------------------------------------------
// measurements: measurements for node a
// -----------------------------------------------------------------------------
int read_g2o(int a, const std::string &filename, int num_nodes,
             measurements_t &measurements);

// *****************************************************************************
// Generate the optimization information for a specific node
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// measurements: relative pose measurements associated with node a
// -----------------------------------------------------------------------------
// intra_measurements: intra-node measurements
// -----------------------------------------------------------------------------
// inter_measurements: inter-node measurements
// -----------------------------------------------------------------------------
// num_poses[0]: the number of poses in node a
// num_poses[1]: the number of poses in node a's neighbours associated with node
//               a
// -----------------------------------------------------------------------------
// offsets[0]: 0
// offsets[1]: (d + 1) * num_poses[0]
// -----------------------------------------------------------------------------
// num_measurements[0]: the number of intra-node measurements in node a
// num_measurements[1]: the number of inter-node measurements associated with
//                      node a
// num_measurements[2]: num_measurements[0] + num_measurements[1]
// -----------------------------------------------------------------------------
// index: index of each pose in matrix X such that index[node][i] gives the
//        index of pose g_i^node in which node can be either node a or the
//        neighbours of node a
// -----------------------------------------------------------------------------
// sent: indices of poses sent to the other nodes such that sent[b][i] gives the
//       index of pose g_i^a in the message sent from a to b
// -----------------------------------------------------------------------------
// recv: indices of poses received from the other nodes such that recv[b][j]
//       gives the index of pose g_j^b in the message sent from b to a
// -----------------------------------------------------------------------------
int generate_data_info(
    int a, const measurements_t &measurements,
    measurements_t &intra_measurements, measurements_t &inter_measurements,
    Eigen::Matrix<int, 2, 1> &num_poses, Eigen::Matrix<int, 2, 1> &offsets,
    Eigen::Matrix<int, 2, 1> &num_measurements,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &sent,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &recv);

// *****************************************************************************
// Construct regular data matrices if the trivial loss is used
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// G, G0, G1 and G2 define the positive-definite matrix of the proximal
// operator of the majorization minimization method
// G^a(X^a|X^a(k) = 0.5 * <G*(X^a-X^a(k)), X^a-X^a(k)> +
//                                        <\nabla_a F, X^a-X^a(k)> + F^a(k)
// in which
// X^a = [t^a R^a]', G = [G0  G1
//                        G1' G2],
// -----------------------------------------------------------------------------
// B, B0 and B1 are matrices such that
//     \nabla F = B^T * B * Z  and  B = [B0
//                                       B1]
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that is used to compute \nabla_a F and F^a(k), B0 is
// related with intra-node measurements, whereas B1 is related with inter-node
// measurements
// -----------------------------------------------------------------------------
// H is a positive semidefinite matrix that is only related with intra-node
// measurements such that
//      H = B0^T * B0
// -----------------------------------------------------------------------------
// M is a matrix that is used to compute the Euclidean gradient \nabla_a F if
// the trivial loss is used:
//                         \nabla_a F = M * Z
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that are used to compute \nabla_a F and F^a(k)
// --------------
// NOTE: X only contains parts of poses of the neighbor nodes that are needed by
//       a to compute \nabla_a F and F^a(k)
// -----------------------------------------------------------------------------
// Q is a negative semidefinite matrix that is used to compute
//      F^a(k+1) - G^a(k+1) =
//                          0.5 * <Q*(Z^a(k+1)-Z^a(k)), (Z^a(k+1)-Z^a(k))>
// in the paper if the trivial loss is used.
// -----------------------------------------------------------------------------
// F is a positive definite matrix that is used to compute F^a(0) if the trivial
// loss is used:
//                     F^a(0) = 0.5 * <F*Z^a(0), Z^a(0)>
// -----------------------------------------------------------------------------
int construct_quadratic_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_poses,
    const Eigen::Matrix<int, 2, 1> &offsets,
    const Eigen::Matrix<int, 2, 1> &num_measurements,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &G, SparseMatrix &G0, SparseMatrix &G1,
    SparseMatrix &G2, SparseMatrix &B, SparseMatrix &B0, SparseMatrix &B1,
    SparseMatrix &H, SparseMatrix &M, SparseMatrix &Q, SparseMatrix &F);

// *****************************************************************************
// Construct regular data matrices if the trivial loss is used
// *****************************************************************************
// M define the positive-definite matrix such that
//                       F = 0.5 * <M * X, X>
// -----------------------------------------------------------------------------
// B, B0 and B1 are matrices such that
//     \nabla F = B^T * B * Z  and  B = [B0
//                                       B1]
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that is used to compute \nabla_a F and F^a(k), B0 is
// related with intra-node measurements, whereas B1 is related with inter-node
// measurements
// -----------------------------------------------------------------------------
int construct_data_matrix(const measurements_t &intra_measurements,
                          const measurements_t &inter_measurements,
                          const std::vector<std::map<int, int>> &g_index,
                          SparseMatrix &M, SparseMatrix &B0, SparseMatrix &B1);

// *****************************************************************************
// Construct simplified data matrices if the trivial loss is used
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// G, G0, G1 and G2 define the positive-definite matrix for the proximal
// operator of the majorization minimization method
// G^a(X^a|X^a(k) = 0.5 * <G*(X^a-X^a(k)), X^a-X^a(k)> +
//                                        <\nabla_a F, X^a-X^a(k)> + F^a(k)
//                = 0.5 * <G*X^a, X^a> + <g^a(k), X^a-X^a(k)> + f^a(k)
// in which
// X^a = [t^a R^a]', G = [G0  G1
//                        G1' G2]
// -----------------------------------------------------------------------------
// B, B0 and B1 are matrices such that
//     \nabla F = B^T * B * Z  and  B = [B0
//                                       B1]
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that is used to compute \nabla_a F and F^a(k), B0 is
// related with intra-node measurements, whereas B1 is related with inter-node
// measurements
// -----------------------------------------------------------------------------
// D is a matrix such that
//                            D = G - B0^T * B0
// -----------------------------------------------------------------------------
// If the trivial loss is used, S is a matrix to compute g^a(k) such that
//                        g^a(k) = S*Z = M*Z - G*X
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that are used to compute \nabla_a F and F^a(k)
// --------------
// NOTE: X only contains parts of poses of the neighbor nodes that are needed by
//       node a to compute \nabla_a F and F^a(k)
// -----------------------------------------------------------------------------
// Q is a negative semidefinite matrix that is used to compute
//      F^a(k+1) - G^a(k+1) =
//                          0.5 * <Q*(Z^a(k+1)-Z^a(k)), (Z^a(k+1)-Z^a(k))>
// in the paper if the trivial loss is used.
// -----------------------------------------------------------------------------
// P is a symmetric matrix that is used to compute
//    0.5 * <G*X^a(k), X^a(k)> - <M*Z^a(k), X^a(k)> = 0.5 * <P*Z^a(k), Z^a(k)>
// -----------------------------------------------------------------------------
// P0 is a symmetric matrix that is used to compute f^a(0)
//                 f^a(0) = 0.5 * <P0 * Z^a(0), Z^a(0)>
// -----------------------------------------------------------------------------
// H defines the positive-definite matrix of the proximal operator for the
// auxiliary majorization minimization method
// H^a(X^a|Z^a(k) = 0.5 * <H*(X^a-X^a(k)), X^a-X^a(k)> +
//                                        <\nabla_a F, X^a-X^a(k)> + F^a(k)
// in which X^a = [t^a R^a]'
// -----------------------------------------------------------------------------
// T and N are the matrices to recover the translation for the auxiliary problem
//                      t^a = t^a(k) - N * R - T * \nabla_t F
// -----------------------------------------------------------------------------
// U is the matrix used to construct the auxiliary problem
//                        min <R^a, U*Z>
// -----------------------------------------------------------------------------
// If the trivial loss is used, V is a matrix such that
//                         V*Z = M*Z - H*X
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that are used to compute \nabla_a F and F^a(k)
// -----------------------------------------------------------------------------
int simplify_quadratic_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_poses,
    const Eigen::Matrix<int, 2, 1> &offsets,
    const Eigen::Matrix<int, 2, 1> &num_measurements,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &G, SparseMatrix &G0, SparseMatrix &G1,
    SparseMatrix &G2, SparseMatrix &B, SparseMatrix &B0, SparseMatrix &B1,
    SparseMatrix &D, SparseMatrix &S, SparseMatrix &Q, SparseMatrix &P,
    SparseMatrix &P0, SparseMatrix &H, DiagonalMatrix &T, SparseMatrix &N,
    SparseMatrix &U, SparseMatrix &V);

// *****************************************************************************
// Construct simplified data matrices if non-trivial losses are used
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// G, G0, G1 and G2 define the positive-definite matrix for the proximal
// operator of the majorization minimization method
// G^a(X^a|X^a(k) = 0.5 * <G*(X^a-X^a(k)), X^a-X^a(k)> +
//                                        <\nabla_a F, X^a-X^a(k)> + F^a(k)
//                = 0.5 * <G*X^a, X^a> + <g^a(k), X^a-X^a(k)> + f^a(k)
// in which
// X^a = [t^a R^a]', G = [G0  G1
//                        G1' G2]
// -----------------------------------------------------------------------------
// B, B0 and B1 are matrices such that
//     \nabla F = B^T * B * Z  and  B = [B0
//                                       B1]
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that is used to compute \nabla_a F and F^a(k), B0 is
// related with intra-node measurements, whereas B1 is related with inter-node
// measurements
// -----------------------------------------------------------------------------
// D is a matrix such that
//                              D = G - B0^T*B0
// -----------------------------------------------------------------------------
// Q is a positive semidefinite matrix that is used to compute
//   G^a(k+1) - F^a(k+1) =
//          0.5 * (0.5 * <Q*(Z^a(k+1)-Z^a(k)), (Z^a(k+1)-Z^a(k))> +
//                           <\nabla F'^a, Z^a(k+1)-Z^a(k)> + F'^a(k) -
//                           F'^a(k+1))
// in the paper if non-trivial losses are used.
// -----------------------------------------------------------------------------
// P0 is a matrix such that
//                        <P0*Z^a, Z^a> = <B0*Z^a, B0*Z^a>
// -----------------------------------------------------------------------------
// H defines the positive-definite matrix of the proximal operator for the
// auxiliary majorization minimization method
// H^a(X^a|Z^a(k) = 0.5 * <H*(X^a-X^a(k)), X^a-X^a(k)> +
//                                        <\nabla_a F, X^a-X^a(k)> + F^a(k)
// in which X^a = [t^a R^a]'
// -----------------------------------------------------------------------------
// T and N are the matrices to recover the translation for the auxiliary problem
//                      t^a = t^a(k) - N^T * R - T * \nabla_t F
// -----------------------------------------------------------------------------
// N and T are the matrices such that the auxiliary problem can be formulated as
//                 max <N * \nabla_t F + V * X -\nabla_R F, R>
// -----------------------------------------------------------------------------
int simplify_regular_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_poses,
    const Eigen::Matrix<int, 2, 1> &offsets,
    const Eigen::Matrix<int, 2, 1> &num_measurements,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &G, SparseMatrix &G0, SparseMatrix &G1,
    SparseMatrix &G2, SparseMatrix &B, SparseMatrix &B0, SparseMatrix &B1,
    SparseMatrix &D, SparseMatrix &Q, SparseMatrix &P0, SparseMatrix &H,
    DiagonalMatrix &T, SparseMatrix &N, SparseMatrix &V);

// *****************************************************************************
// Construct simplified data matrices if non-trivial losses are used
// *****************************************************************************
// a: index of the node
// -----------------------------------------------------------------------------
// M = B0^T * B0
// -----------------------------------------------------------------------------
// B, B0 and B1 are matrices such that
//     \nabla F = B^T * B * Z  and  B = [B0
//                                       B1]
// in which Z = [t^a t^b ... R^a R^b ...]' and (t^b, R^b) ... are poses of the
// neighbor nodes of a that is used to compute \nabla_a F and F^a(k), B0 is
// related with intra-node measurements, whereas B1 is related with inter-node
// measurements
// -----------------------------------------------------------------------------
// E is related with rotational measurements and F is related with translational
// measurements
// -----------------------------------------------------------------------------
// Q is a positive semidefinite matrix that is used to compute
//   G^a(k+1) - F^a(k+1) =
//          0.5 * (0.5 * <Q*(Z^a(k+1)-Z^a(k)), (Z^a(k+1)-Z^a(k))> +
//                           <\nabla F'^a, Z^a(k+1)-Z^a(k)> + F'^a(k) -
//                           F'^a(k+1))
// in the paper if non-trivial losses are used.
// -----------------------------------------------------------------------------
// S defines the positive-definite matrix related with the intra-node
// measurements for auxiliary proimal operator
// -----------------------------------------------------------------------------
// T and N are the matrices to recover the translation for the auxiliary problem
//                   t^a = t^a(k) - N^T * R - T * \nabla_t F
// -----------------------------------------------------------------------------
// N and T are the matrices such that the auxiliary problem can be formulated as
//                 max <N * \nabla_t F + V * X -\nabla_R F, R>
// -----------------------------------------------------------------------------
int simplify_regular_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Eigen::Matrix<int, 2, 1> &inter_n, Scalar xi, SparseMatrix &M,
    std::array<std::array<SparseMatrix, 2>, 2> &Ms, SparseMatrix &B0,
    SparseMatrix &B1, SparseMatrix &D0, SparseMatrix &Q0, SparseMatrix &E,
    SparseMatrix &F, DiagonalMatrix &T0, SparseMatrix &N0, SparseMatrix &V0,
    std::array<std::array<int, 2>, 2> &outer_index_G,
    std::array<std::array<std::vector<int>, 2>, 2> &inner_index_G,
    std::array<std::array<int, 2>, 2> &inner_nnz_G,
    std::array<std::array<int, 2>, 2> &outer_index_Gs,
    std::array<std::array<std::vector<int>, 2>, 2> &inner_index_Gs,
    std::array<std::array<int, 2>, 2> &inner_nnz_Gs,
    std::array<std::array<int, 2>, 2> &outer_index_D,
    std::array<std::array<std::vector<int>, 2>, 2> &inner_index_D,
    std::array<std::array<int, 2>, 2> &inner_nnz_D,
    std::array<std::array<int, 2>, 2> &outer_index_Q,
    std::array<std::array<std::vector<int>, 2>, 2> &inner_index_Q,
    std::array<std::array<int, 2>, 2> &inner_nnz_Q, int &OuterIndexT,
    std::vector<int> &inner_index_T, int &inner_nnz_T, int &outer_index_N,
    std::vector<int> &inner_index_N, int &inner_nnz_N, int &outer_index_V,
    std::vector<int> &inner_index_V, int &inner_nnz_V);

// *****************************************************************************
// Communicate estimates from neighboring nodes
// *****************************************************************************
// file: yaml file
// -----------------------------------------------------------------------------
int load_yaml_file(const std::string &file, Options &options,
                   std::string &dataset_path, std::string &res_path);

// *****************************************************************************
// Communicate estimates from neighboring nodes
// *****************************************************************************
// problems: problem info for each node
// -----------------------------------------------------------------------------
// xs: estimates of all the nodes
// -----------------------------------------------------------------------------
template <typename Problem>
int communicate(const std::vector<std::shared_ptr<const Problem>> &problems,
                std::vector<Matrix> &xs) {
  assert(problems.size() == xs.size());

  if (problems.size() != xs.size()) {
    LOG(ERROR) << "Inconsistent sizes of problems and xs." << std::endl;

    return -1;
  }

  int num_nodes = problems.size();

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    const auto &n = problems[alpha]->n();
    const auto &d = problems[alpha]->d();

    assert((d + 1) * n[0] == xs[alpha].rows() ||
           (d + 1) * (n[0] + n[1]) == xs[alpha].rows());

    assert(d == xs[alpha].cols());

    if (((d + 1) * n[0] != xs[alpha].rows() &&
         (d + 1) * (n[0] + n[1]) != xs[alpha].rows()) ||
        d != xs[alpha].cols()) {
      LOG(ERROR) << "Inconsistent size of xs[" << alpha << "]." << std::endl;

      return -1;
    }
  }

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    const auto &n = problems[alpha]->n();
    const auto &s = problems[alpha]->s();
    const auto &d = problems[alpha]->d();

    xs[alpha].conservativeResize((d + 1) * (n[0] + n[1]), d);

    for (const auto &info : problems[alpha]->index()) {
      const auto &beta = info.first;

      if (beta == alpha) continue;

      for (const auto &index : info.second) {
        const auto &i = index.second;
        const auto &j = index.first;

        xs[alpha].row(s[1] * (d + 1) + i[1]) = xs[beta].row(j);

        xs[alpha].middleRows(s[1] * (d + 1) + n[1] + i[1] * d, d) =
            xs[beta].middleRows(problems[beta]->n()[0] + j * d, d);
      }
    }
  }

  return 0;
}

// *****************************************************************************
// Project a d-by-d matrix onto SO(d)
// *****************************************************************************
// M: a d-by-d matrix
//------------------------------------------------------------------------------
// R: the projection of M onto SO(d)
//------------------------------------------------------------------------------
template <typename Other>
int project_to_SOd(const Matrix &M, Eigen::MatrixBase<Other> &R) {
  assert(M.rows() == M.cols());
  assert(M.rows() == R.rows());
  assert(M.cols() == R.cols());

  // Compute the SVD of M
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Scalar detU = svd.matrixU().determinant();
  Scalar detV = svd.matrixV().determinant();

  if (detU * detV > 0) {
    R.noalias() = svd.matrixU() * svd.matrixV().transpose();
  } else {
    Matrix Uprime = svd.matrixU();
    Uprime.col(Uprime.cols() - 1) *= -1;
    R.noalias() = Uprime * svd.matrixV().transpose();
  }

  return 0;
}

template <typename Derived, typename Result>
int project_to_SOdn(const Derived &A, Result &R) {
  assert(A.rows() == R.rows());
  assert(A.cols() == R.cols());
  assert(A.rows() % A.cols() == 0);

  const int d = A.cols();
  const int n = A.rows() / d;

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    // Compute the (thin) SVD of the ith block of A
    Eigen::JacobiSVD<Matrix> svd(A.block(i * d, 0, d, d),
                                 Eigen::ComputeFullU | Eigen::ComputeFullV);

    Scalar detU = svd.matrixU().determinant();
    Scalar detV = svd.matrixV().determinant();

    if (detU * detV > 0) {
      R.block(i * d, 0, d, d).noalias() =
          svd.matrixU() * svd.matrixV().transpose();
    } else {
      Matrix Uprime = svd.matrixU();
      Uprime.col(Uprime.cols() - 1) *= -1;
      R.block(i * d, 0, d, d).noalias() = Uprime * svd.matrixV().transpose();
    }
  }

  return 0;
}

template <typename Derived, typename Result>
int project_to_SO2n(const Derived &A, Result &R) {
  //#define USE_DOUBLE_FOR_SO2_PROJECTION
  assert(A.rows() == R.rows());
  assert(A.cols() == R.cols());
  assert(A.rows() % 2 == 0);
  // assert(A.rows() >= 8);
  assert(A.cols() == 2);

  if (A.rows() < 8) return project_to_SOdn(A, R);

  const int count = A.rows() - 8;
  int n = 0;

#pragma omp parallel for
  for (n = 0; n < count; n += 8) {
    internal::project_to_SO2_d(A.template block<8, 2>(n, 0),
                               R.template block<8, 2>(n, 0).derived());
  }

  internal::project_to_SO2_d(A.template bottomLeftCorner<8, 2>(),
                             R.template bottomLeftCorner<8, 2>().derived());

  return 0;
}

template <typename Derived, typename Result>
int project_to_SO3n(const Derived &A, Result &R) {
  assert(A.rows() == R.rows());
  assert(A.cols() == R.cols());
  assert(A.rows() % 3 == 0);
  // assert(A.rows() >= 12);
  assert(A.cols() == 3);

  if (A.rows() < 12) return project_to_SOdn(A, R);

  const int count = A.rows() - 12;
  int n = 0;

#pragma omp parallel for
  for (n = 0; n < count; n += 12) {
    internal::project_to_SO3_d(A.template block<12, 3>(n, 0),
                               R.template block<12, 3>(n, 0).derived());
  }

  internal::project_to_SO3_d(A.template bottomLeftCorner<12, 3>(),
                             R.template bottomLeftCorner<12, 3>().derived());

  return 0;
}
}  // namespace DPGO
