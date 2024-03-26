#pragma once

#include <glog/logging.h>
#include <memory>
#include <set>
#include <string>

#include <Eigen/Sparse>

#include <DChordal/DChordal_types.h>
#include <DChordal/RelativePoseMeasurement.h>

#include <SESync/SESync.h>

namespace DChordal {
class DChordalReducedProblem;

// *****************************************************************************
// Solve local PGO using SESync
// *****************************************************************************
int SESync(int a, const measurements_t &measurements,
           const SESync::SESyncOpts &opts, SESync::SESyncResult &result);

// *****************************************************************************
// Generate the information for a specific node
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
// -----------------------------------------------------------------------------
// index: index of each pose in matrix X such that index[node][i] gives the
//        index of pose g_i^node in which node can be either node a or the
//        neighbours of node a
// -----------------------------------------------------------------------------
// sent: poses sent to the other nodes such that sent[b][i] gives the index of
//       pose g_i^a
// -----------------------------------------------------------------------------
// recv: poses received from the other nodes such that recv[b][j] gives the
//       index of pose g_j^b
// -----------------------------------------------------------------------------
// n_index: index of each pose in matrix X such that n_index[node] gives the
//          index of pose g^node in which node can be either node a or the
//          neighbours of node a
// -----------------------------------------------------------------------------
// n_sent: poses sent to the other nodes such that sent[b] gives the index of
//         pose g^a
// -----------------------------------------------------------------------------
// n_recv: poses received from the other nodes such that recv[b] gives the index
//         of pose g^b
// -----------------------------------------------------------------------------
int generate_data_info(
    int a, const measurements_t &measurements,
    measurements_t &intra_measurements, measurements_t &inter_measurements,
    int &num_neighbors, Eigen::Matrix<int, 2, 1> &num_poses,
    Eigen::Matrix<int, 2, 1> &offsets,
    Eigen::Matrix<int, 2, 1> &num_measurements,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &sent,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &recv,
    std::map<int, int> &n_index, std::map<int, int> &n_sent,
    std::map<int, int> &n_recv);

// *****************************************************************************
// Simplify data matrices for the chordal initialization
// *****************************************************************************
// rotS defines the matrix such that
//              rotS * R' = rotG * R^a' + rotM * R^(k)'
// -----------------------------------------------------------------------------
// tS defines the matrix such that
//              tS * t' = tG * t^a' + tT * R'
// -----------------------------------------------------------------------------
int simplify_data_matrix_reduced_R(
    int a, const measurements_t &inter_measurements, const Matrix &X,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    const std::map<int, int> &n_index, Scalar xi, Matrix &rotG, Matrix &rotS,
    Matrix &rotD, Matrix &rotH, Matrix &rotg, SparseMatrix &rotB, Matrix &rotb);

int precompute_data_matrix_recover_t(int a, const measurements_t &measurements,
                                     int num_poses, SparseMatrix &L,
                                     SparseMatrix &P);

int simplify_data_matrix_reduced_t(
    int a, const measurements_t &inter_measurements, const Matrix &X,
    const Matrix &nR, const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    const std::map<int, int> &n_index, Scalar xi, Matrix &tG, Matrix &tS,
    Matrix &tD, Matrix &tH, Matrix &tg, SparseMatrix &tB, Matrix &tb);

int simplify_data_matrix_R(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &rotG, SparseMatrix &rotS, SparseMatrix &rotD,
    SparseMatrix &rotH, Matrix &rotg, SparseMatrix &rotB, Matrix &rotb,
    SparseMatrix &rotB0, Matrix &rotb0, SparseMatrix &rotB1, Matrix &rotb1);

int simplify_data_matrix_t(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements, const Matrix &R,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &tG, SparseMatrix &tS, SparseMatrix &tD,
    SparseMatrix &tH, Matrix &tg, SparseMatrix &tB, Matrix &tb,
    SparseMatrix &tB0, Matrix &tb0, SparseMatrix &tB1, Matrix &tb1);

template <typename Chordal>
Scalar evaluate_f(const std::vector<std::shared_ptr<Chordal>> &chordals) {
  Scalar fobj = 0;

  for (const auto &chordal : chordals) {
    fobj += (chordal->problem()->B() * chordal->results().Xk +
             chordal->problem()->b())
                .squaredNorm();
  }

  return fobj;
}
// *****************************************************************************
// Recevie estimates from neighboring nodes
// *****************************************************************************
// problems: problem info for each node
// -----------------------------------------------------------------------------
// xs: estimates of all the nodes
// -----------------------------------------------------------------------------
template <typename Problem>
int n_communicate(const std::vector<std::shared_ptr<const Problem>> &problems,
                  std::vector<Matrix> &xs) {
  assert(problems.size() == xs.size());

  if (problems.size() != xs.size()) {
    LOG(ERROR) << "Inconsistent sizes of problems and xs." << std::endl;

    return -1;
  }

  int num_nodes = problems.size();

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    const auto &nn = problems[alpha]->nn();
    const auto &p = problems[alpha]->p();
    const auto &d = problems[alpha]->d();

    assert(p == xs[alpha].rows() || p * (nn + 1) == xs[alpha].rows());

    assert(d == xs[alpha].cols());

    if ((p != xs[alpha].rows() && p * (nn + 1) != xs[alpha].rows()) ||
        d != xs[alpha].cols()) {
      LOG(ERROR) << "Inconsistent size of xs[" << alpha << "]." << std::endl;

      return -1;
    }
  }

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    const auto &nn = problems[alpha]->nn();
    const auto &p = problems[alpha]->p();
    const auto &d = problems[alpha]->d();

    xs[alpha].conservativeResize(p * (nn + 1), d);

    for (const auto &info : problems[alpha]->n_index()) {
      const auto &beta = info.first;
      const auto &i = info.second;

      if (beta == alpha) continue;

      xs[alpha].middleRows(p * i, p) = xs[beta].topRows(p);
    }
  }

  return 0;
}

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
    const auto &p = problems[alpha]->p();
    const auto &d = problems[alpha]->d();

    assert(p * n[0] == xs[alpha].rows() ||
           p * (n[0] + n[1]) == xs[alpha].rows());

    assert(d == xs[alpha].cols());

    if ((p * n[0] != xs[alpha].rows() &&
         p * (n[0] + n[1]) != xs[alpha].rows()) ||
        d != xs[alpha].cols()) {
      LOG(ERROR) << "Inconsistent size of xs[" << alpha << "]." << std::endl;

      return -1;
    }
  }

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    const auto &n = problems[alpha]->n();
    const auto &s = problems[alpha]->s();
    const auto &p = problems[alpha]->p();
    const auto &d = problems[alpha]->d();

    xs[alpha].conservativeResize(p * (n[0] + n[1]), d);

    for (const auto &info : problems[alpha]->index()) {
      const auto &beta = info.first;

      if (beta == alpha) continue;

      for (const auto &index : info.second) {
        const auto &i = index.second;
        const auto &j = index.first;

        xs[alpha].middleRows(s[1] * p + i[1] * p, p) =
            xs[beta].middleRows(j * p, p);
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
}  // namespace DChordal
