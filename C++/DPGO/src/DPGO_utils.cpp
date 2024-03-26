#include <DPGO/DPGO_utils.h>

#include <fstream>
#include <memory>
#include <utility>

namespace DPGO {
int read_g2o_file(int a, const std::string &filename, int &num_poses,
                  measurements_t &measurements) {
  // A single measurement, whose values we will fill in
  RelativePoseMeasurement measurement;

  // A string used to contain the contents of a single line
  std::string line;

  // A string used to extract tokens from each line one-by-one
  std::string token;

  // Preallocate various useful quantities
  Scalar dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
      I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

  int i, j;

  // Open the file for reading
  std::ifstream infile(filename);

  num_poses = 0;

  while (std::getline(infile, line)) {
    // Construct a stream from the string
    std::stringstream strstrm(line);

    // Extract the first token from the string
    strstrm >> token;

    if (token == "EDGE_SE2") {
      // This is a 2D pose measurement

      /** The g2o format specifies a 2D relative pose measurement in the
       * following form:
       *
       * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
       *
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
          I23 >> I33;

      // Fill in elements of this measurement

      // Pose ids
      measurement.i.node = a;
      measurement.i.pose = i;
      measurement.j.node = a;
      measurement.j.pose = j;

      // Raw measurements
      measurement.t = Eigen::Matrix<Scalar, 2, 1>(dx, dy);
      measurement.R = Eigen::Rotation2D<Scalar>(dtheta).toRotationMatrix();

      Eigen::Matrix<Scalar, 2, 2> TranCov;
      TranCov << I11, I12, I12, I22;
      measurement.tau = 2 / TranCov.inverse().trace();

      measurement.kappa = I33;

    } else if (token == "EDGE_SE3:QUAT") {
      // This is a 3D pose measurement

      /** The g2o format specifies a 3D relative pose measurement in the
       * following form:
       *
       * EDGE_SE3:QUAT id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw
       *
       * I11 I12 I13 I14 I15 I16
       *     I22 I23 I24 I25 I26
       *         I33 I34 I35 I36
       *             I44 I45 I46
       *                 I55 I56
       *                     I66
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
          I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
          I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

      // Fill in elements of the measurement

      // Pose ids
      measurement.i.node = a;
      measurement.i.pose = i;
      measurement.j.node = a;
      measurement.j.pose = j;

      // Raw measurements
      measurement.t = Eigen::Matrix<Scalar, 3, 1>(dx, dy, dz);
      measurement.R =
          Eigen::Quaternion<Scalar>(dqw, dqx, dqy, dqz).toRotationMatrix();

      // Compute precisions

      // Compute and store the optimal (information-divergence-minimizing) value
      // of the parameter tau
      Eigen::Matrix<Scalar, 3, 3> TranCov;
      TranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;
      measurement.tau = 3 / TranCov.inverse().trace();

      // Compute and store the optimal (information-divergence-minimizing value
      // of the parameter kappa

      Eigen::Matrix<Scalar, 3, 3> RotCov;
      RotCov << I44, I45, I46, I45, I55, I56, I46, I56, I66;
      measurement.kappa = 3 / (2 * RotCov.inverse().trace());

    } else if ((token == "VERTEX_SE2") || (token == "VERTEX_SE3:QUAT")) {
      // This is just initialization information, so do nothing
      continue;
    } else {
      std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
      assert(false);
    }

    // Update maximum value of poses found so far
    int max_pair = std::max<int>(measurement.i.pose, measurement.j.pose);

    num_poses = ((max_pair > num_poses) ? max_pair : num_poses);
    measurements.push_back(measurement);
  }  // while

  infile.close();

  num_poses++;  // Account for the use of zero-based indexing

  return 0;
}

int read_g2o(const std::string &filename, int num_nodes, int &num_poses,
             std::vector<measurements_t> &measurements,
             std::vector<std::map<int, int>> &g_index) {
  measurements_t m_measurements;

  read_g2o_file(0, filename, num_poses, m_measurements);

  const int n_num_poses = num_poses / num_nodes;
  const int inc_n = num_poses - num_nodes * n_num_poses;
  const int inc = inc_n * (n_num_poses + 1);

  auto index = [inc_n, inc, n_num_poses](int i) -> Index {
    if (i < inc) {
      return Index(i / (n_num_poses + 1), i % (n_num_poses + 1));
    } else {
      i -= inc;
      return Index(i / n_num_poses + inc_n, i % n_num_poses);
    }
  };

  measurements.clear();
  g_index.clear();
  measurements.resize(num_nodes);
  g_index.resize(num_nodes);

  for (auto const &m_measurement : m_measurements) {
    RelativePoseMeasurement measurement;

    measurement.i = index(m_measurement.i.pose);
    measurement.j = index(m_measurement.j.pose);

    if (g_index[measurement.i.node].count(measurement.i.pose) == 0)
      g_index[measurement.i.node][measurement.i.pose] = m_measurement.i.pose;

    if (g_index[measurement.j.node].count(measurement.j.pose) == 0)
      g_index[measurement.j.node][measurement.j.pose] = m_measurement.j.pose;

    if (measurement.i.node >= num_nodes) {
      assert(0 && "[ERROR]: Should not occur.");
      measurement.i.pose += n_num_poses * (measurement.i.node - num_nodes + 1);
      measurement.i.node = num_nodes - 1;
    }

    if (measurement.j.node >= num_nodes) {
      assert(0 && "[ERROR]: Should not occur.");
      measurement.j.pose += n_num_poses * (measurement.j.node - num_nodes + 1);
      measurement.j.node = num_nodes - 1;
    }

    measurement.kappa = m_measurement.kappa;
    measurement.R = m_measurement.R;
    measurement.tau = m_measurement.tau;
    measurement.t = m_measurement.t;

    measurements[measurement.i.node].push_back(measurement);

    if (measurement.i.node != measurement.j.node) {
      measurements[measurement.j.node].push_back(measurement);
    }
  }

  return 0;
}

int read_g2o(const std::string &filename, int num_nodes, int &num_poses,
             std::vector<measurements_t> &measurements,
             measurements_t &intra_measurements,
             measurements_t &inter_measurements,
             std::vector<std::map<int, int>> &g_index) {
  measurements_t m_measurements;

  read_g2o_file(0, filename, num_poses, m_measurements);

  const int n_num_poses = num_poses / num_nodes;
  const int inc_n = num_poses - num_nodes * n_num_poses;
  const int inc = inc_n * (n_num_poses + 1);

  auto index = [inc_n, inc, n_num_poses](int i) -> Index {
    if (i < inc) {
      return Index(i / (n_num_poses + 1), i % (n_num_poses + 1));
    } else {
      i -= inc;
      return Index(i / n_num_poses + inc_n, i % n_num_poses);
    }
  };

  measurements.clear();
  g_index.clear();
  measurements.resize(num_nodes);
  g_index.resize(num_nodes);

  for (auto const &m_measurement : m_measurements) {
    RelativePoseMeasurement measurement;

    measurement.i = index(m_measurement.i.pose);
    measurement.j = index(m_measurement.j.pose);

    if (g_index[measurement.i.node].count(measurement.i.pose) == 0)
      g_index[measurement.i.node][measurement.i.pose] = m_measurement.i.pose;

    if (g_index[measurement.j.node].count(measurement.j.pose) == 0)
      g_index[measurement.j.node][measurement.j.pose] = m_measurement.j.pose;

    if (measurement.i.node >= num_nodes) {
      assert(0 && "[ERROR]: Should not occur.");
      measurement.i.pose += n_num_poses * (measurement.i.node - num_nodes + 1);
      measurement.i.node = num_nodes - 1;
    }

    if (measurement.j.node >= num_nodes) {
      assert(0 && "[ERROR]: Should not occur.");
      measurement.j.pose += n_num_poses * (measurement.j.node - num_nodes + 1);
      measurement.j.node = num_nodes - 1;
    }

    measurement.kappa = m_measurement.kappa;
    measurement.R = m_measurement.R;
    measurement.tau = m_measurement.tau;
    measurement.t = m_measurement.t;

    measurements[measurement.i.node].push_back(measurement);

    if (measurement.i.node == measurement.j.node) {
      intra_measurements.push_back(measurement);
    } else {
      inter_measurements.push_back(measurement);
      measurements[measurement.j.node].push_back(measurement);
    }
  }

  return 0;
}

int read_g2o(int a, const std::string &filename, int num_nodes,
             measurements_t &measurements) {
  int num_poses;
  measurements_t m_measurements;

  read_g2o_file(0, filename, num_poses, m_measurements);

  const int n_num_poses = num_poses / num_nodes;
  const int inc_n = num_poses - num_nodes * n_num_poses;
  const int inc = inc_n * (n_num_poses + 1);

  auto index = [inc_n, inc, n_num_poses](int i) -> Index {
    if (i < inc) {
      return Index(i / (n_num_poses + 1), i % (n_num_poses + 1));
    } else {
      i -= inc;
      return Index(i / n_num_poses + inc_n, i % n_num_poses);
    }
  };

  measurements.clear();

  for (auto const &m_measurement : m_measurements) {
    RelativePoseMeasurement measurement;

    measurement.i = Index(m_measurement.i.pose / n_num_poses,
                          m_measurement.i.pose % n_num_poses);
    measurement.j = Index(m_measurement.j.pose / n_num_poses,
                          m_measurement.j.pose % n_num_poses);

    if (measurement.i.node >= num_nodes) {
      measurement.i.pose += n_num_poses * (measurement.i.node - num_nodes + 1);
      measurement.i.node = num_nodes - 1;
    }

    if (measurement.j.node >= num_nodes) {
      measurement.j.pose += n_num_poses * (measurement.j.node - num_nodes + 1);
      measurement.j.node = num_nodes - 1;
    }

    if (measurement.i.node == a || measurement.j.node == a) continue;

    measurement.kappa = m_measurement.kappa;
    measurement.R = m_measurement.R;
    measurement.tau = m_measurement.tau;
    measurement.t = m_measurement.t;

    measurements.push_back(measurement);
  }

  return 0;
}

int generate_data_info(
    int a, const measurements_t &measurements,
    measurements_t &intra_measurements, measurements_t &inter_measurements,
    Eigen::Matrix<int, 2, 1> &num_poses, Eigen::Matrix<int, 2, 1> &offsets,
    Eigen::Matrix<int, 2, 1> &num_measurements,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &sent,
    std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &recv) {
  intra_measurements.clear();
  inter_measurements.clear();
  index.clear();
  sent.clear();
  recv.clear();
  num_poses.setZero();
  num_measurements.setZero();

  if (measurements.empty()) {
    LOG(WARNING) << "No measurements are specified for node " << a << "."
                 << std::endl;

    return -1;
  }

  std::list<Index> indices;
  std::map<int, std::set<int>> m_sent;
  for (int m = 0; m < measurements.size(); m++) {
    const auto &measurement = measurements[m];

    assert((measurement.i.node == a || measurement.j.node == a));

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "The measurement is not associated with node " << a << "."
                 << std::endl;

      continue;
    }

    const Index &i = measurement.i;
    const Index &j = measurement.j;

    if (i.node == a && j.node == a) {
      intra_measurements.push_back(measurement);
    } else {
      inter_measurements.push_back(measurement);
    }

    auto &alpha = index[i.node];

    if (alpha.find(i.pose) == alpha.end()) {
      alpha[i.pose] = {0, 0};
      indices.push_back(i);
    }

    auto &beta = index[j.node];

    if (beta.find(j.pose) == beta.end()) {
      beta[j.pose] = {0, 0};
      indices.push_back(j);
    }

    if (i.node != a) {
      assert(j.node == a);
      m_sent[i.node].insert(j.pose);
    }

    if (j.node != a) {
      assert(i.node == a);
      m_sent[j.node].insert(i.pose);
    }
  }

  num_measurements[0] = intra_measurements.size();
  num_measurements[1] = inter_measurements.size();

  indices.sort([a](Index const &lhs, Index const &rhs) -> bool {
    if (lhs.node == rhs.node)
      return lhs.pose < rhs.pose;
    else if (lhs.node == a)
      return true;
    else if (rhs.node == a)
      return false;
    else
      return lhs.node < rhs.node;
  });

  num_poses.setZero();

  for (const auto &i : indices) {
    if (i.node == a)
      index[i.node][i.pose] = {0, num_poses[0]++};
    else
      index[i.node][i.pose] = {1, num_poses[1]++};
  }

  int d = intra_measurements.size() ? intra_measurements[0].t.size()
                                    : inter_measurements[0].t.size();

  offsets[0] = 0;
  offsets[1] = num_poses[0];

  auto &alpha = index[a];

  for (auto &s : m_sent) {
    auto &beta = sent[s.first];

    for (auto &i : s.second) beta[i] = alpha[i];
  }

  recv = index;
  recv.erase(a);

  return 0;
}

int construct_data_matrix(const measurements_t &intra_measurements,
                          const measurements_t &inter_measurements,
                          const std::vector<std::map<int, int>> &g_index,
                          SparseMatrix &M, SparseMatrix &B0, SparseMatrix &B1) {
  M.resize(0, 0);
  B0.resize(0, 0);
  B1.resize(0, 0);

  const int num_nodes = g_index.size();

  int num_poses = 0;

  for (const auto &index : g_index) num_poses += index.size();

  std::list<Eigen::Triplet<Scalar>> triplets_M;
  std::list<Eigen::Triplet<Scalar>> triplets_B0;
  std::list<Eigen::Triplet<Scalar>> triplets_B1;

  int d = intra_measurements.size() ? intra_measurements[0].t.size()
                                    : inter_measurements[0].t.size();

  const int &num_intra_m = intra_measurements.size();

  for (int m = 0; m < num_intra_m; m++) {
    const auto &measurement = intra_measurements[m];

    assert(measurement.i.node < num_nodes);
    assert(measurement.j.node < num_nodes);

    if (measurement.i.node >= num_nodes ||
        g_index[measurement.i.node].count(measurement.i.pose) == 0) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                 << ", " << measurement.i.pose << "]" << std::endl;

      exit(-1);
    }

    if (measurement.j.node >= num_nodes ||
        g_index[measurement.j.node].count(measurement.j.pose) == 0) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                 << ", " << measurement.j.pose << "]" << std::endl;

      exit(-1);
    }

    int i = g_index[measurement.i.node].at(measurement.i.pose);
    int j = g_index[measurement.j.node].at(measurement.j.pose);

    if (i >= num_poses) {
      LOG(ERROR) << "Inconsistent index for pose " << i << std::endl;

      exit(-1);
    }

    if (j >= num_poses) {
      LOG(ERROR) << "Inconsistent index for pose " << j << std::endl;

      exit(-1);
    }

    triplets_M.emplace_back(i, i, measurement.tau);
    triplets_M.emplace_back(j, j, measurement.tau);
    triplets_M.emplace_back(i, j, -measurement.tau);
    triplets_M.emplace_back(j, i, -measurement.tau);

    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(i, num_poses + i * d + k,
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(j, num_poses + i * d + k,
                              -measurement.tau * measurement.t(k));
    }

    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_poses + i * d + k, i,
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(num_poses + i * d + k, j,
                              -measurement.tau * measurement.t(k));
    }

    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_poses + i * d + k, num_poses + i * d + k,
                              measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_poses + j * d + k, num_poses + j * d + k,
                              measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(num_poses + i * d + r, num_poses + j * d + c,
                                -measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(num_poses + j * d + r, num_poses + i * d + c,
                                -measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(
            num_poses + i * d + r, num_poses + i * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // update B0
    int l = (d + 1) * m;

    Scalar sqrt_tau = std::sqrt(measurement.tau);
    Scalar sqrt_kappa = std::sqrt(measurement.kappa);

    // update translation w.r.t. translation
    // update ith block
    triplets_B0.emplace_back(l, i, sqrt_tau);

    // update jth block
    triplets_B0.emplace_back(l, j, -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B0.emplace_back(l, num_poses + i * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B0.emplace_back(l + r + 1, num_poses + i * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B0.emplace_back(l + k + 1, num_poses + j * d + k, -sqrt_kappa);
    }
  }

  const int &num_inter_m = inter_measurements.size();

  for (int m = 0; m < num_inter_m; m++) {
    const auto &measurement = inter_measurements[m];

    assert(measurement.i.node < num_nodes);
    assert(measurement.j.node < num_nodes);

    if (measurement.i.node >= num_nodes ||
        g_index[measurement.i.node].count(measurement.i.pose) == 0) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                 << ", " << measurement.i.pose << "]" << std::endl;

      exit(-1);
    }

    if (measurement.j.node >= num_nodes ||
        g_index[measurement.j.node].count(measurement.j.pose) == 0) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                 << ", " << measurement.j.pose << "]" << std::endl;

      exit(-1);
    }

    int i = g_index[measurement.i.node].at(measurement.i.pose);
    int j = g_index[measurement.j.node].at(measurement.j.pose);

    if (i >= num_poses) {
      LOG(ERROR) << "Inconsistent index for pose " << i << std::endl;

      exit(-1);
    }

    if (j >= num_poses) {
      LOG(ERROR) << "Inconsistent index for pose " << j << std::endl;

      exit(-1);
    }

    triplets_M.emplace_back(i, i, measurement.tau);
    triplets_M.emplace_back(j, j, measurement.tau);
    triplets_M.emplace_back(i, j, -measurement.tau);
    triplets_M.emplace_back(j, i, -measurement.tau);

    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(i, num_poses + i * d + k,
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(j, num_poses + i * d + k,
                              -measurement.tau * measurement.t(k));
    }

    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_poses + i * d + k, i,
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(num_poses + i * d + k, j,
                              -measurement.tau * measurement.t(k));
    }
    //
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_poses + i * d + k, num_poses + i * d + k,
                              measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_poses + j * d + k, num_poses + j * d + k,
                              measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(num_poses + i * d + r, num_poses + j * d + c,
                                -measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(num_poses + j * d + r, num_poses + i * d + c,
                                -measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(
            num_poses + i * d + r, num_poses + i * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // update B0
    int l = (d + 1) * m;

    Scalar sqrt_tau = std::sqrt(measurement.tau);
    Scalar sqrt_kappa = std::sqrt(measurement.kappa);

    // update translation w.r.t. translation
    // update ith block
    triplets_B1.emplace_back(l, i, sqrt_tau);

    // update jth block
    triplets_B1.emplace_back(l, j, -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B1.emplace_back(l, num_poses + i * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B1.emplace_back(l + r + 1, num_poses + i * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B1.emplace_back(l + k + 1, num_poses + j * d + k, -sqrt_kappa);
    }
  }

  M.resize((d + 1) * num_poses, (d + 1) * num_poses);
  B0.resize((d + 1) * num_intra_m, (d + 1) * num_poses);
  B1.resize((d + 1) * num_inter_m, (d + 1) * num_poses);

  M.setFromTriplets(triplets_M.begin(), triplets_M.end());
  B0.setFromTriplets(triplets_B0.begin(), triplets_B0.end());
  B1.setFromTriplets(triplets_B1.begin(), triplets_B1.end());

  return 0;
}

int construct_quadratic_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,  // the number of poses
    const Eigen::Matrix<int, 2, 1> &num_s,  // the offsets of indices
    const Eigen::Matrix<int, 2, 1> &num_m,  // the number of measurements
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &G, SparseMatrix &G0, SparseMatrix &G1,
    SparseMatrix &G2, SparseMatrix &B, SparseMatrix &B0, SparseMatrix &B1,
    SparseMatrix &H, SparseMatrix &M, SparseMatrix &Q, SparseMatrix &F) {
  G.resize(0, 0);
  G0.resize(0, 0);
  G1.resize(0, 0);
  G2.resize(0, 0);
  B.resize(0, 0);
  B0.resize(0, 0);
  B1.resize(0, 0);
  H.resize(0, 0);
  M.resize(0, 0);
  Q.resize(0, 0);
  F.resize(0, 0);

  if (intra_measurements.empty() && inter_measurements.empty()) {
    LOG(WARNING) << "No measurements are specified for node " << a << "."
                 << std::endl;

    exit(-1);
  }

  assert(intra_measurements.size() == num_m[0]);

  if (intra_measurements.size() != num_m[0]) {
    LOG(ERROR) << "Inconsistent number of intra-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  assert(inter_measurements.size() == num_m[1]);

  if (inter_measurements.size() != num_m[1]) {
    LOG(ERROR) << "Inconsistent number of inter-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  std::list<Eigen::Triplet<Scalar>> triplets_G;
  std::list<Eigen::Triplet<Scalar>> triplets_G0;
  std::list<Eigen::Triplet<Scalar>> triplets_G1;
  std::list<Eigen::Triplet<Scalar>> triplets_G2;
  std::list<Eigen::Triplet<Scalar>> triplets_B;
  std::list<Eigen::Triplet<Scalar>> triplets_B0;
  std::list<Eigen::Triplet<Scalar>> triplets_B1;
  std::list<Eigen::Triplet<Scalar>> triplets_H;
  std::list<Eigen::Triplet<Scalar>> triplets_M;
  std::list<Eigen::Triplet<Scalar>> triplets_Q;
  std::list<Eigen::Triplet<Scalar>> triplets_F;

  int d = intra_measurements.size() ? intra_measurements[0].t.size()
                                    : inter_measurements[0].t.size();

  int s[2], n[2], i[2];

  int l;

  Scalar sqrt_tau, sqrt_kappa;

  auto aa = index.find(a);
  assert(aa != index.end());

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  auto &index_a = aa->second;

  for (int m = 0; m < num_m[0]; m++) {
    auto const &measurement = intra_measurements[m];

    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a || measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      exit(-1);
    }

    auto ii = index_a.find(measurement.i.pose);
    auto jj = index_a.find(measurement.j.pose);
    assert(ii != index_a.end() && jj != index_a.end());

    if (ii == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                 << ", " << measurement.i.pose << "]" << std::endl;

      exit(-1);
    }

    if (jj == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                 << ", " << measurement.j.pose << "]" << std::endl;

      exit(-1);
    }

    s[0] = (d + 1) * num_s[ii->second[0]];
    s[1] = (d + 1) * num_s[jj->second[0]];

    n[0] = num_n[ii->second[0]];
    n[1] = num_n[jj->second[0]];

    i[0] = ii->second[1];
    i[1] = jj->second[1];

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_G0.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_G0.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_G0.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_G0.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_G.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_G.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_G.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_G.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_F.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_F.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_F.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_F.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_H.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_H.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_H.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_H.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_M.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_M.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_M.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_M.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_G1.emplace_back(s[0] + i[0], i[0] * d + k,
                               measurement.tau * measurement.t(k));
      triplets_G1.emplace_back(s[1] + i[1], i[0] * d + k,
                               -measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
      triplets_H.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_H.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
      triplets_M.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));
      triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));
      triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i[0] * d + k, i[0] * d + k, measurement.kappa);
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);
      triplets_F.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);
      triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);
      triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i[1] * d + k, i[1] * d + k, measurement.kappa);
      triplets_G.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);
      triplets_F.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);
      triplets_H.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);
      triplets_M.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);
    }

    // Elements of ii block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(i[0] * d + r, i[1] * d + c,
                                 -measurement.kappa * measurement.R(r, c));
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
        triplets_F.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
        triplets_H.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
        triplets_M.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
      }

    // Elements of jj block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(i[1] * d + r, i[0] * d + c,
                                 -measurement.kappa * measurement.R(c, r));
        triplets_G.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
        triplets_F.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
        triplets_H.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
        triplets_M.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(
            i[0] * d + r, i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_G.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_F.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_H.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_M.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // update B and B0
    // update translation w.r.t. translation
    // update ith block
    triplets_B.emplace_back(l, s[0] + i[0], sqrt_tau);
    triplets_B0.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B.emplace_back(l, s[1] + i[1] - sqrt_tau);
    triplets_B0.emplace_back(l, s[1] + i[1] - sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                              sqrt_tau * measurement.t(k));
      triplets_B0.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                sqrt_kappa * measurement.R(c, r));
        triplets_B0.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                              -sqrt_kappa);
      triplets_B0.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }
  }

  const int l0 = (d + 1) * num_m[0];

  for (int m = 0; m < num_m[1]; m++) {
    auto const &measurement = inter_measurements[m];

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a inter-node measurement for node " << a << "."
                 << std::endl;

      continue;
    }

    if (measurement.i.node == a) {
      const auto &b = measurement.j.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_a.find(measurement.i.pose);
      assert(ii != index_a.end());

      if (ii == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_b.find(measurement.j.pose);
      assert(jj != index_b.end());

      if (jj == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    } else {
      const auto &b = measurement.i.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_b.find(measurement.i.pose);
      assert(ii != index_b.end());

      if (ii == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_a.find(measurement.j.pose);
      assert(jj != index_a.end());

      if (jj == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    }

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_Q.emplace_back(s[0] + i[0], s[0] + i[0], -0.5 * measurement.tau);
    triplets_Q.emplace_back(s[1] + i[1], s[1] + i[1], -0.5 * measurement.tau);
    triplets_Q.emplace_back(s[0] + i[0], s[1] + i[1], -0.5 * measurement.tau);
    triplets_Q.emplace_back(s[1] + i[1], s[0] + i[0], -0.5 * measurement.tau);
    triplets_F.emplace_back(s[0] + i[0], s[0] + i[0], 0.5 * measurement.tau);
    triplets_F.emplace_back(s[1] + i[1], s[1] + i[1], 0.5 * measurement.tau);
    triplets_F.emplace_back(s[0] + i[0], s[1] + i[1], -0.5 * measurement.tau);
    triplets_F.emplace_back(s[1] + i[1], s[0] + i[0], -0.5 * measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_Q.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              0.5 * measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              0.5 * measurement.tau * measurement.t(k));
      triplets_F.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -0.5 * measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.kappa);
      triplets_F.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k,
                              0.5 * measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k,
                              -0.5 * measurement.kappa);
      triplets_F.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k,
                              0.5 * measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -0.5 * measurement.kappa * measurement.R(r, c));
        triplets_F.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -0.5 * measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -0.5 * measurement.kappa * measurement.R(c, r));
        triplets_F.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -0.5 * measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            -0.5 * measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_F.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            0.5 * measurement.tau * measurement.t(r) * measurement.t(c));
      }

    if (measurement.i.node == a) {
      // Add elements for G0
      triplets_G0.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
      triplets_G.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

      triplets_M.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
      triplets_M.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_G1.emplace_back(s[0] + i[0], i[0] * d + k,
                                 2 * measurement.tau * measurement.t(k));
        triplets_G.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
        triplets_M.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                measurement.tau * measurement.t(k));
      }

      // Add elements for G1' (lower-left block)
      for (int k = 0; k < d; k++) {
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
        triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                measurement.tau * measurement.t(k));
        triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of ith block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_G2.emplace_back(i[0] * d + k, i[0] * d + k,
                                 2 * measurement.kappa);
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
        triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k, measurement.kappa);
      }

      // Elements of ij block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_M.emplace_back(s[0] + n[0] + i[0] * d + r,
                                  s[1] + n[1] + i[1] * d + c,
                                  -measurement.kappa * measurement.R(r, c));
        }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_G2.emplace_back(
              i[0] * d + r, i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_G.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_M.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              measurement.tau * measurement.t(r) * measurement.t(c));
        }
    } else {
      assert(measurement.j.node == a);
      // Add elements for G0
      triplets_G0.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);
      triplets_G.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);
      triplets_M.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
      triplets_M.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_M.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of jth block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_G2.emplace_back(i[1] * d + k, i[1] * d + k,
                                 2 * measurement.kappa);
        triplets_G.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
        triplets_M.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k, measurement.kappa);
      }

      // Elements of ji block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_M.emplace_back(s[1] + n[1] + i[1] * d + r,
                                  s[0] + n[0] + i[0] * d + c,
                                  -measurement.kappa * measurement.R(c, r));
        }
    }

    // update B and B1
    // update translation w.r.t. translation
    // update ith block
    triplets_B.emplace_back(l + l0, s[0] + i[0], sqrt_tau);
    triplets_B1.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B.emplace_back(l + l0, s[1] + i[1], -sqrt_tau);
    triplets_B1.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + l0, s[0] + n[0] + i[0] * d + k,
                              sqrt_tau * measurement.t(k));
      triplets_B1.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B.emplace_back(l + l0 + r + 1, s[0] + n[0] + i[0] * d + c,
                                sqrt_kappa * measurement.R(c, r));
        triplets_B1.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + l0 + k + 1, s[1] + n[1] + i[1] * d + k,
                              -sqrt_kappa);
      triplets_B1.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }
  }

  // add xi to the diagonal of G, G0, G2, Q
  for (int i = 0; i < num_n[0]; i++) {
    triplets_G.emplace_back(i, i, xi);
    triplets_G0.emplace_back(i, i, xi);
    triplets_Q.emplace_back(i, i, xi);

    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i * d + k, i * d + k, xi);
      triplets_G.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_Q.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, -xi);
    }
  }

  G.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  G0.resize(num_n[0], num_n[0]);
  G1.resize(num_n[0], d * num_n[0]);
  G2.resize(d * num_n[0], d * num_n[0]);
  B.resize((d + 1) * (num_m[0] + num_m[1]), (d + 1) * (num_n[0] + num_n[1]));
  B0.resize((d + 1) * num_m[0], (d + 1) * (num_n[0] + num_n[1]));
  B1.resize((d + 1) * num_m[1], (d + 1) * (num_n[0] + num_n[1]));
  H.resize((d + 1) * num_n[0], (d + 1) * (num_n[0] + num_n[1]));
  M.resize((d + 1) * num_n[0], (d + 1) * (num_n[0] + num_n[1]));
  Q.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));
  F.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));

  G.setFromTriplets(triplets_G.begin(), triplets_G.end());
  G0.setFromTriplets(triplets_G0.begin(), triplets_G0.end());
  G1.setFromTriplets(triplets_G1.begin(), triplets_G1.end());
  G2.setFromTriplets(triplets_G2.begin(), triplets_G2.end());
  B.setFromTriplets(triplets_B.begin(), triplets_B.end());
  B0.setFromTriplets(triplets_B0.begin(), triplets_B0.end());
  B1.setFromTriplets(triplets_B1.begin(), triplets_B1.end());
  H.setFromTriplets(triplets_H.begin(), triplets_H.end());
  M.setFromTriplets(triplets_M.begin(), triplets_M.end());
  Q.setFromTriplets(triplets_Q.begin(), triplets_Q.end());
  F.setFromTriplets(triplets_F.begin(), triplets_F.end());

  return 0;
}

int simplify_quadratic_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &G, SparseMatrix &G0, SparseMatrix &G1,
    SparseMatrix &G2, SparseMatrix &B, SparseMatrix &B0, SparseMatrix &B1,
    SparseMatrix &D, SparseMatrix &S, SparseMatrix &Q, SparseMatrix &P,
    SparseMatrix &P0, SparseMatrix &H, DiagonalMatrix &T, SparseMatrix &N,
    SparseMatrix &U, SparseMatrix &V) {
  G.resize(0, 0);
  G0.resize(0, 0);
  G1.resize(0, 0);
  G2.resize(0, 0);
  B.resize(0, 0);
  B0.resize(0, 0);
  B1.resize(0, 0);
  D.resize(0, 0);
  S.resize(0, 0);
  Q.resize(0, 0);
  P.resize(0, 0);
  P0.resize(0, 0);

  // auxiliary matrix
  H.resize(0, 0);
  T.resize(0);
  N.resize(0, 0);
  U.resize(0, 0);
  V.resize(0, 0);

  if (intra_measurements.empty() && inter_measurements.empty()) {
    LOG(WARNING) << "No measurements are specified for node " << a << "."
                 << std::endl;

    exit(-1);
  }

  assert(intra_measurements.size() == num_m[0]);

  if (intra_measurements.size() != num_m[0]) {
    LOG(ERROR) << "Inconsistent number of intra-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  assert(inter_measurements.size() == num_m[1]);

  if (inter_measurements.size() != num_m[1]) {
    LOG(ERROR) << "Inconsistent number of inter-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  std::list<Eigen::Triplet<Scalar>> triplets_G;
  std::list<Eigen::Triplet<Scalar>> triplets_G0;
  std::list<Eigen::Triplet<Scalar>> triplets_G1;
  std::list<Eigen::Triplet<Scalar>> triplets_G2;
  std::list<Eigen::Triplet<Scalar>> triplets_B;
  std::list<Eigen::Triplet<Scalar>> triplets_B0;
  std::list<Eigen::Triplet<Scalar>> triplets_B1;
  std::list<Eigen::Triplet<Scalar>> triplets_D;
  std::list<Eigen::Triplet<Scalar>> triplets_S;
  std::list<Eigen::Triplet<Scalar>> triplets_Q;
  std::list<Eigen::Triplet<Scalar>> triplets_P;
  std::list<Eigen::Triplet<Scalar>> triplets_P0;
  std::list<Eigen::Triplet<Scalar>> triplets_H;
  std::list<Eigen::Triplet<Scalar>> triplets_N;
  std::list<Eigen::Triplet<Scalar>> triplets_V;

  int d = intra_measurements.size() ? intra_measurements[0].t.size()
                                    : inter_measurements[0].t.size();

  T.setZero(num_n[0]);

  DiagonalMatrix::DiagonalVectorType &DiagT = T.diagonal();

  int s[2];
  int n[2];
  int i[2];

  int l;

  Scalar sqrt_tau, sqrt_kappa;

  auto aa = index.find(a);
  assert(aa != index.end());

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  auto &index_a = aa->second;

  for (int m = 0; m < num_m[0]; m++) {
    auto const &measurement = intra_measurements[m];

    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a || measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      exit(-1);
    }

    auto ii = index_a.find(measurement.i.pose);
    auto jj = index_a.find(measurement.j.pose);
    assert(ii != index_a.end() && jj != index_a.end());

    if (ii == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                 << ", " << measurement.i.pose << "]" << std::endl;

      exit(-1);
    }

    if (jj == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                 << ", " << measurement.j.pose << "]" << std::endl;

      exit(-1);
    }

    s[0] = (d + 1) * num_s[ii->second[0]];
    s[1] = (d + 1) * num_s[jj->second[0]];

    n[0] = num_n[ii->second[0]];
    n[1] = num_n[jj->second[0]];

    i[0] = ii->second[1];
    i[1] = jj->second[1];

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_G0.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_G0.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_G0.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_G0.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_G.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_G.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_G.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_G.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_P.emplace_back(s[0] + i[0], s[0] + i[0], -measurement.tau);
    triplets_P.emplace_back(s[1] + i[1], s[1] + i[1], -measurement.tau);
    triplets_P.emplace_back(s[0] + i[0], s[1] + i[1], measurement.tau);
    triplets_P.emplace_back(s[1] + i[1], s[0] + i[0], measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_G1.emplace_back(s[0] + i[0], i[0] * d + k,
                               measurement.tau * measurement.t(k));
      triplets_G1.emplace_back(s[1] + i[1], i[0] * d + k,
                               -measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
      triplets_P.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
      triplets_P.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));
      triplets_P.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              -measurement.tau * measurement.t(k));
      triplets_P.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i[0] * d + k, i[0] * d + k, measurement.kappa);
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);
      triplets_P.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, -measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i[1] * d + k, i[1] * d + k, measurement.kappa);
      triplets_G.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);
      triplets_P.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, -measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(i[0] * d + r, i[1] * d + c,
                                 -measurement.kappa * measurement.R(r, c));
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
        triplets_P.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(i[1] * d + r, i[0] * d + c,
                                 -measurement.kappa * measurement.R(c, r));
        triplets_G.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
        triplets_P.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(
            i[0] * d + r, i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_G.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_P.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            -measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // update B and B0
    // update translation w.r.t. translation
    // update ith block
    triplets_B.emplace_back(l, s[0] + i[0], sqrt_tau);
    triplets_B0.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B.emplace_back(l, s[1] + i[1], -sqrt_tau);
    triplets_B0.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                              sqrt_tau * measurement.t(k));
      triplets_B0.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                sqrt_kappa * measurement.R(c, r));
        triplets_B0.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                              -sqrt_kappa);
      triplets_B0.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }

    // update the auxiliary data matrix
    {
      DiagT[s[0] + i[0]] += 2 * measurement.tau;
      DiagT[s[1] + i[1]] += 2 * measurement.tau;

      triplets_H.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
      triplets_H.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

      triplets_V.emplace_back(s[0] + i[0], s[0] + i[0], -measurement.tau);
      triplets_V.emplace_back(s[1] + i[1], s[1] + i[1], -measurement.tau);
      triplets_V.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
      triplets_V.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
        triplets_N.emplace_back(s[0] + i[0], i[0] * d + k,
                                2 * measurement.tau * measurement.t[k]);
        triplets_V.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                -measurement.tau * measurement.t(k));
        triplets_V.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G1' (lower-left block)
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
        triplets_V.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                -measurement.tau * measurement.t(k));
        triplets_V.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of the ith and jth block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
        triplets_H.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
        triplets_V.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k, -measurement.kappa);
        triplets_V.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k, -measurement.kappa);
      }

      // Elements of ii block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_H.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_V.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              -measurement.tau * measurement.t(r) * measurement.t(c));
        }

      // Elements of ij block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_V.emplace_back(s[0] + n[0] + i[0] * d + r,
                                  s[1] + n[1] + i[1] * d + c,
                                  -measurement.kappa * measurement.R(r, c));
        }

      // Elements of ji block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_V.emplace_back(s[1] + n[1] + i[1] * d + r,
                                  s[0] + n[0] + i[0] * d + c,
                                  -measurement.kappa * measurement.R(c, r));
        }
    }
  }

  const int l0 = (d + 1) * num_m[0];

  for (int m = 0; m < num_m[1]; m++) {
    auto const &measurement = inter_measurements[m];

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a inter-node measurement for node " << a << "."
                 << std::endl;

      continue;
    }

    if (measurement.i.node == a) {
      const auto &b = measurement.j.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_a.find(measurement.i.pose);
      assert(ii != index_a.end());

      if (ii == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_b.find(measurement.j.pose);
      assert(jj != index_b.end());

      if (jj == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    } else {
      const auto &b = measurement.i.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_b.find(measurement.i.pose);
      assert(ii != index_b.end());

      if (ii == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_a.find(measurement.j.pose);
      assert(jj != index_a.end());

      if (jj == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    }

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_Q.emplace_back(s[0] + i[0], s[0] + i[0], -0.5 * measurement.tau);
    triplets_Q.emplace_back(s[1] + i[1], s[1] + i[1], -0.5 * measurement.tau);
    triplets_Q.emplace_back(s[0] + i[0], s[1] + i[1], -0.5 * measurement.tau);
    triplets_Q.emplace_back(s[1] + i[1], s[0] + i[0], -0.5 * measurement.tau);

    triplets_P.emplace_back(s[0] + i[0], s[1] + i[1], measurement.tau);
    triplets_P.emplace_back(s[1] + i[1], s[0] + i[0], measurement.tau);

    triplets_P0.emplace_back(s[0] + i[0], s[0] + i[0], 0.5 * measurement.tau);
    triplets_P0.emplace_back(s[1] + i[1], s[1] + i[1], 0.5 * measurement.tau);
    triplets_P0.emplace_back(s[0] + i[0], s[1] + i[1], 0.5 * measurement.tau);
    triplets_P0.emplace_back(s[1] + i[1], s[0] + i[0], 0.5 * measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_Q.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_P.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                               0.5 * measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                               0.5 * measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -0.5 * measurement.tau * measurement.t(k));
      triplets_P.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                               0.5 * measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                               0.5 * measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k,
                              -0.5 * measurement.kappa);
      triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + k,
                               s[0] + n[0] + i[0] * d + k,
                               0.5 * measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k,
                              -0.5 * measurement.kappa);
      triplets_P0.emplace_back(s[1] + n[1] + i[1] * d + k,
                               s[1] + n[1] + i[1] * d + k,
                               0.5 * measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -0.5 * measurement.kappa * measurement.R(r, c));
        triplets_P.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                measurement.kappa * measurement.R(r, c));
        triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + r,
                                 s[1] + n[1] + i[1] * d + c,
                                 0.5 * measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -0.5 * measurement.kappa * measurement.R(c, r));
        triplets_P.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                measurement.kappa * measurement.R(c, r));
        triplets_P0.emplace_back(s[1] + n[1] + i[1] * d + r,
                                 s[0] + n[0] + i[0] * d + c,
                                 0.5 * measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            -0.5 * measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_P0.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            0.5 * measurement.tau * measurement.t(r) * measurement.t(c));
      }

    if (measurement.i.node == a) {
      // Add elements for G0
      triplets_G0.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
      triplets_G.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

      triplets_D.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

      triplets_S.emplace_back(s[0] + i[0], s[0] + i[0], -measurement.tau);
      triplets_S.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_G1.emplace_back(s[0] + i[0], i[0] * d + k,
                                 2 * measurement.tau * measurement.t(k));
        triplets_G.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
        triplets_D.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
        triplets_S.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G1' (lower-left block)
      for (int k = 0; k < d; k++) {
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
        triplets_D.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
        triplets_S.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                -measurement.tau * measurement.t(k));
        triplets_S.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of ith block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_G2.emplace_back(i[0] * d + k, i[0] * d + k,
                                 2 * measurement.kappa);
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
        triplets_D.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
        triplets_S.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k, -measurement.kappa);
      }

      // Elements of ii block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_G2.emplace_back(
              i[0] * d + r, i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_G.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_D.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_S.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              -measurement.tau * measurement.t(r) * measurement.t(c));
        }

      // Elements of ij block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_S.emplace_back(s[0] + n[0] + i[0] * d + r,
                                  s[1] + n[1] + i[1] * d + c,
                                  -measurement.kappa * measurement.R(r, c));
        }

      // update the auxiliary data matrix
      {
        DiagT[s[0] + i[0]] += 2 * measurement.tau;

        triplets_H.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

        triplets_V.emplace_back(s[0] + i[0], s[0] + i[0], -measurement.tau);
        triplets_V.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);

        // Add elements for G1 (upper-right block)
        for (int k = 0; k < d; k++) {
          triplets_H.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                  2 * measurement.tau * measurement.t(k));
          triplets_N.emplace_back(s[0] + i[0], i[0] * d + k,
                                  2 * measurement.tau * measurement.t[k]);
          triplets_V.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                  -measurement.tau * measurement.t(k));
        }

        // Add elements for G1' (lower-left block)
        for (int k = 0; k < d; k++) {
          triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                  2 * measurement.tau * measurement.t(k));
          triplets_V.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                  -measurement.tau * measurement.t(k));
          triplets_V.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                                  -measurement.tau * measurement.t(k));
        }

        // Add elements for G2
        // Elements of the ith block-diagonal
        for (int k = 0; k < d; k++) {
          triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k,
                                  s[0] + n[0] + i[0] * d + k,
                                  2 * measurement.kappa);
          triplets_V.emplace_back(s[0] + n[0] + i[0] * d + k,
                                  s[0] + n[0] + i[0] * d + k,
                                  -measurement.kappa);
        }

        // Elements of ii block
        for (int r = 0; r < d; r++)
          for (int c = 0; c < d; c++) {
            triplets_H.emplace_back(
                s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
                2 * measurement.tau * measurement.t(r) * measurement.t(c));
            triplets_V.emplace_back(
                s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
                -measurement.tau * measurement.t(r) * measurement.t(c));
          }

        // Elements of ij block
        for (int r = 0; r < d; r++)
          for (int c = 0; c < d; c++) {
            triplets_V.emplace_back(s[0] + n[0] + i[0] * d + r,
                                    s[1] + n[1] + i[1] * d + c,
                                    -measurement.kappa * measurement.R(r, c));
          }
      }
    } else {
      assert(measurement.j.node == a);
      // Add elements for G0
      triplets_G0.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);
      triplets_G.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

      triplets_D.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

      triplets_S.emplace_back(s[1] + i[1], s[1] + i[1], -measurement.tau);
      triplets_S.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_S.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                                -measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of jth block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_G2.emplace_back(i[1] * d + k, i[1] * d + k,
                                 2 * measurement.kappa);
        triplets_G.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
        triplets_D.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
        triplets_S.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k, -measurement.kappa);
      }

      // Elements of ji block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_S.emplace_back(s[1] + n[1] + i[1] * d + r,
                                  s[0] + n[0] + i[0] * d + c,
                                  -measurement.kappa * measurement.R(c, r));
        }

      // update the auxiliary data matrix
      {
        DiagT[s[1] + i[1]] += 2 * measurement.tau;

        triplets_H.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

        triplets_V.emplace_back(s[1] + i[1], s[1] + i[1], -measurement.tau);
        triplets_V.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

        // Add elements for G1 (upper-right block)
        for (int k = 0; k < d; k++) {
          triplets_V.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                                  -measurement.tau * measurement.t(k));
        }

        // Add elements for G2
        // Elements of the jth block-diagonal
        for (int k = 0; k < d; k++) {
          triplets_H.emplace_back(s[1] + n[1] + i[1] * d + k,
                                  s[1] + n[1] + i[1] * d + k,
                                  2 * measurement.kappa);
          triplets_V.emplace_back(s[1] + n[1] + i[1] * d + k,
                                  s[1] + n[1] + i[1] * d + k,
                                  -measurement.kappa);
        }

        // Elements of ji block
        for (int r = 0; r < d; r++)
          for (int c = 0; c < d; c++) {
            triplets_V.emplace_back(s[1] + n[1] + i[1] * d + r,
                                    s[0] + n[0] + i[0] * d + c,
                                    -measurement.kappa * measurement.R(c, r));
          }
      }
    }

    // update B and B1
    // update translation w.r.t. translation
    // update ith block
    triplets_B.emplace_back(l + l0, s[0] + i[0], sqrt_tau);
    triplets_B1.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B.emplace_back(l + l0, s[1] + i[1], -sqrt_tau);
    triplets_B1.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + l0, s[0] + n[0] + i[0] * d + k,
                              sqrt_tau * measurement.t(k));
      triplets_B1.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B.emplace_back(l + l0 + r + 1, s[0] + n[0] + i[0] * d + c,
                                sqrt_kappa * measurement.R(c, r));
        triplets_B1.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + l0 + k + 1, s[1] + n[1] + i[1] * d + k,
                              -sqrt_kappa);
      triplets_B1.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }
  }

  // add xi to the diagonal of G, G0, G2, P, S and Q should also be updated
  // accordingly
  for (int i = 0; i < num_n[0]; i++) {
    triplets_G.emplace_back(i, i, xi);
    triplets_G0.emplace_back(i, i, xi);
    triplets_D.emplace_back(i, i, xi);
    triplets_S.emplace_back(i, i, -xi);
    triplets_Q.emplace_back(i, i, -xi);
    triplets_P.emplace_back(i, i, xi);
    triplets_P0.emplace_back(i, i, xi);

    // update the auxiliary data matrix
    DiagT[i] += 1.5 * xi;
    triplets_H.emplace_back(i, i, 1.5 * xi);
    triplets_V.emplace_back(i, i, -1.5 * xi);

    for (int k = 0; k < d; k++) {
      triplets_G.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_G2.emplace_back(i * d + k, i * d + k, xi);
      triplets_D.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_S.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, -xi);
      triplets_Q.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, -xi);
      triplets_P.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_P0.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);

      // update the auxiliary data matrix
      {
        triplets_H.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k,
                                1.5 * xi);
        triplets_V.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k,
                                -1.5 * xi);
      }
    }
  }

  G.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  G0.resize(num_n[0], num_n[0]);
  G1.resize(num_n[0], d * num_n[0]);
  G2.resize(d * num_n[0], d * num_n[0]);
  B.resize((d + 1) * (num_m[0] + num_m[1]), (d + 1) * (num_n[0] + num_n[1]));
  B0.resize((d + 1) * num_m[0], (d + 1) * (num_n[0] + num_n[1]));
  B1.resize((d + 1) * num_m[1], (d + 1) * (num_n[0] + num_n[1]));
  D.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  S.resize((d + 1) * num_n[0], (d + 1) * (num_n[0] + num_n[1]));
  Q.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));
  P.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));
  P0.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));

  G.setFromTriplets(triplets_G.begin(), triplets_G.end());
  G0.setFromTriplets(triplets_G0.begin(), triplets_G0.end());
  G1.setFromTriplets(triplets_G1.begin(), triplets_G1.end());
  G2.setFromTriplets(triplets_G2.begin(), triplets_G2.end());
  B.setFromTriplets(triplets_B.begin(), triplets_B.end());
  B0.setFromTriplets(triplets_B0.begin(), triplets_B0.end());
  B1.setFromTriplets(triplets_B1.begin(), triplets_B1.end());
  D.setFromTriplets(triplets_D.begin(), triplets_D.end());
  S.setFromTriplets(triplets_S.begin(), triplets_S.end());
  Q.setFromTriplets(triplets_Q.begin(), triplets_Q.end());
  P.setFromTriplets(triplets_P.begin(), triplets_P.end());
  P0.setFromTriplets(triplets_P0.begin(), triplets_P0.end());

  // update the auxiliary data matrix
  H.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  N.resize(num_n[0], d * num_n[0]);
  V.resize((d + 1) * num_n[0], (d + 1) * (num_n[0] + num_n[1]));

  H.setFromTriplets(triplets_H.begin(), triplets_H.end());
  N.setFromTriplets(triplets_N.begin(), triplets_N.end());
  V.setFromTriplets(triplets_V.begin(), triplets_V.end());

  T = T.inverse();

  N = T * N;

  U = N.transpose() * V.topRows(num_n[0]);
  U -= V.bottomRows(d * num_n[0]);

  return 0;
}

int simplify_regular_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &G, SparseMatrix &G0, SparseMatrix &G1,
    SparseMatrix &G2, SparseMatrix &B, SparseMatrix &B0, SparseMatrix &B1,
    SparseMatrix &D, SparseMatrix &Q, SparseMatrix &P0, SparseMatrix &H,
    DiagonalMatrix &T, SparseMatrix &N, SparseMatrix &V) {
  G.resize(0, 0);
  G0.resize(0, 0);
  G1.resize(0, 0);
  G2.resize(0, 0);
  B.resize(0, 0);
  B0.resize(0, 0);
  B1.resize(0, 0);
  D.resize(0, 0);
  Q.resize(0, 0);
  P0.resize(0, 0);

  // auxiliary matrix
  H.resize(0, 0);
  T.resize(0);
  N.resize(0, 0);
  V.resize(0, 0);

  if (intra_measurements.empty() && inter_measurements.empty()) {
    LOG(WARNING) << "No measurements are specified for node " << a << "."
                 << std::endl;

    exit(-1);
  }

  assert(intra_measurements.size() == num_m[0]);

  if (intra_measurements.size() != num_m[0]) {
    LOG(ERROR) << "Inconsistent number of intra-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  assert(inter_measurements.size() == num_m[1]);

  if (inter_measurements.size() != num_m[1]) {
    LOG(ERROR) << "Inconsistent number of inter-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  std::list<Eigen::Triplet<Scalar>> triplets_G;
  std::list<Eigen::Triplet<Scalar>> triplets_G0;
  std::list<Eigen::Triplet<Scalar>> triplets_G1;
  std::list<Eigen::Triplet<Scalar>> triplets_G2;
  std::list<Eigen::Triplet<Scalar>> triplets_B;
  std::list<Eigen::Triplet<Scalar>> triplets_B0;
  std::list<Eigen::Triplet<Scalar>> triplets_B1;
  std::list<Eigen::Triplet<Scalar>> triplets_D;
  std::list<Eigen::Triplet<Scalar>> triplets_Q;
  std::list<Eigen::Triplet<Scalar>> triplets_P0;
  std::list<Eigen::Triplet<Scalar>> triplets_H;
  std::list<Eigen::Triplet<Scalar>> triplets_N;

  int d = intra_measurements.size() ? intra_measurements[0].t.size()
                                    : inter_measurements[0].t.size();

  T.setZero(num_n[0]);

  DiagonalMatrix::DiagonalVectorType &DiagT = T.diagonal();

  int s[2];
  int n[2];
  int i[2];

  int l;

  Scalar sqrt_tau, sqrt_kappa;

  auto aa = index.find(a);
  assert(aa != index.end());

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  auto &index_a = aa->second;

  for (int m = 0; m < num_m[0]; m++) {
    auto const &measurement = intra_measurements[m];

    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a || measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      exit(-1);
    }

    auto ii = index_a.find(measurement.i.pose);
    auto jj = index_a.find(measurement.j.pose);
    assert(ii != index_a.end() && jj != index_a.end());

    if (ii == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                 << ", " << measurement.i.pose << "]" << std::endl;

      exit(-1);
    }

    if (jj == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                 << ", " << measurement.j.pose << "]" << std::endl;

      exit(-1);
    }

    s[0] = (d + 1) * num_s[ii->second[0]];
    s[1] = (d + 1) * num_s[jj->second[0]];

    n[0] = num_n[ii->second[0]];
    n[1] = num_n[jj->second[0]];

    i[0] = ii->second[1];
    i[1] = jj->second[1];

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_G0.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_G0.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_G0.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_G0.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_G.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_G.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_G.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_G.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_P0.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_P0.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_P0.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_P0.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_G1.emplace_back(s[0] + i[0], i[0] * d + k,
                               measurement.tau * measurement.t(k));
      triplets_G1.emplace_back(s[1] + i[1], i[0] * d + k,
                               -measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                               measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                               -measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                               measurement.tau * measurement.t(k));
      triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                               -measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i[0] * d + k, i[0] * d + k, measurement.kappa);
      triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);
      triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + k,
                               s[0] + n[0] + i[0] * d + k, measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_G2.emplace_back(i[1] * d + k, i[1] * d + k, measurement.kappa);
      triplets_G.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);
      triplets_P0.emplace_back(s[1] + n[1] + i[1] * d + k,
                               s[1] + n[1] + i[1] * d + k, measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(i[0] * d + r, i[1] * d + c,
                                 -measurement.kappa * measurement.R(r, c));
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
        triplets_P0.emplace_back(s[0] + n[0] + i[0] * d + r,
                                 s[1] + n[1] + i[1] * d + c,
                                 -measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(i[1] * d + r, i[0] * d + c,
                                 -measurement.kappa * measurement.R(c, r));
        triplets_G.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
        triplets_P0.emplace_back(s[1] + n[1] + i[1] * d + r,
                                 s[0] + n[0] + i[0] * d + c,
                                 -measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_G2.emplace_back(
            i[0] * d + r, i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_G.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_P0.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // update B and B0
    // update translation w.r.t. translation
    // update ith block
    triplets_B.emplace_back(l, s[0] + i[0], sqrt_tau);
    triplets_B0.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B.emplace_back(l, s[1] + i[1], -sqrt_tau);
    triplets_B0.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                              sqrt_tau * measurement.t(k));
      triplets_B0.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                sqrt_kappa * measurement.R(c, r));
        triplets_B0.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                              -sqrt_kappa);
      triplets_B0.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }

    // update the auxiliary data matrix
    DiagT[s[0] + i[0]] += 2 * measurement.tau;
    DiagT[s[1] + i[1]] += 2 * measurement.tau;

    triplets_H.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
    triplets_H.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_H.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              2 * measurement.tau * measurement.t(k));
      triplets_N.emplace_back(s[0] + i[0], i[0] * d + k,
                              2 * measurement.tau * measurement.t[k]);
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              2 * measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of the ith and jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k,
                              2 * measurement.kappa);
      triplets_H.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k,
                              2 * measurement.kappa);
    }

    // Elements of ii block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_H.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            2 * measurement.tau * measurement.t(r) * measurement.t(c));
      }
  }

  const int l0 = (d + 1) * num_m[0];

  for (int m = 0; m < num_m[1]; m++) {
    auto const &measurement = inter_measurements[m];

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a inter-node measurement for node " << a << "."
                 << std::endl;

      continue;
    }

    if (measurement.i.node == a) {
      const auto &b = measurement.j.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_a.find(measurement.i.pose);
      assert(ii != index_a.end());

      if (ii == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_b.find(measurement.j.pose);
      assert(jj != index_b.end());

      if (jj == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    } else {
      const auto &b = measurement.i.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_b.find(measurement.i.pose);
      assert(ii != index_b.end());

      if (ii == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_a.find(measurement.j.pose);
      assert(jj != index_a.end());

      if (jj == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    }

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_Q.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
    triplets_Q.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              2 * measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              2 * measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k,
                              2 * measurement.kappa);
    }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_Q.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            2 * measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_Q.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k,
                              2 * measurement.kappa);
    }

    if (measurement.i.node == a) {
      // Add elements for G0
      triplets_G0.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
      triplets_G.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);
      triplets_D.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_G1.emplace_back(s[0] + i[0], i[0] * d + k,
                                 2 * measurement.tau * measurement.t(k));
        triplets_G.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
        triplets_D.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
      }

      // Add elements for G1' (lower-left block)
      for (int k = 0; k < d; k++) {
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
        triplets_D.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of ith block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_G2.emplace_back(i[0] * d + k, i[0] * d + k,
                                 2 * measurement.kappa);
        triplets_G.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
        triplets_D.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
      }

      // Elements of ii block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_G2.emplace_back(
              i[0] * d + r, i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_G.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
          triplets_D.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
        }

      // update the auxiliary data matrix
      DiagT[s[0] + i[0]] += 2 * measurement.tau;

      triplets_H.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

      // Add elements for G1 (upper-right block)
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                                2 * measurement.tau * measurement.t(k));
        triplets_N.emplace_back(s[0] + i[0], i[0] * d + k,
                                2 * measurement.tau * measurement.t[k]);
      }

      // Add elements for G1' (lower-left block)
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                                2 * measurement.tau * measurement.t(k));
      }

      // Add elements for G2
      // Elements of the ith block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[0] + n[0] + i[0] * d + k,
                                s[0] + n[0] + i[0] * d + k,
                                2 * measurement.kappa);
      }

      // Elements of ii block
      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_H.emplace_back(
              s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
              2 * measurement.tau * measurement.t(r) * measurement.t(c));
        }
    } else {
      assert(measurement.j.node == a);
      // Add elements for G0
      triplets_G0.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);
      triplets_G.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);
      triplets_D.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

      // Add elements for G2
      // Elements of jth block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_G2.emplace_back(i[1] * d + k, i[1] * d + k,
                                 2 * measurement.kappa);
        triplets_G.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
        triplets_D.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
      }

      // update the auxiliary data matrix
      DiagT[s[1] + i[1]] += 2 * measurement.tau;

      triplets_H.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

      // Add elements for G2
      // Elements of the jth block-diagonal
      for (int k = 0; k < d; k++) {
        triplets_H.emplace_back(s[1] + n[1] + i[1] * d + k,
                                s[1] + n[1] + i[1] * d + k,
                                2 * measurement.kappa);
      }
    }

    // update B and B1
    // update translation w.r.t. translation
    // update ith block
    triplets_B.emplace_back(l + l0, s[0] + i[0], sqrt_tau);
    triplets_B1.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B.emplace_back(l + l0, s[1] + i[1], -sqrt_tau);
    triplets_B1.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + l0, s[0] + n[0] + i[0] * d + k,
                              sqrt_tau * measurement.t(k));
      triplets_B1.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B.emplace_back(l + l0 + r + 1, s[0] + n[0] + i[0] * d + c,
                                sqrt_kappa * measurement.R(c, r));
        triplets_B1.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B.emplace_back(l + l0 + k + 1, s[1] + n[1] + i[1] * d + k,
                              -sqrt_kappa);
      triplets_B1.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }
  }

  // add xi to the diagonal of G, G0, G2, P, S and Q should also be updated
  // accordingly
  for (int i = 0; i < num_n[0]; i++) {
    triplets_G.emplace_back(i, i, xi);
    triplets_G0.emplace_back(i, i, xi);
    triplets_D.emplace_back(i, i, xi);
    triplets_Q.emplace_back(i, i, 2.0 * xi);

    // update the auxiliary data matrix
    DiagT[i] += 1.5 * xi;
    triplets_H.emplace_back(i, i, 1.5 * xi);

    for (int k = 0; k < d; k++) {
      triplets_G.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_G2.emplace_back(i * d + k, i * d + k, xi);
      triplets_D.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_Q.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k,
                              2.0 * xi);

      // update the auxiliary data matrix
      triplets_H.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k,
                              1.5 * xi);
    }
  }

  G.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  G0.resize(num_n[0], num_n[0]);
  G1.resize(num_n[0], d * num_n[0]);
  G2.resize(d * num_n[0], d * num_n[0]);
  B.resize((d + 1) * (num_m[0] + num_m[1]), (d + 1) * (num_n[0] + num_n[1]));
  B0.resize((d + 1) * num_m[0], (d + 1) * (num_n[0] + num_n[1]));
  B1.resize((d + 1) * num_m[1], (d + 1) * (num_n[0] + num_n[1]));
  D.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  Q.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));
  P0.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));

  G.setFromTriplets(triplets_G.begin(), triplets_G.end());
  G0.setFromTriplets(triplets_G0.begin(), triplets_G0.end());
  G1.setFromTriplets(triplets_G1.begin(), triplets_G1.end());
  G2.setFromTriplets(triplets_G2.begin(), triplets_G2.end());
  B.setFromTriplets(triplets_B.begin(), triplets_B.end());
  B0.setFromTriplets(triplets_B0.begin(), triplets_B0.end());
  B1.setFromTriplets(triplets_B1.begin(), triplets_B1.end());
  D.setFromTriplets(triplets_D.begin(), triplets_D.end());
  Q.setFromTriplets(triplets_Q.begin(), triplets_Q.end());
  P0.setFromTriplets(triplets_P0.begin(), triplets_P0.end());

  // update the auxiliary data matrix
  H.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  N.resize(num_n[0], d * num_n[0]);

  H.setFromTriplets(triplets_H.begin(), triplets_H.end());
  N.setFromTriplets(triplets_N.begin(), triplets_N.end());

  T = T.inverse();

  N = T * N;

  SparseMatrix K = T * H.block(0, num_n[0], num_n[0], d * num_n[0]);
  V = -H.block(num_n[0], 0, d * num_n[0], num_n[0]) * K;
  V += H.block(num_n[0], num_n[0], d * num_n[0], d * num_n[0]);

  return 0;
}

int simplify_regular_data_matrix(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Eigen::Matrix<int, 2, 1> &num_inter_n, Scalar xi, SparseMatrix &M,
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
    std::array<std::array<int, 2>, 2> &inner_nnz_Q, int &outer_index_T,
    std::vector<int> &inner_index_T, int &inner_nnz_T, int &outer_index_N,
    std::vector<int> &inner_index_N, int &inner_nnz_N, int &outer_index_V,
    std::vector<int> &inner_index_V, int &inner_nnz_V) {
  M.resize(0, 0);
  B0.resize(0, 0);
  B1.resize(0, 0);

  D0.resize(0, 0);
  Q0.resize(0, 0);

  E.resize(0, 0);
  F.resize(0, 0);

  N0.resize(0, 0);
  V0.resize(0, 0);

  if (intra_measurements.empty() && inter_measurements.empty()) {
    LOG(WARNING) << "No measurements are specified for node " << a << "."
                 << std::endl;

    exit(-1);
  }

  assert(intra_measurements.size() == num_m[0]);

  if (intra_measurements.size() != num_m[0]) {
    LOG(ERROR) << "Inconsistent number of intra-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  assert(inter_measurements.size() == num_m[1]);

  if (inter_measurements.size() != num_m[1]) {
    LOG(ERROR) << "Inconsistent number of inter-node measurements for node "
               << a << "." << std::endl;

    exit(-1);
  }

  std::list<Eigen::Triplet<Scalar>> triplets_M;
  std::list<Eigen::Triplet<Scalar>> triplets_Ms[2][2];
  std::list<Eigen::Triplet<Scalar>> triplets_D0;
  std::list<Eigen::Triplet<Scalar>> triplets_Q0;
  std::list<Eigen::Triplet<Scalar>> triplets_B0;
  std::list<Eigen::Triplet<Scalar>> triplets_B1;
  std::list<Eigen::Triplet<Scalar>> triplets_E;
  std::list<Eigen::Triplet<Scalar>> triplets_F;
  std::list<Eigen::Triplet<Scalar>> triplets_N0;
  std::list<Eigen::Triplet<Scalar>> triplets_V0;

  int d = intra_measurements.size() ? intra_measurements[0].t.size()
                                    : inter_measurements[0].t.size();

  T0.setZero(num_n[0]);

  DiagonalMatrix::DiagonalVectorType &DiagT0 = T0.diagonal();

  int s[2];
  int n[2];
  int i[2];

  int l;

  const Scalar sqrt_two = std::sqrt(2.0);

  Scalar sqrt_tau, sqrt_kappa;

  auto aa = index.find(a);
  assert(aa != index.end());

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  auto &index_a = aa->second;

  // if pose n has an out measurement
  std::vector<int> out(num_n[0], 0);

  for (int m = 0; m < num_m[0]; m++) {
    auto const &measurement = intra_measurements[m];

    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a || measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      exit(-1);
    }

    auto ii = index_a.find(measurement.i.pose);
    auto jj = index_a.find(measurement.j.pose);
    assert(ii != index_a.end() && jj != index_a.end());

    if (ii == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                 << ", " << measurement.i.pose << "]" << std::endl;

      exit(-1);
    }

    if (jj == index_a.end()) {
      LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                 << ", " << measurement.j.pose << "]" << std::endl;

      exit(-1);
    }

    s[0] = (d + 1) * num_s[ii->second[0]];
    s[1] = (d + 1) * num_s[jj->second[0]];

    n[0] = num_n[ii->second[0]];
    n[1] = num_n[jj->second[0]];

    i[0] = ii->second[1];
    i[1] = jj->second[1];

    l = (d + 1) * m;

    out[i[0]] = 1;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    // Add elements for G0
    triplets_M.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_M.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_M.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_M.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_Ms[0][0].emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_Ms[0][0].emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_Ms[0][0].emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_Ms[0][0].emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    // Add elements for G1 (upper-right block)
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(s[0] + i[0], s[0] + n[0] + i[0] * d + k,
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(s[1] + i[1], s[0] + n[0] + i[0] * d + k,
                              -measurement.tau * measurement.t(k));

      triplets_Ms[0][1].emplace_back(s[0] + i[0], s[0] + i[0] * d + k,
                                     measurement.tau * measurement.t(k));
      triplets_Ms[0][1].emplace_back(s[1] + i[1], s[0] + i[0] * d + k,
                                     -measurement.tau * measurement.t(k));
    }

    // Add elements for G1' (lower-left block)
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k, s[0] + i[0],
                              measurement.tau * measurement.t(k));
      triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k, s[1] + i[1],
                              -measurement.tau * measurement.t(k));

      triplets_Ms[1][0].emplace_back(s[0] + i[0] * d + k, s[0] + i[0],
                                     measurement.tau * measurement.t(k));
      triplets_Ms[1][0].emplace_back(s[0] + i[0] * d + k, s[1] + i[1],
                                     -measurement.tau * measurement.t(k));
    }

    // Add elements for G2
    // Elements of ith block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(s[0] + n[0] + i[0] * d + k,
                              s[0] + n[0] + i[0] * d + k, measurement.kappa);

      triplets_Ms[1][1].emplace_back(s[0] + i[0] * d + k, s[0] + i[0] * d + k,
                                     measurement.kappa);
    }

    // Elements of jth block-diagonal
    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(s[1] + n[1] + i[1] * d + k,
                              s[1] + n[1] + i[1] * d + k, measurement.kappa);

      triplets_Ms[1][1].emplace_back(s[1] + i[1] * d + k, s[1] + i[1] * d + k,
                                     measurement.kappa);
    }

    // Elements of ij block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(s[0] + n[0] + i[0] * d + r,
                                s[1] + n[1] + i[1] * d + c,
                                -measurement.kappa * measurement.R(r, c));
        triplets_Ms[1][1].emplace_back(
            s[0] + i[0] * d + r, s[1] + i[1] * d + c,
            -measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(s[1] + n[1] + i[1] * d + r,
                                s[0] + n[0] + i[0] * d + c,
                                -measurement.kappa * measurement.R(c, r));
        triplets_Ms[1][1].emplace_back(
            s[1] + i[1] * d + r, s[0] + i[0] * d + c,
            -measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(
            s[0] + n[0] + i[0] * d + r, s[0] + n[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
        triplets_Ms[1][1].emplace_back(
            s[0] + i[0] * d + r, s[0] + i[0] * d + c,
            measurement.tau * measurement.t(r) * measurement.t(c));
      }

    // update B and B0
    // update translation w.r.t. translation
    // update ith block
    triplets_B0.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B0.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B0.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B0.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B0.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }

    // update the auxiliary data matrix
    DiagT0[s[0] + i[0]] += 2 * measurement.tau;
    DiagT0[s[1] + i[1]] += 2 * measurement.tau;

    for (int k = 0; k < d; k++) {
      triplets_N0.emplace_back(s[0] + i[0], i[0] * d + k,
                               2 * measurement.tau * measurement.t[k]);
    }

    for (int k = 0; k < d; k++) {
      triplets_V0.emplace_back(i[0] * d + k, i[0] * d + k,
                               2 * measurement.kappa);
      triplets_V0.emplace_back(i[1] * d + k, i[1] * d + k,
                               2 * measurement.kappa);
    }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_V0.emplace_back(
            i[0] * d + r, i[0] * d + c,
            2 * measurement.tau * measurement.t(r) * measurement.t(c));
      }
  }

  // setup indices for poses related with inter-node measurements
  std::vector<int> inter_index_n[2]{std::vector<int>(num_n[0], -1),
                                    std::vector<int>(num_n[1], -1)};
  std::vector<int> inter_out[2] = {std::vector<int>(num_n[0], 0),
                                   std::vector<int>(num_n[1], 0)};
  num_inter_n = {0, 0};

  for (int m = 0; m < num_m[1]; m++) {
    auto const &measurement = inter_measurements[m];

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a inter-node measurement for node " << a << "."
                 << std::endl;

      continue;
    }

    if (measurement.i.node == a) {
      const auto &b = measurement.j.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_a.find(measurement.i.pose);
      assert(ii != index_a.end());

      if (ii == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_b.find(measurement.j.pose);
      assert(jj != index_b.end());

      if (jj == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      inter_index_n[0][ii->second[1]] = 0;
      inter_index_n[1][jj->second[1]] = 0;

      out[ii->second[1]] = 1;
      inter_out[0][ii->second[1]] = 1;
    } else {
      const auto &b = measurement.i.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_b.find(measurement.i.pose);
      assert(ii != index_b.end());

      if (ii == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_a.find(measurement.j.pose);
      assert(jj != index_a.end());

      if (jj == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      inter_index_n[1][ii->second[1]] = 0;
      inter_index_n[0][jj->second[1]] = 0;

      inter_out[1][ii->second[1]] = 1;
    }
  }

  int num_inter_nout[2] = {0, 0};

  // specify the sparse pattern for E and F
  std::vector<int> inter_inner_index_n[2][3] = {{{0}, {0}, {0}},
                                                {{0}, {0}, {0}}};

  for (int i = 0; i < num_n[0]; i++) {
    if (inter_index_n[0][i] == 0) {
      num_inter_nout[0] += inter_out[0][i];
      inter_index_n[0][i] = num_inter_n[0]++;

      inter_inner_index_n[0][0].push_back(num_inter_n[0]);
      inter_inner_index_n[0][1].push_back(num_inter_nout[0] * d);
      inter_inner_index_n[0][2].push_back(num_inter_n[0] * d +
                                          num_inter_nout[0] * (d * d - d));
    }
  }

  for (int i = 0; i < num_n[1]; i++) {
    if (inter_index_n[1][i] == 0) {
      num_inter_nout[1] += inter_out[1][i];
      inter_index_n[1][i] = num_inter_n[1]++;

      inter_inner_index_n[1][0].push_back(num_inter_n[1]);
      inter_inner_index_n[1][1].push_back(num_inter_nout[1] * d);
      inter_inner_index_n[1][2].push_back(num_inter_n[1] * d +
                                          num_inter_nout[1] * (d * d - d));
    }
  }

  if (num_inter_n[1] != num_n[1]) {
    LOG(WARNING) << "Inconsistent number of poses for neighboring poses."
                 << std::endl;
  }

  // process inter-node measurements
  int inter_i[2] = {0, 0};
  int inter_out_i[2] = {0, 0};

  for (int m = 0; m < num_m[1]; m++) {
    auto const &measurement = inter_measurements[m];

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a inter-node measurement for node " << a << "."
                 << std::endl;

      continue;
    }

    if (measurement.i.node == a) {
      const auto &b = measurement.j.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_a.find(measurement.i.pose);
      assert(ii != index_a.end());

      if (ii == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_b.find(measurement.j.pose);
      assert(jj != index_b.end());

      if (jj == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];

      // inter_n[0] = num_inter_n[ii->second[0]];
      // inter_n[1] = num_inter_n[jj->second[0]];

      inter_i[0] = inter_index_n[ii->second[0]][ii->second[1]];
      inter_i[1] = inter_index_n[jj->second[0]][jj->second[1]];

      inter_out_i[0] = inter_out[ii->second[0]][ii->second[1]];
      inter_out_i[1] = inter_out[jj->second[0]][jj->second[1]];
    } else {
      const auto &b = measurement.i.node;

      auto bb = index.find(b);
      assert(bb != index.end());

      if (bb == index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      auto &index_b = bb->second;

      auto ii = index_b.find(measurement.i.pose);
      assert(ii != index_b.end());

      if (ii == index_b.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.i.node
                   << ", " << measurement.i.pose << "]" << std::endl;

        exit(-1);
      }

      auto jj = index_a.find(measurement.j.pose);
      assert(jj != index_a.end());

      if (jj == index_a.end()) {
        LOG(ERROR) << "No index is specified for pose [" << measurement.j.node
                   << ", " << measurement.j.pose << "]" << std::endl;

        exit(-1);
      }

      s[0] = (d + 1) * num_s[ii->second[0]];
      s[1] = (d + 1) * num_s[jj->second[0]];

      n[0] = num_n[ii->second[0]];
      n[1] = num_n[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];

      inter_i[0] = inter_index_n[ii->second[0]][ii->second[1]];
      inter_i[1] = inter_index_n[jj->second[0]][jj->second[1]];

      inter_out_i[0] = inter_out[ii->second[0]][ii->second[1]];
      inter_out_i[1] = inter_out[jj->second[0]][jj->second[1]];
    }

    l = (d + 1) * m;

    sqrt_tau = std::sqrt(measurement.tau);
    sqrt_kappa = std::sqrt(measurement.kappa);

    if (measurement.i.node == a) {
      triplets_E.emplace_back(inter_i[0], m, 2 * measurement.tau);
      triplets_F.emplace_back(inter_i[1], m, 2 * measurement.tau);

      for (int k = 0; k < d; k++) {
        triplets_E.emplace_back(
            num_inter_n[0] + inter_inner_index_n[0][1][inter_i[0]] + k, m,
            2 * measurement.tau * measurement.t[k]);
      }

      for (int k = 0; k < d; k++) {
        triplets_E.emplace_back(num_inter_n[0] + num_inter_nout[0] * d +
                                    inter_inner_index_n[0][2][inter_i[0]] +
                                    k * d + k,
                                m, 2 * measurement.kappa);

        if (inter_out_i[1]) {
          triplets_F.emplace_back(num_inter_n[1] + num_inter_nout[1] * d +
                                      inter_inner_index_n[1][2][inter_i[1]] +
                                      k * d + k,
                                  m, 2 * measurement.kappa);
        } else {
          triplets_F.emplace_back(num_inter_n[1] + num_inter_nout[1] * d +
                                      inter_inner_index_n[1][2][inter_i[1]] + k,
                                  m, 2 * measurement.kappa);
        }
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_E.emplace_back(
              num_inter_n[0] + num_inter_nout[0] * d +
                  inter_inner_index_n[0][2][inter_i[0]] + r * d + c,
              m, 2 * measurement.tau * measurement.t[r] * measurement.t[c]);
        }
    } else {
      triplets_F.emplace_back(inter_i[0], m, 2 * measurement.tau);
      triplets_E.emplace_back(inter_i[1], m, 2 * measurement.tau);

      for (int k = 0; k < d; k++) {
        triplets_F.emplace_back(
            num_inter_n[1] + inter_inner_index_n[1][1][inter_i[0]] + k, m,
            2 * measurement.tau * measurement.t[k]);
      }

      for (int k = 0; k < d; k++) {
        triplets_F.emplace_back(num_inter_n[1] + num_inter_nout[1] * d +
                                    inter_inner_index_n[1][2][inter_i[0]] +
                                    k * d + k,
                                m, 2 * measurement.kappa);

        if (inter_out_i[1]) {
          triplets_E.emplace_back(num_inter_n[0] + num_inter_nout[0] * d +
                                      inter_inner_index_n[0][2][inter_i[1]] +
                                      k * d + k,
                                  m, 2 * measurement.kappa);
        } else {
          triplets_E.emplace_back(num_inter_n[0] + num_inter_nout[0] * d +
                                      inter_inner_index_n[0][2][inter_i[1]] + k,
                                  m, 2 * measurement.kappa);
        }
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_F.emplace_back(
              num_inter_n[1] + num_inter_nout[1] * d +
                  inter_inner_index_n[1][2][inter_i[0]] + r * d + c,
              m, 2 * measurement.tau * measurement.t[r] * measurement.t[c]);
        }
    }

    // update B and B1
    // update translation w.r.t. translation
    // update ith block
    triplets_B1.emplace_back(l, s[0] + i[0], sqrt_tau);

    // update jth block
    triplets_B1.emplace_back(l, s[1] + i[1], -sqrt_tau);

    // update translation w.r.t. rotation
    for (int k = 0; k < d; k++) {
      triplets_B1.emplace_back(l, s[0] + n[0] + i[0] * d + k,
                               sqrt_tau * measurement.t(k));
    }

    // update rotation w.r.t rotation
    // update ith block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_B1.emplace_back(l + r + 1, s[0] + n[0] + i[0] * d + c,
                                 sqrt_kappa * measurement.R(c, r));
      }

    // update jth block
    for (int k = 0; k < d; k++) {
      triplets_B1.emplace_back(l + k + 1, s[1] + n[1] + i[1] * d + k,
                               -sqrt_kappa);
    }
  }

  // add xi to the diagonal of G, G0, G2, P, S and Q should also be updated
  // accordingly
  for (int i = 0; i < num_n[0]; i++) {
    DiagT0[i] += 0.5 * xi;

    triplets_M.emplace_back(i, i, xi);
    triplets_Ms[0][0].emplace_back(i, i, xi);
    triplets_D0.emplace_back(i, i, xi);
    triplets_Q0.emplace_back(i, i, 2.0 * xi);

    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_Ms[1][1].emplace_back(i * d + k, i * d + k, xi);
      triplets_D0.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k, xi);
      triplets_Q0.emplace_back(num_n[0] + i * d + k, num_n[0] + i * d + k,
                               2.0 * xi);
      triplets_V0.emplace_back(i * d + k, i * d + k, 0.5 * xi);
    }

    for (int k = 0; k < d; k++) {
      triplets_N0.emplace_back(i, i * d + k, 0);
    }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_V0.emplace_back(i * d + r, i * d + c, 0);
      }

    if (inter_out[0][i] == 0) continue;

    triplets_M.emplace_back(i, i, 0);
    triplets_Ms[0][0].emplace_back(i, i, 0);

    for (int k = 0; k < d; k++) {
      triplets_M.emplace_back(i, num_n[0] + i * d + k, 0);
      triplets_M.emplace_back(num_n[0] + i * d + k, i, 0);

      triplets_Ms[0][1].emplace_back(i, i * d + k, 0);
      triplets_Ms[1][0].emplace_back(i * d + k, i, 0);

      triplets_D0.emplace_back(i, num_n[0] + i * d + k, 0);
      triplets_D0.emplace_back(num_n[0] + i * d + k, i, 0);

      triplets_Q0.emplace_back(i, num_n[0] + i * d + k, 0);
      triplets_Q0.emplace_back(num_n[0] + i * d + k, i, 0);
    }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_M.emplace_back(num_n[0] + i * d + r, num_n[0] + i * d + c, 0);

        triplets_Ms[1][1].emplace_back(i * d + r, i * d + c, 0);

        triplets_D0.emplace_back(num_n[0] + i * d + r, num_n[0] + i * d + c, 0);

        triplets_Q0.emplace_back(num_n[0] + i * d + r, num_n[0] + i * d + c, 0);
      }
  }

  for (int i = 0; i < num_n[1]; i++) {
    triplets_Q0.emplace_back((d + 1) * num_s[1] + i, (d + 1) * num_s[1] + i, 0);

    if (inter_out[1][i]) {
      for (int k = 0; k < d; k++) {
        triplets_Q0.emplace_back((d + 1) * num_s[1] + i,
                                 (d + 1) * num_s[1] + num_n[1] + d * i + k, 0);
        triplets_Q0.emplace_back((d + 1) * num_s[1] + num_n[1] + d * i + k,
                                 (d + 1) * num_s[1] + i, 0);
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_Q0.emplace_back((d + 1) * num_s[1] + num_n[1] + i * d + r,
                                   (d + 1) * num_s[1] + num_n[1] + i * d + c,
                                   0);
        }
    } else {
      for (int k = 0; k < d; k++) {
        triplets_Q0.emplace_back((d + 1) * num_s[1] + num_n[1] + i * d + k,
                                 (d + 1) * num_s[1] + num_n[1] + i * d + k, 0);
      }
    }
  }

  M.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  Ms[0][0].resize(num_n[0], num_n[0]);
  Ms[0][1].resize(num_n[0], d * num_n[0]);
  Ms[1][0].resize(d * num_n[0], num_n[0]);
  Ms[1][1].resize(d * num_n[0], d * num_n[0]);
  B0.resize((d + 1) * num_m[0], (d + 1) * (num_n[0] + num_n[1]));
  B1.resize((d + 1) * num_m[1], (d + 1) * (num_n[0] + num_n[1]));
  D0.resize((d + 1) * num_n[0], (d + 1) * num_n[0]);
  Q0.resize((d + 1) * (num_n[0] + num_n[1]), (d + 1) * (num_n[0] + num_n[1]));
  E.resize((d + 1) * num_inter_n[0] + d * d * num_inter_nout[0], num_m[1]);
  F.resize((d + 1) * num_inter_n[1] + d * d * num_inter_nout[1], num_m[1]);
  N0.resize(num_n[0], d * num_n[0]);
  V0.resize(d * num_n[0], d * num_n[0]);

  M.setFromTriplets(triplets_M.begin(), triplets_M.end());
  Ms[0][0].setFromTriplets(triplets_Ms[0][0].begin(), triplets_Ms[0][0].end());
  Ms[0][1].setFromTriplets(triplets_Ms[0][1].begin(), triplets_Ms[0][1].end());
  Ms[1][0].setFromTriplets(triplets_Ms[1][0].begin(), triplets_Ms[1][0].end());
  Ms[1][1].setFromTriplets(triplets_Ms[1][1].begin(), triplets_Ms[1][1].end());
  B0.setFromTriplets(triplets_B0.begin(), triplets_B0.end());
  B1.setFromTriplets(triplets_B1.begin(), triplets_B1.end());
  D0.setFromTriplets(triplets_D0.begin(), triplets_D0.end());
  Q0.setFromTriplets(triplets_Q0.begin(), triplets_Q0.end());
  E.setFromTriplets(triplets_E.begin(), triplets_E.end());
  F.setFromTriplets(triplets_F.begin(), triplets_F.end());
  N0.setFromTriplets(triplets_N0.begin(), triplets_N0.end());
  V0.setFromTriplets(triplets_V0.begin(), triplets_V0.end());

  // update the index
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 2; c++) {
      inner_index_G[r][c].clear();
      inner_index_Gs[r][c].clear();
      inner_index_D[r][c].clear();
      inner_index_Q[r][c].clear();
    }

  outer_index_G[0][0] = 0;
  outer_index_G[0][1] = num_inter_n[0];
  outer_index_G[1][0] = num_inter_n[0];
  outer_index_G[1][1] = num_inter_n[0] + num_inter_nout[0] * d;

  inner_nnz_G[0][0] = num_inter_n[0];
  inner_nnz_G[0][1] = num_inter_nout[0] * d;
  inner_nnz_G[1][0] = num_inter_nout[0] * d;
  inner_nnz_G[1][1] = num_inter_n[0] * d + num_inter_nout[0] * (d * d - d);

  outer_index_Gs[0][0] = 0;
  outer_index_Gs[0][1] = num_inter_n[0];
  outer_index_Gs[1][0] = num_inter_n[0];
  outer_index_Gs[1][1] = num_inter_n[0] + num_inter_nout[0] * d;

  inner_nnz_Gs[0][0] = num_inter_n[0];
  inner_nnz_Gs[0][1] = num_inter_nout[0] * d;
  inner_nnz_Gs[1][0] = num_inter_nout[0] * d;
  inner_nnz_Gs[1][1] = num_inter_n[0] * d + num_inter_nout[0] * (d * d - d);

  outer_index_D[0][0] = 0;
  outer_index_D[0][1] = num_inter_n[0];
  outer_index_D[1][0] = num_inter_n[0];
  outer_index_D[1][1] = num_inter_n[0] + num_inter_nout[0] * d;

  inner_nnz_D[0][0] = num_inter_n[0];
  inner_nnz_D[0][1] = num_inter_nout[0] * d;
  inner_nnz_D[1][0] = num_inter_nout[0] * d;
  inner_nnz_D[1][1] = num_inter_n[0] * d + num_inter_nout[0] * (d * d - d);

  outer_index_Q[0][0] = 0;
  outer_index_Q[0][1] = num_inter_n[1];
  outer_index_Q[1][0] = num_inter_n[1];
  outer_index_Q[1][1] = num_inter_n[1] + num_inter_nout[1] * d;

  inner_nnz_Q[0][0] = num_inter_n[1];
  inner_nnz_Q[0][1] = num_inter_nout[1] * d;
  inner_nnz_Q[1][0] = num_inter_nout[1] * d;
  inner_nnz_Q[1][1] = num_inter_n[1] * d + num_inter_nout[1] * (d * d - d);

  outer_index_T = 0;
  inner_index_T.clear();
  inner_nnz_T = num_inter_n[0];

  outer_index_N = num_inter_n[0];
  inner_index_N.clear();
  inner_nnz_N = num_inter_nout[0] * d;

  outer_index_V = num_inter_n[0] + num_inter_nout[0] * d;
  inner_index_V.clear();
  inner_nnz_V = num_inter_n[0] * d + num_inter_nout[0] * (d * d - d);

  for (int i = 0; i < num_n[0]; i++) {
    if (inter_index_n[0][i] < 0) continue;

    inner_index_G[0][0].push_back(&M.coeffRef(i, i) - M.valuePtr());
    inner_index_Gs[0][0].push_back(&Ms[0][0].coeffRef(i, i) -
                                   Ms[0][0].valuePtr());
    inner_index_D[0][0].push_back(&D0.coeffRef(i, i) - D0.valuePtr());
    inner_index_T.push_back(i);

    if (inter_out[0][i]) {
      for (int k = 0; k < d; k++) {
        inner_index_G[0][1].push_back(&M.coeffRef(i, num_n[0] + i * d + k) -
                                      M.valuePtr());
        inner_index_G[1][0].push_back(&M.coeffRef(num_n[0] + i * d + k, i) -
                                      M.valuePtr());

        inner_index_Gs[0][1].push_back(&Ms[0][1].coeffRef(i, i * d + k) -
                                       Ms[0][1].valuePtr());
        inner_index_Gs[1][0].push_back(&Ms[1][0].coeffRef(i * d + k, i) -
                                       Ms[1][0].valuePtr());

        inner_index_D[0][1].push_back(&D0.coeffRef(i, num_n[0] + i * d + k) -
                                      D0.valuePtr());
        inner_index_D[1][0].push_back(&D0.coeffRef(num_n[0] + i * d + k, i) -
                                      D0.valuePtr());

        inner_index_N.push_back(&N0.coeffRef(i, i * d + k) - N0.valuePtr());
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          inner_index_G[1][1].push_back(
              &M.coeffRef(num_n[0] + i * d + r, num_n[0] + i * d + c) -
              M.valuePtr());

          inner_index_Gs[1][1].push_back(
              &Ms[1][1].coeffRef(i * d + r, i * d + c) - Ms[1][1].valuePtr());

          inner_index_D[1][1].push_back(
              &D0.coeffRef(num_n[0] + i * d + r, num_n[0] + i * d + c) -
              D0.valuePtr());

          inner_index_V.push_back(&V0.coeffRef(i * d + r, i * d + c) -
                                  V0.valuePtr());
        }
    } else {
      for (int k = 0; k < d; k++) {
        inner_index_G[1][1].push_back(
            &M.coeffRef(num_n[0] + i * d + k, num_n[0] + i * d + k) -
            M.valuePtr());

        inner_index_Gs[1][1].push_back(
            &Ms[1][1].coeffRef(i * d + k, i * d + k) - Ms[1][1].valuePtr());

        inner_index_D[1][1].push_back(
            &D0.coeffRef(num_n[0] + i * d + k, num_n[0] + i * d + k) -
            D0.valuePtr());

        inner_index_V.push_back(&V0.coeffRef(i * d + k, i * d + k) -
                                V0.valuePtr());
      }
    }

    assert(inner_index_G[0][0].size() ==
           inter_inner_index_n[0][0][inter_index_n[0][i] + 1]);
    assert(inner_index_G[0][1].size() ==
           inter_inner_index_n[0][1][inter_index_n[0][i] + 1]);
    assert(inner_index_G[1][0].size() ==
           inter_inner_index_n[0][1][inter_index_n[0][i] + 1]);
    assert(inner_index_G[1][1].size() ==
           inter_inner_index_n[0][2][inter_index_n[0][i] + 1]);
    assert(inner_index_V.size() ==
           inter_inner_index_n[0][2][inter_index_n[0][i] + 1]);
  }

  for (int i = 0; i < num_n[1]; i++) {
    assert(inter_index_n[1][i] >= 0);

    inner_index_Q[0][0].push_back(
        &Q0.coeffRef((d + 1) * num_s[1] + i, (d + 1) * num_s[1] + i) -
        Q0.valuePtr());

    if (inter_out[1][i]) {
      for (int k = 0; k < d; k++) {
        inner_index_Q[0][1].push_back(
            &Q0.coeffRef((d + 1) * num_s[1] + i,
                         (d + 1) * num_s[1] + num_n[1] + i * d + k) -
            Q0.valuePtr());
        inner_index_Q[1][0].push_back(
            &Q0.coeffRef((d + 1) * num_s[1] + num_n[1] + i * d + k,
                         (d + 1) * num_s[1] + i) -
            Q0.valuePtr());
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          inner_index_Q[1][1].push_back(
              &Q0.coeffRef((d + 1) * num_s[1] + num_n[1] + i * d + r,
                           (d + 1) * num_s[1] + num_n[1] + i * d + c) -
              Q0.valuePtr());
        }
    } else {
      for (int k = 0; k < d; k++) {
        inner_index_Q[1][1].push_back(
            &Q0.coeffRef((d + 1) * num_s[1] + num_n[1] + i * d + k,
                         (d + 1) * num_s[1] + num_n[1] + i * d + k) -
            Q0.valuePtr());
      }
    }
  }

  return 0;
}
}  // namespace DPGO
