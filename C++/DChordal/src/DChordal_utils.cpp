#include <DChordal/DChordalReducedProblem.h>
#include <DChordal/DChordal_utils.h>

#include <DPGO/DPGO_utils.h>
#include <SESync/RelativePoseMeasurement.h>
#include <SESync/SESync.h>

#include <utility>

namespace DChordal {
int SESync(int a, const measurements_t &measurements,
           const SESync::SESyncOpts &opts, SESync::SESyncResult &result) {
  SESync::measurements_t SESyncMeasurements;

  for (const auto &measurement : measurements) {
    if (measurement.i.node != a || measurement.j.node != a) {
      continue;
    }

    SESyncMeasurements.push_back(SESync::RelativePoseMeasurement(
        measurement.i.pose, measurement.j.pose, measurement.R, measurement.t,
        measurement.kappa, measurement.tau));
  }

  result = SESync::SESync(SESyncMeasurements, opts);

  return 0;
}

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
    std::map<int, int> &n_recv) {
  DPGO::generate_data_info(a, measurements, intra_measurements,
                           inter_measurements, num_poses, offsets,
                           num_measurements, index, sent, recv);

  n_index.clear();
  n_sent.clear();
  n_recv.clear();

  int count = 1;

  for (const auto &node : index) {
    if (node.first == a) {
      n_index.insert({node.first, 0});
    } else {
      n_sent.insert({node.first, 0});
      n_recv.insert({node.first, count});
      n_index.insert({node.first, count});
      count++;
    }
  }

  num_neighbors = n_index.size() - 1;

  return 0;
}

int simplify_data_matrix_reduced_R(
    int a, const measurements_t &inter_measurements, const Matrix &X,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    const std::map<int, int> &n_index, Scalar xi, Matrix &rotG, Matrix &rotS,
    Matrix &rotD, Matrix &rotH, Matrix &rotg, SparseMatrix &rotB,
    Matrix &rotb) {
  rotG.resize(0, 0);
  rotS.resize(0, 0);
  rotD.resize(0, 0);
  rotH.resize(0, 0);
  rotg.resize(0, 0);
  rotB.resize(0, 0);
  rotb.resize(0, 0);

  const int num_nodes = n_index.size();
  const int num_measurements = inter_measurements.size();

  const int d = inter_measurements[0].t.size();
  const int num_poses = num_n[0] + num_n[1];

  assert(X.rows() == num_poses * (d + 1) && X.cols() == d);

  if (X.rows() != num_poses * (d + 1) || X.cols() != d) {
    LOG(ERROR) << "Inconsistent number of poses for node " << a << "."
               << std::endl;

    return -1;
  }

  rotG.setZero(d, d);
  rotS.setZero(d, num_nodes * d);
  rotD.setZero(d, d);
  rotH.setZero(d, num_nodes * d);
  rotB.resize(0, 0);
  rotg.setZero(d, d);

  std::list<Eigen::Triplet<Scalar>> triplets_rotB;
  rotb.setZero(num_measurements * d, d);

  int s[2];
  int n[2];
  int i[2];
  int ni[2];

  int l;

  Scalar sqrt_kappa;

  auto aa = index.find(a);
  assert(aa != index.end());

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  auto &index_a = aa->second;

  Matrix nR(d, d);

  for (int m = 0; m < num_measurements; m++) {
    auto const &measurement = inter_measurements[m];

    if ((measurement.i.node != a && measurement.j.node != a) ||
        (measurement.i.node == a && measurement.j.node == a)) {
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

      auto nbb = n_index.find(measurement.j.node);
      assert(nbb != n_index.end());

      if (nbb == n_index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      ni[0] = 0;
      ni[1] = nbb->second;
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

      auto nbb = n_index.find(measurement.i.node);
      assert(nbb != n_index.end());

      if (nbb == n_index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      ni[0] = nbb->second;
      ni[1] = 0;
    }

    const auto &Ri = X.middleRows(s[0] + n[0] + d * i[0], d);
    const auto &Rj = X.middleRows(s[1] + n[1] + d * i[1], d);

    nR.noalias() = Ri.transpose() * measurement.R * Rj;

    l = d * m;
    sqrt_kappa = std::sqrt(measurement.kappa);

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++)
        triplets_rotB.emplace_back(m * d + r, ni[0] * d + c,
                                   sqrt_kappa * nR(c, r));

    for (int k = 0; k < d; k++)
      triplets_rotB.emplace_back(m * d + k, ni[1] * d + k, -sqrt_kappa);

    if (measurement.i.node == a) {
      for (int k = 0; k < d; k++) {
        rotG(ni[0] * d + k, ni[0] * d + k) += 2 * measurement.kappa;
        rotS(ni[0] * d + k, ni[0] * d + k) -= measurement.kappa;
        rotD(ni[0] * d + k, ni[0] * d + k) += measurement.kappa;
        rotH(ni[0] * d + k, ni[0] * d + k) += measurement.kappa;
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          rotH(ni[0] * d + r, ni[1] * d + c) -= measurement.kappa * nR(r, c);
          rotS(ni[0] * d + r, ni[1] * d + c) -= measurement.kappa * nR(r, c);
        }
    } else {
      for (int k = 0; k < d; k++) {
        rotG(ni[1] * d + k, ni[1] * d + k) += 2 * measurement.kappa;
        rotS(ni[1] * d + k, ni[1] * d + k) -= measurement.kappa;
        rotD(ni[1] * d + k, ni[1] * d + k) += measurement.kappa;
        rotH(ni[1] * d + k, ni[1] * d + k) += measurement.kappa;
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          rotS(ni[1] * d + r, ni[0] * d + c) -= measurement.kappa * nR(c, r);
          rotH(ni[1] * d + r, ni[0] * d + c) -= measurement.kappa * nR(c, r);
        }
    }
  }

  for (int k = 0; k < d; k++) {
    rotG(k, k) += xi;
    rotS(k, k) -= xi;
  }

  rotB.resize(num_measurements * d, num_nodes * d);
  rotB.setFromTriplets(triplets_rotB.begin(), triplets_rotB.end());

  return 0;
}

int precompute_data_matrix_recover_t(int a,
                                     const measurements_t &measurements,
                                     int num_poses, SparseMatrix &L,
                                     SparseMatrix &P) {
  L.resize(0, 0);
  P.resize(0, 0);

  assert(!measurements.empty());

  if (measurements.empty()) {
    LOG(ERROR) <<"No measurements are specified." <<std::endl;

    return -1;
  }

  const int d = measurements.empty() ? 0 : measurements[0].t.size();

  std::list<Eigen::Triplet<Scalar>> triplets_L;
  std::list<Eigen::Triplet<Scalar>> triplets_P;

  for (const auto& measurement: measurements) {
    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      continue;
    }

    int i[2] = {measurement.i.pose, measurement.j.pose};

    triplets_L.emplace_back(i[0], i[0], measurement.tau);
    triplets_L.emplace_back(i[1], i[1], measurement.tau);
    triplets_L.emplace_back(i[0], i[1], -measurement.tau);
    triplets_L.emplace_back(i[1], i[0], -measurement.tau);

    for (int k = 0; k < d; k++) {
      triplets_P.emplace_back(i[0], i[0] * d + k, measurement.tau * measurement.t[k]);
      triplets_P.emplace_back(i[1], i[0] * d + k, -measurement.tau * measurement.t[k]);
    }
  }

  triplets_L.emplace_back(0, 0, 100);

  L.resize(num_poses, num_poses);
  P.resize(num_poses, num_poses * d);

  L.setFromTriplets(triplets_L.begin(), triplets_L.end());
  P.setFromTriplets(triplets_P.begin(), triplets_P.end());

  return 0;
}

int simplify_data_matrix_reduced_t(
    int a, const measurements_t &inter_measurements, const Matrix &X,
    const Matrix &nR, const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    const std::map<int, int> &n_index, Scalar xi, Matrix &tG, Matrix &tS,
    Matrix &tD, Matrix &tH, Matrix &tg, SparseMatrix &tB, Matrix &tb) {
  tG.resize(0, 0);
  tS.resize(0, 0);
  tD.resize(0, 0);
  tH.resize(0, 0);
  tg.resize(0, 0);
  tB.resize(0, 0);
  tb.resize(0, 0);

  const int num_nodes = n_index.size();
  const int num_measurements = inter_measurements.size();

  const int d = inter_measurements[0].t.size();
  const int num_poses = num_n[0] + num_n[1];

  assert(X.rows() == num_poses * (d + 1) && X.cols() == d &&
         nR.rows() == num_nodes * d && nR.cols() == d);

  if (X.rows() != num_poses * (d + 1) || X.cols() != d ||
      nR.rows() != num_nodes * d || nR.cols() != d) {
    LOG(ERROR) << "Inconsistent number of poses for node " << a << "."
               << std::endl;

    return -1;
  }

  tG.setZero(1, 1);
  tS.setZero(1, num_nodes);
  tD.setZero(1, 1);
  tH.setZero(1, num_nodes);
  tB.resize(0, 0);
  tg.setZero(1, d);

  std::list<Eigen::Triplet<Scalar>> triplets_tB;
  tb.setZero(num_measurements, d);

  int s[2];
  int n[2];
  int i[2];
  int ni[2];

  Scalar sqrt_tau;

  auto aa = index.find(a);
  auto &index_a = aa->second;

  Vector nt;

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  for (int m = 0; m < num_measurements; m++) {
    auto const &measurement = inter_measurements[m];

    if ((measurement.i.node != a && measurement.j.node != a) ||
        (measurement.i.node == a && measurement.j.node == a)) {
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

      auto nbb = n_index.find(measurement.j.node);
      assert(nbb != n_index.end());

      if (nbb == n_index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      ni[0] = 0;
      ni[1] = nbb->second;
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

      auto nbb = n_index.find(measurement.i.node);
      assert(nbb != n_index.end());

      if (nbb == n_index.end()) {
        LOG(ERROR) << "No index is specified for node " << b << "."
                   << std::endl;

        exit(-1);
      }

      ni[0] = nbb->second;
      ni[1] = 0;
    }

    const auto &Ri = X.middleRows(s[0] + n[0] + d * i[0], d);
    const auto &Rj = X.middleRows(s[1] + n[1] + d * i[1], d);

    Vector ti = X.row(s[0] + i[0]);
    ti.noalias() += Ri.transpose() * measurement.t;
    const auto &tj = X.row(s[1] + i[1]).transpose();

    const auto &nRi = nR.middleRows(ni[0] * d, d);
    const auto &nRj = nR.middleRows(ni[1] * d, d);

    nt.noalias() = nRi.transpose() * ti;
    nt.noalias() -= nRj.transpose() * tj;

    sqrt_tau = std::sqrt(measurement.tau);

    if (measurement.i.node == a) {
      triplets_tB.emplace_back(m, ni[0], sqrt_tau);
      triplets_tB.emplace_back(m, ni[1], -sqrt_tau);
      tb.row(m) = sqrt_tau * nt;

      tG(ni[0], ni[0]) += 2 * measurement.tau;
      tS(ni[0], ni[0]) -= measurement.tau;
      tS(ni[0], ni[1]) -= measurement.tau;

      tD(ni[0], ni[0]) += measurement.tau;
      tH(ni[0], ni[0]) += measurement.tau;
      tH(ni[0], ni[1]) -= measurement.tau;

      tg += measurement.tau * nt.transpose();
    } else {
      triplets_tB.emplace_back(m, ni[0], -sqrt_tau);
      triplets_tB.emplace_back(m, ni[1], sqrt_tau);
      tb.row(m) = -sqrt_tau * nt;

      tG(ni[1], ni[1]) += 2 * measurement.tau;
      tS(ni[1], ni[1]) -= measurement.tau;
      tS(ni[1], ni[0]) -= measurement.tau;

      tD(ni[1], ni[1]) += measurement.tau;
      tH(ni[1], ni[1]) += measurement.tau;
      tH(ni[1], ni[0]) -= measurement.tau;

      tg -= measurement.tau * nt.transpose();
    }
  }

  tG(0, 0) += xi;
  tS(0, 0) -= xi;

  tB.resize(num_measurements, num_nodes);
  tB.setFromTriplets(triplets_tB.begin(), triplets_tB.end());

  return 0;
}

int simplify_data_matrix_R(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &rotG, SparseMatrix &rotS, SparseMatrix &rotD,
    SparseMatrix &rotH, Matrix &rotg, SparseMatrix &rotB, Matrix &rotb,
    SparseMatrix &rotB0, Matrix &rotb0, SparseMatrix &rotB1, Matrix &rotb1) {
  rotG.resize(0, 0);
  rotS.resize(0, 0);
  rotD.resize(0, 0);
  rotH.resize(0, 0);
  rotg.resize(0, 0);
  rotB.resize(0, 0);
  rotb.resize(0, 0);
  rotB0.resize(0, 0);
  rotb0.resize(0, 0);
  rotB1.resize(0, 0);
  rotb1.resize(0, 0);

  const int d = inter_measurements[0].t.size();
  const int num_poses = num_n[0] + num_n[1];

  rotg.setZero(d * num_n[0], d);
  rotb.setZero(d * (num_m[0] + num_m[1]), d);
  rotb0.setZero(d * num_m[0], d);
  rotb1.setZero(d * num_m[1], d);

  std::list<Eigen::Triplet<Scalar>> triplets_rotG;
  std::list<Eigen::Triplet<Scalar>> triplets_rotS;
  std::list<Eigen::Triplet<Scalar>> triplets_rotD;
  std::list<Eigen::Triplet<Scalar>> triplets_rotH;
  std::list<Eigen::Triplet<Scalar>> triplets_rotB;
  std::list<Eigen::Triplet<Scalar>> triplets_rotB0;
  std::list<Eigen::Triplet<Scalar>> triplets_rotB1;

  int s[2];
  int i[2];

  Scalar sqrt_kappa;

  auto aa = index.find(a);
  auto &index_a = aa->second;

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  for (int m = 0; m < num_m[0]; m++) {
    auto const &measurement = intra_measurements[m];

    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      continue;
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

    s[0] = num_s[ii->second[0]];
    s[1] = num_s[jj->second[0]];

    i[0] = ii->second[1];
    i[1] = jj->second[1];

    sqrt_kappa = std::sqrt(measurement.kappa);

    for (int k = 0; k < d; k++) {
      triplets_rotG.emplace_back(i[0] * d + k, i[0] * d + k, measurement.kappa);
      triplets_rotD.emplace_back(i[0] * d + k, i[0] * d + k, measurement.kappa);
      triplets_rotH.emplace_back(i[0] * d + k, i[0] * d + k, measurement.kappa);
    }

    for (int k = 0; k < d; k++) {
      triplets_rotG.emplace_back(i[1] * d + k, i[1] * d + k, measurement.kappa);
      triplets_rotD.emplace_back(i[1] * d + k, i[1] * d + k, measurement.kappa);
      triplets_rotH.emplace_back(i[1] * d + k, i[1] * d + k, measurement.kappa);
    }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_rotG.emplace_back(i[0] * d + r, i[1] * d + c,
                                   -measurement.kappa * measurement.R(r, c));
        triplets_rotD.emplace_back(i[0] * d + r, i[1] * d + c,
                                   -measurement.kappa * measurement.R(r, c));
        triplets_rotH.emplace_back(i[0] * d + r, i[1] * d + c,
                                   -measurement.kappa * measurement.R(r, c));
      }

    // Elements of ji block
    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_rotG.emplace_back(i[1] * d + r, i[0] * d + c,
                                   -measurement.kappa * measurement.R(c, r));
        triplets_rotD.emplace_back(i[1] * d + r, i[0] * d + c,
                                   -measurement.kappa * measurement.R(c, r));
        triplets_rotH.emplace_back(i[1] * d + r, i[0] * d + c,
                                   -measurement.kappa * measurement.R(c, r));
      }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_rotB.emplace_back(d * m + r, i[0] * d + c,
                                   sqrt_kappa * measurement.R(c, r));
        triplets_rotB0.emplace_back(d * m + r, i[0] * d + c,
                                    sqrt_kappa * measurement.R(c, r));
      }

    for (int k = 0; k < d; k++) {
      triplets_rotB.emplace_back(d * m + k, i[1] * d + k, -sqrt_kappa);
      triplets_rotB0.emplace_back(d * m + k, i[1] * d + k, -sqrt_kappa);
    }
  }

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

      s[0] = d * num_s[ii->second[0]];
      s[1] = d * num_s[jj->second[0]];

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

      s[0] = d * num_s[ii->second[0]];
      s[1] = d * num_s[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    }

    sqrt_kappa = std::sqrt(measurement.kappa);

    if (measurement.i.node == a) {
      for (int k = 0; k < d; k++) {
        triplets_rotG.emplace_back(s[0] + i[0] * d + k, s[0] + i[0] * d + k,
                                   2 * measurement.kappa);
        triplets_rotS.emplace_back(s[0] + i[0] * d + k, s[0] + i[0] * d + k,
                                   -measurement.kappa);
        triplets_rotD.emplace_back(s[0] + i[0] * d + k, s[0] + i[0] * d + k,
                                   measurement.kappa);
        triplets_rotH.emplace_back(s[0] + i[0] * d + k, s[0] + i[0] * d + k,
                                   measurement.kappa);
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_rotS.emplace_back(s[0] + i[0] * d + r, s[1] + i[1] * d + c,
                                     -measurement.kappa * measurement.R(r, c));
          triplets_rotH.emplace_back(s[0] + i[0] * d + r, s[1] + i[1] * d + c,
                                     -measurement.kappa * measurement.R(r, c));
        }
    } else {
      for (int k = 0; k < d; k++) {
        triplets_rotG.emplace_back(s[1] + i[1] * d + k, s[1] + i[1] * d + k,
                                   2 * measurement.kappa);
        triplets_rotS.emplace_back(s[1] + i[1] * d + k, s[1] + i[1] * d + k,
                                   -measurement.kappa);
        triplets_rotD.emplace_back(s[1] + i[1] * d + k, s[1] + i[1] * d + k,
                                   measurement.kappa);
        triplets_rotH.emplace_back(s[1] + i[1] * d + k, s[1] + i[1] * d + k,
                                   measurement.kappa);
      }

      for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++) {
          triplets_rotS.emplace_back(s[1] + i[1] * d + r, s[0] + i[0] * d + c,
                                     -measurement.kappa * measurement.R(c, r));
          triplets_rotH.emplace_back(s[1] + i[1] * d + r, s[0] + i[0] * d + c,
                                     -measurement.kappa * measurement.R(c, r));
        }
    }

    for (int r = 0; r < d; r++)
      for (int c = 0; c < d; c++) {
        triplets_rotB.emplace_back((num_m[0] + m) * d + r, s[0] + i[0] * d + c,
                                   sqrt_kappa * measurement.R(c, r));
        triplets_rotB1.emplace_back(m * d + r, s[0] + i[0] * d + c,
                                    sqrt_kappa * measurement.R(c, r));
      }

    for (int k = 0; k < d; k++) {
      triplets_rotB.emplace_back((num_m[0] + m) * d + k, s[1] + i[1] * d + k,
                                 -sqrt_kappa);
      triplets_rotB1.emplace_back(m * d + k, s[1] + i[1] * d + k, -sqrt_kappa);
    }
  }

  for (int i = 0; i < d * num_n[0]; i++) {
    triplets_rotG.emplace_back(i, i, xi);
    triplets_rotG.emplace_back(i, i, -xi);
  }

  rotG.resize(num_n[0] * d, num_n[0] * d);
  rotS.resize(num_n[0] * d, (num_n[0] + num_n[1]) * d);
  rotD.resize(num_n[0] * d, num_n[0] * d);
  rotH.resize(num_n[0] * d, (num_n[0] + num_n[1]) * d);
  rotB.resize((num_m[0] + num_m[1]) * d, (num_n[0] + num_n[1]) * d);
  rotB0.resize(num_m[0] * d, (num_n[0] + num_n[1]) * d);
  rotB1.resize(num_m[1] * d, (num_n[0] + num_n[1]) * d);

  rotG.setFromTriplets(triplets_rotG.begin(), triplets_rotG.end());
  rotS.setFromTriplets(triplets_rotS.begin(), triplets_rotS.end());
  rotD.setFromTriplets(triplets_rotD.begin(), triplets_rotD.end());
  rotH.setFromTriplets(triplets_rotH.begin(), triplets_rotH.end());
  rotB.setFromTriplets(triplets_rotB.begin(), triplets_rotB.end());
  rotB0.setFromTriplets(triplets_rotB0.begin(), triplets_rotB0.end());
  rotB1.setFromTriplets(triplets_rotB1.begin(), triplets_rotB1.end());

  return 0;
}

int simplify_data_matrix_t(
    int a, const measurements_t &intra_measurements,
    const measurements_t &inter_measurements, const Matrix &R,
    const Eigen::Matrix<int, 2, 1> &num_n,
    const Eigen::Matrix<int, 2, 1> &num_s,
    const Eigen::Matrix<int, 2, 1> &num_m,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>> &index,
    Scalar xi, SparseMatrix &tG, SparseMatrix &tS, SparseMatrix &tD,
    SparseMatrix &tH, Matrix &tg, SparseMatrix &tB, Matrix &tb,
    SparseMatrix &tB0, Matrix &tb0, SparseMatrix &tB1, Matrix &tb1) {
  tG.resize(0, 0);
  tS.resize(0, 0);
  tD.resize(0, 0);
  tH.resize(0, 0);
  tg.resize(0, 0);
  tB.resize(0, 0);
  tb.resize(0, 0);
  tB0.resize(0, 0);
  tb0.resize(0, 0);
  tB1.resize(0, 0);
  tb1.resize(0, 0);

  const int d = inter_measurements[0].t.size();
  const int num_poses = num_n[0] + num_n[1];

  assert(R.rows() == num_poses * d && R.cols() == d);

  if (R.rows() != num_poses * d || R.cols() != d) {
    LOG(ERROR) << "Inconsistent number of poses for node " << a << "."
               << std::endl;

    return -1;
  }

  tg.setZero(num_n[0], d);
  tb.setZero(num_m[0] + num_m[1], d);
  tb0.setZero(num_m[0], d);
  tb1.setZero(num_m[1], d);

  std::list<Eigen::Triplet<Scalar>> triplets_tG;
  std::list<Eigen::Triplet<Scalar>> triplets_tS;
  std::list<Eigen::Triplet<Scalar>> triplets_tD;
  std::list<Eigen::Triplet<Scalar>> triplets_tH;
  std::list<Eigen::Triplet<Scalar>> triplets_tB;
  std::list<Eigen::Triplet<Scalar>> triplets_tB0;
  std::list<Eigen::Triplet<Scalar>> triplets_tB1;

  int s[2];
  int i[2];

  Scalar sqrt_tau;

  auto aa = index.find(a);
  auto &index_a = aa->second;

  Vector nt;

  if (aa == index.end()) {
    LOG(ERROR) << "No index is specified for node " << a << "." << std::endl;

    exit(-1);
  }

  for (int m = 0; m < num_m[0]; m++) {
    auto const &measurement = intra_measurements[m];

    assert(measurement.i.node == a && measurement.j.node == a);

    if (measurement.i.node != a && measurement.j.node != a) {
      LOG(ERROR) << "Not a intra-node measurement for node " << a << "."
                 << std::endl;

      continue;
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

    s[0] = num_s[ii->second[0]];
    s[1] = num_s[jj->second[0]];

    i[0] = ii->second[1];
    i[1] = jj->second[1];

    const auto &Ri = R.middleRows((s[0] + i[0]) * d, d);

    Vector nt = Ri.transpose() * measurement.t;
    sqrt_tau = std::sqrt(measurement.tau);

    triplets_tG.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_tG.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_tG.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_tG.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_tD.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_tD.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_tD.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_tD.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    triplets_tH.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
    triplets_tH.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
    triplets_tH.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);
    triplets_tH.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

    tg.row(s[0] + i[0]) += measurement.tau * nt.transpose();
    tg.row(s[1] + i[1]) -= measurement.tau * nt.transpose();

    triplets_tB.emplace_back(m, s[0] + i[0], sqrt_tau);
    triplets_tB.emplace_back(m, s[1] + i[1], -sqrt_tau);

    tb.row(m) += sqrt_tau * nt.transpose();

    triplets_tB0.emplace_back(m, s[0] + i[0], sqrt_tau);
    triplets_tB0.emplace_back(m, s[1] + i[1], -sqrt_tau);

    tb0.row(m) += sqrt_tau * nt.transpose();
  }

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

      s[0] = num_s[ii->second[0]];
      s[1] = num_s[jj->second[0]];

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

      s[0] = num_s[ii->second[0]];
      s[1] = num_s[jj->second[0]];

      i[0] = ii->second[1];
      i[1] = jj->second[1];
    }

    const auto &Ri = R.middleRows((s[0] + i[0]) * d, d);

    Vector nt = Ri.transpose() * measurement.t;
    sqrt_tau = std::sqrt(measurement.tau);

    if (measurement.i.node == a) {
      triplets_tG.emplace_back(s[0] + i[0], s[0] + i[0], 2 * measurement.tau);

      triplets_tS.emplace_back(s[0] + i[0], s[0] + i[0], -measurement.tau);
      triplets_tS.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);

      triplets_tD.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);

      triplets_tH.emplace_back(s[0] + i[0], s[0] + i[0], measurement.tau);
      triplets_tH.emplace_back(s[0] + i[0], s[1] + i[1], -measurement.tau);

      tg.row(s[0] + i[0]) += measurement.tau * nt.transpose();
    } else {
      triplets_tG.emplace_back(s[1] + i[1], s[1] + i[1], 2 * measurement.tau);

      triplets_tS.emplace_back(s[1] + i[1], s[1] + i[1], -measurement.tau);
      triplets_tS.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

      triplets_tD.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);

      triplets_tH.emplace_back(s[1] + i[1], s[1] + i[1], measurement.tau);
      triplets_tH.emplace_back(s[1] + i[1], s[0] + i[0], -measurement.tau);

      tg.row(s[1] + i[1]) -= measurement.tau * nt.transpose();
    }

    triplets_tB.emplace_back(num_m[0] + m, s[0] + i[0], sqrt_tau);
    triplets_tB.emplace_back(num_m[0] + m, s[1] + i[1], -sqrt_tau);

    tb.row(num_m[0] + m) += sqrt_tau * nt.transpose();

    triplets_tB1.emplace_back(m, s[0] + i[0], sqrt_tau);
    triplets_tB1.emplace_back(m, s[1] + i[1], -sqrt_tau);

    tb1.row(m) += sqrt_tau * nt.transpose();
  }

  for (int i = 0; i < num_n[0]; i++) {
    triplets_tG.emplace_back(i, i, xi);
    triplets_tS.emplace_back(i, i, -xi);
  }

  tG.resize(num_n[0], num_n[0]);
  tS.resize(num_n[0], num_n[0] + num_n[1]);
  tD.resize(num_n[0], num_n[0]);
  tH.resize(num_n[0], num_n[0] + num_n[1]);
  tB.resize(num_m[0] + num_m[1], num_n[0] + num_n[1]);
  tB0.resize(num_m[0], num_n[0] + num_n[1]);
  tB1.resize(num_m[1], num_n[0] + num_n[1]);

  tG.setFromTriplets(triplets_tG.begin(), triplets_tG.end());
  tS.setFromTriplets(triplets_tS.begin(), triplets_tS.end());
  tD.setFromTriplets(triplets_tD.begin(), triplets_tD.end());
  tH.setFromTriplets(triplets_tH.begin(), triplets_tH.end());
  tB.setFromTriplets(triplets_tB.begin(), triplets_tB.end());
  tB0.setFromTriplets(triplets_tB0.begin(), triplets_tB0.end());
  tB1.setFromTriplets(triplets_tB1.begin(), triplets_tB1.end());

  return 0;
}
}  // namespace DChordal
