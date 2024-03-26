#include <DPGO/PCM.h>
#include <glog/logging.h>

namespace DPGO {
int PCM::update(
    int alpha, int beta, const measurements_t& inter_measurements,
    const Eigen::Matrix<int, 2, 1>& num_n,
    const Eigen::Matrix<int, 2, 1>& num_s,
    const std::map<int, std::map<int, Eigen::Matrix<int, 2, 1>>>& index,
    const Matrix& X, const Options& options) {
  reset();

  opts_ = options;

  auto& opts = opts_;
  auto& num_m = num_m_;
  auto& adjacency_matrix = adjancecy_matrix_;

  num_m = 0;

  auto& measurements = measurements_;
  measurements.clear();

  for (const auto& measurement : inter_measurements) {
    if ((measurement.i.node != alpha || measurement.j.node != beta) &&
        (measurement.i.node != beta || measurement.j.node != alpha)) {
      continue;
    }

    measurements.push_back(measurement);

    num_m++;
  }

  int d = inter_measurements.size() ? inter_measurements[0].t.size() : 0;

  adjacency_matrix.setZero(num_m, num_m);

  if (index.count(alpha) == 0 || index.count(beta) == 0) {
    LOG(ERROR) << "There are no indices for node " << alpha << " or " << beta
               << "." << std::endl;

    exit(-1);
  }

  const auto& index_a = index.at(alpha);
  const auto& index_b = index.at(beta);

  Vector tij, tjj, tji, tii;
  Matrix Rij, Rjj, Rji, Rii;
  Scalar kappa, tau;

  Vector ti[2], tj[2], tai[2], taj[2];
  Matrix Ri[2], Rj[2], Rai[2], Raj[2];

  int s[2], n[2], i[2];

  for (int p = 0; p < num_m; p++) {
    adjacency_matrix(p, p) = 1;

    const auto& measurement_p = measurements[p];

    s[0] = 0;
    s[1] = (d + 1) * num_s[1];

    n[0] = num_n[0];
    n[1] = num_n[1];

    if (measurement_p.i.node == alpha && measurement_p.j.node == beta) {
      auto search_i = index_a.find(measurement_p.i.pose);

      if (search_i == index_a.end()) {
        LOG(ERROR) << "There is no index for pose [" << alpha << ", "
                   << measurement_p.i.pose << "]." << std::endl;

        exit(-1);
      }

      auto search_j = index_b.find(measurement_p.j.pose);

      if (search_j == index_b.end()) {
        LOG(ERROR) << "There is no index for pose [" << beta << ", "
                   << measurement_p.j.pose << "]." << std::endl;

        exit(-1);
      }

      i[0] = search_i->second[1];
      i[1] = search_j->second[1];

      tij = measurement_p.t;
      Rij = measurement_p.R;
    } else if (measurement_p.i.node == beta && measurement_p.j.node == alpha) {
      auto search_i = index_a.find(measurement_p.j.pose);

      if (search_i == index_a.end()) {
        LOG(ERROR) << "There is no index for pose [" << alpha << ", "
                   << measurement_p.j.pose << "]." << std::endl;

        exit(-1);
      }

      auto search_j = index_b.find(measurement_p.i.pose);

      if (search_i == index_a.end()) {
        LOG(ERROR) << "There is no index for pose [" << beta << ", "
                   << measurement_p.i.pose << "]." << std::endl;

        exit(-1);
      }

      i[0] = search_i->second[1];
      i[1] = search_j->second[1];

      tij.noalias() = -measurement_p.R.transpose() * measurement_p.t;
      Rij.noalias() = measurement_p.R.transpose();
    } else {
      assert(0);
      LOG(ERROR) << "Inconsitent measurements for nodes " << alpha << " and "
                 << beta << std::endl;

      exit(-1);
    }

    ti[0] = X.row(s[0] + i[0]);
    tj[0] = X.row(s[1] + i[1]);

    Ri[0] = X.middleRows(s[0] + n[0] + d * i[0], d).transpose();
    Rj[0] = X.middleRows(s[1] + n[1] + d * i[1], d).transpose();

    for (int q = p + 1; q < num_m; q++) {
      const auto& measurement_q = measurements[q];

      if (measurement_q.i.node == alpha && measurement_q.j.node == beta) {
        auto search_i = index_a.find(measurement_q.i.pose);

        if (search_i == index_a.end()) {
          LOG(ERROR) << "There is no index for pose [" << alpha << ", "
                     << measurement_q.i.pose << "]." << std::endl;

          exit(-1);
        }

        auto search_j = index_b.find(measurement_q.j.pose);

        if (search_j == index_b.end()) {
          LOG(ERROR) << "There is no index for pose [" << beta << ", "
                     << measurement_q.j.pose << "]." << std::endl;

          exit(-1);
        }

        i[0] = search_i->second[1];
        i[1] = search_j->second[1];

        tji.noalias() = -measurement_q.R.transpose() * measurement_q.t;
        Rji.noalias() = measurement_q.R.transpose();
      } else if (measurement_q.i.node == beta &&
                 measurement_q.j.node == alpha) {
        auto search_i = index_a.find(measurement_q.j.pose);

        if (search_i == index_a.end()) {
          LOG(ERROR) << "There is no index for pose [" << alpha << ", "
                     << measurement_q.j.pose << "]." << std::endl;

          exit(-1);
        }

        auto search_j = index_b.find(measurement_q.i.pose);

        if (search_i == index_a.end()) {
          LOG(ERROR) << "There is no index for pose [" << beta << ", "
                     << measurement_q.i.pose << "]." << std::endl;

          exit(-1);
        }

        i[0] = search_i->second[1];
        i[1] = search_j->second[1];

        tji = measurement_q.t;
        Rji = measurement_q.R;
      } else {
        assert(0);
        LOG(ERROR) << "Inconsitent measurements for nodes " << alpha << " and "
                   << beta << std::endl;

        exit(-1);
      }

      ti[1] = X.row(s[0] + i[0]);
      tj[1] = X.row(s[1] + i[1]);

      Ri[1] = X.middleRows(s[0] + n[0] + d * i[0], d).transpose();
      Rj[1] = X.middleRows(s[1] + n[1] + d * i[1], d).transpose();

      if (options.weighted) {
        kappa = 0.5 * (measurement_p.kappa + measurement_q.kappa);
        tau = 0.5 * (measurement_p.tau + measurement_q.tau);
      } else {
        kappa = 1;
        tau = 1;
      }

      Rii.noalias() = Ri[1].transpose() * Ri[0];
      tii.noalias() = Ri[1].transpose() * (ti[0] - ti[1]);

      Rjj.noalias() = Rj[0].transpose() * Rj[1];
      tjj.noalias() = Rj[0].transpose() * (tj[1] - tj[0]);

      Raj[0] = Rij;
      taj[0] = tij;

      Raj[1].noalias() = Raj[0] * Rjj;
      taj[1] = taj[0];
      taj[1].noalias() += Raj[0] * tjj;

      Rai[1].noalias() = Raj[1] * Rji;
      tai[1] = taj[1];
      tai[1].noalias() += Raj[1] * tji;

      Rai[0].noalias() = Rai[1] * Rii;
      tai[0] = tai[1];
      tai[0].noalias() += Rai[1] * tii;

      Scalar error =
          std::sqrt(kappa * (Rai[0] - Matrix::Identity(d, d)).squaredNorm() +
                    tau * tai[0].squaredNorm());

      adjacency_matrix(p, q) = adjacency_matrix(q, p) = error <= opts.tolerance;
    }
  }

  return 0;
}

const std::vector<bool>& PCM::solveExact() const {
  auto& results = results_;

  results = exact_solver_->find_max_clique(adjancecy_matrix_);

  return results;
}

const std::vector<bool>& PCM::solveHeuristic() const {
  auto& results = results_;

  results = heruistic_solver_->find_max_clique(adjancecy_matrix_);

  return results;
}
}  // namespace DPGO
