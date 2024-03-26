#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <DPGO/DPGOProblem.h>
#include <DPGO/DPGO_types.h>
#include <DPGO/RelativePoseMeasurement.h>

namespace DPGO {
class DPGOHash {
public:
  DPGOHash() {}
  DPGOHash(std::shared_ptr<const DPGOProblem> problem,
            const Options &options = Options());
  DPGOHash(int node, const measurements_t &measurements,
            const Options &options = Options());
  int initialize(const Matrix &x) const;

  int receive(const std::map<int, Matrix> &msg) const;

  int update() const;

  int iterate() const;

  template <typename PGO>
  int communicate(const std::vector<std::shared_ptr<PGO>> &pgos) const {
    const auto &problem = problem_;

    const auto &n = problem->n();
    const auto &s = problem->s();
    const auto &d = problem->d();

    const auto &alpha = problem->node();

    for (const auto &info : problem->index()) {
      const auto &beta = info.first;

      assert(beta < pgos.size());

      if (beta >= pgos.size()) {
        LOG(ERROR) << "No information for node[" << beta << "]." << std::endl;

        return -1;
      }

      for (const auto &index : info.second) {
        const auto &j = index.first;

        assert(j < pgos[beta]->problem()->n()[0]);
        assert(d == pgos[beta]->problem()->d());

        if (j >= pgos[beta]->problem()->n()[0] ||
            d != pgos[beta]->problem()->d()) {
          LOG(ERROR) << "Inconsistent size of node[" << beta << "]."
                     << std::endl;
        }
      }

      auto &results = results_;

      for (const auto &info : problem->index()) {
        const auto &beta = info.first;

        if (beta == alpha)
          continue;

        for (const auto &index : info.second) {
          const auto &i = index.second;
          const auto &j = index.first;

          assert(j < pgos[beta]->problem()->n()[0]);

          results.Xk.row(s[1] * (d + 1) + i[1]) =
              pgos[beta]->results().Xk.row(j);
          results.Xk.middleRows(s[1] * (d + 1) + n[1] + i[1] * d, d) =
              pgos[beta]->results().Xk.middleRows(
                  pgos[beta]->problem()->n()[0] + j * d, d);
        }
      }
    }

    return 0;
  }

  std::shared_ptr<const DPGOProblem> problem() const { return problem_; }

  DPGOResult const &results() const { return results_; }

  Options const &options() const { return options_; }

  int set_options(Options const &options) {
    options_ = options;
    return 0;
  }

private:
  int mm_pgo() const;
  int amm_pgo() const;

protected:
  std::shared_ptr<const DPGOProblem> problem_;
  mutable DPGOResult results_;
  Options options_;
};
} // namespace DPGO
