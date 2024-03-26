#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <DChordal/DChordalProblem.h>
#include <DChordal/DChordal_types.h>
#include <DChordal/RelativePoseMeasurement.h>

namespace DChordal {
class DChordal {
 public:
  DChordal() {}
  DChordal(std::shared_ptr<const DChordalProblem> problem,
           const Options& options = Options());

  int initialize(const Matrix& X) const;
  int update() const;

  int receive(const std::map<int, Matrix>& msg) const;

  int iterate() const;

  template <typename Chordal>
  int communicate(const std::vector<std::shared_ptr<Chordal>>& chordals) const {
    const auto& problem = problem_;

    const auto& n = problem->n();
    const auto& s = problem->s();
    const auto& p = problem->p();
    const auto& d = problem->d();

    const auto& alpha = problem->node();

    for (const auto& info : problem->index()) {
      const auto& beta = info.first;

      assert(beta < chordals.size());

      if (beta >= chordals.size()) {
          LOG(ERROR) << "No information for node[" << beta << "]."
                     << std::endl;

          return -1;
      }

      for (const auto& index : info.second) {
        const auto& j = index.first;

        assert(j < chordals[beta]->problem()->n()[0]);
        assert(d == chordals[beta]->problem()->d());

        if (j >= chordals[beta]->problem()->n()[0] ||
            d != chordals[beta]->problem()->d()) {
          LOG(ERROR) << "Inconsistent size of node[" << beta << "]."
                     << std::endl;
        }
      }
    }

    auto& results = results_;

    for (const auto& info : problem->index()) {
      const auto& beta = info.first;

      if (beta == alpha) continue;

      for (const auto& index : info.second) {
        const auto& i = index.second;
        const auto& j = index.first;

        assert(j < chordals[beta]->problem()->n()[0]);

        results.Xk.middleRows(s[1] * p + i[1] * p, p) =
            chordals[beta]->results().Xk.middleRows(j * p, p);
      }
    }

    results.updated = false;

    return 0;
  }

  std::shared_ptr<const DChordalProblem> problem() const { return problem_; }

  Result const& results() const { return results_; }

  Options const& options() const { return options_; }

  int set_options(Options const& options) {
    options_ = options;
    return 0;
  }

 protected:
  std::shared_ptr<const DChordalProblem> problem_;
  mutable Result results_;
  Options options_;
};

class DChordal_R : public DChordal {
 public:
  DChordal_R() {}
  DChordal_R(std::shared_ptr<const DChordalProblem_R> problem,
             const Options& options = Options());
  DChordal_R(int node, const measurements_t& measurements,
             const Options& options = Options());

  int setup();

 protected:
  std::shared_ptr<const DChordalProblem_R> problem_R_;
};

class DChordal_t : public DChordal {
 public:
  DChordal_t() {}
  DChordal_t(std::shared_ptr<const DChordalProblem_t> problem,
             const Options& options = Options());
  DChordal_t(int node, const measurements_t& measurements,
             const Options& options = Options());

  int setup(const Matrix& R);

 protected:
  std::shared_ptr<const DChordalProblem_t> problem_t_;
};
}  // namespace DChordal
