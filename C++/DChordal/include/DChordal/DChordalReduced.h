#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <DChordal/DChordalReducedProblem.h>
#include <DChordal/DChordal_types.h>
#include <DChordal/RelativePoseMeasurement.h>

namespace DChordal {
class DChordalReduced {
 public:
  DChordalReduced() {}
  DChordalReduced(std::shared_ptr<const DChordalReducedProblem> problem,
                  const Options& options = Options());
  int receive(const std::map<int, Matrix>& msg) const;

  int initialize(const Matrix& X) const;
  int update() const;

  int n_receive(const std::map<int, Matrix>& msg) const;

  template <typename Chordal>
  int n_communicate(
      const std::vector<std::shared_ptr<Chordal>>& chordals) const {
    const auto& problem = problem_;

    const auto& nn = problem->nn();
    const auto& p = problem->p();
    const auto& d = problem->d();

    const auto& alpha = problem->node();

    for (const auto& info : problem->n_index()) {
      const auto& beta = info.first;

      assert(beta < chordals.size());

      if (beta >= chordals.size()) {
        LOG(ERROR) << "No information for node[" << beta << "]." << std::endl;

        return -1;
      }
    }

    auto& results = results_;

    for (const auto& info : problem->n_index()) {
      const auto& beta = info.first;
      const auto& i = info.second;

      if (beta == alpha) continue;

      results.Xk.middleRows(i * p, p) = chordals[beta]->results().Xk.topRows(p);
    }

    results.updated = false;

    return 0;
  }

  virtual int iterate() const;

  std::shared_ptr<const DChordalReducedProblem> problem() const {
    return problem_;
  }

  Result const& results() const { return results_; }

  Options const& options() const { return options_; }

  int set_options(Options const& options) {
    options_ = options;
    return 0;
  }

 protected:
  std::shared_ptr<const DChordalReducedProblem> problem_;
  mutable Result results_;
  Options options_;
};

class DChordalReduced_R : public DChordalReduced {
 public:
  DChordalReduced_R() {}
  DChordalReduced_R(std::shared_ptr<const DChordalReducedProblem_R> problem,
                    const Options& options = Options());
  DChordalReduced_R(int node, const measurements_t& measurements,
                    const Options& options = Options());

  int setup(const Matrix& X);

 protected:
  std::shared_ptr<const DChordalReducedProblem_R> problem_R_;
};

class DChordalReduced_t : public DChordalReduced {
 public:
  DChordalReduced_t() {}
  DChordalReduced_t(std::shared_ptr<const DChordalReducedProblem_t> problem,
                    const Options& options = Options());
  DChordalReduced_t(int node, const measurements_t& measurements,
                    const Options& options = Options());

  int setup(const Matrix& X, const Matrix& R);

 protected:
  std::shared_ptr<const DChordalReducedProblem_t> problem_t_;
};
}  // namespace DChordal
