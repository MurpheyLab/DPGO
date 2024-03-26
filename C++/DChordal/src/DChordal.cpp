#include <DChordal/DChordal.h>

namespace DChordal {
DChordal::DChordal(std::shared_ptr<const DChordalProblem> problem,
                   const Options& options)
    : problem_(problem), options_(options) {}

int DChordal::initialize(const Matrix& X) const {
  const auto& problem = problem_;
  const int& d = problem->d();
  const int& p = problem->p();
  const auto& n = problem->n();

  assert(problem->check() == 0);
  assert(X.rows() == p * (n[0] + n[1]) && X.cols() == d);

  if (problem->check()) {
    LOG(ERROR) << "The problem is not set up." << std::endl;

    return -1;
  }

  if (X.rows() != p * (n[0] + n[1]) || X.cols() != d) {
    LOG(ERROR) << "Inconsistent poses for node " << problem->node()
               << std::endl;

    return -1;
  }

  results_.clear();
  results_.Xk = X;
  results_.Xak = X.topRows(p * n[0]);
  results_.updated = false;

  results_.X.resize(1);
  results_.g.resize(1);
  results_.s.resize(1);

  results_.s[0] = 1.0;

  return 0;
}

int DChordal::receive(const std::map<int, Matrix>& msg) const {
  auto& results = results_;
  auto& problem = problem_;

  const auto& recv = problem->recv();
  const auto& n = problem->n();
  const int& p = problem->p();
  const int& d = problem->d();

  for (const auto& beta : msg) {
    int b = beta.first;

    auto search = recv.find(b);

    assert(search != recv.end());

    if (search != recv.end()) {
      results.updated = false;

      int num_poses = search->second.size();

      assert(p * num_poses == beta.second.rows());
      assert(d == beta.second.cols());

      int i = search->second.begin()->second[1];

      results.Xk.middleRows(p * (n[0] + i), p * num_poses) = beta.second;
    } else {
      LOG(ERROR) << "Can not find information for node " << b << std::endl;
    }
  }

  return 0;
}

int DChordal::iterate() const {
  const auto& options = options_;

  auto& results = results_;
  int& iter = results.iters;

  if (iter > options.max_iterations) {
    return 1;
  }

  const auto& problem = problem_;
  const int& n0 = problem->n()[0];
  const int& n1 = problem->n()[1];
  const int& d = problem->d();
  const int& p = problem->p();

  if (options_.verbose) {
    std::cout
        << "==================================================================="
        << std::endl;
    std::cout << "Iteration " << iter << std::endl;
    std::cout
        << "==================================================================="
        << std::endl;
  }

  auto start_time = Stopwatch::tick();

  Matrix Y(d * (n0 + n1), d), g(n0, d);
  Matrix Ysol(d * n0, d);

  Scalar s0 = results.s[iter];
  Scalar s1 = 0.5 + 0.5 * std::sqrt(4.0 * s0 * s0 + 1.0);
  Scalar gamma = (s0 - 1) / s1;
  results.s.push_back(s1);

  if (iter == 0) {
    Y = results.X[0];
  } else {
    Y = (1.0 + gamma) * results.X[iter] - gamma * results.X[iter - 1];
  }

  problem->iterate(Y, g, results.Xak);

  results.Xk.topRows(p * n0) = results.Xak;

  iter++;

  return 0;
}

int DChordal::update() const {
  auto& results = results_;

  if (results.updated) return 0;

  const auto& problem = problem_;
  const auto& options = options_;

  const int& iter = results.iters;

  const int& n0 = problem->n()[0];
  const int& n1 = problem->n()[1];
  const int& d = problem->d();

  results.X.resize(iter + 1);
  results.g.resize(iter + 1);

  results.X[iter] = results.Xk;

  problem->evaluate_g(results.X[iter], results.g[iter]);

  return 0;
}

DChordal_R::DChordal_R(std::shared_ptr<const DChordalProblem_R> problem,
                       const Options& options)
    : DChordal(problem, options), problem_R_(problem) {}

DChordal_R::DChordal_R(int node, const measurements_t& measurements,
                       const Options& options)
    : DChordal_R(
          std::make_shared<const DChordalProblem_R>(
              node, measurements, options.reg_G, options.loss, options.loss_reg),
          options) {}

int DChordal_R::setup() {
  results_.clear();

  return problem_R_->setup();
}

DChordal_t::DChordal_t(std::shared_ptr<const DChordalProblem_t> problem,
                       const Options& options)
    : DChordal(problem, options), problem_t_(problem) {}

DChordal_t::DChordal_t(int node, const measurements_t& measurements,
                       const Options& options)
    : DChordal_t(
          std::make_shared<const DChordalProblem_t>(
              node, measurements, options.reg_G, options.loss, options.loss_reg),
          options) {}

int DChordal_t::setup(const Matrix& R) {
  results_.clear();

  return problem_t_->setup(R);
}
}  // namespace DChordal
