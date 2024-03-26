#include <DChordal/DChordalReduced.h>
#include <Optimization/Util/Stopwatch.h>
#include <memory>

namespace DChordal {
DChordalReduced::DChordalReduced(
    std::shared_ptr<DChordalReducedProblem const> problem,
    Options const& options)
    : problem_(problem), options_(options) {}

int DChordalReduced::receive(const std::map<int, Matrix>& msg) const {
  auto& results = results_;
  auto& problem = problem_;

  const auto& n_recv = problem->n_recv();
  const int& d = problem->d();
  const int& p = problem->p();

  for (auto const& beta : msg) {
    int b = beta.first;

    auto search = n_recv.find(b);

    if (search != n_recv.end()) {
      results.updated = false;

      assert(d == beta.second.rows());
      assert(d == beta.second.cols());

      int i = search->second;

      results.Xk.middleRows(i * p, p) = beta.second;
    }
  }

  return 0;
}

int DChordalReduced::initialize(const Matrix& X) const {
  const auto& problem = problem_;
  const int& d = problem->d();
  const int& p = problem->p();
  const int& n = problem->nn() + 1;

  assert(problem->check() == 0);
  assert(X.rows() == p * n && X.cols() == d);

  if (problem->check()) {
    LOG(ERROR) << "The problem is not set up." << std::endl;

    return -1;
  }

  if (X.rows() != p * n || X.cols() != d) {
    LOG(ERROR) << "Inconsistent poses for node " << problem->node()
               << std::endl;

    return -1;
  }

  results_.clear();
  results_.Xk = X;
  results_.Xak = X.topRows(p);
  results_.updated = false;

  results_.X.resize(1);
  results_.g.resize(1);
  results_.s.resize(1);

  results_.s[0] = 1.0;

  return 0;
}

int DChordalReduced::n_receive(const std::map<int, Matrix>& msg) const {
  const auto& problem = problem_;
  const auto& n_recv = problem->n_recv();

  assert(n_recv.size() == msg.size());

  if (n_recv.size() != msg.size()) {
    LOG(ERROR) << "Inconsisent messages for node " << problem->node()
               << std::endl;

    return -1;
  }

  int p = problem->p();

  auto& results = results_;

  auto it_n = n_recv.begin();
  auto it_x = msg.begin();

  while (it_n != n_recv.end() && it_x != msg.end()) {
    assert(it_n->first == it_x->first);

    if (it_n->first != it_x->first) {
      LOG(ERROR) << "Inconsisent messages for node " << problem->node()
                 << std::endl;

      return -1;
    }

    results.Xk.middleRows(p * it_n->second, p) = it_x->second;

    it_n++;
    it_x++;
  }

  return 0;
}

int DChordalReduced::iterate() const {
  const auto& options = options_;

  auto& results = results_;
  int& iter = results.iters;

  if (iter > options.max_iterations) {
    return 1;
  }

  const auto& problem = problem_;
  const int& n0 = 1;
  const int& n1 = problem->nn();
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

#if 1
  Matrix Y, g;

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
#else
  Matrix Df;

  problem->evaluate_Df(results.X[iter], Df);
  problem->jacobi(results.Xak, Df, results.Xak);
#endif

  results.Xk.topRows(p) = results.Xak;

  iter++;

  return 0;
}

int DChordalReduced::update() const {
  auto& results = results_;

  if (results.updated) return 0;

  const auto& problem = problem_;
  const auto& options = options_;

  const int& iter = results.iters;

  const int& n0 = 1;
  const int& n1 = problem->nn();
  const int& d = problem->d();

  results.X.resize(iter + 1);
  results.g.resize(iter + 1);

  results.X[iter] = results.Xk;

  problem->evaluate_g(results.X[iter], results.g[iter]);

  return 0;
}

DChordalReduced_R::DChordalReduced_R(
    std::shared_ptr<const DChordalReducedProblem_R> problem,
    const Options& options)
    : DChordalReduced(problem, options), problem_R_(problem) {}

DChordalReduced_R::DChordalReduced_R(int node,
                                     const measurements_t& measurements,
                                     const Options& options)
    : DChordalReduced_R(std::make_shared<const DChordalReducedProblem_R>(
                            node, measurements, options.reg_G, options.loss,
                            options.loss_reg),
                        options) {}

int DChordalReduced_R::setup(const Matrix& X) {
  results_.clear();

  return problem_R_->setup(X);
}

DChordalReduced_t::DChordalReduced_t(
    std::shared_ptr<const DChordalReducedProblem_t> problem,
    const Options& options)
    : DChordalReduced(problem, options), problem_t_(problem) {}

DChordalReduced_t::DChordalReduced_t(int node,
                                     const measurements_t& measurements,
                                     const Options& options)
    : DChordalReduced_t(std::make_shared<const DChordalReducedProblem_t>(
                            node, measurements, options.reg_G, options.loss,
                            options.loss_reg),
                        options) {}

int DChordalReduced_t::setup(const Matrix& X, const Matrix& R) {
  results_.clear();

  return problem_t_->setup(X, R);
}
}  // namespace DChordal
