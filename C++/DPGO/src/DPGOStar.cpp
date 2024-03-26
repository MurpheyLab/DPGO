#include <DPGO/DPGOStar.h>
#include <Optimization/Riemannian/TNT.h>
#include <Optimization/Util/Stopwatch.h>
#include <memory>

namespace DPGO {
DPGOStar::DPGOStar(int num_nodes, const std::string &filename,
                   const Options &options)
    : num_nodes_(num_nodes), options_(options), problems_(num_nodes) {
  std::vector<DPGO::measurements_t> measurements;

  DPGO::read_g2o(filename, num_nodes_, num_poses_, measurements,
                 intra_measurements_, inter_measurements_, g_index_);

  d_ = intra_measurements_.size() ? intra_measurements_[0].t.size()
                                  : inter_measurements_[0].t.size();

  num_measurements_[0] = intra_measurements_.size();
  num_measurements_[1] = inter_measurements_.size();

  DPGO::construct_data_matrix(intra_measurements_, inter_measurements_,
                              g_index_, M_, B0_, B1_);

  auto &problems = problems_;

  for (int node = 0; node < num_nodes; node++) {
    problems[node] = std::make_shared<DPGOProblem>(
        node, measurements[node], options.regularizer, options.loss,
        options.preconditioner,
        options.reg_Cholesky_precon_max_condition_number, options.rescale,
        options.loss_reg);
  }

  Xk_.setZero((d_ + 1) * num_poses_, d_);
  Xkh_.setZero((d_ + 1) * num_poses_, d_);
  Xkp_.setZero((d_ + 1) * num_poses_, d_);

  SP_.set_d(d_);
  SP_.set_n(num_poses_);
}

DPGOStar::DPGOStar(const std::vector<measurements_t> &measurements,
                   const Options &options)
    : num_nodes_(measurements.size()), options_(options),
      problems_(measurements.size()) {
  intra_measurements_.clear();
  inter_measurements_.clear();
  g_index_.clear();
  g_index_.resize(num_nodes_);

  for (int alpha = 0; alpha < num_nodes_; alpha++) {
    for (const auto &measurement : measurements[alpha]) {
      assert(measurement.i.node == alpha || measurement.j.node == alpha);

      if (measurement.i.node != alpha && measurement.j.node != alpha) {
        LOG(ERROR) << "Inconsistent measurement for node " << alpha << "."
                   << std::endl;

        exit(-1);
      }

      if (measurement.i.node == measurement.j.node) {
        intra_measurements_.push_back(measurement);
      } else if (measurement.i.node == alpha) {
        inter_measurements_.push_back(measurement);
      }

      g_index_[measurement.i.node][measurement.i.pose] = -1;
      g_index_[measurement.j.node][measurement.j.pose] = -1;
    }
  }

  d_ = intra_measurements_.size() ? intra_measurements_[0].t.size()
                                  : inter_measurements_[0].t.size();

  num_measurements_[0] = intra_measurements_.size();
  num_measurements_[1] = inter_measurements_.size();

  num_poses_ = 0;

  for (auto &index_a : g_index_) {
    for (auto &index : index_a) {
      index.second = num_poses_++;
    }
  }

  DPGO::construct_data_matrix(intra_measurements_, inter_measurements_,
                              g_index_, M_, B0_, B1_);

  auto &problems = problems_;

  for (int node = 0; node < num_nodes_; node++) {
    problems[node] = std::make_shared<DPGOProblem>(
        node, measurements[node], options.regularizer, options.loss,
        options.preconditioner,
        options.reg_Cholesky_precon_max_condition_number, options.rescale,
        options.loss_reg);
  }

  Xk_.setZero((d_ + 1) * num_poses_, d_);
  Xkh_.setZero((d_ + 1) * num_poses_, d_);
  Xkp_.setZero((d_ + 1) * num_poses_, d_);

  SP_.set_d(d_);
  SP_.set_n(num_poses_);
}

int DPGOStar::initialize(const Matrix &X) const {
  auto &Xk = Xk_;
  auto &fobj = fobj_;
  auto &F = F_;

  results_.resize(num_nodes_);

  for (int node = 0; node < num_nodes_; node++) {
    initialize_n(node, X);
  }

  Xk = X;
  evaluate_f(Xk, fobj);
  F = fobj;

  return 0;
}

int DPGOStar::iterate() const {
  const auto &num_nodes = num_nodes_;
  const auto &num_poses = num_poses_;
  const auto &options = options_;

  const auto &M = M_;
  const auto &B0 = B0_;
  const auto &B1 = B1_;

  auto &Xkh = Xkh_;
  auto &Xk = Xk_;
  auto &Xkp = Xkp_;
  auto &fobj = fobj_;
  auto &F = F_;

  Scalar fobjh;

  for (int node = 0; node < num_nodes; node++) {
    amm_pgo_n(node);
  }

  evaluate_f(Xkh, fobjh);

  if (fobjh > F - options.psi * (Xkh - Xk).squaredNorm()) {
    for (int node = 0; node < num_nodes; node++) {
      pm_pgo_n(node);
    }

    evaluate_f(Xkh, fobjh);
  }

  evaluate_f(Xkp, fobj);

  if (fobj > F - options.psi * (Xkp - Xk).squaredNorm()) {
    for (int node = 0; node < num_nodes; node++) {
      auto &results = results_[node];

      mm_pgo_n(node);
      results.s[results.iters + 1] =
          std::max(0.5 * results.s[results.iters + 1], 1.0);
    }

    evaluate_f(Xkp, fobj);
  }

  if (F - fobj < options.phi * (F - fobjh)) {
    for (int node = 0; node < num_nodes; node++) {
      const auto &problem = problems_[node];
      const auto &n0 = problem->n()[0];
      const auto &d = d_;

      auto &results = results_[node];

      auto t = results.Xak.topRows(n0);
      auto R = results.Xak.bottomRows(d * n0);
      R = results.Xakh.bottomRows(d * n0);
      problem->recover_translations(R, results.g[results.iters], t);

      const int &index = g_index_.at(node).begin()->second;

      Xkp.middleRows(index, n0) = results.Xak.topRows(n0);
      Xkp.middleRows(num_poses + d * index, d * n0) =
          results.Xak.bottomRows(d * n0);
    }

    evaluate_f(Xkp, fobj);
  }

  for (int node = 0; node < num_nodes; node++) {
    const auto &problem = problems_[node];
    const auto &n0 = problem->n()[0];
    const auto &d = d_;

    auto &results = results_[node];

    results.iters++;

    results.Xk.topRows((d + 1) * n0) = results.Xak;

    results.updated = false;
  }

  Xk.swap(Xkp);

  F = F * (1 - options.eta[0]) + fobj * options.eta[0];

  return 0;
}

int DPGOStar::communicate() const {
  const auto &num_nodes = num_nodes_;

  for (int node = 0; node < num_nodes; node++) {
    communicate_n(node);
  }

  return 0;
}

int DPGOStar::update() const {
  const auto &num_nodes = num_nodes_;

  for (int node = 0; node < num_nodes; node++) {
    update_n(node);
  }

  return 0;
}

int DPGOStar::initialize_n(int node, const Matrix &X) const {
  const auto &problem = problems_[node];
  const auto &g_index = g_index_;
  const auto &num_poses = num_poses_;

  const auto &n = problem->n();
  const auto &s = problem->s();
  const auto &d = problem->d();

  auto &results = results_[node];
  results.clear();

  results.Xk.resize((d + 1) * (n[0] + n[1]), d);

  for (auto &index_nn : problem->index()) {
    int beta = index_nn.first;

    const auto &g_index_nn = g_index.at(beta);

    for (auto &j : index_nn.second) {
      int nn = g_index_nn.at(j.first);

      const auto &index = j.second;

      results.Xk.row(s[index[0]] * (d + 1) + index[1]) = X.row(nn);
      results.Xk.middleRows(s[index[0]] * (d + 1) + n[index[0]] + d * index[1],
                            d) = X.middleRows(num_poses + d * nn, d);
    }
  }

  results.Xak = results.Xk.topRows((d + 1) * n[0]);

  results.Xakh.resize((d + 1) * n[0], d);

  results.gamma = 0;

  results.updated = false;

  return 0;
}

int DPGOStar::communicate_n(int node) const {
  const auto &problem = problems_[node];
  const auto &g_index = g_index_;

  const auto &n = problem->n();
  const auto &s = problem->s();
  const auto &num_poses = num_poses_;
  const auto &d = d_;

  const auto &Xk = Xk_;

  auto &results = results_[node];

  for (const auto &index_nn : problem->index()) {
    int beta = index_nn.first;

    if (beta == node)
      continue;

    const auto &g_index_nn = g_index.at(beta);

    for (auto &j : index_nn.second) {
      int nn = g_index_nn.at(j.first);

      const auto &index = j.second;

      assert(index[0] == 1);

      results.Xk.row(s[index[0]] * (d + 1) + index[1]) = Xk.row(nn);
      results.Xk.middleRows(s[index[0]] * (d + 1) + n[index[0]] + d * index[1],
                            d) = Xk.middleRows(num_poses + d * nn, d);
    }
  }

  results.updated = false;

  return 0;
}

int DPGOStar::update_n(int node) const {
  auto &results = results_[node];

  if (results.updated)
    return 0;

  const auto &problem = problems_[node];
  const auto &options = options_;

  const int &iter = results.iters;

  const int &n0 = problem->n()[0];
  const int &n1 = problem->n()[1];
  const int &d = problem->d();

  results.X.resize(iter + 1);
  results.fobj.resize(iter + 1);
  results.Dfobj.resize(iter + 1);
  results.g.resize(iter + 1);
  results.f.resize(iter + 1);
  results.G.resize(iter + 1);

  results.X[iter] = results.Xk;

  if (options.loss == Loss::None && SIMPLE) {
    problem->evaluate_none_g_and_f0(results.X[iter], results.g[iter],
                                    results.f[iter]);
    problem->evaluate_G(results.Xak, results.g[iter], results.f[iter],
                        results.fobj[iter]);
    results.Gk = results.fobj[iter];
  } else if (options.rescale == Rescale::Static) {
    problem->evaluate_g_and_f0(
        results.X[iter], results.g[iter], results.f[iter], results.Dfobj[iter],
        results.fobj[iter], results.DfobjE, results.fobjE);
    results.Gk = results.fobj[iter];
  } else {
    problem->evaluate_g_and_f0_rescale(
        results.X[iter], results.g[iter], results.f[iter], results.Dfobj[iter],
        results.fobj[iter], results.DfobjE, results.fobjE,
        results.rescale_count, options.max_rescale_count);
  }

  if (options.loss == Loss::None && SIMPLE) {
    problem->full_Riemannian_gradient_G(results.Xak, results.g[iter],
                                        results.Dfobj[iter], results.gradF);
  } else {
    problem->full_tangent_space_projection(results.Xak, results.Dfobj[iter],
                                           results.gradF);
  }

  results.gradFnorm = results.gradF.norm();

  results.G[iter] = results.Gk;

  if (options.scheme == Scheme::AMM) {
    results.s.resize(iter + 2);

    if (iter == 0) {
      results.s[0] = 1;
    }

    Scalar &s0 = results.s[iter];
    Scalar &s1 = results.s[iter + 1];

    s1 = 0.5 + 0.5 * std::sqrt(4.0 * s0 * s0 + 1.0);

    results.gamma = (s0 - 1) / s1;
  }

  results.Fk[0] = results.fobj[iter];
  results.Fk[1] = results.fobj[iter];

  results.updated = true;

  return 0;
}

int DPGOStar::amm_pgo_n(int node) const {
  assert(results_[node].updated == true &&
         "The optimizer has not been updated.");

  const auto &problem = problems_[node];
  const auto &options = options_;

  const int &n0 = problem->n()[0];
  const int &n1 = problem->n()[1];
  const int &d = problem->d();
  const int &num_poses = num_poses_;

  auto &results = results_[node];
  auto &Xkh = Xkh_;
  auto &Xkp = Xkp_;

  int &iter = results.iters;

  Matrix Y((d + 1) * n0, d);
  Matrix g((d + 1) * n0, d);
  Matrix Df((d + 1) * n0, d);

  if (iter == 0) {
    Y = results.Xk;
    g = results.g[iter];
    Df = results.Dfobj[iter];
  } else {
    Y = results.X[iter] +
        results.gamma * (results.X[iter] - results.X[iter - 1]);

    if (options.loss == Loss::None && SIMPLE) {
      g = results.g[iter] +
          results.gamma * (results.g[iter] - results.g[iter - 1]);
      Df = results.Dfobj[iter] +
           results.gamma * (results.Dfobj[iter] - results.Dfobj[iter - 1]);
    } else {
      problem->evaluate_g_and_Df(Y, g, Df);
    }
  }

  const Scalar &f = results.f[iter];

  /// Riemannian Optimization
  Optimization::Objective<Matrix, Scalar, Matrix> Fobj =
      [&problem, &g, &f](const Matrix &Y, const Matrix &NablaF_Y) {
        Scalar fobj;
        problem->evaluate_G(Y, g, f, fobj);

        return fobj;
      };

  // Local quadratic model constructor
  Optimization::Riemannian::QuadraticModel<Matrix, Matrix, Matrix> QM =
      [&problem,
       &g](const Matrix &Y, Matrix &grad,
           Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix>
               &HessOp,
           Matrix &NablaF_Y) {
        // Compute and cache Euclidean gradient at the current iterate
        problem->reduced_Euclidean_gradient_G(Y, g, NablaF_Y);

        // Compute Riemannian gradient from Euclidean gradient
        problem->reduced_Riemannian_gradient_G(Y, NablaF_Y, grad);

        // Define linear operator for computing Riemannian Hessian-vector
        // products (cf. eq. (44) in the SE-Sync tech report)
        HessOp = [problem, &g](const Matrix &Y, const Matrix &Ydot,
                               const Matrix &NablaF_Y) {
          Matrix Hess;
          problem->reduced_Riemannian_Hessian_vector_product(Y, NablaF_Y, Ydot,
                                                             Hess);
          return Hess;
        };
      };

  // Riemannian metric
  // We consider a realization of the product of SO(d) as an
  // embedded submanifold of R^{dn x d}; consequently, the induced Riemannian
  // metric is simply the usual Euclidean inner product
  Optimization::Riemannian::RiemannianMetric<Matrix, Matrix, Scalar, Matrix>
      metric =
          [](const Matrix &Y, const Matrix &V1, const Matrix &V2,
             const Matrix &NablaF_Y) { return (V1 * V2.transpose()).trace(); };

  // Retraction operator
  Optimization::Riemannian::Retraction<Matrix, Matrix, Matrix> retraction =
      [&problem, &g](const Matrix &Y, const Matrix &Ydot,
                     const Matrix &NablaF_Y) {
        Matrix Yplus;
        problem->retract(Y, Ydot, g, Yplus);
        return Yplus;
      };

  // Preconditioning operator (optional)
  std::optional<
      Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix>>
      precon;
  if (problem->preconditioner() == Preconditioner::None)
    precon = std::nullopt;
  else {
    Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix> precon_op =
        [&problem](const Matrix &Y, const Matrix &Ydot,
                   const Matrix &NablaF_Y) {
          return problem->precondition(Y, Ydot);
        };
    precon = precon_op;
  }

  // Configure optimization parameters
  Optimization::Riemannian::TNTParams<Scalar> params;
  params.gradient_tolerance = options.grad_norm_tol;
  params.preconditioned_gradient_tolerance =
      options.preconditioned_grad_norm_tol;
  params.relative_decrease_tolerance = options.rel_func_decrease_tol;
  params.stepsize_tolerance = options.stepsize_tol;
  params.max_iterations = options.max_iterations;
  params.max_iterations_accepted = options.max_iterations_accepted;
  params.max_TPCG_iterations = options.max_tCG_iterations;
  params.kappa_fgr = options.STPCG_kappa;
  params.theta = options.STPCG_theta;
  params.log_iterates = options.log_iterates;
  params.verbose = options.verbose;

  bool refined = ((results.gradFnorm * results.gradFnorm / results.fobj[iter]) >
                  options.accepted_delta);

  const auto &Fk = results.Fk;
  Scalar Gkh;
  Matrix DfR;

  // initialization
  problem->proximal(Y, Df, results.Xakh);

  auto t = results.Xak.topRows(n0);
  auto R = results.Xak.bottomRows(d * n0);
  R = results.Xakh.bottomRows(d * n0);
  problem->recover_translations(R, g, t);

  if (refined) {
    auto Results =
        Optimization::Riemannian::TNT<Matrix, Matrix, Scalar, Matrix>(
            Fobj, QM, metric, retraction, results.Xak, DfR, precon, params,
            options.user_function);

    results.Xak.swap(Results.x);
  }

  const int &index = g_index_.at(node).begin()->second;

  Xkh.middleRows(index, n0) = results.Xakh.topRows(n0);
  Xkh.middleRows(num_poses + d * index, d * n0) =
      results.Xakh.bottomRows(d * n0);

  Xkp.middleRows(index, n0) = results.Xak.topRows(n0);
  Xkp.middleRows(num_poses + d * index, d * n0) =
      results.Xak.bottomRows(d * n0);

  return 0;
}

int DPGOStar::mm_pgo_n(int node) const {
  assert(results_[node].updated == true &&
         "The optimizer has not been updated.");

  const auto &problem = problems_[node];
  const auto &options = options_;

  const int &n0 = problem->n()[0];
  const int &d = problem->d();
  const int &num_poses = num_poses_;

  auto &results = results_[node];
  auto &Xkp = Xkp_;

  int &iter = results.iters;

  const auto &g = results.g[iter];
  const auto &Df = results.Dfobj[iter];
  const auto &f = results.f[iter];

  /// Riemannian Optimization
  Optimization::Objective<Matrix, Scalar, Matrix> Fobj =
      [&problem, &g, &f](const Matrix &Y, const Matrix &NablaF_Y) {
        Scalar fobj;
        problem->evaluate_G(Y, g, f, fobj);

        return fobj;
      };

  // Local quadratic model constructor
  Optimization::Riemannian::QuadraticModel<Matrix, Matrix, Matrix> QM =
      [&problem,
       &g](const Matrix &Y, Matrix &grad,
           Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix>
               &HessOp,
           Matrix &NablaF_Y) {
        // Compute and cache Euclidean gradient at the current iterate
        problem->reduced_Euclidean_gradient_G(Y, g, NablaF_Y);

        // Compute Riemannian gradient from Euclidean gradient
        problem->reduced_Riemannian_gradient_G(Y, NablaF_Y, grad);

        // Define linear operator for computing Riemannian Hessian-vector
        // products (cf. eq. (44) in the SE-Sync tech report)
        HessOp = [problem, &g](const Matrix &Y, const Matrix &Ydot,
                               const Matrix &NablaF_Y) {
          Matrix Hess;
          problem->reduced_Riemannian_Hessian_vector_product(Y, NablaF_Y, Ydot,
                                                             Hess);
          return Hess;
        };
      };

  // Riemannian metric
  // We consider a realization of the product of SO(d) as an
  // embedded submanifold of R^{dn x d}; consequently, the induced Riemannian
  // metric is simply the usual Euclidean inner product
  Optimization::Riemannian::RiemannianMetric<Matrix, Matrix, Scalar, Matrix>
      metric =
          [](const Matrix &Y, const Matrix &V1, const Matrix &V2,
             const Matrix &NablaF_Y) { return (V1 * V2.transpose()).trace(); };

  // Retraction operator
  Optimization::Riemannian::Retraction<Matrix, Matrix, Matrix> retraction =
      [&problem, &g](const Matrix &Y, const Matrix &Ydot,
                     const Matrix &NablaF_Y) {
        Matrix Yplus;
        problem->retract(Y, Ydot, g, Yplus);
        return Yplus;
      };

  // Preconditioning operator (optional)
  std::optional<
      Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix>>
      precon;
  if (problem->preconditioner() == Preconditioner::None)
    precon = std::nullopt;
  else {
    Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix> precon_op =
        [&problem](const Matrix &Y, const Matrix &Ydot,
                   const Matrix &NablaF_Y) {
          return problem->precondition(Y, Ydot);
        };
    precon = precon_op;
  }

  // Configure optimization parameters
  Optimization::Riemannian::TNTParams<Scalar> params;
  params.gradient_tolerance = options.grad_norm_tol;
  params.preconditioned_gradient_tolerance =
      options.preconditioned_grad_norm_tol;
  params.relative_decrease_tolerance = options.rel_func_decrease_tol;
  params.stepsize_tolerance = options.stepsize_tol;
  params.max_iterations = options.max_iterations;
  params.max_iterations_accepted = options.max_iterations_accepted;
  params.max_TPCG_iterations = options.max_tCG_iterations;
  params.kappa_fgr = options.STPCG_kappa;
  params.theta = options.STPCG_theta;
  params.log_iterates = options.log_iterates;
  params.verbose = options.verbose;

  bool refined = ((results.gradFnorm * results.gradFnorm / results.fobj[iter]) >
                  options.accepted_delta);

  /// Run optimization!
  auto t = results.Xak.topRows(n0);
  auto R = results.Xak.bottomRows(d * n0);
  R = results.Xakh.bottomRows(d * n0);
  problem->recover_translations(R, g, t);

  if (refined) {
    Matrix DfR;

    auto Results =
        Optimization::Riemannian::TNT<Matrix, Matrix, Scalar, Matrix>(
            Fobj, QM, metric, retraction, results.Xak, DfR, precon, params,
            options.user_function);

    results.Xak = std::move(Results.x);
    results.Gk = std::move(Results.f);
  } else {
    problem->evaluate_G(results.Xak, results.g[iter], f, results.Gk);
  }

  const int &index = g_index_.at(node).begin()->second;

  Xkp.middleRows(index, n0) = results.Xak.topRows(n0);
  Xkp.middleRows(num_poses + d * index, d * n0) =
      results.Xak.bottomRows(d * n0);

  return 0;
}

int DPGOStar::pm_pgo_n(int node) const {
  const auto &problem = problems_[node];
  const auto &options = options_;

  const int &n0 = problem->n()[0];
  const int &d = problem->d();
  const int &num_poses = num_poses_;

  auto &results = results_[node];
  auto &Xkh = Xkh_;

  int &iter = results.iters;

  const auto &g = results.g[iter];
  const auto &Df = results.Dfobj[iter];

  /// Run optimization!
  problem->proximal(results.Xk, Df, results.Xakh);

  const int &index = g_index_.at(node).begin()->second;

  Xkh.middleRows(index, n0) = results.Xakh.topRows(n0);
  Xkh.middleRows(num_poses + d * index, d * n0) =
      results.Xakh.bottomRows(d * n0);

  return 0;
}

int DPGOStar::evaluate_f(const Matrix &X, Scalar &fobj) const {
  assert(X.rows() == (d_ + 1) * num_poses_);
  assert(X.cols() == d_);

  const auto &options = options_;

  if (options.loss == Loss::None) {
    const auto &M = M_;

    fobj = 0.5 * (X.transpose() * M * X).trace();
  } else {
    const auto &B0 = B0_;
    const auto &B1 = B1_;
    const auto &d = d_;
    const auto &m = num_measurements_;

    Matrix Error;
    Error.noalias() = X.transpose() * B0.transpose();

    fobj = 0.5 * Error.squaredNorm();

    Error.noalias() = X.transpose() * B1.transpose();
    Array ErrSqNorm = Eigen::Map<Matrix>(Error.data(), (d + 1) * d, m[1])
                          .colwise()
                          .squaredNorm();

    if (options.loss == Loss::Huber) {
      Array ReScale = ErrSqNorm.max(options.loss_reg).sqrt();
      Scalar sqrt_loss_reg = std::sqrt(options.loss_reg);

      fobj +=
          0.5 *
          (2 * sqrt_loss_reg * ReScale - options.loss_reg).min(ErrSqNorm).sum();
    } else if (options.loss == Loss::GemanMcClure) {
      fobj += 0.5 * options.loss_reg *
              (ErrSqNorm / (ErrSqNorm + options.loss_reg)).sum();
    } else if (options.loss == Loss::Welsch) {
      fobj += 0.5 *
              (options.loss_reg * m[1] -
               options.loss_reg * (-ErrSqNorm / options.loss_reg).exp().sum());
    } else {
      assert(0 && "Invalid loss kernels for pose graph optimization.");
      LOG(ERROR) << "Invalid loss kernels for pose graph optimization."
                 << std::endl;
    }
  }

  return 0;
}

int DPGOStar::evaluate_grad(const Matrix &X, Matrix &grad) const {
  assert(X.rows() == (d_ + 1) * num_poses_);
  assert(X.cols() == d_);

  const auto &options = options_;
  const auto &d = d_;
  const auto &n = num_poses_;

  Matrix Df;

  if (options.loss == Loss::None) {
    const auto &M = M_;

    Df.noalias() = M_ * X;
  } else {
    const auto &B0 = B0_;
    const auto &B1 = B1_;
    const auto &m = num_measurements_;

    Matrix Error;
    Error.noalias() = X.transpose() * B0.transpose();
    Df.noalias() = B0.transpose() * Error.transpose();

    Vector DiagReg((d + 1) * m[1]);

    Error.noalias() = X.transpose() * B1.transpose();
    Array ErrSqNorm = Eigen::Map<Matrix>(Error.data(), (d + 1) * d, m[1])
                          .colwise()
                          .squaredNorm();

    if (options.loss == Loss::Huber) {
      Scalar sqrt_loss_reg = std::sqrt(options.loss_reg);

      Array ReScale = ErrSqNorm.max(options.loss_reg).sqrt();
      Eigen::Map<Matrix> Reg(DiagReg.data(), d + 1, m[1]);
      Reg.row(0) = sqrt_loss_reg / ReScale;
      Reg.rowwise() = Reg.row(0);
    } else if (options.loss == Loss::GemanMcClure) {
      Scalar squared_loss_reg = options.loss_reg * options.loss_reg;

      Eigen::Map<Matrix> Reg(DiagReg.data(), d + 1, m[1]);
      Reg.row(0) = squared_loss_reg / (ErrSqNorm + options.loss_reg).square();
      Reg.rowwise() = Reg.row(0);
    } else if (options.loss == Loss::Welsch) {
      Eigen::Map<Matrix> Reg(DiagReg.data(), d + 1, m[1]);
      Reg.row(0) = (-ErrSqNorm / options.loss_reg).exp();
      Reg.rowwise() = Reg.row(0);
    } else {
      assert(0 && "Invalid loss kernels for pose graph optimization.");
      LOG(ERROR) << "Invalid loss kernels for pose graph optimization."
                 << std::endl;
    }

    Error.noalias() = Error * DiagReg.asDiagonal();

    Df.noalias() += B1.transpose() * Error.transpose();
  }

  const auto &R = X.bottomRows(d * n);

  grad = Df;

  auto gradR = grad.bottomRows(n * d);
  SP_.Proj(R, Df.bottomRows(d * n), gradR);

  return 0;
}
} // namespace DPGO
