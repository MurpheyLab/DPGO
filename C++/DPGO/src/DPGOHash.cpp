#include <DPGO/DPGOHash.h>
#include <Optimization/Riemannian/TNT.h>
#include <Optimization/Util/Stopwatch.h>
#include <memory>

namespace DPGO {
DPGOHash::DPGOHash(std::shared_ptr<const DPGOProblem> problem,
                     const Options &options)
    : problem_(problem), options_(options) {}

DPGOHash::DPGOHash(int node, const measurements_t &measurements,
                     const Options &options)
    : options_(options) {
  problem_ = std::make_shared<DPGOProblem>(
      node, measurements, options.regularizer, options.loss,
      options.preconditioner, options.reg_Cholesky_precon_max_condition_number,
      options.rescale, options.loss_reg);
}

int DPGOHash::initialize(const Matrix &X) const {
  const auto &problem = problem_;

  const auto &n = problem->n();
  const int &d = problem->d();

  assert(X.rows() == (d + 1) * (n[0] + n[1]));
  assert(X.cols() == d);

  auto &results = results_;

  results.clear();
  results.Xk = X;

  results.Xak = results.Xk.topRows((d + 1) * n[0]);

  results.Xakh.resize((d + 1) * n[0], d);

  results.gamma = 0;

  results.updated = false;

  return 0;
}

int DPGOHash::receive(const std::map<int, Matrix> &msg) const {
  auto &results = results_;
  auto &problem = problem_;

  const auto &recv = problem->recv();
  const auto &n = problem->n();
  const auto &s = problem->s();

  int d = problem->d();

  for (const auto &beta : msg) {
    int b = beta.first;

    auto search = recv.find(b);

    assert(search != recv.end());

    if (search != recv.end()) {
      results.updated = false;

      int num_poses = search->second.size();

      assert((d + 1) * num_poses == beta.second.rows());
      assert(d == beta.second.cols());

      int i = search->second.begin()->second[1];

      results.Xk.middleRows(s[1] * (d + 1) + i, num_poses) =
          beta.second.topRows(num_poses);
      results.Xk.middleRows(s[1] * (d + 1) + n[1] + i * d, num_poses * d) =
          beta.second.bottomRows(num_poses * d);
    } else {
      LOG(ERROR) << "Can not find information for node " << b << std::endl;
    }
  }

  return 0;
}

int DPGOHash::update() const {
  auto &results = results_;

  if (results.updated)
    return 0;

  const auto &problem = problem_;
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
    if (iter == 0) {
      problem->evaluate_none_g_and_f0(results.X[iter], results.g[iter],
                                      results.f[iter]);
      problem->evaluate_G(results.Xak, results.g[iter], results.f[iter],
                          results.fobj[iter]);
    } else {
      problem->evaluate_none_g_and_f(results.X[iter], results.X[iter - 1],
                                     results.Gk, results.g[iter],
                                     results.f[iter], results.fobj[iter]);
    }
  } else if (options.rescale == Rescale::Static) {
    if (iter == 0) {
      problem->evaluate_g_and_f0(results.X[iter], results.g[iter],
                                 results.f[iter], results.Dfobj[iter],
                                 results.fobj[iter], results.DfobjE,
                                 results.fobjE);
    } else {
      problem->evaluate_g_and_f(
          results.X[iter], results.X[iter - 1], results.Gk, results.DfobjE,
          results.fobjE, results.g[iter], results.f[iter], results.Dfobj[iter],
          results.fobj[iter], results.DfobjE, results.fobjE);
    }
  } else {
    if (iter == 0) {
      problem->evaluate_g_and_f0_rescale(
          results.X[iter], results.g[iter], results.f[iter],
          results.Dfobj[iter], results.fobj[iter], results.DfobjE,
          results.fobjE, results.rescale_count, options.max_rescale_count);
    } else {
      problem->evaluate_g_and_f_rescale(
          results.X[iter], results.X[iter - 1], results.Gk, results.DfobjE,
          results.fobjE, results.g[iter], results.f[iter], results.Dfobj[iter],
          results.fobj[iter], results.DfobjE, results.fobjE,
          results.rescale_count, options.max_rescale_count);
    }
  }

  if (iter == 0) {
    results.Fk[0] = results.fobj[iter];
    results.Fk[1] = results.fobj[iter];
    results.Gk = results.fobj[iter];
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
    results.F[0].resize(iter + 1);
    results.F[1].resize(iter + 1);

    if (iter == 0) {
      results_.s[0] = 1;
      results.oscillations.push_back(1);
    }

    Scalar &s0 = results.s[iter];
    Scalar &s1 = results.s[iter + 1];

    s1 = 0.5 + 0.5 * std::sqrt(4.0 * s0 * s0 + 1.0);

    results.gamma = (s0 - 1) / s1;

    if (results.fobj[iter] <= results.Fk[1]) {
      // decrease soft_restart_hits[0] by 2
      results.soft_restart_hits[0] = results.soft_restart_hits[0] > 2
                                         ? results.soft_restart_hits[0] - 2
                                         : 0;
    } else {
      // increase soft_restart_hits[1] by 1
      results.soft_restart_hits[0]++;
    }

    if (iter > 0) {
      if (results.fobj[iter] <= results.fobj[iter - 1]) {
        // set soft_restart_hits[1] to 0
        results.soft_restart_hits[1] = 0;
        results.oscillations.push_back(1);
      } else {
        // increase soft_restart_hits[1] by 1
        results.soft_restart_hits[1]++;
        results.oscillations.push_back(0);
      }

      results.num_oscillations +=
          results.oscillations[iter] != results.oscillations[iter - 1];
    }

    if (iter > options.oscillation_cnt_period) {
      results.num_oscillations -=
          results.oscillations[iter - options.oscillation_cnt_period] !=
          results.oscillations[iter - options.oscillation_cnt_period - 1];
    }

    results.Fk[0] = results.Fk[0] * (1 - options.eta[0]) +
                    results.fobj[iter] * options.eta[0];
    results.Fk[1] =
        std::max(results.fobj[iter], results.Fk[1] * (1 - options.eta[1]) +
                                         results.fobj[iter] * options.eta[1]);

    results.F[0][iter] = results.Fk[0];
    results.F[1][iter] = results.Fk[1];
  } else {
    results.Fk[0] = results.fobj[iter];
    results.Fk[1] = results.fobj[iter];
  }

  results.updated = true;

  return 0;
}

int DPGOHash::amm_pgo() const {
  assert(options_.scheme == Scheme::AMM &&
         "You are not expected to use the AMM scheme.");
  assert(results_.updated == true && "The optimizer has not been updated.");

  auto &results = results_;

  const auto &problem = problem_;
  const auto &options = options_;

  const int &n0 = problem->n()[0];
  const int &n1 = problem->n()[1];
  const int &d = problem->d();

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

  bool refined =
      (((results.gradFnorm * results.gradFnorm / results.fobj[iter]) >
        options.accepted_delta) ||
       (results.num_oscillations >= options.max_oscillations)) &&
      (options.max_iterations > 0) && (options.max_iterations_accepted > 0);

  /// Run optimization!
  const auto &Fk = results.Fk;
  Scalar Gkh;
  Matrix DfR;

  // initialization
  problem->proximal(Y, Df, results.Xakh);
  problem->evaluate_G(results.Xakh, results.g[iter], f, Gkh);

  Scalar minG =
      Fk[0] - options.psi * (results.Xakh - results.Xak).squaredNorm();

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

  problem->evaluate_G(results.Xak, results.g[iter], f, results.Gk);

  // adaptive restart
  if (Gkh > minG) {
    problem->proximal(results.Xk, results.Dfobj[iter], results.Xakh);
    problem->evaluate_G(results.Xakh, results.g[iter], f, Gkh);
  }

  bool hard_restart = results.Gk > Fk[0];
  bool soft_restart =
      // soft adaptive restart if soft_restart_hits[0] >=
      // max_soft_restart_hits[0]
      (results.Gk > Fk[1] &&
       results.soft_restart_hits[0] >= options.max_soft_restart_hits[0]) ||
      // soft adaptive restart if soft_restart_hits[1] >=
      // max_soft_restart_hits[1]
      (results.Gk > results.fobj[iter] &&
       results.soft_restart_hits[1] > options.max_soft_restart_hits[1]);

  if (hard_restart || soft_restart) {
    g = results.g[iter];

    if (Gkh <= results.fobj[iter]) {
      results.Xak = results.Xakh;
    } else {
      problem->proximal(results.Xk, results.Dfobj[iter], results.Xak);
    }
    auto t = results.Xak.topRows(n0);
    auto R = results.Xak.bottomRows(d * n0);
    problem->recover_translations(R, results.g[iter], t);

    if (refined) {
      auto Results =
          Optimization::Riemannian::TNT<Matrix, Matrix, Scalar, Matrix>(
              Fobj, QM, metric, retraction, results.Xak, DfR, precon, params,
              options.user_function);

      results.Xak.swap(Results.x);
      results.Gk = Results.f;
    } else {
      problem->evaluate_G(results.Xak, results.g[iter], f, results.Gk);
    }

    if (hard_restart) {
      results.s[iter + 1] = std::max(0.5 * results.s[iter + 1], 1.0);
    }

    results.soft_restart_hits[0] /= 3;
    results.soft_restart_hits[1] = 0;
  }

  if ((Fk[0] - results.Gk) < options.phi * (Fk[0] - Gkh)) {
    auto t = results.Xak.topRows(n0);
    auto R = results.Xak.bottomRows(d * n0);
    R = results.Xakh.bottomRows(d * n0);
    problem->recover_translations(R, g, t);

    problem->evaluate_G(results.Xak, results.g[iter], f, results.Gk);
  }

  return 0;
}

int DPGOHash::mm_pgo() const {
  assert(options_.scheme == Scheme::MM &&
         "You are not expected to use the MM scheme.");
  assert(results_.updated == true && "The optimizer has not been updated.");

  auto &results = results_;

  const auto &problem = problem_;
  const auto &options = options_;

  const int &n0 = problem->n()[0];
  const int &n1 = problem->n()[1];
  const int &d = problem->d();

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
                  options.accepted_delta) &&
                 (options.max_iterations > 0) &&
                 (options.max_iterations_accepted > 0);

  /// Run optimization!
  // initialization
  problem->proximal(results.Xk, Df, results.Xakh);

#if 1
  auto t = results.Xakh.topRows(n0);
  auto R = results.Xakh.bottomRows(d * n0);
  problem->recover_translations(R, g, t);

  if (refined) {
    Matrix DfR;

    auto Results =
        Optimization::Riemannian::TNT<Matrix, Matrix, Scalar, Matrix>(
            Fobj, QM, metric, retraction, results.Xakh, DfR, precon, params,
            options.user_function);

    results.Xak = std::move(Results.x);
    results.Gk = std::move(Results.f);
  } else {
    results.Xak = results.Xakh;
    problem->evaluate_G(results.Xak, results.g[iter], f, results.Gk);
  }
#else
  problem->evaluate_G(results.Xakh, g, f, results.Gk);
  results.Xak = results.Xakh;
#endif

  return 0;
}

int DPGOHash::iterate() const {
  const auto &options = options_;

  auto &results = results_;
  auto &iter = results.iters;

  const auto &problem = problem_;
  const int &n0 = problem->n()[0];
  const int &n1 = problem->n()[1];
  const int &d = problem->SOd().get_d();

  if (options_.verbose) {
    std::cout
        << "==================================================================="
        << std::endl;
    std::cout << "Iteration " << iter << std::endl;
    std::cout
        << "==================================================================="
        << std::endl;
  }

  auto start = Stopwatch::tick();
  if (options.scheme == Scheme::AMM) {
    amm_pgo();
  } else {
    mm_pgo();
  }
  auto elapsed_time = Stopwatch::tock(start);

  iter++;

  results.Xk.topRows((d + 1) * n0) = results.Xak;

  results.updated = false;

  if (options.verbose) {
    // Display some output to the user
    std::cout << std::endl
              << "Objective value F(Y) = " << results.Fk
              << "!  Elapsed computation time: " << elapsed_time << " seconds"
              << std::endl
              << std::endl;
  }

  return 0;
}
} // namespace DPGO
