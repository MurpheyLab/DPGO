#include <boost/program_options.hpp>

#include <DChordal/DChordal.h>
#include <DChordal/DChordalReduced.h>

#include <DPGO/DPGOHash.h>
#include <DPGO/DPGOStar.h>
#include <DPGO/DPGO_utils.h>

#include <SESync/SESyncProblem.h>
#include <SESync/SESync_utils.h>

#include <fstream>
#include <iomanip>
#include <memory>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " [input .g2o file]" << std::endl;
    exit(1);
  }

  boost::program_options::options_description desc("Program options");
  desc.add_options()                   // solver options
      ("help", "produce help message") // produce help message
      ("dataset", boost::program_options::value<std::string>(),
       "path to pose graph dataset") // path to pose graph dataset
      ("num_nodes", boost::program_options::value<int>(),
       "number of nodes") // number of nodes
      ("iters", boost::program_options::value<int>()->default_value(1000),
       "number of iterations") // maximum number of iterations
      ("dist_init",
       boost::program_options::value<bool>()->default_value("true"),
       "distributed (\"true\") or centralized (\"false\") initialization") // distributed
                                                                           // or
                                                                           // centralized
                                                                           // initialization
      ("loss",
       boost::program_options::value<std::string>()->default_value("trivial"),
       "loss type (\"trivial\", \"huber\" or \"welsch\")") // loss types
      ("accelerated",
       boost::program_options::value<bool>()->default_value(true),
       "whether accelerated or not") // whether accelerated or not
      ("save", boost::program_options::value<bool>()->default_value(true),
       "whether to save the optimization results or not"); // whether to save
                                                           // the optimization
                                                           // results or not

  boost::program_options::variables_map program_options;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc),
      program_options);

  if (program_options.count("help")) {
    std::cout << desc << "\n";
    exit(0);
  }

  if (program_options.count("dataset") == false) {
    LOG(ERROR) << "No dataset has been specfied." << std::endl;
    exit(-1);
  }

  if (program_options.count("num_nodes") == false) {
    LOG(ERROR) << "No number of nodes has been specfied." << std::endl;
    exit(-1);
  }

  std::string filename = program_options["dataset"].as<std::string>();
  int num_nodes = program_options["num_nodes"].as<int>();
  int num_iters = program_options["iters"].as<int>();
  bool accelerated = program_options["accelerated"].as<bool>();
  bool dist_chordal = program_options["dist_init"].as<bool>();
  std::string loss_type = program_options["loss"].as<std::string>();
  bool save = program_options["save"].as<bool>();

  DPGO::Loss loss;

  if (loss_type == "trivial") {
    loss = DPGO::Loss::None;
  } else if (loss_type == "huber") {
    loss = DPGO::Loss::Huber;
  } else if (loss_type == "welsch") {
    loss = DPGO::Loss::Welsch;
  } else {
    LOG(ERROR)
        << " The loss type can only be \"trivial\", \"huber\" or \"welsch\"."
        << std::endl;
    exit(-1);
  }

  int num_poses;
  int d;

  // Distributed PGO
  std::vector<std::shared_ptr<DPGO::DPGOHash>> dpgo_hash(num_nodes);
  std::vector<std::shared_ptr<const DPGO::DPGOProblem>> problems(num_nodes);

  std::vector<std::map<int, int>> g_index;
  std::vector<DPGO::measurements_t> measurements;
  DPGO::read_g2o(filename, num_nodes, num_poses, measurements, g_index);

  DPGO::Options options;

  options.rescale = DPGO::Rescale::Static;
  options.loss = loss;
  options.loss_reg = 0.25;
  options.verbose = false;
  options.scheme = accelerated ? DPGO::Scheme::AMM : DPGO::Scheme::MM;
  options.STPCG_kappa = 0.05;
  options.STPCG_theta = 0.9;
  options.eta[0] = 5e-4;
  options.eta[1] = 2.5e-2;
  options.max_soft_restart_hits[0] = 10;
  options.max_soft_restart_hits[1] = 25;
  options.max_iterations = 10;
  options.max_iterations_accepted = 1;
  options.grad_norm_tol = 1e-3;
  options.preconditioned_grad_norm_tol = 1e-4;
  options.regularizer = 1e-11;

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    dpgo_hash[alpha] =
        std::make_shared<DPGO::DPGOHash>(alpha, measurements[alpha], options);
    problems[alpha] = dpgo_hash[alpha]->problem();
  }

  // Initialization
  std::vector<DPGO::Matrix> Xk(num_nodes);
  DPGO::Matrix X;

  d = 0;

  for (const auto &measurement : measurements) {
    if (measurement.empty()) {
      continue;
    } else {
      d = measurement[0].t.size();

      break;
    }
  }

  if (dist_chordal) {
    SESync::SESyncOpts SESyncOptions;
    SESyncOptions.verbose = true;

    std::vector<DChordal::Matrix> xs(num_nodes);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      int n = g_index[alpha].size();

      SESync::SESyncResult result;
      DChordal::SESync(alpha, measurements[alpha], SESyncOptions, result);

      xs[alpha].noalias() =
          result.xhat.transpose() * result.xhat.middleCols(n, d);
    }

    std::vector<std::shared_ptr<DChordal::DChordalReduced_R>> dchordal_red_R(
        num_nodes);
    std::vector<std::shared_ptr<const DChordal::DChordalReducedProblem>>
        problems_red_R(num_nodes);
    DChordal::Options options;

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dchordal_red_R[alpha] = std::make_shared<DChordal::DChordalReduced_R>(
          alpha, measurements[alpha], options);

      problems_red_R[alpha] = dchordal_red_R[alpha]->problem();

      const auto &n = dchordal_red_R[alpha]->problem()->n();
      const auto &d = dchordal_red_R[alpha]->problem()->d();
      const auto &recv = dchordal_red_R[alpha]->problem()->recv();

      xs[alpha].conservativeResize((d + 1) * (n[0] + n[1]), d);
    }

    d = problems_red_R[0]->d();

    std::vector<DChordal::Matrix> rots_n(num_nodes);
    std::vector<DChordal::Matrix> rots(num_nodes);

    // rotation initialization
    std::cout << "===============================================" << std::endl;
    std::cout << "Initialize the reduced rotation" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    DPGO::communicate(problems_red_R, xs);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      const auto &d = problems_red_R[alpha]->d();

      dchordal_red_R[alpha]->setup(xs[alpha]);

      DChordal::Matrix R(d * problems_red_R[alpha]->nn() + d, d);

      for (int n = 0; n <= problems_red_R[alpha]->nn(); n++) {
        R.middleRows(d * n, d).setIdentity();
      }

      dchordal_red_R[alpha]->initialize(R);
    }

    for (int iter = 0; iter < 100; iter++) {
      auto fobj = DChordal::evaluate_f(dchordal_red_R);

      if (iter % 20 == 0)
        std::cout << iter << ": " << std::setprecision(16) << 0.5 * fobj
                  << std::endl;

      for (int alpha = 1; alpha < num_nodes; alpha++) {
        dchordal_red_R[alpha]->update();
        dchordal_red_R[alpha]->iterate();
      }

      for (int alpha = 0; alpha < num_nodes; alpha++) {
        dchordal_red_R[alpha]->n_communicate(dchordal_red_R);
      }
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      rots_n[alpha] = dchordal_red_R[alpha]->results().Xak;
      DChordal::project_to_SOd(rots_n[alpha], rots_n[alpha]);
    }

    std::cout << "===============================================" << std::endl;
    std::cout << "Initialize the rotation" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::vector<std::shared_ptr<DChordal::DChordal_R>> dchordal_R(num_nodes);
    std::vector<std::shared_ptr<const DChordal::DChordalProblem>> problems_R(
        num_nodes);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dchordal_R[alpha] = std::make_shared<DChordal::DChordal_R>(
          alpha, measurements[alpha], options);

      problems_R[alpha] = dchordal_R[alpha]->problem();

      const auto &n = problems_R[alpha]->n();

      rots[alpha].setZero(d * (n[0] + n[1]), d);
      rots[alpha].topRows(d * n[0]).noalias() =
          xs[alpha].middleRows(n[0], d * n[0]) * rots_n[alpha];
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      const auto &s = problems_R[alpha]->s();

      for (const auto &info : problems_R[alpha]->index()) {
        int beta = info.first;

        if (beta == alpha)
          continue;

        for (const auto &index : info.second) {
          const auto &i = index.second;
          const auto &j = index.first;

          rots[alpha].middleRows(d * (s[1] + i[1]), d) =
              rots[beta].middleRows(d * j, d);
        }
      }
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dchordal_R[alpha]->setup();
      dchordal_R[alpha]->initialize(rots[alpha]);
    }

    for (int iter = 0; iter < 400; iter++) {
      auto fobj = DChordal::evaluate_f(dchordal_R);

      if (iter % 20 == 0)
        std::cout << iter << ": " << 0.5 * fobj << std::endl;

      for (int alpha = 0; alpha < num_nodes; alpha++) {
        dchordal_R[alpha]->update();
        dchordal_R[alpha]->iterate();
      }

      for (int alpha = 0; alpha < num_nodes; alpha++) {
        dchordal_R[alpha]->communicate(dchordal_R);
      }
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      const auto &n = problems_R[alpha]->n();

      rots[alpha] = dchordal_R[alpha]->results().Xak;

      if (d == 2) {
        DPGO::project_to_SO2n(rots[alpha], rots[alpha]);
      } else if (d == 3) {
        DPGO::project_to_SO3n(rots[alpha], rots[alpha]);
      } else {
        DPGO::project_to_SOdn(rots[alpha], rots[alpha]);
      }

      rots_n[alpha] = rots[alpha].topRows(d);

      xs[alpha].middleRows(n[0], d * n[0]).noalias() =
          rots[alpha].topRows(d * n[0]) * rots_n[alpha].transpose();
    }

    // translation initialization
    std::cout << "===============================================" << std::endl;
    std::cout << "Initialize the reduced translation" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    std::vector<std::shared_ptr<DChordal::DChordalReduced_t>> dchordal_red_t(
        num_nodes);
    std::vector<std::shared_ptr<const DChordal::DChordalReducedProblem>>
        problems_red_t(num_nodes);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dchordal_red_t[alpha] = std::make_shared<DChordal::DChordalReduced_t>(
          alpha, measurements[alpha], options);

      problems_red_t[alpha] = dchordal_red_t[alpha]->problem();

      auto problem_t =
          std::dynamic_pointer_cast<const DChordal::DChordalReducedProblem_t>(
              problems_red_t[alpha]);

      const auto &n = problems_red_t[alpha]->n();

      auto t = xs[alpha].topRows(n[0]);
      auto R = xs[alpha].middleRows(n[0], d * n[0]);

      problem_t->recover_translations(R, t);
    }

    DPGO::communicate(problems_red_t, xs);
    DChordal::n_communicate(problems_red_R, rots_n);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      DChordal::Matrix t =
          DChordal::Matrix::Zero(problems_red_t[alpha]->nn() + 1, d);

      dchordal_red_t[alpha]->setup(xs[alpha], rots_n[alpha]);
      dchordal_red_t[alpha]->initialize(t);
    }

    for (int iter = 0; iter < 150; iter++) {
      auto fobj = DChordal::evaluate_f(dchordal_red_t);

      if (iter % 20 == 0)
        std::cout << iter << ": " << 0.5 * fobj << std::endl;

      for (int alpha = 1; alpha < num_nodes; alpha++) {
        dchordal_red_t[alpha]->update();
        dchordal_red_t[alpha]->iterate();
      }

      for (int alpha = 0; alpha < num_nodes; alpha++) {
        dchordal_red_t[alpha]->n_communicate(dchordal_red_t);
      }
    }

    std::cout << "===============================================" << std::endl;
    std::cout << "Initialize the translation" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    std::vector<std::shared_ptr<DChordal::DChordal_t>> dchordal_t(num_nodes);
    std::vector<std::shared_ptr<const DChordal::DChordalProblem>> problems_t(
        num_nodes);

    std::vector<DChordal::Matrix> ts(num_nodes);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dchordal_t[alpha] = std::make_shared<DChordal::DChordal_t>(
          alpha, measurements[alpha], options);

      problems_t[alpha] = dchordal_t[alpha]->problem();
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      const auto &n = problems_t[alpha]->n();

      ts[alpha].noalias() = xs[alpha].topRows(n[0]) * rots_n[alpha].topRows(d);
      ts[alpha].rowwise() += dchordal_red_t[alpha]->results().Xak.row(0);
    }

    DChordal::communicate(problems_R, rots);
    DChordal::communicate(problems_t, ts);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dchordal_t[alpha]->setup(rots[alpha]);
      dchordal_t[alpha]->initialize(ts[alpha]);
    }

    for (int iter = 0; iter < 250; iter++) {
      auto fobj = DChordal::evaluate_f(dchordal_t);

      if (iter % 20 == 0)
        std::cout << iter << ": " << 0.5 * fobj << std::endl;

      for (int alpha = 0; alpha < num_nodes; alpha++) {
        dchordal_t[alpha]->update();
        dchordal_t[alpha]->iterate();
      }

      for (int alpha = 0; alpha < num_nodes; alpha++) {
        dchordal_t[alpha]->communicate(dchordal_t);
      }
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      const auto &n = problems_t[alpha]->n()[0];

      Xk[alpha].setZero((d + 1) * n, d);
      Xk[alpha].topRows(n) = dchordal_t[alpha]->results().Xak;
      Xk[alpha].bottomRows(d * n) = rots[alpha].topRows(d * n);
    }
  } else {
    std::cout << "===============================================" << std::endl;
    std::cout << "Initialization" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    size_t n_poses;
    const auto &m_measurements = SESync::read_g2o_file(filename, n_poses);
    num_poses = n_poses;

    SESync::SparseMatrix B1, B2, B3;
    SESync::construct_B_matrices(m_measurements, B1, B2, B3);

    X.setZero((d + 1) * num_poses, d);
    X.bottomRows(d * num_poses) =
        SESync::chordal_initialization(d, B3).transpose();
    X.topRows(num_poses) = SESync::recover_translations(
                               B1, B2, X.bottomRows(d * num_poses).transpose())
                               .transpose();

    for (int alpha = 0, index = 0; alpha < num_nodes; alpha++) {
      const auto &n = problems[alpha]->n()[0];

      Xk[alpha].setZero((d + 1) * n, d);
      Xk[alpha].topRows(n) = X.middleRows(index, n);
      Xk[alpha].bottomRows(d * n) = X.middleRows(num_poses + d * index, d * n);

      index += n;
    }
  }

  DPGO::communicate(problems, Xk);

  DPGO::DPGOStar dpgo_star(num_nodes, filename, options);

  std::vector<std::map<int, Eigen::MatrixXd>> msgs(num_nodes);

  double fobj = 0, grad = 0;
  DPGO::Matrix gradF;

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    dpgo_hash[alpha]->initialize(Xk[alpha]);
    dpgo_hash[alpha]->update();

    for (auto &info : problems[alpha]->recv()) {
      msgs[alpha][info.first].resize((d + 1) * info.second.size(), d);
    }
  }

  X.setZero((d + 1) * num_poses, d);

  for (int alpha = 0; alpha < num_nodes; alpha++) {
    const auto &n = problems[alpha]->n();
    const auto &s = problems[alpha]->s();

    int i = g_index[alpha].begin()->second;

    X.middleRows(i, n[0]) = dpgo_hash[alpha]->results().Xk.topRows(n[0]);
    X.middleRows(num_poses + d * i, d * n[0]) =
        dpgo_hash[alpha]->results().Xk.middleRows(n[0], d * n[0]);
  }

  dpgo_star.evaluate_f(X, fobj);
  fobj *= 2;

  dpgo_star.evaluate_grad(X, gradF);
  grad = 2 * gradF.norm();

  std::list<DPGO::Vector> results;
  results.push_back(Eigen::Vector4d{0, 0, fobj, grad});

  double time = 0;

  std::cout << "===============================================" << std::endl;
  std::cout << "Distributed PGO" << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;

  for (int iter = 0; iter < num_iters; iter++) {
    std::cout << iter << ": " << std::setprecision(20) << fobj << " " << grad
              << std::endl;

    auto opt_start = Stopwatch::tick();
    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dpgo_hash[alpha]->iterate();
    }
    time += Stopwatch::tock(opt_start);

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      const auto &n = problems[alpha]->n();
      const auto &s = problems[alpha]->s();

      int i = g_index[alpha].begin()->second;

      X.middleRows(i, n[0]) = dpgo_hash[alpha]->results().Xk.topRows(n[0]);
      X.middleRows(num_poses + d * i, d * n[0]) =
          dpgo_hash[alpha]->results().Xk.middleRows(n[0], d * n[0]);
    }

    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dpgo_hash[alpha]->communicate(dpgo_hash);
    }

    auto update_start = Stopwatch::tick();
    for (int alpha = 0; alpha < num_nodes; alpha++) {
      dpgo_hash[alpha]->update();
    }
    time += Stopwatch::tock(update_start);

    dpgo_star.evaluate_f(X, fobj);
    fobj *= 2;

    dpgo_star.evaluate_grad(X, gradF);
    grad = 2 * gradF.norm();

    results.push_back(
        Eigen::Vector4d{DPGO::Scalar(iter) + 1, time, fobj, grad});
  }

  std::cout << "---------------------------------------" << std::endl;
  std::cout << "final objective: " << fobj << std::endl;
  std::cout << "final gradient: " << grad << std::endl;
  std::cout << "time: " << time / num_nodes << " s/node." << std::endl;

  if (save) {
    std::string resfile = "results_chordal_" + std::to_string(num_nodes) + "_" +
                          (accelerated ? "amm" : "mm") + ".txt";
    std::ofstream output(resfile);

    if (!output.is_open())
      return -1;

    for (const auto &res : results) {
      output << int(res[0]) << " " << std::setprecision(16) << res[1] << " "
             << std::setprecision(16) << res[2] << " " << std::setprecision(16)
             << res[3] << std::endl;
    }

    output.close();

    DPGO::Vector t = X.row(0);
    X.topRows(num_poses).rowwise() -= t.transpose();

    DPGO::Matrix R = X.middleRows(num_poses, d).transpose();
    X = X * R;

    output.open("./estimates_" + loss_type + ".txt");

    if (!output.is_open())
      return -1;

    output << X << std::endl;

    output.close();
  }

  return 0;
}
