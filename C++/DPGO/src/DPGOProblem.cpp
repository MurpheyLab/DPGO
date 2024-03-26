#include <DPGO/DPGOProblem.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#define USE_G 1

namespace DPGO {
constexpr Scalar DPGOProblem::max_rescale_;
constexpr Scalar DPGOProblem::min_rescale_;

DPGOProblem::DPGOProblem(int node, const measurements_t& measurements,
                         Scalar reg, const Loss& loss,
                         const Preconditioner& preconditioner,
                         Scalar reg_chol_precon_max_cond, Rescale rescale,
                         Scalar loss_reg)
    : node_(node),
      measurements_(measurements),
      reg_G_(reg),
      loss_(loss),
      rescale_(rescale),
      preconditioner_(preconditioner),
      reg_Chol_precon_max_cond_(reg_chol_precon_max_cond),
      loss_reg_(loss_reg),
      squared_loss_reg_(loss_reg * loss_reg),
      sqrt_loss_reg_(std::sqrt(loss_reg)) {
  generate_data_info(node_, measurements, intra_measurements_,
                     inter_measurements_, n_, s_, m_, index_, sent_, recv_);

  d_ = (!measurements.empty() ? measurements[0].t.size() : 0);

  DiagReScale_.setOnes(m_[1]);

  if (loss_ == Loss::None && SIMPLE) {
    simplify_quadratic_data_matrix(
        node_, intra_measurements_, inter_measurements_, n_, s_, m_, index_,
        reg_G_, G_, mG_[0][0], mG_[0][1], mG_[1][1], B_, mB_[0], mB_[1], D_, S_,
        Q_, P_, P0_, H_, DiagT_, ScaleN_, U_, V_);

    mG_[1][0] = mG_[0][1].transpose();
  } else if (rescale_ == Rescale::Static) {
    simplify_regular_data_matrix(
        node_, intra_measurements_, inter_measurements_, n_, s_, m_, index_,
        reg_G_, G_, mG_[0][0], mG_[0][1], mG_[1][1], B_, mB_[0], mB_[1], D_, Q_,
        P0_, H_, DiagT_, ScaleN_, V_);

    mG_[1][0] = mG_[0][1].transpose();
  } else {
    Eigen::Matrix<int, 2, 1> num_inter_n;

    simplify_regular_data_matrix(
        node_, intra_measurements_, inter_measurements_, n_, s_, m_, index_,
        num_inter_n, reg_G_, M_, mM_, mB_[0], mB_[1], D0_, Q0_, E_, F_, DiagT0_,
        N0_, V0_, outer_index_G_, inner_index_G_, inner_nnz_G_, outer_index_mG_,
        inner_index_mG_, inner_nnz_mG_, outer_index_D_, inner_index_D_,
        inner_nnz_D_, outer_index_Q_, inner_index_Q_, inner_nnz_Q_,
        outer_index_T_, inner_index_T_, inner_nnz_T_, outer_index_N_,
        inner_index_N_, inner_nnz_N_, outer_index_V_, inner_index_V_,
        inner_nnz_V_);

    G_ = M_;
    mG_ = mM_;
    D_ = D0_;
    Q_ = Q0_;
    DiagT_ = DiagT0_;
    N_ = N0_;
    V_ = V0_;
    ScaleN_ = N0_;

    new (&Gval_) Eigen::Map<Vector>(G_.valuePtr(), G_.nonZeros());

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        new (&mGval_[i][j])
            Eigen::Map<Vector>(mG_[i][j].valuePtr(), mG_[i][j].nonZeros());
      }

    new (&Dval_) Eigen::Map<Vector>(D_.valuePtr(), D_.nonZeros());
    new (&Qval_) Eigen::Map<Vector>(Q_.valuePtr(), Q_.nonZeros());
    new (&Tval_) Eigen::Map<Vector>(DiagT_.diagonal().data(), n_[0]);
    new (&Nval_) Eigen::Map<Vector>(N_.valuePtr(), N_.nonZeros());
    new (&Vval_) Eigen::Map<Vector>(V_.valuePtr(), V_.nonZeros());
    new (&ScaleNval_)
        Eigen::Map<Vector>(ScaleN_.valuePtr(), ScaleN_.nonZeros());

    update_quadratic_mat(DiagReScale_);
  }

  size_ = (d_ + 1) * n_;

  SP_.set_d(d_);
  SP_.set_n(n_[0]);

  L_.compute(mG_[0][0]);

  /** Compute and cache preconditioning matrices, if required */
  if (preconditioner_ == Preconditioner::Jacobi) {
    Vector diag = mG_[1][1].diagonal();
    Jacobi_precon_ = diag.cwiseInverse().asDiagonal();
  } else if (preconditioner_ == Preconditioner::IncompleteCholesky)
    iChol_precon_ = new IncompleteCholeskyFactorization(mG_[1][1]);
  else if (preconditioner_ == Preconditioner::RegularizedCholesky) {
    // Compute maximum eigenvalue of LGrho

    // NB: Spectra's built-in SparseSymProduct matrix assumes that input
    // matrices are stored in COLUMN-MAJOR order
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> rotG_col_major(mG_[1][1]);

    Spectra::SparseSymMatProd<Scalar> op(rotG_col_major);
    Spectra::SymEigsSolver<Scalar, Spectra::LARGEST_MAGN,
                           Spectra::SparseSymMatProd<Scalar>>
        max_eig_solver(&op, 1, 3);
    max_eig_solver.init();

    int max_iterations = 10000;
    Scalar tol = 1e-4;  // We only require a relatively loose estimate here ...
    int nconv = max_eig_solver.compute(max_iterations, tol);

    Scalar lambda_max = max_eig_solver.eigenvalues()(0);
    reg_Chol_precon_.compute(
        mG_[1][1] +
        SparseMatrix(Vector::Constant(mG_[1][1].rows(),
                                      lambda_max / reg_Chol_precon_max_cond_)
                         .asDiagonal()));
  }
}

int DPGOProblem::retract(const Matrix& Y, const Matrix& Ydot, const Matrix& g,
                         Matrix& Yplus) const {
  Yplus.resize(size_[0], d_);

  auto const& t = Y.topRows(n_[0]);
  auto const& R = Y.bottomRows(d_ * n_[0]);
  auto tplus = Yplus.topRows(n_[0]);
  auto Rplus = Yplus.bottomRows(d_ * n_[0]);

  SP_.retract(R, Ydot, Rplus);

  Matrix temp = g.topRows(n_[0]);
  temp.noalias() += mG_[0][1] * Rplus;
  tplus.noalias() = -L_.solve(temp);

  return 0;
}

int DPGOProblem::full_tangent_space_projection(const Matrix& Y,
                                               const Matrix& Ydot,
                                               Matrix& gradF_Y) const {
  assert(Y.rows() == size_[0]);
  assert(Y.cols() == d_);
  assert(Ydot.rows() == size_[0]);
  assert(Ydot.cols() == d_);

  auto const& R = Y.bottomRows(d_ * n_[0]);

  gradF_Y.resize(size_[0], d_);
  gradF_Y.topRows(n_[0]) = Ydot.topRows(n_[0]);

  auto gradF_R = gradF_Y.bottomRows(n_[0] * d_);
  SP_.Proj(R, Ydot.bottomRows(d_ * n_[0]), gradF_R);

  return 0;
}

int DPGOProblem::reduced_tangent_space_projection(const Matrix& Y,
                                                  const Matrix& Ydot,
                                                  Matrix& gradF_Y) const {
  assert(Y.rows() == size_[0]);
  assert(Y.cols() == d_);
  assert(Ydot.rows() == d_ * n_[0]);
  assert(Ydot.cols() == d_);

  auto const& R = Y.bottomRows(d_ * n_[0]);

  gradF_Y.resize(n_[0] * d_, d_);
  SP_.Proj(R, Ydot, gradF_Y);

  return 0;
}

int DPGOProblem::evaluate_G(const Matrix& Y, const Matrix& g, Scalar f,
                            Scalar& G) const {
  assert(Y.rows() == size_[0]);

  Matrix temp = g;

#if USE_G
  temp.noalias() += 0.5 * G_ * Y;
#else
  if (loss_ == Loss::None && SIMPLE) {
    temp.noalias() += 0.5 * G_ * Y;
  } else {
    const auto& t = Y.topRows(n_[0]);
    const auto& R = Y.bottomRows(d_ * n_[0]);

    temp.topRows(n_[0]).noalias() += 0.5 * mG_[0][0] * t;
    temp.topRows(n_[0]).noalias() += 0.5 * mG_[0][1] * R;
    temp.bottomRows(d_ * n_[0]).noalias() += 0.5 * mG_[1][0] * t;
    temp.bottomRows(d_ * n_[0]).noalias() += 0.5 * mG_[1][1] * R;
  }
#endif

  G = (Y.transpose() * temp).trace() + f;

  return 0;
}

int DPGOProblem::evaluate_G(const Matrix& Y, const Matrix& Y0,
                            const Matrix& Dfobj, Scalar fobj, Scalar& G) const {
  assert(Y.rows() >= size_[0]);
  assert(Y0.rows() >= size_[0]);

  Matrix Z = Y.topRows(size_[0]) - Y0.topRows(size_[0]);
  Matrix temp = Dfobj;

  temp.noalias() += 0.5 * G_ * Z;

  G = (Z.transpose() * temp).trace() + fobj;

  return 0;
}

int DPGOProblem::evaluate_g_and_f0(const Matrix& Z, Matrix& g, Scalar& f0,
                                   Matrix& Dfobj, Scalar& fobj, Matrix& DfobjE,
                                   Scalar& fobjE) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  Vector DiagReg;

  evaluate_E(Z, DiagReg, DfobjE, fobjE);

  g = DfobjE.topRows(size_[0]);

  Matrix temp = D_ * Z.topRows(size_[0]);

  g.noalias() -= temp;

  temp *= 0.5;
  temp -= DfobjE.topRows(size_[0]);

  f0 = 0.5 * fobjE;
  f0 += (Z.topRows(size_[0]).transpose() * temp).trace();

#if USE_G
  temp.noalias() = G_ * Z.topRows(size_[0]);
#else
  const auto& t = Z.topRows(n_[0]);
  const auto& R = Z.middleRows(n_[0], d_ * n_[0]);

  temp.resize((d_ + 1) * n_[0], d_);
  temp.topRows(n_[0]).noalias() = mG_[0][0] * t;
  temp.topRows(n_[0]).noalias() += mG_[0][1] * R;
  temp.bottomRows(d_ * n_[0]).noalias() = mG_[1][0] * t;
  temp.bottomRows(d_ * n_[0]).noalias() += mG_[1][1] * R;
#endif

  Dfobj = g;
  Dfobj.noalias() += temp;

  temp *= 0.5;
  temp += g;

  fobj = f0;
  fobj += (Z.topRows(size_[0]).transpose() * temp).trace();

  return 0;
}

int DPGOProblem::evaluate_none_g_and_f0(const Matrix& Z, Matrix& g,
                                        Scalar& f0) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  assert(loss_ == Loss::None);

  if (loss_ == Loss::None && SIMPLE) {
    g.noalias() = S_ * Z;
    f0 = 0.5 * (Z.transpose() * P0_ * Z).trace();
  } else {
    LOG(ERROR) << "This method only applies to non-robust kernels."
               << std::endl;

    exit(-1);
  }

  return 0;
}

int DPGOProblem::evaluate_g_and_f0_rescale(const Matrix& Z, Matrix& g,
                                           Scalar& f0, Matrix& Dfobj,
                                           Scalar& fobj, Matrix& DfobjE,
                                           Scalar& fobjE, int& rescale_count,
                                           int max_rescale_count) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  Vector DiagReg;

  evaluate_E(Z, DiagReg, DfobjE, fobjE);

  if (rescale_ == Rescale::Dynamic) {
    Eigen::Map<Vector, 0, Eigen::InnerStride<>> Reg(
        DiagReg.data(), m_[1], 1, Eigen::InnerStride<>(d_ + 1));

    bool rescaled = (rescale_count >= max_rescale_count) ||
                    (Reg.array() > DiagReScale_.array()).sum();

    if (rescaled) {
      DiagReScale_ = 1.25 * Reg;
      DiagReScale_ = DiagReScale_.cwiseMin(max_rescale_);
      DiagReScale_ = DiagReScale_.cwiseMax(min_rescale_);

      update_quadratic_mat(DiagReScale_);

      L_.factorize(mG_[0][0]);

      rescale_count = 0;
    } else {
      rescale_count++;
    }
  }

  g = DfobjE.topRows(size_[0]);

  Matrix temp = D_ * Z.topRows(size_[0]);

  g.noalias() -= temp;

  temp *= 0.5;
  temp -= DfobjE.topRows(size_[0]);

  f0 = 0.5 * fobjE;
  f0 += (Z.topRows(size_[0]).transpose() * temp).trace();

#if USE_G
  temp.noalias() = G_ * Z.topRows(size_[0]);
#else
  const auto& t = Z.topRows(n_[0]);
  const auto& R = Z.middleRows(n_[0], d_ * n_[0]);

  temp.resize((d_ + 1) * n_[0], d_);
  temp.topRows(n_[0]).noalias() = mG_[0][0] * t;
  temp.topRows(n_[0]).noalias() += mG_[0][1] * R;
  temp.bottomRows(d_ * n_[0]).noalias() = mG_[1][0] * t;
  temp.bottomRows(d_ * n_[0]).noalias() += mG_[1][1] * R;
#endif

  Dfobj = g;
  Dfobj.noalias() += temp;

  temp *= 0.5;
  temp += g;

  fobj = f0;
  fobj += (Z.topRows(size_[0]).transpose() * temp).trace();

  return 0;
}

int DPGOProblem::evaluate_g_and_f(const Matrix& Z, const Matrix& Z0, Scalar G,
                                  const Matrix& DfobjE0, Scalar fobjE0,
                                  Matrix& g, Scalar& f, Matrix& Dfobj,
                                  Scalar& fobj, Matrix& DfobjE,
                                  Scalar& fobjE) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  const auto& X = Z.topRows(size_[0]);

  if (loss_ == Loss::None && SIMPLE) {
    g.noalias() = S_ * Z;

    Dfobj = g;
    Dfobj.noalias() += G_ * X;

    Matrix Y = Z - Z0;

    fobj = G;
    fobj += 0.5 * (Y.transpose() * Q_ * Y).trace();
    f = fobj + 0.5 * (X.transpose() * P_ * X).trace();

    Vector DiagReg;
    evaluate_E(Z, DiagReg, DfobjE, fobjE);
  } else {
    Matrix Y = Z - Z0;
    Matrix temp = DfobjE0;
    temp.noalias() += 0.5 * Q_ * Y;

    fobj = G - 0.5 * fobjE0;
    fobj -= 0.5 * (Y.transpose() * temp).trace();

    Vector DiagReg;
    evaluate_E(Z, DiagReg, DfobjE, fobjE);

    fobj += 0.5 * fobjE;

    g = DfobjE.topRows(size_[0]);
    g.noalias() -= D_ * Z.topRows(size_[0]);

#if USE_G
    temp.noalias() = G_ * X;
#else
    const auto& t = Z.topRows(n_[0]);
    const auto& R = Z.middleRows(n_[0], d_ * n_[0]);

    temp.resize((d_ + 1) * n_[0], d_);
    temp.topRows(n_[0]).noalias() = mG_[0][0] * t;
    temp.topRows(n_[0]).noalias() += mG_[0][1] * R;
    temp.bottomRows(d_ * n_[0]).noalias() = mG_[1][0] * t;
    temp.bottomRows(d_ * n_[0]).noalias() += mG_[1][1] * R;
#endif

    Dfobj = g;
    Dfobj.noalias() += temp;

    temp *= 0.5;
    temp += g;

    f = fobj;
    f -= (X.transpose() * temp).trace();
  }

  return 0;
}

int DPGOProblem::evaluate_g_and_f_rescale(const Matrix& Z, const Matrix& Z0,
                                          Scalar G, const Matrix& DfobjE0,
                                          Scalar fobjE0, Matrix& g, Scalar& f,
                                          Matrix& Dfobj, Scalar& fobj,
                                          Matrix& DfobjE, Scalar& fobjE,
                                          int& rescale_count,
                                          int max_rescale_count) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  const auto& X = Z.topRows(size_[0]);

  if (loss_ == Loss::None && SIMPLE) {
    g.noalias() = S_ * Z;

    Dfobj = g;
    Dfobj.noalias() += G_ * X;

    Matrix Y = Z - Z0;

    fobj = G;
    fobj += 0.5 * (Y.transpose() * Q_ * Y).trace();
    f = fobj + 0.5 * (X.transpose() * P_ * X).trace();

    Vector DiagReg;
    evaluate_E(Z, DiagReg, DfobjE, fobjE);
  } else {
    Matrix Y = Z - Z0;
    Matrix temp = DfobjE0;
    temp.noalias() += 0.5 * Q_ * Y;

    fobj = G - 0.5 * fobjE0;
    fobj -= 0.5 * (Y.transpose() * temp).trace();

    Vector DiagReg;
    evaluate_E(Z, DiagReg, DfobjE, fobjE);

    fobj += 0.5 * fobjE;

    if (rescale_ == Rescale::Dynamic) {
      Eigen::Map<Vector, 0, Eigen::InnerStride<>> Reg(
          DiagReg.data(), m_[1], 1, Eigen::InnerStride<>(d_ + 1));

      bool rescaled = (rescale_count >= max_rescale_count) ||
                      (Reg.array() > DiagReScale_.array()).sum();

      if (rescaled) {
        DiagReScale_ = 1.25 * Reg;
        DiagReScale_ = DiagReScale_.cwiseMin(max_rescale_);
        DiagReScale_ = DiagReScale_.cwiseMax(min_rescale_);

        update_quadratic_mat(DiagReScale_);

        L_.factorize(mG_[0][0]);

        rescale_count = 0;
      } else {
        rescale_count++;
      }
    }

    g = DfobjE.topRows(size_[0]);
    g.noalias() -= D_ * Z.topRows(size_[0]);

#if USE_G
    temp.noalias() = G_ * X;
#else
    const auto& t = Z.topRows(n_[0]);
    const auto& R = Z.middleRows(n_[0], d_ * n_[0]);

    temp.resize((d_ + 1) * n_[0], d_);
    temp.topRows(n_[0]).noalias() = mG_[0][0] * t;
    temp.topRows(n_[0]).noalias() += mG_[0][1] * R;
    temp.bottomRows(d_ * n_[0]).noalias() = mG_[1][0] * t;
    temp.bottomRows(d_ * n_[0]).noalias() += mG_[1][1] * R;
#endif

    Dfobj = g;
    Dfobj.noalias() += temp;

    temp *= 0.5;
    temp += g;

    f = fobj;
    f -= (X.transpose() * temp).trace();
  }

  return 0;
}

int DPGOProblem::evaluate_none_g_and_f(const Matrix& Z, const Matrix& Z0,
                                       Scalar G, Matrix& g, Scalar& f,
                                       Scalar& fobj) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);
  assert(Z0.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z0.cols() == d_);

  assert(loss_ == Loss::None);

  if (loss_ == Loss::None && SIMPLE) {
    g.noalias() = S_ * Z;

    Matrix Y = Z - Z0;

    fobj = G;
    fobj += 0.5 * (Y.transpose() * Q_ * Y).trace();
    f = fobj + 0.5 * (Z.transpose() * P_ * Z).trace();
  } else {
    LOG(ERROR) << "This method only applies to non-robust kernels."
               << std::endl;

    exit(-1);
  }

  return 0;
}

int DPGOProblem::evaluate_g_and_Df(const Matrix& Z, Matrix& g,
                                   Matrix& Df) const {
  evaluate_g(Z, g);
  evaluate_Df(Z, g, Df);

  return 0;
}

int DPGOProblem::reduced_Riemannian_Hessian_vector_product(
    const Matrix& Y, const Matrix& nablaF_Y, const Matrix& Ydot,
    Matrix& Hess) const {
  assert(Y.rows() == size_[0]);
  assert(Y.cols() == d_);
  assert(nablaF_Y.rows() == d_ * n_[0]);
  assert(nablaF_Y.cols() == d_);
  assert(Ydot.rows() == d_ * n_[0]);
  assert(Ydot.cols() == d_);

  auto const& R = Y.bottomRows(d_ * n_[0]);
  Eigen::MatrixXd temp(d_ * n_[0], d_);

  Hess.resize(d_ * n_[0], d_);

  auto const& Rdot = Ydot;
  Matrix tdot = -L_.solve(mG_[0][1] * Rdot);

  Matrix E(d_ * n_[0], d_);
  E.noalias() = mG_[1][0] * tdot + mG_[1][1] * Rdot;
  SP_.SymBlockDiagProduct(Rdot, R, nablaF_Y, temp);
  E -= temp;
  SP_.Proj(R, E, Hess);

  return 0;
}

Matrix DPGOProblem::precondition(const Matrix& Y, const Matrix& Ydot) const {
  Matrix Ysol;

  switch (preconditioner_) {
    case Preconditioner::Jacobi:
      reduced_tangent_space_projection(Y, Jacobi_precon_ * Ydot, Ysol);
      return Ysol;

    case Preconditioner::IncompleteCholesky:
      reduced_tangent_space_projection(Y, iChol_precon_->solve(Ydot), Ysol);
      return Ysol;

    case Preconditioner::RegularizedCholesky:
      reduced_tangent_space_projection(Y, reg_Chol_precon_.solve(Ydot), Ysol);
      return Ysol;

    default:
      return Ydot;
  };
}

int DPGOProblem::proximal(const Matrix& Z, const Matrix& Df, Matrix& X) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);
  assert(Df.rows() == size_[0]);
  assert(Df.cols() == d_);

  X.resize(size_[0], d_);

  const auto& t0 = Z.topRows(n_[0]);
  const auto& R0 = Z.middleRows(n_[0], d_ * n_[0]);
  auto t = X.topRows(n_[0]);
  auto R = X.bottomRows(d_ * n_[0]);

  Matrix M;

  if (loss_ == Loss::None && SIMPLE) {
    M.noalias() = U_ * Z;
  } else {
    M = -Df.bottomRows(d_ * n_[0]);
    M.noalias() += ScaleN_.transpose() * Df.topRows(n_[0]);
    M.noalias() += V_ * Z.middleRows(n_[0], d_ * n_[0]);
  }

  SP_.project(M, R);

  // recover_translations(R, DF, t);

  t = t0;
  t.noalias() -= ScaleN_ * (R - R0);
  t.noalias() -= DiagT_ * Df.topRows(n_[0]);

  return 0;
}

int DPGOProblem::evaluate_E(const Matrix& Z, Vector& DiagReg, Matrix& DfobjE,
                            Scalar& fobjE) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  Matrix Error = Z.transpose() * mB_[1].transpose();

  DiagReg.resize((d_ + 1) * m_[1]);

  Array mErrNorm = Eigen::Map<Matrix>(Error.data(), (d_ + 1) * d_, m_[1])
                       .colwise()
                       .squaredNorm();

  if (loss_ == Loss::None) {
    DiagReg.setOnes();

    fobjE = 0.5 * mErrNorm.sum();
  } else if (loss_ == Loss::Huber) {
    Array ReScale = mErrNorm.max(loss_reg_).sqrt();
    Eigen::Map<Matrix> Reg(DiagReg.data(), d_ + 1, m_[1]);
    Reg.row(0) = sqrt_loss_reg_ / ReScale;
    Reg.rowwise() = Reg.row(0);

    fobjE =
        0.5 * (2 * sqrt_loss_reg_ * ReScale - loss_reg_).min(mErrNorm).sum();
  } else if (loss_ == Loss::GemanMcClure) {
    Eigen::Map<Matrix> Reg(DiagReg.data(), d_ + 1, m_[1]);
    Reg.row(0) = squared_loss_reg_ / (mErrNorm + loss_reg_).square();
    Reg.rowwise() = Reg.row(0);

    fobjE = 0.5 * loss_reg_ * (mErrNorm / (mErrNorm + loss_reg_)).sum();
  } else if (loss_ == Loss::Welsch) {
    Eigen::Map<Matrix> Reg(DiagReg.data(), d_ + 1, m_[1]);
    Reg.row(0) = (-mErrNorm / loss_reg_).exp();
    Reg.rowwise() = Reg.row(0);

    fobjE = 0.5 * (loss_reg_ * m_[1] - loss_reg_ * Reg.row(0).sum());
  } else {
    assert(0 && "Invalid loss kernels for pose graph optimization.");
    LOG(ERROR) << "Invalid loss kernels for pose graph optimization."
               << std::endl;
  }

  Error.noalias() = Error * DiagReg.asDiagonal();
  DfobjE.noalias() = mB_[1].transpose() * Error.transpose();

  return 0;
}

int DPGOProblem::evaluate_g(const Matrix& Z, Matrix& g) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);

  if (loss_ == Loss::None && SIMPLE) {
    g.noalias() = S_ * Z;
  } else {
    Matrix Error = Z.transpose() * mB_[1].transpose();

    Vector DiagReg((d_ + 1) * m_[1]);

    Array mErrNorm = Eigen::Map<Matrix>(Error.data(), (d_ + 1) * d_, m_[1])
                         .colwise()
                         .squaredNorm();

    if (loss_ == Loss::None) {
      DiagReg.setOnes();
    } else if (loss_ == Loss::Huber) {
      Eigen::Map<Matrix> Reg(DiagReg.data(), d_ + 1, m_[1]);
      Reg.row(0) = sqrt_loss_reg_ / mErrNorm.max(loss_reg_).sqrt();
      Reg.rowwise() = Reg.row(0);
    } else if (loss_ == Loss::GemanMcClure) {
      Eigen::Map<Matrix> Reg(DiagReg.data(), d_ + 1, m_[1]);
      Reg.row(0) = squared_loss_reg_ / (mErrNorm + loss_reg_).square();
      Reg.rowwise() = Reg.row(0);
    } else if (loss_ == Loss::Welsch) {
      Eigen::Map<Matrix> Reg(DiagReg.data(), d_ + 1, m_[1]);
      Reg.row(0) = (-mErrNorm / loss_reg_).exp();
      Reg.rowwise() = Reg.row(0);
    } else {
      assert(0 && "Invalid loss kernels for pose graph optimization.");
      LOG(ERROR) << "Invalid loss kernels for pose graph optimization."
                 << std::endl;
    }

    Error.noalias() = Error * DiagReg.asDiagonal();

    g.noalias() = mB_[1].leftCols(size_[0]).transpose() * Error.transpose();
    g.noalias() -= D_ * Z.topRows(size_[0]);
  }

  return 0;
}

int DPGOProblem::evaluate_Df(const Matrix& Z, const Matrix& g,
                             Matrix& Df) const {
  assert(Z.rows() == (d_ + 1) * (n_[0] + n_[1]));
  assert(Z.cols() == d_);
  assert(g.rows() == size_[0]);
  assert(g.cols() == d_);

  Df = g;

#if USE_G
  Df.noalias() += G_ * Z.topRows(size_[0]);
#else
  const auto& t = Z.topRows(n_[0]);
  const auto& R = Z.middleRows(n_[0], d_ * n_[0]);

  Df.topRows(n_[0]).noalias() += mG_[0][0] * t;
  Df.topRows(n_[0]).noalias() += mG_[0][1] * R;
  Df.bottomRows(d_ * n_[0]).noalias() += mG_[1][0] * t;
  Df.bottomRows(d_ * n_[0]).noalias() += mG_[1][1] * R;
#endif

  return 0;
}

int DPGOProblem::update_quadratic_mat(const Vector& DiagReScale) const {
  if (loss_ == Loss::None && SIMPLE) return 0;

  DiagReScale_ = DiagReScale;

  std::copy_n(M_.valuePtr(), G_.nonZeros(), Gval_.data());

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      std::copy_n(mM_[i][j].valuePtr(), mG_[i][j].nonZeros(),
                  mGval_[i][j].data());
    }

  std::copy_n(DiagT0_.diagonal().data(), DiagT_.diagonal().size(),
              Tval_.data());
  std::copy_n(N0_.valuePtr(), N_.nonZeros(), Nval_.data());
  std::copy_n(V0_.valuePtr(), V_.nonZeros(), Vval_.data());
  std::copy_n(D0_.valuePtr(), D_.nonZeros(), Dval_.data());
  std::copy_n(Q0_.valuePtr(), Q_.nonZeros(), Qval_.data());

  static const int outer_start = 0;

  Vector Evals, Fvals;

  Evals.noalias() = E_ * DiagReScale;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      Gval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
          Gval_.size(), 1, inner_nnz_G_[i][j], &outer_start,
          inner_index_G_[i][j].data(), Evals.data() + outer_index_G_[i][j],
          &inner_nnz_G_[i][j]);
    }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      mGval_[i][j] += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
          mGval_[i][j].size(), 1, inner_nnz_mG_[i][j], &outer_start,
          inner_index_mG_[i][j].data(), Evals.data() + outer_index_mG_[i][j],
          &inner_nnz_mG_[i][j]);
    }

  Tval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
      n_[0], 1, inner_nnz_T_, &outer_start, inner_index_T_.data(),
      Evals.data() + outer_index_T_, &inner_nnz_T_);

  Tval_ = Tval_.cwiseInverse();

  Nval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
      Nval_.size(), 1, inner_nnz_N_, &outer_start, inner_index_N_.data(),
      Evals.data() + outer_index_N_, &inner_nnz_N_);

  Eigen::Map<DPGO::Matrix>(ScaleNval_.data(), d_, n_[0]).noalias() =
      Eigen::Map<DPGO::Matrix>(Nval_.data(), d_, n_[0]) * DiagT_;

  Vval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
      Vval_.size(), 1, inner_nnz_V_, &outer_start, inner_index_V_.data(),
      Evals.data() + outer_index_V_, &inner_nnz_V_);

  Eigen::Map<DPGO::Matrix>(Vval_.data(), d_, d_ * n_[0]).noalias() -=
      Eigen::Map<DPGO::Matrix>(Nval_.data(), d_, n_[0]) * ScaleN_;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      Dval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
          Dval_.nonZeros(), 1, inner_nnz_D_[i][j], &outer_start,
          inner_index_D_[i][j].data(), Evals.data() + outer_index_D_[i][j],
          &inner_nnz_D_[i][j]);
    }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      Qval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
          Qval_.nonZeros(), 1, inner_nnz_D_[i][j], &outer_start,
          inner_index_D_[i][j].data(), Evals.data() + outer_index_D_[i][j],
          &inner_nnz_D_[i][j]);
    }

  Fvals.noalias() = F_ * DiagReScale;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      Qval_ += Eigen::Map<const Eigen::SparseMatrix<DPGO::Scalar>>(
          Qval_.nonZeros(), 1, inner_nnz_Q_[i][j], &outer_start,
          inner_index_Q_[i][j].data(), Fvals.data() + outer_index_Q_[i][j],
          &inner_nnz_Q_[i][j]);
    }

  return 0;
}
}  // namespace DPGO
