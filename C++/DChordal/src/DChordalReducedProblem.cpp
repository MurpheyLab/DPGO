#include <DChordal/DChordalReducedProblem.h>

namespace DChordal {
DChordalReducedProblem::DChordalReducedProblem(
    int node, const measurements_t& measurements, Scalar reg_G,
    const Loss& loss, Scalar loss_reg) {
  generate_data_info(node, measurements, intra_measurements_,
                     inter_measurements_, nn_, n_, s_, m_, index_, sent_, recv_,
                     n_index_, n_sent_, n_recv_);

  const auto& d = measurements[0].t.size();

  assert(reg_G >= 0 && loss_reg > 0);

  if (reg_G < 0 || loss_reg <= 0) {
    LOG(ERROR) << "Inconsistent regularizers for node " << node << std::endl;
    reset();

    exit(-1);
  }

  node_ = node;
  reg_G_ = reg_G;
  loss_ = loss;
  loss_reg_ = loss_reg;
  d_ = d;
  squared_loss_reg_ = loss_reg * loss_reg;
  sqrt_loss_reg_ = std::sqrt(loss_reg);
  ready_ = false;
}

int DChordalReducedProblem::reset() const {
  G_.resize(0, 0);
  S_.resize(0, 0);
  g_.resize(0, 0);
  B_.resize(0, 0);
  b_.resize(0, 0);

  ready_ = false;

  return 0;
}

DChordalReducedProblem_R::DChordalReducedProblem_R(
    int node, const measurements_t& measurements, Scalar reg_G,
    const Loss& loss, Scalar loss_reg)
    : DChordalReducedProblem(node, measurements, reg_G, loss, loss_reg) {
  p_ = d_;
}

int DChordalReducedProblem_R::setup(const Matrix& X) const {
  assert(X.rows() == (n_[0] + n_[1]) * (d_ + 1));
  assert(X.cols() == d_);

  if (X.rows() != (n_[0] + n_[1]) * (d_ + 1) || X.cols() != d_) {
    LOG(ERROR) << "Inconsistent poses for node " << node_ << std::endl;
  }

  R_.setZero((n_[0] + n_[1]) * d_, d_);
  R_.topRows(d_ * n_[0]) = X.middleRows((d_ + 1) * s_[0] + n_[0], d_ * n_[0]);
  R_.bottomRows(d_ * n_[1]) =
      X.middleRows((d_ + 1) * s_[1] + n_[1], d_ * n_[1]);

  if (simplify_data_matrix_reduced_R(node_, inter_measurements_, X, n_, s_,
                                     index_, n_index_, reg_G_, G_, S_, D_, H_,
                                     g_, B_, b_)) {
    reset();

    return -1;
  }

  Ginv_ = G_.inverse();
  Dinv_ = D_.inverse();

  ready_ = true;

  return 0;
}

DChordalReducedProblem_t::DChordalReducedProblem_t(
    int node, const measurements_t& measurements, Scalar reg_G,
    const Loss& loss, Scalar loss_reg)
    : DChordalReducedProblem(node, measurements, reg_G, loss, loss_reg) {
  precompute_data_matrix_recover_t(node, intra_measurements_, n_[0], L_, P_);

  LL_.compute(L_);

  p_ = 1;
}

int DChordalReducedProblem_t::setup(const Matrix& X, const Matrix& nR) const {
  assert(X.rows() == (n_[0] + n_[1]) * (d_ + 1));
  assert(nR.rows() == (nn_ + 1) * d_);
  assert(X.cols() == d_);
  assert(nR.cols() == d_);

  if (X.rows() != (n_[0] + n_[1]) * (d_ + 1) || nR.rows() != (nn_ + 1) * d_ ||
      X.cols() != d_ || nR.cols() != d_) {
    LOG(ERROR) << "Inconsistent poses for node " << node_ << std::endl;
  }

  X_ = X;
  nR_ = nR;

  if (simplify_data_matrix_reduced_t(node_, inter_measurements_, X_, nR, n_, s_,
                                     index_, n_index_, reg_G_, G_, S_, D_, H_,
                                     g_, B_, b_)) {
    reset();

    exit(-1);
  }

  ready_ = true;

  return 0;
}
}  // namespace DChordal
