#include <DChordal/DChordalProblem.h>
#include "DPGO/DPGO_utils.h"

namespace DChordal {
DChordalProblem::DChordalProblem(int node, const measurements_t& measurements,
                                 Scalar reg_G, Loss loss, Scalar loss_reg)
    : node_(node),
      reg_G_(reg_G),
      loss_(loss),
      loss_reg_(loss_reg),
      squared_loss_reg_(loss_reg * loss_reg),
      sqrt_loss_reg_(std::sqrt(loss_reg)),
      ready_(false) {
  assert(reg_G >= 0 && loss_reg > 0);

  d_ = (!measurements.empty() ? measurements[0].t.size() : 0);

  generate_data_info(node_, measurements, intra_measurements_,
                     inter_measurements_, n_, s_, m_, index_, sent_, recv_);

  if (reg_G < 0 || loss_reg <= 0) {
    LOG(ERROR) << "Inconsistent regularizers for node " << node << std::endl;
    reset();

    exit(-1);
  }
}

int DChordalProblem::reset() const {
  G_.resize(0, 0);
  S_.resize(0, 0);
  g_.resize(0, 0);
  B_.resize(0, 0);
  mB_[0].resize(0, 0);
  mB_[1].resize(0, 0);
  b_.resize(0, 0);
  mb_[0].resize(0, 0);
  mb_[1].resize(0, 0);
  ready_ = false;

  return 0;
}

DChordalProblem_R::DChordalProblem_R(int node,
                                     const measurements_t& measurements,
                                     Scalar reg_G, Loss loss, Scalar loss_reg)
    : DChordalProblem(node, measurements, reg_G, loss, loss_reg) {
  p_ = d_;
}

int DChordalProblem_R::setup() const {
  if (simplify_data_matrix_R(node_, intra_measurements_, inter_measurements_,
                             n_, s_, m_, index_, reg_G_, G_, S_, D_, H_, g_, B_,
                             b_, mB_[0], mb_[0], mB_[1], mb_[1])) {
    reset();

    return -1;
  }

  if (node_ == 0) {
    g_ = G_.bottomLeftCorner((n_[0] - 1) * d_, d_);
    LG_.compute(G_.bottomRightCorner((n_[0] - 1) * d_, (n_[0] - 1) * d_));
  } else {
    LG_.compute(G_);
  }

  ready_ = true;

  return 0;
}

DChordalProblem_t::DChordalProblem_t(int node,
                                     const measurements_t& measurements,
                                     Scalar reg_G, Loss loss, Scalar loss_reg)
    : DChordalProblem(node, measurements, reg_G, loss, loss_reg) {
  p_ = 1;
}

int DChordalProblem_t::setup(const Matrix& R) const {
  assert(R.rows() == (n_[0] + n_[1]) * d_);
  assert(R.cols() == d_);

  if (R.rows() != (n_[0] + n_[1]) * d_ || R.cols() != d_) {
    LOG(ERROR) << "Inconsistent poses for node " << node_ << std::endl;

    return -1;
  }

  R_ = R;

  if (simplify_data_matrix_t(node_, intra_measurements_, inter_measurements_,
                             R_, n_, s_, m_, index_, reg_G_, G_, S_, D_, H_, g_,
                             B_, b_, mB_[0], mb_[0], mB_[1], mb_[1])) {
    reset();

    return -1;
  }

  LG_.compute(G_);

  ready_ = true;

  return 0;
}
}  // namespace DChordal
