#pragma once

#include <x86intrin.h>

namespace DPGO {
namespace internal {

template <typename Scalar>
struct traits {};

template <>
struct traits<double> {
  static const __m256d szero;
  static const __m256d sone;
  static const __m256d smtwo;
  static const __m256d sone_half;
  static const __m256d stiny_number;
  static const __m256d ssmall_number;
  static const __m256d ssine_pi_over_eight;
  static const __m256d scosine_pi_over_eight;
  static const __m256d sfour_gamma_squared;
};

template <>
struct traits<float> {
  static const __m256 szero;
  static const __m256 sone;
  static const __m256 smtwo;
  static const __m256 sone_half;
  static const __m256 stiny_number;
  static const __m256 ssmall_number;
  static const __m256 ssine_pi_over_eight;
  static const __m256 scosine_pi_over_eight;
  static const __m256 sfour_gamma_squared;
};
}  // namespace internal
}  // namespace DPGO
