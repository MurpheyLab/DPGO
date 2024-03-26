#pragma once

#include <Eigen/Dense>
#include <DPGO/internal/svd3x3.h>
#include <DPGO/internal/traits.h>

namespace DPGO {
namespace internal {
#define T __m256d
#define Sone traits<double>::sone
#define Smtwo traits<double>::smtwo
#define Szero traits<double>::szero
#define Ssine_pi_over_eight traits<double>::ssine_pi_over_eight
#define Scosine_pi_over_eight traits<double>::scosine_pi_over_eight
#define Sone_half traits<double>::sone_half
#define Ssmall_number traits<double>::ssmall_number
#define Stiny_number traits<double>::stiny_number
#define Sfour_gamma_squared traits<double>::sfour_gamma_squared
#define add(a, b) _mm256_add_pd(a, b)
#define sub(a, b) _mm256_sub_pd(a, b)
#define mul(a, b) _mm256_mul_pd(a, b)
#define fma(a, b, c) _mm256_fmadd_pd(a, b, c)
#define fms(a, b, c) _mm256_fmsub_pd(a, b, c)
//#define rsqrt(a) _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(a)))
#define rsqrt(a) _mm256_div_pd(Sone, _mm256_sqrt_pd(a))
#define cmp(a, b, op) _mm256_cmp_pd(a, b, op)
#define max(a, b) _mm256_max_pd(a, b)
#define sand(a, b) _mm256_and_pd(a, b)
#define sxor(a, b) _mm256_xor_pd(a, b)
#define blend(a, b, c) _mm256_blendv_pd(a, b, c)
#define set(a) _mm256_set1_pd(a)

void svd(const T& a11, const T& a12, const T& a13, const T& a21, const T& a22,
         const T& a23, const T& a31, const T& a32, const T& a33,  // input A
         T& u11, T& u12, T& u13, T& u21, T& u22, T& u23, T& u31, T& u32,
         T& u33,                  // output U
         T& s11, T& s22, T& s33,  // ostpst S
         T& v11, T& v12, T& v13, T& v21, T& v22, T& v23, T& v31, T& v32,
         T& v33  // ovtpvt V
) {
  T& Sa11 = s11;
  T& Sa22 = s22;
  T& Sa33 = s33;
  T Sa12, Sa13, Sa21, Sa23, Sa31, Sa32;
  T Sc, Ss, Sch, Ssh;
  T Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
  T Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
  T Sqvs, Sqvx, Sqvy, Sqvz;

  Sa11 = a11;
  Sa12 = a12;
  Sa13 = a13;
  Sa21 = a21;
  Sa22 = a22;
  Sa23 = a23;
  Sa31 = a31;
  Sa32 = a32;
  Sa33 = a33;

  T& Su11 = u11;
  T& Su21 = u21;
  T& Su31 = u31;
  T& Su12 = u12;
  T& Su22 = u22;
  T& Su32 = u32;
  T& Su13 = u13;
  T& Su23 = u23;
  T& Su33 = u33;

  T& Sv11 = v11;
  T& Sv21 = v21;
  T& Sv31 = v31;
  T& Sv12 = v12;
  T& Sv22 = v22;
  T& Sv32 = v32;
  T& Sv13 = v13;
  T& Sv23 = v23;
  T& Sv33 = v33;

  SVD3X3_COMPUTE_ATA

  Sqvs = Sone;
  Sqvx = Szero;
  Sqvy = Szero;
  Sqvz = Szero;

  for (int i = 0; i < 8; i++) {
    // First Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss11, Ss21, Ss31, Ss22, Ss32, Ss33, Sqvx, Sqvy,
                             Sqvz, Stmp1, Stmp2, Stmp3)

    // Second Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss22, Ss32, Ss21, Ss33, Ss31, Ss11, Sqvy, Sqvz,
                             Sqvx, Stmp2, Stmp3, Stmp1)

    // Third Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss33, Ss31, Ss32, Ss11, Ss21, Ss22, Sqvz, Sqvx,
                             Sqvy, Stmp3, Stmp1, Stmp2)
  }

  SVD3X3_COMPUTE_MATRIX_V

  SVD3X3_MULTIPLY_WITH_V

  SVD3X3_SORT_SINGULAR_VALUES

  Su11 = Sone;
  Su12 = Szero;
  Su13 = Szero;
  Su21 = Szero;
  Su22 = Sone;
  Su23 = Szero;
  Su31 = Szero;
  Su32 = Szero;
  Su33 = Sone;

  // First Givens rotation
  SVD3X3_QR(Sa11, Sa21, Sa11, Sa21, Sa12, Sa22, Sa13, Sa23, Su11, Su12, Su21,
            Su22, Su31, Su32)

  // Second Givens rotation
  SVD3X3_QR(Sa11, Sa31, Sa11, Sa31, Sa12, Sa32, Sa13, Sa33, Su11, Su13, Su21,
            Su23, Su31, Su33)

  // Third Givens Rotation
  SVD3X3_QR(Sa22, Sa32, Sa21, Sa31, Sa22, Sa32, Sa23, Sa33, Su12, Su13, Su22,
            Su23, Su32, Su33)
}

void svd(const Eigen::Matrix<double, 12, 3>& A, Eigen::Matrix<double, 12, 3>& U,
         Eigen::Matrix<double, 12, 1>& S, Eigen::Matrix<double, 12, 3>& V) {
  double temp[2][9][4];

  for (int k = 0; k < 4; k++) {
    int src = 3 * k;
    for (int i = 0; i < 3; i++) {
      int des = 3 * i;
      for (int j = 0; j < 3; j++) {
        temp[0][des + j][k] = A(src + i, j);
      }
    }
  }

  T a[9], u[9], s[3], v[9];

  for (int i = 0; i < 9; i++) {
    a[i] = _mm256_loadu_pd(temp[0][i]);
  }

  svd(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],   // A
      u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],   // U
      s[0], s[1], s[2],                                       // S
      v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);  // V

  for (int i = 0; i < 9; i++) {
    _mm256_storeu_pd(temp[0][i], u[i]);
    _mm256_storeu_pd(temp[1][i], v[i]);
  }

  for (int k = 0; k < 4; k++) {
    int des = 3 * k;
    for (int i = 0; i < 3; i++) {
      int src = 3 * i;
      for (int j = 0; j < 3; j++) {
        U(des + i, j) = temp[0][src + j][k];
        V(des + i, j) = temp[1][src + j][k];
      }
    }
  }

  for (int k = 0; k < 4; k++) {
    int des = 3 * k;
    for (int j = 0; j < 3; j++) {
      S[des + j] = s[j][k];
    }
  }
}

#undef T
#undef Sone
#undef Smtwo
#undef Szero
#undef Ssine_pi_over_eight
#undef Scosine_pi_over_eight
#undef Sone_half
#undef Ssmall_number
#undef Stiny_number
#undef Sfour_gamma_squared
#undef add
#undef sub
#undef mul
#undef fma
#undef fms
#undef rsqrt
#undef cmp
#undef max
#undef sand
#undef sxor
#undef blend
#undef set

#define T __m256
#define Sone traits<float>::sone
#define Smtwo traits<float>::smtwo
#define Szero traits<float>::szero
#define Ssine_pi_over_eight traits<float>::ssine_pi_over_eight
#define Scosine_pi_over_eight traits<float>::scosine_pi_over_eight
#define Sone_half traits<float>::sone_half
#define Ssmall_number traits<float>::ssmall_number
#define Stiny_number traits<float>::stiny_number
#define Sfour_gamma_squared traits<float>::sfour_gamma_squared
#define add(a, b) _mm256_add_ps(a, b)
#define sub(a, b) _mm256_sub_ps(a, b)
#define mul(a, b) _mm256_mul_ps(a, b)
#define fma(a, b, c) _mm256_fmadd_ps(a, b, c)
#define fms(a, b, c) _mm256_fmsub_ps(a, b, c)
#define rsqrt(a) _mm256_rsqrt_ps(a)
#define cmp(a, b, op) _mm256_cmp_ps(a, b, op)
#define max(a, b) _mm256_max_ps(a, b)
#define sand(a, b) _mm256_and_ps(a, b)
#define sxor(a, b) _mm256_xor_ps(a, b)
#define blend(a, b, c) _mm256_blendv_ps(a, b, c)
#define set(a) _mm256_set1_ps(a)

void svd(const T& a11, const T& a12, const T& a13, const T& a21, const T& a22,
         const T& a23, const T& a31, const T& a32, const T& a33,  // input A
         T& u11, T& u12, T& u13, T& u21, T& u22, T& u23, T& u31, T& u32,
         T& u33,                  // output U
         T& s11, T& s22, T& s33,  // ostpst S
         T& v11, T& v12, T& v13, T& v21, T& v22, T& v23, T& v31, T& v32,
         T& v33  // ovtpvt V
) {
  T& Sa11 = s11;
  T& Sa22 = s22;
  T& Sa33 = s33;
  T Sa12, Sa13, Sa21, Sa23, Sa31, Sa32;
  T Sc, Ss, Sch, Ssh;
  T Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
  T Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
  T Sqvs, Sqvx, Sqvy, Sqvz;

  Sa11 = a11;
  Sa12 = a12;
  Sa13 = a13;
  Sa21 = a21;
  Sa22 = a22;
  Sa23 = a23;
  Sa31 = a31;
  Sa32 = a32;
  Sa33 = a33;

  T& Su11 = u11;
  T& Su21 = u21;
  T& Su31 = u31;
  T& Su12 = u12;
  T& Su22 = u22;
  T& Su32 = u32;
  T& Su13 = u13;
  T& Su23 = u23;
  T& Su33 = u33;

  T& Sv11 = v11;
  T& Sv21 = v21;
  T& Sv31 = v31;
  T& Sv12 = v12;
  T& Sv22 = v22;
  T& Sv32 = v32;
  T& Sv13 = v13;
  T& Sv23 = v23;
  T& Sv33 = v33;

  // SVD3X3_COMPUTE_ATA

  Ss11 = mul(Sa11, Sa11);
  Ss11 = fma(Sa21, Sa21, Ss11);
  Ss11 = fma(Sa31, Sa31, Ss11);

  Ss21 = mul(Sa12, Sa11);
  Ss21 = fma(Sa22, Sa21, Ss21);
  Ss21 = fma(Sa32, Sa31, Ss21);

  Ss31 = mul(Sa13, Sa11);
  Ss31 = fma(Sa23, Sa21, Ss31);
  Ss31 = fma(Sa33, Sa31, Ss31);

  Ss22 = mul(Sa12, Sa12);
  Ss22 = fma(Sa22, Sa22, Ss22);
  Ss22 = fma(Sa32, Sa32, Ss22);

  Ss32 = mul(Sa13, Sa12);
  Ss32 = fma(Sa23, Sa22, Ss32);
  Ss32 = fma(Sa33, Sa32, Ss32);

  Ss33 = mul(Sa13, Sa13);
  Ss33 = fma(Sa23, Sa23, Ss33);
  Ss33 = fma(Sa33, Sa33, Ss33);

  Sqvs = Sone;
  Sqvx = Szero;
  Sqvy = Szero;
  Sqvz = Szero;

  for (int i = 0; i < 8; i++) {
    // First Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss11, Ss21, Ss31, Ss22, Ss32, Ss33, Sqvx, Sqvy,
                             Sqvz, Stmp1, Stmp2, Stmp3)

    // Second Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss22, Ss32, Ss21, Ss33, Ss31, Ss11, Sqvy, Sqvz,
                             Sqvx, Stmp2, Stmp3, Stmp1)

    // Third Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss33, Ss31, Ss32, Ss11, Ss21, Ss22, Sqvz, Sqvx,
                             Sqvy, Stmp3, Stmp1, Stmp2)
  }

  SVD3X3_ACCURATE_COMPUTE_MATRIX_V

  SVD3X3_MULTIPLY_WITH_V

  SVD3X3_SORT_SINGULAR_VALUES

  Su11 = Sone;
  Su12 = Szero;
  Su13 = Szero;
  Su21 = Szero;
  Su22 = Sone;
  Su23 = Szero;
  Su31 = Szero;
  Su32 = Szero;
  Su33 = Sone;

  // First Givens rotation
  SVD3X3_ACCURATE_QR(Sa11, Sa21, Sa11, Sa21, Sa12, Sa22, Sa13, Sa23, Su11, Su12,
                     Su21, Su22, Su31, Su32)

  // Second Givens rotation
  SVD3X3_ACCURATE_QR(Sa11, Sa31, Sa11, Sa31, Sa12, Sa32, Sa13, Sa33, Su11, Su13,
                     Su21, Su23, Su31, Su33)

  // Third Givens Rotation
  SVD3X3_ACCURATE_QR(Sa22, Sa32, Sa21, Sa31, Sa22, Sa32, Sa23, Sa33, Su12, Su13,
                     Su22, Su23, Su32, Su33)
}

void svd(const Eigen::Matrix<double, 24, 3>& A, Eigen::Matrix<double, 24, 3>& U,
         Eigen::Matrix<double, 24, 1>& S, Eigen::Matrix<double, 24, 3>& V) {
  float temp[2][9][8];

  for (int k = 0; k < 8; k++) {
    int src = 3 * k;
    for (int i = 0; i < 3; i++) {
      int des = 3 * i;
      for (int j = 0; j < 3; j++) {
        temp[0][des + j][k] = A(src + i, j);
      }
    }
  }

  T a[9], u[9], s[3], v[9];

  for (int i = 0; i < 9; i++) {
    a[i] = _mm256_loadu_ps(temp[0][i]);
  }

  svd(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],   // A
      u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],   // U
      s[0], s[1], s[2],                                       // S
      v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);  // V

  for (int i = 0; i < 9; i++) {
    _mm256_storeu_ps(temp[0][i], u[i]);
    _mm256_storeu_ps(temp[1][i], v[i]);
  }

  for (int k = 0; k < 8; k++) {
    int des = 3 * k;
    for (int i = 0; i < 3; i++) {
      int src = 3 * i;
      for (int j = 0; j < 3; j++) {
        U(des + i, j) = temp[0][src + j][k];
        V(des + i, j) = temp[1][src + j][k];
      }
    }
  }

  for (int k = 0; k < 8; k++) {
    int des = 3 * k;
    for (int j = 0; j < 3; j++) {
      S[des + j] = s[j][k];
    }
  }
}

void svd(const Eigen::Matrix<float, 24, 3>& A, Eigen::Matrix<float, 24, 3>& U,
         Eigen::Matrix<float, 24, 1>& S, Eigen::Matrix<float, 24, 3>& V) {
  float temp[2][9][8];

  for (int k = 0; k < 8; k++) {
    int src = 3 * k;
    for (int i = 0; i < 3; i++) {
      int des = 3 * i;
      for (int j = 0; j < 3; j++) {
        temp[0][des + j][k] = A(src + i, j);
      }
    }
  }

  T a[9], u[9], s[3], v[9];

  for (int i = 0; i < 9; i++) {
    a[i] = _mm256_loadu_ps(temp[0][i]);
  }

  svd(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],   // A
      u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],   // U
      s[0], s[1], s[2],                                       // S
      v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);  // V

  for (int i = 0; i < 9; i++) {
    _mm256_storeu_ps(temp[0][i], u[i]);
    _mm256_storeu_ps(temp[1][i], v[i]);
  }

  for (int k = 0; k < 8; k++) {
    int des = 3 * k;
    for (int i = 0; i < 3; i++) {
      int src = 3 * i;
      for (int j = 0; j < 3; j++) {
        U(des + i, j) = temp[0][src + j][k];
        V(des + i, j) = temp[1][src + j][k];
      }
    }
  }

  for (int k = 0; k < 8; k++) {
    int des = 3 * k;
    for (int j = 0; j < 3; j++) {
      S[des + j] = s[j][k];
    }
  }
}

#undef T
#undef Sone
#undef Smtwo
#undef Szero
#undef Ssine_pi_over_eight
#undef Scosine_pi_over_eight
#undef Sone_half
#undef Ssmall_number
#undef Stiny_number
#undef Sfour_gamma_squared
#undef add
#undef sub
#undef mul
#undef fmadd
#undef fmsub
#undef rsqrt
#undef cmp
#undef max
#undef sand
#undef sxor
#undef blend
#undef set
}  // namespace internal
}  // namespace DPGO
