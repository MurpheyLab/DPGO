#include <Eigen/Dense>
#include <DPGO/internal/project_to_SOd.h>
#include <DPGO/internal/project_to_SO2.h>
#include <DPGO/internal/project_to_SO3.h>
#include <DPGO/internal/traits.h>

namespace DPGO {
namespace internal {
#define T __m256d
#define Sone traits<double>::sone
#define Szero traits<double>::szero
#define Sone_half traits<double>::sone_half
#define Stiny_number traits<double>::stiny_number
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
void project_to_SO2(const T& a11, const T& a12, const T& a21, const T& a22,
                    T& u11, T& u21) {
  T Sc, Ss;
  T Stmp1, Stmp2;

  PROJECT_TO_SO2
}

#undef T
#undef Sone
#undef Szero
#undef Sone_half
#undef Stiny_number
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
#define Szero traits<float>::szero
#define Sone_half traits<float>::sone_half
#define Stiny_number traits<float>::stiny_number
#define add(a, b) _mm256_add_ps(a, b)
#define sub(a, b) _mm256_sub_ps(a, b)
#define mul(a, b) _mm256_mul_ps(a, b)
#define fma(a, b, c) _mm256_fmadd_ps(a, b, c)
#define fms(a, b, c) _mm256_fmsub_ps(a, b, c)
#define rsqrt(a) _mm256_div_ps(Sone, _mm256_sqrt_ps(a))
//#define rsqrt(a) _mm256_rsqrt_ps(a)
#define cmp(a, b, op) _mm256_cmp_ps(a, b, op)
#define max(a, b) _mm256_max_ps(a, b)
#define sand(a, b) _mm256_and_ps(a, b)
#define sxor(a, b) _mm256_xor_ps(a, b)
#define blend(a, b, c) _mm256_blendv_ps(a, b, c)
#define set(a) _mm256_set1_ps(a)
void project_to_SO2(const T& a11, const T& a12, const T& a21, const T& a22,
                    T& u11, T& u21) {
  T Sc, Ss;
  T Stmp1, Stmp2;

  PROJECT_TO_SO2
}

#undef T
#undef Sone
#undef Szero
#undef Sone_half
#undef Stiny_number
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

void project_to_SO3(const T& a11, const T& a12, const T& a13, const T& a21,
                    const T& a22, const T& a23, const T& a31, const T& a32,
                    const T& a33,  // input A
                    T& u11, T& u12, T& u13, T& u21, T& u22, T& u23, T& u31,
                    T& u32, T& u33  // output U
) {
  T Sa11, Sa12, Sa13, Sa21, Sa22, Sa23, Sa31, Sa32, Sa33;
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

  T Su11, Su21, Su31, Su12, Su22, Su32, Su13, Su23, Su33;
  T Sv11, Sv21, Sv31, Sv12, Sv22, Sv32, Sv13, Sv23, Sv33;

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

  PROJECT_TO_SO3_COMPUTE_U
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

void project_to_SO3(const T& a11, const T& a12, const T& a13, const T& a21,
                    const T& a22, const T& a23, const T& a31, const T& a32,
                    const T& a33,  // input A
                    T& u11, T& u12, T& u13, T& u21, T& u22, T& u23, T& u31,
                    T& u32, T& u33  // output U
) {
  T Sa11, Sa12, Sa13, Sa21, Sa22, Sa23, Sa31, Sa32, Sa33;
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

  T Su11, Su21, Su31, Su12, Su22, Su32, Su13, Su23, Su33;
  T Sv11, Sv21, Sv31, Sv12, Sv22, Sv32, Sv13, Sv23, Sv33;

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

  PROJECT_TO_SO3_COMPUTE_U
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
}  // namespace internal
}  // namespace DPGO
