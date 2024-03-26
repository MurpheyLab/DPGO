#pragma once

#include <Eigen/Dense>
#include <DPGO/internal/traits.h>

namespace DPGO {
namespace internal {
#define T __m256d
void project_to_SO2(const T& a11, const T& a12, const T& a21, const T& a22,
                    T& u11, T& u21);

template <typename Derived, typename Result>
void project_to_SO2_d(const Derived& A, Result& U) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 8, 2)
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Result, 8, 2)

  double temp[4][4];

  for (int k = 0; k < 4; k++) {
    int src = 2 * k;

    for (int i = 0; i < 2; i++) {
      int des = 2 * i;

      for (int j = 0; j < 2; j++) {
        temp[des + j][k] = A(src + i, j);
      }
    }
  }

  T a[4], u[2];

  for (int i = 0; i < 4; i++) {
    a[i] = _mm256_loadu_pd(temp[i]);
  }

  project_to_SO2(a[0], a[1], a[2], a[3], /* input A */
                 u[0], u[1] /* output U */);

  for (int i = 0; i < 2; i++) {
    _mm256_storeu_pd(temp[i], u[i]);
  }

  for (int k = 0, des = 0; k < 4; k++) {
    U(des, 0) = temp[0][k];
    U(des, 1) = -temp[1][k];

    des++;
    U(des, 0) = temp[1][k];
    U(des, 1) = temp[0][k];

    des++;
  }
}

void project_to_SO3(const T& a11, const T& a12, const T& a13, const T& a21,
                    const T& a22, const T& a23, const T& a31, const T& a32,
                    const T& a33,  // input A
                    T& u11, T& u12, T& u13, T& u21, T& u22, T& u23, T& u31,
                    T& u32, T& u33);  // output U

template <typename Derived, typename Result>
void project_to_SO3_d(const Derived& A, Result& U) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 12, 3)
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Result, 12, 3)

  double temp[9][4];

  for (int k = 0; k < 4; k++) {
    int src = 3 * k;
    for (int i = 0; i < 3; i++) {
      int des = 3 * i;
      for (int j = 0; j < 3; j++) {
        temp[des + j][k] = A(src + i, j);
      }
    }
  }

  T a[9], u[9];

  for (int i = 0; i < 9; i++) {
    a[i] = _mm256_loadu_pd(temp[i]);
  }

  project_to_SO3(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],   // A
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8]);  // U

  for (int i = 0; i < 9; i++) {
    _mm256_storeu_pd(temp[i], u[i]);
  }

  for (int k = 0; k < 4; k++) {
    int des = 3 * k;
    for (int i = 0; i < 3; i++) {
      int src = 3 * i;
      for (int j = 0; j < 3; j++) {
        U(des + i, j) = temp[src + j][k];
      }
    }
  }
}
#undef T

#define T __m256
void project_to_SO2(const T& a11, const T& a12, const T& a21, const T& a22,
                    T& u11, T& u21);

template <typename Derived, typename Result>
void project_to_SO2_s(const Derived& A, Result& U) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 16, 2)
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Result, 16, 2)

  float temp[4][8];

  for (int k = 0; k < 8; k++) {
    int src = 2 * k;

    for (int i = 0; i < 2; i++) {
      int des = 2 * i;

      for (int j = 0; j < 2; j++) {
        temp[des + j][k] = A(src + i, j);
      }
    }
  }

  T a[4], u[2];

  for (int i = 0; i < 4; i++) {
    a[i] = _mm256_loadu_ps(temp[i]);
  }

  project_to_SO2(a[0], a[1], a[2], a[3], /* input A */
                 u[0], u[1] /* output U */);

  for (int i = 0; i < 2; i++) {
    _mm256_storeu_ps(temp[i], u[i]);
  }

  for (int k = 0, des = 0; k < 8; k++) {
    U(des, 0) = temp[0][k];
    U(des, 1) = -temp[1][k];

    des++;
    U(des, 0) = temp[1][k];
    U(des, 1) = temp[0][k];

    des++;
  }
}

void project_to_SO3(const T& a11, const T& a12, const T& a13, const T& a21,
                    const T& a22, const T& a23, const T& a31, const T& a32,
                    const T& a33,  // input A
                    T& u11, T& u12, T& u13, T& u21, T& u22, T& u23, T& u31,
                    T& u32, T& u33);  // output U

template <typename Derived, typename Result>
void project_to_SO3_s(const Derived& A, Result& U) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 24, 3)
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Result, 24, 3)

  float temp[9][8];

  for (int k = 0; k < 8; k++) {
    int src = 3 * k;
    for (int i = 0; i < 3; i++) {
      int des = 3 * i;
      for (int j = 0; j < 3; j++) {
        temp[des + j][k] = A(src + i, j);
      }
    }
  }

  T a[9], u[9];

  for (int i = 0; i < 9; i++) {
    a[i] = _mm256_loadu_ps(temp[i]);
  }

  project_to_SO3(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],   // A
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8]);  // U

  for (int i = 0; i < 9; i++) {
    _mm256_storeu_ps(temp[i], u[i]);
  }

  for (int k = 0; k < 8; k++) {
    int des = 3 * k;
    for (int i = 0; i < 3; i++) {
      int src = 3 * i;
      for (int j = 0; j < 3; j++) {
        U(des + i, j) = temp[src + j][k];
      }
    }
  }
}
#undef T
}  // namespace internal
}  // namespace DPGO
