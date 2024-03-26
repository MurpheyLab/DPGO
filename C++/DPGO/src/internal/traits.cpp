#include <DPGO/internal/traits.h>
#include <cmath>

namespace DPGO {
namespace internal {
const __m256d traits<double>::szero = _mm256_set1_pd(0.0);
const __m256d traits<double>::sone = _mm256_set1_pd(1.0);
const __m256d traits<double>::smtwo = _mm256_set1_pd(-2.0);
const __m256d traits<double>::sone_half = _mm256_set1_pd(0.5);
const __m256d traits<double>::stiny_number = _mm256_set1_pd(1.0e-32);
const __m256d traits<double>::ssmall_number = _mm256_set1_pd(1.0e-16);
const __m256d traits<double>::ssine_pi_over_eight =
    _mm256_set1_pd(0.5 * sqrt(2.0 - sqrt(2.0)));
const __m256d traits<double>::scosine_pi_over_eight =
    _mm256_set1_pd(0.5 * sqrt(2.0 + sqrt(2.0)));
const __m256d traits<double>::sfour_gamma_squared =
    _mm256_set1_pd(sqrt(8.0) + 3.0);

const __m256 traits<float>::szero = _mm256_set1_ps(0.0);
const __m256 traits<float>::sone = _mm256_set1_ps(1.0);
const __m256 traits<float>::smtwo = _mm256_set1_ps(-2.0);
const __m256 traits<float>::sone_half = _mm256_set1_ps(0.5);
const __m256 traits<float>::stiny_number = _mm256_set1_ps(1.0e-20);
const __m256 traits<float>::ssmall_number = _mm256_set1_ps(1.0e-12);
const __m256 traits<float>::ssine_pi_over_eight =
    _mm256_set1_ps(0.5 * sqrt(2.0 - sqrt(2.0)));
const __m256 traits<float>::scosine_pi_over_eight =
    _mm256_set1_ps(0.5 * sqrt(2.0 + sqrt(2.0)));
const __m256 traits<float>::sfour_gamma_squared =
    _mm256_set1_ps(sqrt(8.0) + 3.0);
}  // namespace internal
}  // namespace DPGO
