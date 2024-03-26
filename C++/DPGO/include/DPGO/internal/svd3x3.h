#pragma once
// compute A^T*A
#define SVD3X3_COMPUTE_ATA      \
  Ss11 = mul(Sa11, Sa11);       \
  Ss11 = fma(Sa21, Sa21, Ss11); \
  Ss11 = fma(Sa31, Sa31, Ss11); \
                                \
  Ss21 = mul(Sa12, Sa11);       \
  Ss21 = fma(Sa22, Sa21, Ss21); \
  Ss21 = fma(Sa32, Sa31, Ss21); \
                                \
  Ss31 = mul(Sa13, Sa11);       \
  Ss31 = fma(Sa23, Sa21, Ss31); \
  Ss31 = fma(Sa33, Sa31, Ss31); \
                                \
  Ss22 = mul(Sa12, Sa12);       \
  Ss22 = fma(Sa22, Sa22, Ss22); \
  Ss22 = fma(Sa32, Sa32, Ss22); \
                                \
  Ss32 = mul(Sa13, Sa12);       \
  Ss32 = fma(Sa23, Sa22, Ss32); \
  Ss32 = fma(Sa33, Sa32, Ss32); \
                                \
  Ss33 = mul(Sa13, Sa13);       \
  Ss33 = fma(Sa23, Sa23, Ss33); \
  Ss33 = fma(Sa33, Sa33, Ss33);

#define SVD3X3_JACOBI_CONJUATION(SS11, SS21, SS31, SS22, SS32, SS33, SQVX, \
                                 SQVY, SQVZ, STMP1, STMP2, STMP3)          \
  Ssh = mul(SS21, Sone_half);                                              \
  Stmp5 = sub(SS11, SS22);                                                 \
                                                                           \
  Stmp2 = mul(Ssh, Ssh);                                                   \
  Stmp1 = cmp(Stmp2, Stiny_number, _CMP_GE_OS);                            \
                                                                           \
  Ssh = sand(Stmp1, Ssh);                                                  \
  Sch = blend(Sone, Stmp5, Stmp1);                                         \
                                                                           \
  Stmp1 = mul(Ssh, Ssh);                                                   \
  Stmp2 = mul(Sch, Sch);                                                   \
  Stmp3 = add(Stmp1, Stmp2);                                               \
  Stmp4 = rsqrt(Stmp3);                                                    \
                                                                           \
  /* Ss = mul(Stmp4, Sone_half); */                                        \
  /* Sc = mul(Stmp4, Ss);        */                                        \
  /* Sc = mul(Stmp4, Sc);        */                                        \
  /* Sc = mul(Stmp3, Sc);        */                                        \
  /* Stmp4 = add(Stmp4, Ss);     */                                        \
  /* Stmp4 = sub(Stmp4, Sc);     */                                        \
                                                                           \
  Ssh = mul(Stmp4, Ssh);                                                   \
  Sch = mul(Stmp4, Sch);                                                   \
  Stmp1 = mul(Sfour_gamma_squared, Stmp1);                                 \
  Stmp1 = cmp(Stmp2, Stmp1, _CMP_LE_OS);                                   \
                                                                           \
  Ssh = blend(Ssh, Ssine_pi_over_eight, Stmp1);                            \
  Sch = blend(Sch, Scosine_pi_over_eight, Stmp1);                          \
                                                                           \
  Stmp1 = mul(Ssh, Ssh);                                                   \
  Stmp2 = mul(Sch, Sch);                                                   \
  Sc = sub(Stmp2, Stmp1);                                                  \
  Ss = mul(Sch, Ssh);                                                      \
  Ss = add(Ss, Ss);                                                        \
                                                                           \
  Stmp3 = add(Stmp1, Stmp2);                                               \
  SS33 = mul(SS33, Stmp3);                                                 \
  SS31 = mul(SS31, Stmp3);                                                 \
  SS32 = mul(SS32, Stmp3);                                                 \
  SS33 = mul(SS33, Stmp3);                                                 \
                                                                           \
  Stmp1 = mul(Ss, SS31);                                                   \
  Stmp2 = mul(Ss, SS32);                                                   \
  SS31 = mul(Sc, SS31);                                                    \
  SS32 = mul(Sc, SS32);                                                    \
  SS31 = add(Stmp2, SS31);                                                 \
  SS32 = sub(SS32, Stmp1);                                                 \
                                                                           \
  Stmp2 = mul(Ss, Ss);                                                     \
  Stmp1 = mul(SS22, Stmp2);                                                \
  Stmp3 = mul(SS11, Stmp2);                                                \
  Stmp4 = mul(Sc, Sc);                                                     \
  SS11 = mul(SS11, Stmp4);                                                 \
  SS22 = mul(SS22, Stmp4);                                                 \
  SS11 = add(SS11, Stmp1);                                                 \
  SS22 = add(SS22, Stmp3);                                                 \
  Stmp4 = sub(Stmp4, Stmp2);                                               \
  Stmp2 = add(SS21, SS21);                                                 \
  SS21 = mul(SS21, Stmp4);                                                 \
  Stmp4 = mul(Sc, Ss);                                                     \
  Stmp2 = mul(Stmp2, Stmp4);                                               \
  Stmp5 = mul(Stmp5, Stmp4);                                               \
  SS11 = add(SS11, Stmp2);                                                 \
  SS21 = sub(SS21, Stmp5);                                                 \
  SS22 = sub(SS22, Stmp2);                                                 \
                                                                           \
  Stmp1 = mul(Ssh, Sqvx);                                                  \
  Stmp2 = mul(Ssh, Sqvy);                                                  \
  Stmp3 = mul(Ssh, Sqvz);                                                  \
  Ssh = mul(Ssh, Sqvs);                                                    \
                                                                           \
  Sqvs = mul(Sch, Sqvs);                                                   \
  Sqvx = mul(Sch, Sqvx);                                                   \
  Sqvy = mul(Sch, Sqvy);                                                   \
  Sqvz = mul(Sch, Sqvz);                                                   \
                                                                           \
  SQVZ = add(SQVZ, Ssh);                                                   \
  Sqvs = sub(Sqvs, STMP3);                                                 \
  SQVX = add(SQVX, STMP2);                                                 \
  SQVY = sub(SQVY, STMP1);

#define SVD3X3_COMPUTE_MATRIX_V   \
  Stmp2 = mul(Sqvs, Sqvs);        \
  Stmp2 = fma(Sqvx, Sqvx, Stmp2); \
  Stmp2 = fma(Sqvy, Sqvy, Stmp2); \
  Stmp2 = fma(Sqvz, Sqvz, Stmp2); \
                                  \
  Stmp1 = rsqrt(Stmp2);           \
                                  \
  Sqvs = mul(Sqvs, Stmp1);        \
  Sqvx = mul(Sqvx, Stmp1);        \
  Sqvy = mul(Sqvy, Stmp1);        \
  Sqvz = mul(Sqvz, Stmp1);        \
  Stmp1 = mul(Sqvx, Sqvx);        \
  Stmp2 = mul(Sqvy, Sqvy);        \
  Stmp3 = mul(Sqvz, Sqvz);        \
  Sv11 = mul(Sqvs, Sqvs);         \
  Sv22 = sub(Sv11, Stmp1);        \
  Sv33 = sub(Sv22, Stmp2);        \
  Sv33 = add(Sv33, Stmp3);        \
  Sv22 = add(Sv22, Stmp2);        \
  Sv22 = sub(Sv22, Stmp3);        \
  Sv11 = add(Sv11, Stmp1);        \
  Sv11 = sub(Sv11, Stmp2);        \
  Sv11 = sub(Sv11, Stmp3);        \
  Stmp1 = add(Sqvx, Sqvx);        \
  Stmp2 = add(Sqvy, Sqvy);        \
  Stmp3 = add(Sqvz, Sqvz);        \
  Sv32 = mul(Sqvs, Stmp1);        \
  Sv13 = mul(Sqvs, Stmp2);        \
  Sv21 = mul(Sqvs, Stmp3);        \
  Stmp1 = mul(Sqvy, Stmp1);       \
  Stmp2 = mul(Sqvz, Stmp2);       \
  Stmp3 = mul(Sqvx, Stmp3);       \
  Sv12 = sub(Stmp1, Sv21);        \
  Sv23 = sub(Stmp2, Sv32);        \
  Sv31 = sub(Stmp3, Sv13);        \
  Sv21 = add(Stmp1, Sv21);        \
  Sv32 = add(Stmp2, Sv32);        \
  Sv13 = add(Stmp3, Sv13);

#define SVD3X3_ACCURATE_COMPUTE_MATRIX_V \
  Stmp2 = mul(Sqvs, Sqvs);               \
  Stmp2 = fma(Sqvx, Sqvx, Stmp2);        \
  Stmp2 = fma(Sqvy, Sqvy, Stmp2);        \
  Stmp2 = fma(Sqvz, Sqvz, Stmp2);        \
                                         \
  Stmp1 = rsqrt(Stmp2);                  \
  Stmp4 = mul(Stmp1, Sone_half);         \
  Stmp3 = mul(Stmp1, Stmp4);             \
  Stmp3 = mul(Stmp1, Stmp3);             \
  Stmp3 = mul(Stmp2, Stmp3);             \
  Stmp1 = add(Stmp1, Stmp4);             \
  Stmp1 = sub(Stmp1, Stmp3);             \
                                         \
  Sqvs = mul(Sqvs, Stmp1);               \
  Sqvx = mul(Sqvx, Stmp1);               \
  Sqvy = mul(Sqvy, Stmp1);               \
  Sqvz = mul(Sqvz, Stmp1);               \
  Stmp1 = mul(Sqvx, Sqvx);               \
  Stmp2 = mul(Sqvy, Sqvy);               \
  Stmp3 = mul(Sqvz, Sqvz);               \
  Sv11 = mul(Sqvs, Sqvs);                \
  Sv22 = sub(Sv11, Stmp1);               \
  Sv33 = sub(Sv22, Stmp2);               \
  Sv33 = add(Sv33, Stmp3);               \
  Sv22 = add(Sv22, Stmp2);               \
  Sv22 = sub(Sv22, Stmp3);               \
  Sv11 = add(Sv11, Stmp1);               \
  Sv11 = sub(Sv11, Stmp2);               \
  Sv11 = sub(Sv11, Stmp3);               \
  Stmp1 = add(Sqvx, Sqvx);               \
  Stmp2 = add(Sqvy, Sqvy);               \
  Stmp3 = add(Sqvz, Sqvz);               \
  Sv32 = mul(Sqvs, Stmp1);               \
  Sv13 = mul(Sqvs, Stmp2);               \
  Sv21 = mul(Sqvs, Stmp3);               \
  Stmp1 = mul(Sqvy, Stmp1);              \
  Stmp2 = mul(Sqvz, Stmp2);              \
  Stmp3 = mul(Sqvx, Stmp3);              \
  Sv12 = sub(Stmp1, Sv21);               \
  Sv23 = sub(Stmp2, Sv32);               \
  Sv31 = sub(Stmp3, Sv13);               \
  Sv21 = add(Stmp1, Sv21);               \
  Sv32 = add(Stmp2, Sv32);               \
  Sv13 = add(Stmp3, Sv13);

#define SVD3X3_MULTIPLY_WITH_V   \
  Stmp2 = Sa12;                  \
  Stmp3 = Sa13;                  \
  Sa12 = mul(Sv12, Sa11);        \
  Sa13 = mul(Sv13, Sa11);        \
  Sa11 = mul(Sv11, Sa11);        \
  Sa11 = fma(Sv21, Stmp2, Sa11); \
  Sa11 = fma(Sv31, Stmp3, Sa11); \
  Sa12 = fma(Sv22, Stmp2, Sa12); \
  Sa12 = fma(Sv32, Stmp3, Sa12); \
  Sa13 = fma(Sv23, Stmp2, Sa13); \
  Sa13 = fma(Sv33, Stmp3, Sa13); \
                                 \
  Stmp2 = Sa22;                  \
  Stmp3 = Sa23;                  \
  Sa22 = mul(Sv12, Sa21);        \
  Sa23 = mul(Sv13, Sa21);        \
  Sa21 = mul(Sv11, Sa21);        \
  Sa21 = fma(Sv21, Stmp2, Sa21); \
  Sa21 = fma(Sv31, Stmp3, Sa21); \
  Sa22 = fma(Sv22, Stmp2, Sa22); \
  Sa22 = fma(Sv32, Stmp3, Sa22); \
  Sa23 = fma(Sv23, Stmp2, Sa23); \
  Sa23 = fma(Sv33, Stmp3, Sa23); \
                                 \
  Stmp2 = Sa32;                  \
  Stmp3 = Sa33;                  \
  Sa32 = mul(Sv12, Sa31);        \
  Sa33 = mul(Sv13, Sa31);        \
  Sa31 = mul(Sv11, Sa31);        \
  Sa31 = fma(Sv21, Stmp2, Sa31); \
  Sa31 = fma(Sv31, Stmp3, Sa31); \
  Sa32 = fma(Sv22, Stmp2, Sa32); \
  Sa32 = fma(Sv32, Stmp3, Sa32); \
  Sa33 = fma(Sv23, Stmp2, Sa33); \
  Sa33 = fma(Sv33, Stmp3, Sa33);

#define SVD3X3_SORT_SINGULAR_VALUES      \
  Stmp1 = mul(Sa11, Sa11);               \
  Stmp1 = fma(Sa21, Sa21, Stmp1);        \
  Stmp1 = fma(Sa31, Sa31, Stmp1);        \
                                         \
  Stmp2 = mul(Sa12, Sa12);               \
  Stmp2 = fma(Sa22, Sa22, Stmp2);        \
  Stmp2 = fma(Sa32, Sa32, Stmp2);        \
                                         \
  Stmp3 = mul(Sa13, Sa13);               \
  Stmp3 = fma(Sa23, Sa23, Stmp3);        \
  Stmp3 = fma(Sa33, Sa33, Stmp3);        \
                                         \
  Stmp4 = cmp(Stmp1, Stmp2, _CMP_LT_OS); \
  Stmp5 = sxor(Sa11, Sa12);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa11 = sxor(Sa11, Stmp5);              \
  Sa12 = sxor(Sa12, Stmp5);              \
                                         \
  Stmp5 = sxor(Sa21, Sa22);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa21 = sxor(Sa21, Stmp5);              \
  Sa22 = sxor(Sa22, Stmp5);              \
                                         \
  Stmp5 = sxor(Sa31, Sa32);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa31 = sxor(Sa31, Stmp5);              \
  Sa32 = sxor(Sa32, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv11, Sv12);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv11 = sxor(Sv11, Stmp5);              \
  Sv12 = sxor(Sv12, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv21, Sv22);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv21 = sxor(Sv21, Stmp5);              \
  Sv22 = sxor(Sv22, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv31, Sv32);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv31 = sxor(Sv31, Stmp5);              \
  Sv32 = sxor(Sv32, Stmp5);              \
                                         \
  Stmp5 = sxor(Stmp1, Stmp2);            \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Stmp1 = sxor(Stmp1, Stmp5);            \
  Stmp2 = sxor(Stmp2, Stmp5);            \
                                         \
  Stmp5 = sand(Smtwo, Stmp4);            \
  Stmp4 = add(Sone, Stmp5);              \
                                         \
  Sa12 = mul(Sa12, Stmp4);               \
  Sa22 = mul(Sa22, Stmp4);               \
  Sa32 = mul(Sa32, Stmp4);               \
                                         \
  Sv12 = mul(Sv12, Stmp4);               \
  Sv22 = mul(Sv22, Stmp4);               \
  Sv32 = mul(Sv32, Stmp4);               \
                                         \
  Stmp4 = cmp(Stmp1, Stmp3, _CMP_LT_OS); \
  Stmp5 = sxor(Sa11, Sa13);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa11 = sxor(Sa11, Stmp5);              \
  Sa13 = sxor(Sa13, Stmp5);              \
                                         \
  Stmp5 = sxor(Sa21, Sa23);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa21 = sxor(Sa21, Stmp5);              \
  Sa23 = sxor(Sa23, Stmp5);              \
                                         \
  Stmp5 = sxor(Sa31, Sa33);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa31 = sxor(Sa31, Stmp5);              \
  Sa33 = sxor(Sa33, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv11, Sv13);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv11 = sxor(Sv11, Stmp5);              \
  Sv13 = sxor(Sv13, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv21, Sv23);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv21 = sxor(Sv21, Stmp5);              \
  Sv23 = sxor(Sv23, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv31, Sv33);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv31 = sxor(Sv31, Stmp5);              \
  Sv33 = sxor(Sv33, Stmp5);              \
                                         \
  Stmp5 = sxor(Stmp1, Stmp3);            \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Stmp1 = sxor(Stmp1, Stmp5);            \
  Stmp3 = sxor(Stmp3, Stmp5);            \
                                         \
  Stmp5 = sand(Smtwo, Stmp4);            \
  Stmp4 = add(Sone, Stmp5);              \
                                         \
  Sa11 = mul(Sa11, Stmp4);               \
  Sa21 = mul(Sa21, Stmp4);               \
  Sa31 = mul(Sa31, Stmp4);               \
                                         \
  Sv11 = mul(Sv11, Stmp4);               \
  Sv21 = mul(Sv21, Stmp4);               \
  Sv31 = mul(Sv31, Stmp4);               \
                                         \
  Stmp4 = cmp(Stmp2, Stmp3, _CMP_LT_OS); \
  Stmp5 = sxor(Sa12, Sa13);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa12 = sxor(Sa12, Stmp5);              \
  Sa13 = sxor(Sa13, Stmp5);              \
                                         \
  Stmp5 = sxor(Sa22, Sa23);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa22 = sxor(Sa22, Stmp5);              \
  Sa23 = sxor(Sa23, Stmp5);              \
                                         \
  Stmp5 = sxor(Sa32, Sa33);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sa32 = sxor(Sa32, Stmp5);              \
  Sa33 = sxor(Sa33, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv12, Sv13);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv12 = sxor(Sv12, Stmp5);              \
  Sv13 = sxor(Sv13, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv22, Sv23);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv22 = sxor(Sv22, Stmp5);              \
  Sv23 = sxor(Sv23, Stmp5);              \
                                         \
  Stmp5 = sxor(Sv32, Sv33);              \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Sv32 = sxor(Sv32, Stmp5);              \
  Sv33 = sxor(Sv33, Stmp5);              \
                                         \
  Stmp5 = sxor(Stmp2, Stmp3);            \
  Stmp5 = sand(Stmp5, Stmp4);            \
  Stmp2 = sxor(Stmp2, Stmp5);            \
  Stmp3 = sxor(Stmp3, Stmp5);            \
                                         \
  Stmp5 = sand(Smtwo, Stmp4);            \
  Stmp4 = add(Sone, Stmp5);              \
                                         \
  Sa13 = mul(Sa13, Stmp4);               \
  Sa23 = mul(Sa23, Stmp4);               \
  Sa33 = mul(Sa33, Stmp4);               \
                                         \
  Sv13 = mul(Sv13, Stmp4);               \
  Sv23 = mul(Sv23, Stmp4);               \
  Sv33 = mul(Sv33, Stmp4);

#define SVD3X3_QR(SAPIVOT, SANPIVOT, SA11, SA21, SA12, SA22, SA13, SA23, SU11, \
                  SU12, SU21, SU22, SU31, SU32)                                \
  Ssh = mul(SANPIVOT, SANPIVOT);                                               \
  Ssh = cmp(Ssh, Ssmall_number, _CMP_GE_OS);                                   \
  Ssh = sand(Ssh, SANPIVOT);                                                   \
                                                                               \
  Sch = sub(Szero, SAPIVOT);                                                   \
  Sch = max(Sch, SAPIVOT);                                                     \
  Sch = max(Sch, Ssmall_number);                                               \
  Stmp5 = cmp(SAPIVOT, Szero, _CMP_GE_OS);                                     \
                                                                               \
  Stmp1 = mul(Sch, Sch);                                                       \
  Stmp2 = fma(Ssh, Ssh, Stmp1);                                                \
  Stmp1 = rsqrt(Stmp2);                                                        \
                                                                               \
  Stmp1 = mul(Stmp1, Stmp2);                                                   \
                                                                               \
  Sch = add(Sch, Stmp1);                                                       \
                                                                               \
  Stmp1 = Sch;                                                                 \
  Sch = blend(Ssh, Sch, Stmp5);                                                \
  Ssh = blend(Stmp1, Ssh, Stmp5);                                              \
                                                                               \
  Stmp1 = mul(Sch, Sch);                                                       \
  Stmp2 = fma(Ssh, Ssh, Stmp1);                                                \
  Stmp1 = rsqrt(Stmp2);                                                        \
                                                                               \
  Sch = mul(Sch, Stmp1);                                                       \
  Ssh = mul(Ssh, Stmp1);                                                       \
                                                                               \
  Ss = mul(Ssh, Ssh);                                                          \
  Sc = fms(Sch, Sch, Ss);                                                      \
  Ss = mul(Ssh, Sch);                                                          \
  Ss = add(Ss, Ss);                                                            \
                                                                               \
  Stmp1 = mul(Ss, SA11);                                                       \
  Stmp2 = mul(Ss, SA21);                                                       \
  SA11 = mul(Sc, SA11);                                                        \
  SA21 = mul(Sc, SA21);                                                        \
  SA11 = add(SA11, Stmp2);                                                     \
  SA21 = sub(SA21, Stmp1);                                                     \
                                                                               \
  Stmp1 = mul(Ss, SA12);                                                       \
  Stmp2 = mul(Ss, SA22);                                                       \
  SA12 = mul(Sc, SA12);                                                        \
  SA22 = mul(Sc, SA22);                                                        \
  SA12 = add(SA12, Stmp2);                                                     \
  SA22 = sub(SA22, Stmp1);                                                     \
                                                                               \
  Stmp1 = mul(Ss, SA13);                                                       \
  Stmp2 = mul(Ss, SA23);                                                       \
  SA13 = mul(Sc, SA13);                                                        \
  SA23 = mul(Sc, SA23);                                                        \
  SA13 = add(SA13, Stmp2);                                                     \
  SA23 = sub(SA23, Stmp1);                                                     \
                                                                               \
  Stmp1 = mul(Ss, SU11);                                                       \
  Stmp2 = mul(Ss, SU12);                                                       \
  SU11 = mul(Sc, SU11);                                                        \
  SU12 = mul(Sc, SU12);                                                        \
  SU11 = add(SU11, Stmp2);                                                     \
  SU12 = sub(SU12, Stmp1);                                                     \
                                                                               \
  Stmp1 = mul(Ss, SU21);                                                       \
  Stmp2 = mul(Ss, SU22);                                                       \
  SU21 = mul(Sc, SU21);                                                        \
  SU22 = mul(Sc, SU22);                                                        \
  SU21 = add(SU21, Stmp2);                                                     \
  SU22 = sub(SU22, Stmp1);                                                     \
                                                                               \
  Stmp1 = mul(Ss, SU31);                                                       \
  Stmp2 = mul(Ss, SU32);                                                       \
  SU31 = mul(Sc, SU31);                                                        \
  SU32 = mul(Sc, SU32);                                                        \
  SU31 = add(SU31, Stmp2);                                                     \
  SU32 = sub(SU32, Stmp1);

#define SVD3X3_ACCURATE_QR(SAPIVOT, SANPIVOT, SA11, SA21, SA12, SA22, SA13, \
                           SA23, SU11, SU12, SU21, SU22, SU31, SU32)        \
  Ssh = mul(SANPIVOT, SANPIVOT);                                            \
  Ssh = cmp(Ssh, Ssmall_number, _CMP_GE_OS);                                \
  Ssh = sand(Ssh, SANPIVOT);                                                \
                                                                            \
  Sch = sub(Szero, SAPIVOT);                                                \
  Sch = max(Sch, SAPIVOT);                                                  \
  Sch = max(Sch, Ssmall_number);                                            \
  Stmp5 = cmp(SAPIVOT, Szero, _CMP_GE_OS);                                  \
                                                                            \
  Stmp1 = mul(Sch, Sch);                                                    \
  Stmp2 = fma(Ssh, Ssh, Stmp1);                                             \
  Stmp1 = rsqrt(Stmp2);                                                     \
                                                                            \
  Stmp4 = mul(Stmp1, Sone_half);                                            \
  Stmp3 = mul(Stmp1, Stmp4);                                                \
  Stmp3 = mul(Stmp1, Stmp3);                                                \
  Stmp3 = mul(Stmp2, Stmp3);                                                \
  Stmp1 = add(Stmp1, Stmp4);                                                \
  Stmp1 = sub(Stmp1, Stmp3);                                                \
  Stmp1 = mul(Stmp1, Stmp2);                                                \
                                                                            \
  Sch = add(Sch, Stmp1);                                                    \
                                                                            \
  Stmp1 = Sch;                                                              \
  Sch = blend(Ssh, Sch, Stmp5);                                             \
  Ssh = blend(Stmp1, Ssh, Stmp5);                                           \
                                                                            \
  Stmp1 = mul(Sch, Sch);                                                    \
  Stmp2 = fma(Ssh, Ssh, Stmp1);                                             \
  Stmp1 = rsqrt(Stmp2);                                                     \
                                                                            \
  Stmp4 = mul(Stmp1, Sone_half);                                            \
  Stmp3 = mul(Stmp1, Stmp4);                                                \
  Stmp3 = mul(Stmp1, Stmp3);                                                \
  Stmp3 = mul(Stmp2, Stmp3);                                                \
  Stmp1 = add(Stmp1, Stmp4);                                                \
  Stmp1 = sub(Stmp1, Stmp3);                                                \
                                                                            \
  Sch = mul(Sch, Stmp1);                                                    \
  Ssh = mul(Ssh, Stmp1);                                                    \
                                                                            \
  Ss = mul(Ssh, Ssh);                                                       \
  Sc = fms(Sch, Sch, Ss);                                                   \
  Ss = mul(Ssh, Sch);                                                       \
  Ss = add(Ss, Ss);                                                         \
                                                                            \
  Stmp1 = mul(Ss, SA11);                                                    \
  Stmp2 = mul(Ss, SA21);                                                    \
  SA11 = mul(Sc, SA11);                                                     \
  SA21 = mul(Sc, SA21);                                                     \
  SA11 = add(SA11, Stmp2);                                                  \
  SA21 = sub(SA21, Stmp1);                                                  \
                                                                            \
  Stmp1 = mul(Ss, SA12);                                                    \
  Stmp2 = mul(Ss, SA22);                                                    \
  SA12 = mul(Sc, SA12);                                                     \
  SA22 = mul(Sc, SA22);                                                     \
  SA12 = add(SA12, Stmp2);                                                  \
  SA22 = sub(SA22, Stmp1);                                                  \
                                                                            \
  Stmp1 = mul(Ss, SA13);                                                    \
  Stmp2 = mul(Ss, SA23);                                                    \
  SA13 = mul(Sc, SA13);                                                     \
  SA23 = mul(Sc, SA23);                                                     \
  SA13 = add(SA13, Stmp2);                                                  \
  SA23 = sub(SA23, Stmp1);                                                  \
                                                                            \
  Stmp1 = mul(Ss, SU11);                                                    \
  Stmp2 = mul(Ss, SU12);                                                    \
  SU11 = mul(Sc, SU11);                                                     \
  SU12 = mul(Sc, SU12);                                                     \
  SU11 = add(SU11, Stmp2);                                                  \
  SU12 = sub(SU12, Stmp1);                                                  \
                                                                            \
  Stmp1 = mul(Ss, SU21);                                                    \
  Stmp2 = mul(Ss, SU22);                                                    \
  SU21 = mul(Sc, SU21);                                                     \
  SU22 = mul(Sc, SU22);                                                     \
  SU21 = add(SU21, Stmp2);                                                  \
  SU22 = sub(SU22, Stmp1);                                                  \
                                                                            \
  Stmp1 = mul(Ss, SU31);                                                    \
  Stmp2 = mul(Ss, SU32);                                                    \
  SU31 = mul(Sc, SU31);                                                     \
  SU32 = mul(Sc, SU32);                                                     \
  SU31 = add(SU31, Stmp2);                                                  \
  SU32 = sub(SU32, Stmp1);
