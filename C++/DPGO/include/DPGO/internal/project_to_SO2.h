#pragma once

#define PROJECT_TO_SO2                          \
  Sc = add(a11, a22);                           \
  Ss = sub(a21, a12);                           \
                                                \
  Stmp1 = mul(Sc, Sc);                          \
  Stmp1 = fma(Ss, Ss, Stmp1);                   \
  Stmp2 = cmp(Stmp1, Stiny_number, _CMP_GE_OS); \
                                                \
  Sc = blend(Sone, Sc, Stmp2);                  \
  Ss = blend(Szero, Ss, Stmp2);                 \
  Stmp1 = blend(Sone, Stmp1, Stmp2);            \
                                                \
  Stmp2 = rsqrt(Stmp1);                         \
                                                \
  u11 = mul(Sc, Stmp2);                         \
  u21 = mul(Ss, Stmp2);

#define PROJECT_TO_SO2_ACCURATE_RSQRT           \
  Sc = add(a11, a22);                           \
  Ss = sub(a21, a12);                           \
                                                \
  Stmp1 = mul(Sc, Sc);                          \
  Stmp1 = fma(Ss, Ss, Stmp1);                   \
  Stmp2 = cmp(Stmp1, Stiny_number, _CMP_GE_OS); \
                                                \
  Sc = blend(Sone, Sc, Stmp2);                  \
  Ss = blend(Szero, Ss, Stmp2);                 \
  Stmp1 = blend(Sone, Stmp1, Stmp2);            \
                                                \
  Stmp2 = rsqrt(Stmp1);                         \
                                                \
  u11 = mul(Stmp2, Sone_half);                  \
  u21 = mul(Stmp2, u11);                        \
  u21 = mul(Stmp2, u21);                        \
  u21 = mul(Stmp1, u21);                        \
  Stmp2 = add(Stmp2, u11);                      \
  Stmp2 = sub(Stmp2, u21);                      \
                                                \
  u11 = mul(Sc, Stmp2);                         \
  u21 = mul(Ss, Stmp2);
