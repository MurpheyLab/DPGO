#pragma once

#include "svd3x3.h"

#define PROJECT_TO_SO3_COMPUTE_U \
                                 \
  u11 = mul(Su11, Sv11);         \
  u11 = fma(Su12, Sv12, u11);    \
  u11 = fma(Su13, Sv13, u11);    \
                                 \
  u21 = mul(Su21, Sv11);         \
  u21 = fma(Su22, Sv12, u21);    \
  u21 = fma(Su23, Sv13, u21);    \
                                 \
  u31 = mul(Su31, Sv11);         \
  u31 = fma(Su32, Sv12, u31);    \
  u31 = fma(Su33, Sv13, u31);    \
                                 \
  u12 = mul(Su11, Sv21);         \
  u12 = fma(Su12, Sv22, u12);    \
  u12 = fma(Su13, Sv23, u12);    \
                                 \
  u22 = mul(Su21, Sv21);         \
  u22 = fma(Su22, Sv22, u22);    \
  u22 = fma(Su23, Sv23, u22);    \
                                 \
  u32 = mul(Su31, Sv21);         \
  u32 = fma(Su32, Sv22, u32);    \
  u32 = fma(Su33, Sv23, u32);    \
                                 \
  u13 = mul(Su11, Sv31);         \
  u13 = fma(Su12, Sv32, u13);    \
  u13 = fma(Su13, Sv33, u13);    \
                                 \
  u23 = mul(Su21, Sv31);         \
  u23 = fma(Su22, Sv32, u23);    \
  u23 = fma(Su23, Sv33, u23);    \
                                 \
  u33 = mul(Su31, Sv31);         \
  u33 = fma(Su32, Sv32, u33);    \
  u33 = fma(Su33, Sv33, u33);
