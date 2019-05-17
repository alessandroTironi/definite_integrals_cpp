#pragma once

#include <emmintrin.h>

typedef float real_t;
typedef unsigned long precision_t;
typedef real_t(*function_t)(real_t);
typedef __m128(*function128_t)(__m128);
typedef __m256(*function256_t)(__m256);

// Cavalieri-Simpson rule 
real_t definite_integral_cs(function_t function, const real_t a, const real_t b, precision_t precision);
// Cavalieri-Simpson rule + SSE
real_t definite_integral_cs_sse(function_t function, const real_t a, const real_t b, precision_t precision);
real_t definite_integral_cs_avx2_256(function256_t function, const real_t a, const real_t b, precision_t precision);
// Rectangles approximation
real_t definite_integral_rectangles(function_t function, const real_t a, const real_t b, precision_t precision);
// Rectangles approximation + SSE
real_t definite_integral_rectangles_sse(function_t function, const real_t a, const real_t b, precision_t precision);
real_t definite_integral_rectangles_sse_128(function128_t function, const real_t a, const real_t b, precision_t precision);
real_t definite_integral_rectangles_avx_256(function256_t function, const real_t a, const real_t b, precision_t precision);
// Ad-hoc Gauss function integration
real_t gaussian_prob_sse(const float mean, const float stdev, const real_t a, const real_t b, precision_t precision);
real_t gaussian_prob_avx2(const float mean, const float stdev, const real_t a, const real_t b, precision_t precision);