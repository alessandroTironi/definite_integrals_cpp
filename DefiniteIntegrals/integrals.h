#pragma once

typedef float real_t;
typedef unsigned long precision_t;
typedef real_t(*function_t)(real_t);

// Cavalieri-Simpson rule 
real_t definite_integral_cs(function_t function, real_t a, real_t b, precision_t precision);
// Cavalieri-Simpson rule + SSE
real_t definite_integral_cs_sse(function_t function, real_t a, real_t b, precision_t precision);
// Rectangles approximation
real_t definite_integral_rectangles(function_t function, real_t a, real_t b, precision_t precision);
// Rectangles approximation + SSE
real_t definite_integral_rectangles_sse(function_t function, real_t a, real_t b, precision_t precision);
// Ad-hoc Gauss function integration
real_t gaussian_prob_sse(float mean, float stdev, real_t a, real_t b, precision_t precision);