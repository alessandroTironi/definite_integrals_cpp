#include <emmintrin.h>
#include <iostream>

#include "fmath.hpp"

#include "integrals.h"

#define ALIGN_256 __declspec(align(32))
#define ALIGN_128 __declspec(align(16))

#define ROUND_2(x) x & 0xfffffffe
#define ROUND_4(x) x & 0xfffffffc
#define ROUND_8(x) x & 0xfffffff8

const __m256 m256_INC = _mm256_set_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
const __m128 m128_INC = _mm_set_ps(0.0f, 1.0f, 2.0f, 3.0f);
const __m128 m128_INC1_1 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
const __m128 m128_INC1_2 = _mm_set_ps(5.0f, 6.0f, 7.0f, 8.0f);

real_t definite_integral_rectangles(function_t function, real_t a, real_t b, precision_t nRects)
{
	real_t step = (b - a) / (real_t)nRects;
	real_t half_step = step * 0.5f;

	real_t i = 0.0f;
	int nIt = 0;
	for (real_t s = a + half_step; s < b; s += step)
		i += function(s) * step;
	return i;
}

real_t definite_integral_rectangles_sse(function_t function, real_t a, real_t b, precision_t precision)
{
	// Prevents interval overrun.
	precision = ROUND_4(precision);

	real_t step = (b - a) / (real_t)precision;
	real_t half_step = step * 0.5f;

	__m128 mm_int = _mm_set_ps1(0.0f);
	__m128 mm_step = _mm_set_ps1(step);
	__m128 mm_step_inc = _mm_mul_ps(mm_step, m128_INC);
	__m128 heights;
	__m128 mm_s;

	int nIts = 0;
	float ALIGN_128 v[4];
	for (real_t s = a + half_step; s < b; s = step * 4 + s)
	{
		mm_s = _mm_set_ps1(s);
		_mm_store_ps(v, _mm_add_ps(mm_s, mm_step_inc));
		heights = _mm_set_ps(function(v[0]), function(v[1]),
			function(v[2]), function(v[3]));
		mm_int = _mm_add_ps(mm_int, _mm_mul_ps(heights, mm_step));
	}

	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	return mm_int.m128_f32[0];
}


real_t definite_integral_cs(function_t function, real_t a, real_t b, precision_t precision)
{
	// Prevents interval overrun.
	precision = ROUND_2(precision);

	real_t h = (b - a) / (real_t)precision;
	real_t x0, y0, x1, y1, x2, y2;
	real_t i = 0.0f;
	real_t h_3 = h / 3.f;
	y0 = function(a);
	for (real_t s = a; s < b; s += h * 2)
	{
		x1 = s + h;
		y1 = function(x1);
		x2 = h * 2 + s;
		y2 = function(x2);

		i += h_3 * (y0 + (4 * y1) + y2);
		y0 = y2;
	}
	return i;
}

real_t definite_integral_cs_sse(function_t function, real_t a, real_t b, precision_t precision)
{
	// Prevents interval overrun
	precision = ROUND_8(precision);

	// Utility data (compute once to save CPU time)
	real_t h = (b - a) / (real_t)precision;
	real_t h_3 = h / 3.f;

	// 128bit accumulator
	__m128 mm_int = _mm_set_ps1(0.0f);
	__m128 p, a1, a2, a3;
	__m128 mm_h_3 = _mm_set_ps1(h_3);
	__m128 mm_4 = _mm_set_ps1(4.0f);
	__m128 mm_s[2];
	__m128 mm_h = _mm_set_ps1(h);
	__m128 mm_h_inc_1 = _mm_mul_ps(mm_h, m128_INC1_1);		// from 1 to 4h
	__m128 mm_h_inc_2 = _mm_mul_ps(mm_h, m128_INC1_2);		// from 5h to 8h
	float ALIGN_128 y[9];
	
	y[0] = function(a);
	for (real_t s = a; s < b; s += h * 8)
	{
		mm_s[0] = _mm_add_ps(_mm_set_ps1(s), mm_h_inc_1);
		mm_s[1] = _mm_add_ps(_mm_set_ps1(s), mm_h_inc_2);
		_mm_store_ps(&y[1], mm_s[0]);
		_mm_store_ps(&y[5], mm_s[1]);

		// Computes function values.
		y[1] = function(y[1]);
		y[2] = function(y[2]);
		y[3] = function(y[3]);
		y[4] = function(y[4]);
		y[5] = function(y[5]);
		y[6] = function(y[6]);
		y[7] = function(y[7]);
		y[8] = function(y[8]);

		// Computes current term of CS rule sum.
		a1 = _mm_set_ps(y[0], y[2], y[4], y[6]);
		a2 = _mm_set_ps(y[1], y[3], y[5], y[7]);
		a3 = _mm_set_ps(y[2], y[4], y[6], y[8]);
		p = _mm_mul_ps(mm_h_3,
			_mm_add_ps(a1,
				_mm_add_ps(a3,
					_mm_mul_ps(a2, mm_4)
				)
			)
		);

		mm_int = _mm_add_ps(mm_int, p);
		y[0] = y[8];
	}

	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	return mm_int.m128_f32[0];
}

real_t gaussian_prob_sse(float mean, float stdev, real_t a, real_t b, precision_t precision)
{
	// Prevents interval overrun.
	precision = ROUND_4(precision);

	real_t h = (b - a) / (real_t)precision;
	real_t h_2 = h * 0.5f;
	__m128 mm_int = _mm_set_ps1(0.0f);
	__m128 mm;

	// Computes once the common coefficient of the Gauss function.
	real_t coeff = 1.0f / (stdev * sqrt(2.0f * 3.141f));
	real_t var = stdev * stdev;
	__m128 mm_neg_half = _mm_set_ps1(-0.5f);
	__m128 mm_var = _mm_set_ps1(var);
	__m128 mm_mean = _mm_set_ps1(mean);
	__m128 mm_error;
	__m128 mm_h = _mm_set_ps1(h);
	__m128 mm_h_inc = _mm_mul_ps(mm_h, m128_INC);
	__m128 mm_s;

	for (real_t s = a + h_2; s < b; s += h * 4)
	{
	
		// Computing errors
		mm_error = _mm_add_ps(_mm_set_ps1(s), mm_h_inc);
		mm_error = _mm_sub_ps(mm_error, mm_mean);

		// Computing exponentials
		mm = fmath::exp_ps(
			_mm_div_ps(
				_mm_mul_ps(mm_neg_half,
					_mm_mul_ps(mm_error, mm_error)
				),
				mm_var)
		);

		// Computing area of rects
		mm_int = _mm_add_ps(mm_int, _mm_mul_ps(mm, mm_h));
	}

	// Retrieves result.
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	return coeff * mm_int.m128_f32[0];
}

real_t gaussian_prob_avx2(float mean, float stdev, real_t a, real_t b, precision_t precision)
{
	// Prevents interval overrun.
	precision = ROUND_8(precision);

	real_t h = (b - a) / (real_t)precision;
	real_t h_2 = h * 0.5f;
	__m256 mm_int = _mm256_set1_ps(0.0f);
	__m256 mm;
	__m256 mm_h = _mm256_set1_ps(h);

	// Computes once the common coefficient of the Gauss function.
	real_t coeff = 1.0f / (stdev * sqrt(2.0f * 3.141f));
	real_t var = stdev * stdev;
	__m256 mm_neg_half = _mm256_set1_ps(-0.5f);
	__m256 mm_var = _mm256_set1_ps(var);
	__m256 mm_mean = _mm256_set1_ps(mean);
	__m256 mm_s;
	__m256 mm_error;

	for (real_t s = a + h_2; s < b; s += h * 8)
	{
		// Computing errors
		mm_error = _mm256_mul_ps(_mm256_set1_ps(h), m256_INC);
		mm_s = _mm256_set1_ps(s);
		mm_error = _mm256_sub_ps(_mm256_add_ps(mm_s, mm_error), mm_mean);

		// Computing exponentials
		mm = fmath::exp_ps256(
			_mm256_div_ps(
				_mm256_mul_ps(mm_neg_half,
					_mm256_mul_ps(mm_error, mm_error)
				),
			mm_var)
		);

		// Computing area of rects
		mm_int = _mm256_add_ps(mm_int, _mm256_mul_ps(mm, mm_h));
	}

	__m128* mm128_int = (__m128*) &mm_int;
	__m128 mm_sum4 = _mm_add_ps(mm128_int[0], mm128_int[1]);
	mm_sum4 = _mm_hadd_ps(mm_sum4, mm_sum4);
	mm_sum4 = _mm_hadd_ps(mm_sum4, mm_sum4);
	return coeff * mm_sum4.m128_f32[0];
}