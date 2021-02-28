#include <iostream>

#include "fmath.hpp"

#include "integrals.h"
#include "time_measurement.hpp"

time_stamp timer::ts0;
time_stamp timer::ts1;

#define ALIGN_256 __declspec(align(32))
#define ALIGN_128 __declspec(align(16))

#define ROUND_2(x) x & 0xfffffffe
#define ROUND_4(x) x & 0xfffffffc
#define ROUND_8(x) x & 0xfffffff8
#define ROUND_16(x) x & 0xfffffff0

const __m256 m256_INC = _mm256_set_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
const __m128 m128_INC = _mm_set_ps(0.0f, 1.0f, 2.0f, 3.0f);
const __m128 m128_INC1_1 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
const __m128 m128_INC1_2 = _mm_set_ps(5.0f, 6.0f, 7.0f, 8.0f);
const __m256 m256_cs_pattern_1 = _mm256_set_ps(0.0f, 2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f, 7.0f);
const __m256 m256_cs_pattern_2 = _mm256_set_ps(2.0f, 4.0f, 6.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f);
const __m256 m256_INC2 = _mm256_set_ps(0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.f, 12.0f, 14.0f);
const __m128 m128_INC2 = _mm_set_ps(0.0f, 2.0f, 4.0f, 6.0f);

real_t definite_integral_rectangles(function_t function, const real_t a, const real_t b, precision_t nRects)
{
	SET_TIMESTAMP_0();
	const real_t step = (b - a) / (real_t)nRects;
	const real_t half_step = step * 0.5f;

	real_t i = 0.0f;
	int nIt = 0;
	for (real_t s = a + half_step; s < b; s += step)
		i += function(s);
	SET_TIMESTAMP_1();
	return i * step;
}

real_t definite_integral_rectangles_sse(function_t function, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	// Prevents interval overrun.
	precision = ROUND_4(precision);

	const real_t h = (b - a) / (real_t)precision;
	const real_t h_2 = h * 0.5f;

	__m128 mm_int = _mm_set_ps1(0.0f);
	const __m128 mm_h = _mm_set_ps1(h);
	const __m128 mm_h_inc = _mm_mul_ps(mm_h, m128_INC);
	__m128 heights;

	float ALIGN_128 v[4];
	for (real_t s = a + h_2; s < b; s = h * 4 + s)
	{
		_mm_store_ps(v, _mm_add_ps(_mm_set_ps1(s), mm_h_inc));
		heights = _mm_set_ps(function(v[0]), function(v[1]),
			function(v[2]), function(v[3]));
		mm_int = _mm_add_ps(mm_int, heights);
	}

	mm_int = _mm_mul_ps(mm_h, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	SET_TIMESTAMP_1();
	//return mm_int.m128_f32[0];
	return mm_int[0];
}

real_t definite_integral_rectangles_sse_128(function128_t function, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	precision = ROUND_4(precision);

	const real_t h = (b - a) / (real_t)precision;
	const real_t h_2 = h * 0.5f;

	__m128 mm_int = _mm_set_ps1(0.0f);
	const __m128 mm_h = _mm_set_ps1(h);
	const __m128 mm_h_inc = _mm_mul_ps(mm_h, m128_INC);
	__m128 heights;
	__m128 mm_s;

	for (real_t s = a + h_2; s < b; s = h * 4 + s)
	{
		mm_s = _mm_set_ps1(s);
		heights = function(_mm_add_ps(mm_s, mm_h_inc));
		mm_int = _mm_add_ps(mm_int, _mm_mul_ps(heights, mm_h));
	}

	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	SET_TIMESTAMP_1();
	//return mm_int.m128_f32[0];
	return mm_int[0];
}


real_t definite_integral_rectangles_avx_256(function256_t function, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	precision = ROUND_8(precision);

	const real_t h = (b - a) / (real_t)precision;
	const real_t h_2 = h * 0.5f;

	__m256 mm_int = _mm256_set1_ps(0.0f);
	const __m256 mm_h = _mm256_set1_ps(h);
	const __m256 mm_h_inc = _mm256_mul_ps(mm_h, m256_INC);
	__m256 heights;

	for (real_t s = a + h_2; s < b; s = h * 8 + s)
	{
		heights = function(_mm256_add_ps(_mm256_set1_ps(s), mm_h_inc));
		mm_int = _mm256_add_ps(mm_int, _mm256_mul_ps(heights, mm_h));
	}

	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	SET_TIMESTAMP_1();
	//return mm_int.m256_f32[0];
	return mm_int[0];
}


real_t definite_integral_cs(function_t function, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	// Prevents interval overrun.
	precision = ROUND_2(precision);

	const real_t h = (b - a) / (real_t)precision;
	real_t x0, y0, x1, y1, x2, y2;
	real_t i = 0.0f;
	const real_t double_h = h * 2;
	y0 = function(a);
	for (real_t s = a; s < b; s += h * 2)
	{
		x1 = s + h;
		y1 = function(x1);
		x2 = s + double_h;
		y2 = function(x2);

		i += (y0 + (4 * y1) + y2);
		y0 = y2;
	}

	SET_TIMESTAMP_1();
	return i * h * 0.333333f;
}

real_t definite_integral_cs_sse(function_t function, real_t a, real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	// Prevents interval overrun
	precision = ROUND_8(precision);

	// Utility data (compute once to save CPU time)
	real_t h = (b - a) / (real_t)precision;
	const real_t h_3 = h * 0.333333f;

	// 128bit accumulator
	__m128 mm_int = _mm_set_ps1(0.0f);
	__m128 p, a1, a2, a3;
	const __m128 mm_h_3 = _mm_set_ps1(h_3);
	const __m128 mm_4 = _mm_set_ps1(4.0f);
	__m128 mm_x[3];
	const __m128 mm_h = _mm_set_ps1(h);
	const __m128 mm_h_inc2 = _mm_mul_ps(mm_h, m128_INC2);
	float ALIGN_128 x[12];
	float y0;
	
	y0 = function(a);
	for (real_t s = a; s < b; s += h * 8)
	{
		mm_x[0] = _mm_add_ps(_mm_set_ps1(s), mm_h_inc2);
		mm_x[1] = _mm_add_ps(mm_x[0], mm_h);
		mm_x[2] = _mm_add_ps(mm_x[1], mm_h);
		_mm_store_ps(&x[0], mm_x[0]);
		_mm_store_ps(&x[4], mm_x[1]);
		_mm_store_ps(&x[8], mm_x[2]);

		// Computes current term of CS rule sum. 
		a1 = _mm_set_ps(y0, function(x[1]), function(x[2]), function(x[3]));
		a2 = _mm_set_ps(function(x[4]), function(x[5]), function(x[6]), function(x[7]));
		//a3 = _mm_set_ps(a1.m128_f32[1], a1.m128_f32[2], a1.m128_f32[3], function(x[11]));
		a3 = _mm_set_ps(a1[1], a1[2], a1[3], function(x[11]));
		p = _mm_add_ps(a1,
				_mm_add_ps(a3,
					_mm_mul_ps(a2, mm_4)
				)
		);

		mm_int = _mm_add_ps(mm_int, p);
		//y0 = a3.m128_f32[3];
		y0 = a3[3];
	}

	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);

	SET_TIMESTAMP_1();
	//return mm_int.m128_f32[0] * h_3;
	return mm_int[0] * h_3;
}

/*
	x0 |x1 |x2
	x2 |x3 |x4
	x4 |x5 |x6
	x6 |x7 |x8
	x8 |x9 |x10
	x10|x11|x12
	x12|x13|x14
	x14|x15|x16
*/

real_t definite_integral_cs_avx2_256(function256_t function, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	precision = ROUND_16(precision);

	const float h = (b - a) / (real_t)precision;
	const __m256 mm_h = _mm256_set1_ps(h);
	const __m256 mm_h_3 = _mm256_set1_ps(h * 0.33333f);
	const __m256 mm_h_inc2 = _mm256_mul_ps(mm_h, m256_INC2);
	const __m256 mm_4 = _mm256_set1_ps(4.0f);
	__m256 mm_x[3];
	__m256 mm_y[3];
	__m256 mm_int = _mm256_set1_ps(0.0f);
	
	for (real_t s = a; s < b; s += h * 16)
	{
		mm_x[0] = _mm256_add_ps(_mm256_set1_ps(s), mm_h_inc2);
		mm_x[1] = _mm256_add_ps(mm_x[0], mm_h);
		mm_x[2] = _mm256_add_ps(mm_x[1], mm_h);

		// Computes function value
		mm_y[0] = function(mm_x[0]);
		mm_y[1] = _mm256_mul_ps(function(mm_x[1]), mm_4);
		mm_y[2] = function(mm_x[2]);

		// Updates accumulator
		mm_int = _mm256_add_ps(mm_int, _mm256_mul_ps(mm_h_3,
			_mm256_add_ps(mm_y[0], _mm256_add_ps(mm_y[1], mm_y[2])))
		);
	}

	// Horizontal sum
	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	SET_TIMESTAMP_1();
	//return mm_int.m256_f32[0];
	return mm_int[0];
}

real_t gaussian_prob_sse(const float mean, const float stdev, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();

	// Prevents interval overrun.
	precision = ROUND_4(precision);

	const real_t h = (b - a) / (real_t)precision;
	const real_t h_2 = h * 0.5f;
	__m128 mm_int = _mm_set_ps1(0.0f);
	__m128 mm;

	// Computes once the common coefficient of the Gauss function.
	const real_t coeff = 1.0f / (stdev * sqrt(2.0f * 3.141f));
	const real_t var = stdev * stdev;
	const __m128 mm_neg_half = _mm_set_ps1(-0.5f);
	const __m128 mm_var = _mm_set_ps1(var);
	const __m128 mm_mean = _mm_set_ps1(mean);
	__m128 mm_error;
	const __m128 mm_h_inc = _mm_mul_ps(_mm_set_ps1(h), m128_INC);
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
		mm_int = _mm_add_ps(mm_int, mm);
	}

	// Retrieves result.
	mm_int = _mm_hadd_ps(mm_int, mm_int);
	mm_int = _mm_hadd_ps(mm_int, mm_int);

	SET_TIMESTAMP_1();
	//return coeff * mm_int.m128_f32[0] * h;
	return coeff * mm_int[0] * h;
}

real_t gaussian_prob_avx2(const float mean, const float stdev, const real_t a, const real_t b, precision_t precision)
{
	SET_TIMESTAMP_0();
	// Prevents interval overrun.
	precision = ROUND_8(precision);

	const real_t h = (b - a) / (real_t)precision;
	const real_t h_2 = h * 0.5f;
	__m256 mm_int = _mm256_set1_ps(0.0f);
	__m256 mm;

	// Computes once the common coefficient of the Gauss function.
	const real_t coeff = 1.0f / (stdev * sqrt(2.0f * 3.141f));
	const real_t var = stdev * stdev;
	const __m256 mm_h_inc = _mm256_mul_ps(_mm256_set1_ps(h), m256_INC);
	const __m256 mm_neg_half = _mm256_set1_ps(-0.5f);
	const __m256 mm_var = _mm256_set1_ps(var);
	const __m256 mm_mean = _mm256_set1_ps(mean);
	__m256 mm_error;

	for (real_t s = a + h_2; s < b; s += h * 8)
	{
		// Computing errors
		mm_error = _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps(s), mm_h_inc), mm_mean);

		// Computing exponentials
		mm = fmath::exp_ps256(
			_mm256_div_ps(
				_mm256_mul_ps(mm_neg_half,
					_mm256_mul_ps(mm_error, mm_error)
				),
			mm_var)
		);

		// Computing area of rects
		mm_int = _mm256_add_ps(mm_int, mm);
	}

	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	mm_int = _mm256_hadd_ps(mm_int, mm_int);
	mm_int = _mm256_hadd_ps(mm_int, mm_int);

	SET_TIMESTAMP_1();
	//return coeff * mm_int.m256_f32[0] * h;
	return coeff * mm_int[0] * h;
}
