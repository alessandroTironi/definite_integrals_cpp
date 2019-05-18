#include <iostream>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "fmath.hpp"

#include "time_measurement.hpp"
#include "integrals.h"


// Test parameters
// Precision of the integration (number of intervals)
#define PRECISION 100000

const real_t sqrt_2pi = sqrt(2 * 3.14f);

real_t gauss(real_t x)
{
	// Fixed mean and stdev, for test purposes
	real_t mean = 0.0f;
	real_t stdev = 1.0f;
	real_t e = (x - mean) / stdev;
	return exp(-0.5f * e * e) * 1 / (stdev * sqrt_2pi);
}

__m128 gauss128(__m128 x)
{
	const float mean = 0.0f;
	const __m128 mm_mean = _mm_set_ps1(mean);
	const float stdev = 1.0f;
	const __m128 mm_stdev = _mm_set_ps1(stdev);
	const __m128 mm_e = _mm_div_ps(_mm_sub_ps(x, mm_mean), mm_stdev);
	const __m128 mm_coeff = _mm_set_ps1(1 / (stdev * sqrt(2 * 3.14f)));

	return _mm_mul_ps(
		fmath::exp_ps(
			_mm_mul_ps(_mm_set_ps1(-0.5f),
				_mm_mul_ps(mm_e, mm_e))
		), mm_coeff
	);
}

__m256 gauss256(__m256 x)
{
	float mean = 0.0f;
	__m256 mm_mean = _mm256_set1_ps(mean);
	float stdev = 1.0f;
	__m256 mm_stdev = _mm256_set1_ps(stdev);
	__m256 mm_e = _mm256_div_ps(_mm256_sub_ps(x, mm_mean), mm_stdev);
	__m256 mm_coeff = _mm256_set1_ps(1 / (stdev * sqrt(2 * 3.14f)));

	return _mm256_mul_ps(
		fmath::exp_ps256(
			_mm256_mul_ps(_mm256_set1_ps(-0.5f),
				_mm256_mul_ps(mm_e, mm_e))
		), mm_coeff
	);
}

int main()
{
	std::cout << "Starting performance tests..." << std::endl
		<< "Precision:\t\t\t" << PRECISION << std::endl
		<< "Iterations for each test:\t" << ITERATIONS << std::endl << std::endl;

	//std::cout << "Starting Gaussian function tests..." << std::endl;
	// Gaussian integration with rectangle approximation
	TEST_PERFORMANCES(definite_integral_rectangles(gauss, -3.0f, 3.0f, PRECISION), t2_rect,
		"int(gauss) with rectangles approximation");

	
	// Gaussian integration with rectangle approximation + SSE
	TEST_PERFORMANCES(definite_integral_rectangles_sse(gauss, -3.0f, 3.0f, PRECISION), t2_rect_sse,
		"int(gauss) with rectangles approximation + SSE");

	//TEST_PERFORMANCES(definite_integral_rectangles_sse_128(gauss128, -3.0f, 3.0f, PRECISION), t2_rect128_sse,
	//	"int(gauss) with rectangles approximation (SSE + 128 bit function vectorization)");
	TEST_PERFORMANCES(definite_integral_rectangles_avx_256(gauss256, -3.0f, 3.0f, PRECISION), t2_rect128_sse,
		"int(gauss) with rectangles approximation (AVX2 + 256 bit function vectorization)");

	// Gaussian integration with Cavalieri-Simpson rule approximation
	TEST_PERFORMANCES(definite_integral_cs(gauss, -3.0f, 3.0f, PRECISION), t2_cs,
		"int(gauss) with Cavalieri-Simpson rule approximation");

	// Gaussian integration with Cavalieri-Simpson rule approximation + SSE
	TEST_PERFORMANCES(definite_integral_cs_sse(gauss, -3.0f, 3.0f, PRECISION), t2_cs_sse,
		"int(gauss) with Cavalieri-Simpson rule approximation + SSE");
	TEST_PERFORMANCES(definite_integral_cs_avx2_256(gauss256, -3.0f, 3.0f, PRECISION), t2_cs_avx256,
		"int(gauss) with Cavalieri-Simpson rule approximation (AVX2 + 256-bit vectorized function)");

	// Gaussian intgration with ad-hoc optimization + SSE
	//TEST_PERFORMANCES(gaussian_prob_sse(0.0f, 1.0f, -3.0f, 3.0f, PRECISION ), t3_rect_sse,
	//	"int(gauss) with ad-hoc optimization (SSE)");

	TEST_PERFORMANCES(gaussian_prob_avx2(0.0f, 1.0f, -3.0f, 3.0f, PRECISION), t3_rect_avx2,
		"int(gauss) with ad-hoc optimization (AVX2 with m256 registers)");
}

