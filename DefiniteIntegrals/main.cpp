#include <iostream>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "time_measurement.hpp"
#include "integrals.h"


// Test parameters
// Precision of the integration (number of intervals)
#define PRECISION 10000

real_t parabola(real_t x)
{
	return 2 * x* x + 3 * x + 12;
}

real_t gauss(real_t x)
{
	// Fixed mean and stdev, for test purposes
	real_t mean = 0.0f;
	real_t stdev = 1.0f;
	real_t e = (x - mean) / stdev;
	return exp(-0.5f * e * e) * 1 / (stdev * sqrt(2 * 3.14f));
}

int main()
{
	std::cout << "Starting performance tests..." << std::endl
		<< "Precision:\t\t\t" << PRECISION << std::endl
		<< "Iterations for each test:\t" << ITERATIONS << std::endl << std::endl;

	/*
	std::cout << "Starting parabola tests..." << std::endl;
	// Integration of simple parabola with rectangles approximation
	TEST_PERFORMANCE(definite_integral_rectangles(parabola, 0.0f, 1.0f, PRECISION), t1_rect,
		"int(parabola) with rectangles approximation");

	// Integration of simple parabola with rectangles approximation + SSE
	TEST_PERFORMANCE(definite_integral_rectangles_sse(parabola, 0.0f, 1.0f, PRECISION), t1_rect_sse,
		"int(parabola) with rectangles approximation + SSE");

	// Integration of simple parabola with Cavalieri-Simpson approximation
	TEST_PERFORMANCE(definite_integral_cs(parabola, 0.0f, 1.0f, PRECISION), t1_cs,
		"int(parabola) with Cavalieri-Simpson rule approximation");

	// Integration of simple parabola with Cavalieri-Simpson approximation + SSE
	TEST_PERFORMANCE(definite_integral_cs_sse(parabola, 0.0f, 1.0f, PRECISION), t1_cs_sse,
		"int(parabola) with Cavalieri-Simpson rule approximation + SSE");
	*/

	std::cout << "Starting Gaussian function tests..." << std::endl;
	// Gaussian integration with rectangle approximation
	TEST_PERFORMANCES(definite_integral_rectangles(gauss, -3.0f, 3.0f, PRECISION), t2_rect,
		"int(gauss) with rectangles approximation");

	// Gaussian integration with rectangle approximation + SSE
	TEST_PERFORMANCES(definite_integral_rectangles_sse(gauss, -3.0f, 3.0f, PRECISION), t2_rect_sse,
		"int(gauss) with rectangles approximation + SSE");

	// Gaussian integration with Cavalieri-Simpson rule approximation
	TEST_PERFORMANCES(definite_integral_cs(gauss, -3.0f, 3.0f, PRECISION), t2_cs,
		"int(gauss) with Cavalieri-Simpson rule approximation");

	// Gaussian integration with Cavalieri-Simpson rule approximation + SSE
	TEST_PERFORMANCES(definite_integral_cs_sse(gauss, -3.0f, 3.0f, PRECISION), t2_cs_sse,
		"int(gauss) with Cavalieri-Simpson rule approximation + SSE");

	// Gaussian intgration with ad-hoc optimization + SSE
	TEST_PERFORMANCES(gaussian_prob_sse(0.0f, 1.0f, -3.0f, 3.0f, PRECISION ), t3_rect_sse,
		"int(gauss) with ad-hoc optimization (SSE)");

	TEST_PERFORMANCES(gaussian_prob_avx2(0.0f, 1.0f, -3.0f, 3.0f, PRECISION), t3_rect_avx2,
		"int(gauss) with ad-hoc optimization (AVX2 with m256 registers)");
}

