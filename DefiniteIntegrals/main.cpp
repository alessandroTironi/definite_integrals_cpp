#include <iostream>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "integrals.h"

// Test parameters
// Precision of the integration (number of intervals)
#define PRECISION 10000
// Instances to compute for each function.
#define ITERATIONS 1000

// Macros for efficient testing code.
#define GET_TIMESTAMP(timestamp_name)	time_stamp timestamp_name = time_point_cast<microseconds>(system_clock::now())
#define ELAPSED_TIME(t0, t1) (double)(t1 - t0).count() / 1000.0

// Time measurement utilities
using namespace std::chrono;
using time_stamp = std::chrono::time_point<std::chrono::system_clock,
	std::chrono::microseconds>;

#define TEST_PERFORMANCE(function_call, out_var, test_id)	{										\
	GET_TIMESTAMP(ts0);																				\
	real_t res = 0.0f;																				\
	for (int i = 0; i < ITERATIONS; i++)															\
		res += function_call;																		\
	GET_TIMESTAMP(ts1);																				\
	real_t out_var = res / (real_t)ITERATIONS;														\
	long double t = ELAPSED_TIME(ts0, ts1);															\
	std::cout << test_id << " = " << out_var << ", execution time = " << t << " ms" << std::endl;	\
}


inline real_t parabola(real_t x)
{
	return 2 * x* x + 3 * x + 12;
}

inline real_t gauss(real_t x)
{
	// Fixed mean and stdev, for test purposes
	real_t mean = 10.f;
	real_t stdev = 1.5f;
	real_t e = (x - mean) / stdev;
	return exp(-0.5f * e * e) * 1 / (stdev * sqrt(2 * 3.14f));
}

int main()
{
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

	std::cout << "Starting Gaussian function tests..." << std::endl;
	// Gaussian integration with rectangle approximation
	TEST_PERFORMANCE(definite_integral_rectangles(gauss, 0.0f, 20.0f, PRECISION), t2_rect,
		"int(gauss) with rectangles approximation");

	// Gaussian integration with rectangle approximation + SSE
	TEST_PERFORMANCE(definite_integral_rectangles_sse(gauss, 0.0f, 20.0f, PRECISION), t2_rect_sse,
		"int(gauss) with rectangles approximation + SSE");

	// Gaussian integration with Cavalieri-Simpson rule approximation
	TEST_PERFORMANCE(definite_integral_cs(gauss, 0.0f, 20.0f, PRECISION), t2_cs,
		"int(gauss) with Cavalieri-Simpson rule approximation");

	// Gaussian integration with Cavalieri-Simpson rule approximation + SSE
	TEST_PERFORMANCE(definite_integral_cs_sse(gauss, 0.0f, 20.0f, PRECISION), t2_cs_sse,
		"int(gauss) with Cavalieri-Simpson rule approximation + SSE");

	// Gaussian intgration with ad-hoc optimization + SSE
	TEST_PERFORMANCE(gaussian_prob_sse(10.0f, 1.5f, 0.0f, 20.0f, PRECISION), t3_rect_sse,
		"int(gauss) with ad-hoc optimization (SSE)");
}

