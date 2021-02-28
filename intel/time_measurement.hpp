#ifndef GLOBAL_TIMER_H
#define GLOBAL_TIMER_H
#include <chrono>

// Time measurement utilities
using namespace std::chrono;
using time_stamp = std::chrono::time_point<std::chrono::system_clock,
	std::chrono::nanoseconds>;

class timer
{
public:
	static time_stamp ts0;
	static time_stamp ts1;
};

// Instances to compute for each function.
#define ITERATIONS 10000

// Macros for efficient code testing.
#define CREATE_TIMESTAMP(timestamp_name)	time_stamp timestamp_name = time_point_cast<nanoseconds>(system_clock::now())
#define ELAPSED_TIME(t0, t1) (double)(t1 - t0).count() / 1000.0

#define MAX_FLOAT 100000.0f
#define MIN_FLOAT -100000.0f

#define GET_TIMESTAMP(timestamp_name) timestamp_name = time_point_cast<nanoseconds>(system_clock::now())
#define SET_TIMESTAMP_0() GET_TIMESTAMP(timer::ts0)
#define SET_TIMESTAMP_1() GET_TIMESTAMP(timer::ts1)
#define GET_LAST_ELAPSED_TIME() ELAPSED_TIME(timer::ts0, timer::ts1)

#define TEST_PERFORMANCES(function_call, out_var, test_id)	{										\
	time_stamp ts0, ts1;																			\
	long double avg = 0.0f, min = MAX_FLOAT, max = MIN_FLOAT;										\
	real_t res = 0.0f;																				\
	for (int i = 0; i < ITERATIONS; i++)															\
	{																								\
		res += function_call;																		\
		long double t = GET_LAST_ELAPSED_TIME();													\
		avg += t;																					\
		if (t < min) min = t;																		\
		if (t > max) max = t;																		\
	}																								\
	real_t out_var = res / (real_t)ITERATIONS;														\
	std::cout << test_id << std::endl << "\tResult = " << out_var << std::endl						\
		/*<< "\tTotal execution time = " << avg << " ms" << std::endl								*/\
		<< "\tAverage execution time = " << avg / (long double) ITERATIONS << " ms" << std::endl	\
		/*<< "\tMinimum time = " << min << " ms" << std::endl										*/\
		/*<< "\tMaximum time = " << max << " ms" << std::endl */;										\
}

#endif