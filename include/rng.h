#ifndef RNG_H
#define RNG_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

// nvcc uses __restrict__ instead of C99's restrict keyword
// CUDA Programming Guide v4.1 Appendix B.2.4
#ifdef __CUDACC__
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 32-bit CPU Pseudo-random number generator algorithm.
 */
typedef struct __rng32_type_st * rng32_t;

/**
 * 32-bit CPU Pseudo-random number generator.
 */
typedef struct __rng32_st * rng32;

/**
 * Creates a new PRNG that generates 32-bit pseudo-random integers and floating
 * point numbers using the CPU.
 *
 * @param rng   the newly created PRNG is returned through this pointer
 * @param type  the PRNG algorithm to use
 * @return 0 on success, or ENOMEM if there is not enough memory to create
 *         another PRNG.
 */
int rng32Create(rng32 *, const rng32_t);

/**
 * Destroys a PRNG.
 *
 * @param rng  the PRNG to destroy.
 */
void rng32Destroy(rng32);

/**
 * Seeds the PRNG.
 *
 * @param rng   the PRNG to seed.
 * @param seed  the seed.
 */
void rng32Set(const rng32, uint32_t);

/**
 * Fills a vector with 32-bit pseudo-random integers.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng32Get(const rng32, uint32_t *, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over (0, 1), i.e. 0 < x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng32GetOpenOpen(const rng32, float *, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over (0, 1], i.e. 0 < x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng32GetOpenClose(const rng32, float *, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over [0, 1), i.e. 0 <= x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng32GetCloseOpen(const rng32, float *, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over [0, 1], i.e. 0 <= x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng32GetCloseClose(const rng32, float *, size_t);

/**
 * 32-bit PRNG type wrapping stdlib.h's rand() function.
 */
extern const rng32_t std_rand_t;

/* These generators are based on code from
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 * (source URLs relative to http://www.math.sci.hiroshima-u.ac.jp/~m-mat/) */
/**
 * Implementation of the standard 32-bit Mersenne Twister PRNG
 */
extern const rng32_t mt32_19937_t;  // MT/MT2002/CODES/mt19937ar.tgz

/**
 * SIMD-Oriented Fast Mersenne Twister for CPUs.
 */
extern const rng32_t sfmt_607_t;    // bin/dl/dl.cgi?SFMT:SFMT-src-1.3.3.tar.gz
extern const rng32_t sfmt_1279_t;
extern const rng32_t sfmt_2281_t;
extern const rng32_t sfmt_4253_t;
extern const rng32_t sfmt_11213_t;
extern const rng32_t sfmt_19937_t;
extern const rng32_t sfmt_44497_t;
extern const rng32_t sfmt_86243_t;
extern const rng32_t sfmt_132049_t;
extern const rng32_t sfmt_216091_t;


/**
 * 64-bit CPU Pseudo-random number generator algorithm.
 */
typedef struct __rng64_type_st * rng64_t;

/**
 * 64-bit CPU Pseudo-random number generator.
 */
typedef struct __rng64_st * rng64;

/**
 * Creates a new PRNG that generates 64-bit pseudo-random integers and double
 * precision floating point numbers using the CPU.
 *
 * @param rng   the newly created PRNG is returned through this pointer
 * @param type  the PRNG algorithm to use
 * @return 0 on success, or ENOMEM if there is not enough memory to create
 *         another PRNG.
 */
int rng64Create(rng64 *, const rng64_t);

/**
 * Destroys a PRNG.
 *
 * @param rng  the PRNG to destroy.
 */
void rng64Destroy(rng64);

/**
 * Seeds the PRNG.
 *
 * @param rng   the PRNG to seed.
 * @param seed  the seed.
 */
void rng64Set(const rng64, uint64_t);

/**
 * Fills a vector with 64-bit pseudo-random integers.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng64Get(const rng64, uint64_t *, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over (0, 1), i.e. 0 < x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng64GetOpenOpen(const rng64, double *, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over (0, 1], i.e. 0 < x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng64GetOpenClose(const rng64, double *, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over [0, 1), i.e. 0 <= x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng64GetCloseOpen(const rng64, double *, size_t);

/**
 * Fills a vector with 64-bit pseudo-random floating point numbers distributed
 * over [0, 1], i.e. 0 <= x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
void rng64GetCloseClose(const rng64, double *, size_t);

/* 64-bit CPU PRNGs based on 64-bit Mersenne Twister and 64-bit SIMD-Oriented
 * Fast Mersenne Twister */
/**
 * 64-bit implementation of the standard Mersenne Twister PRNG
 */
extern const rng64_t mt64_19937_t;  // MT/mt19937-64.tgz

/**
 * 64-bit SIMD-Oriented Fast Mersenne Twister for CPUs.
 */
extern const rng64_t dsfmt_521_t;   // bin/dl/dl.cgi?SFMT:dSFMT-src-2.1.tar.gz
extern const rng64_t dsfmt_1279_t;
extern const rng64_t dsfmt_2203_t;
extern const rng64_t dsfmt_4253_t;
extern const rng64_t dsfmt_11213_t;
extern const rng64_t dsfmt_19937_t;
extern const rng64_t dsfmt_44497_t;
extern const rng64_t dsfmt_86243_t;
extern const rng64_t dsfmt_132049_t;
extern const rng64_t dsfmt_216091_t;


/**
 * 32-bit GPU Pseudo-random number generator algorithm.
 */
typedef struct __curng32_type_st * CUrng32_t;

/**
 * 32-bit GPU Pseudo-random number generator.
 */
typedef struct __curng32_st * CUrng32;

/**
 * Creates a new PRNG that generates 32-bit pseudo-random integers and floating
 * point numbers using the GPU.
 *
 * @param rng   the newly created PRNG is returned through this pointer
 * @param type  the PRNG algorithm to use
 * @return CUDA_SUCCESS on success, or CUDA_ERROR_OUT_OF_MEMORY if there is not
 *         enough memory to create another PRNG.
 */
CUresult cuRng32Create(CUrng32 *, const CUrng32_t);

/**
 * Destroys a PRNG.
 *
 * @param rng  the PRNG to destroy.
 */
CUresult cuRng32Destroy(CUrng32);

/**
 * Seeds the PRNG.
 *
 * @param rng   the PRNG to seed.
 * @param seed  the seed.
 */
CUresult cuRng32Set(const CUrng32, uint32_t);

/**
 * Fills a vector with 32-bit pseudo-random integers.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng32Get(const CUrng32, CUdeviceptr, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over (0, 1), i.e. 0 < x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng32GetOpenOpen(const CUrng32, CUdeviceptr, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over (0, 1], i.e. 0 < x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng32GetOpenClose(const CUrng32, CUdeviceptr, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over [0, 1), i.e. 0 <= x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng32GetCloseOpen(const CUrng32, CUdeviceptr, size_t);

/**
 * Fills a vector with 32-bit pseudo-random floating point numbers distributed
 * over [0, 1], i.e. 0 <= x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng32GetCloseClose(const CUrng32, CUdeviceptr, size_t);

/** 32 CUDA PRNG based on Mersenne Twister for Graphics Processors */
extern const CUrng32_t mtgp32_11213_t;  // bin/dl/dl.cgi?MTGP:MTGP-src-1.0.2.tar.gz


/**
 * 64-bit GPU Pseudo-random number generator algorithm.
 */
typedef struct __curng64_type_st * CUrng64_t;

/**
 * 64-bit GPU Pseudo-random number generator.
 */
typedef struct __curng64_st * CUrng64;

/**
 * Creates a new PRNG that generates 64-bit pseudo-random integers and double
 * precision floating point numbers using the GPU.
 *
 * @param rng   the newly created PRNG is returned through this pointer
 * @param type  the PRNG algorithm to use
 * @return CUDA_SUCCESS on success, or CUDA_ERROR_OUT_OF_MEMORY if there is not
 *         enough memory to create another PRNG.
 */
CUresult cuRng64Create(CUrng64 *, const CUrng64_t);

/**
 * Destroys a PRNG.
 *
 * @param rng  the PRNG to destroy.
 */
CUresult cuRng64Destroy(CUrng64);

/**
 * Seeds the PRNG.
 *
 * @param rng   the PRNG to seed.
 * @param seed  the seed.
 */
CUresult cuRng64Set(const CUrng64, uint64_t);

/**
 * Fills a vector with 64-bit pseudo-random integers.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng64Get(const CUrng64, CUdeviceptr, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over (0, 1), i.e. 0 < x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng64GetOpenOpen(const CUrng64, CUdeviceptr, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over (0, 1], i.e. 0 < x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng64GetOpenClose(const CUrng64, CUdeviceptr, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over [0, 1), i.e. 0 <= x < 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng64GetCloseOpen(const CUrng64, CUdeviceptr, size_t);

/**
 * Fills a vector with 64-bit pseudo-random double precision floating point
 * numbers distributed over [0, 1], i.e. 0 <= x <= 1.
 *
 * @param rng  the PRNG
 * @param x    the vector
 * @param n    the size of the vector
 */
CUresult cuRng64GetCloseClose(const CUrng64, CUdeviceptr, size_t);

/** 64-bit CUDA PRNGs based on Mersenne Twister for Graphics Processors */
extern const CUrng64_t mtgp64_11213_t;

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#undef restrict
#endif

#endif
