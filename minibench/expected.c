#include <stdio.h>
#include <stdbool.h>
#include <math.h>

// single (laptop)
// #define GPU_FLOPS (25.3E9)
// #define CPU_FLOPS (17.1E9)
// #define BANDWIDTH_HTOD (double)(573 << 20)
// #define BANDWIDTH_DTOH (double)(430 << 20)
//
// #define LATENCY_HTOD (0.000018)
// #define LATENCY_DTOH (0.000018)

// single (PC)
// #define GPU_FLOPS (705.15E9)
// #define CPU_FLOPS (53.08E9)
// #define BANDWIDTH_HTOD (double)(1500 << 20)
// #define BANDWIDTH_DTOH (double)(1447 << 20)

// double (PC)
#define GPU_FLOPS (132.84E9)
#define CPU_FLOPS (41.64E9)
#define BANDWIDTH_HTOD (double)(750 << 20)
#define BANDWIDTH_DTOH (double)(723 << 20)

#define LATENCY_HTOD (0.00001)
#define LATENCY_DTOH (0.000057)

typedef enum __side {left, right} side;
typedef enum __direction {htod, dtoh} direction;

static inline size_t gemm_flops(size_t m, size_t k, size_t n) { return 2 * m * k * n; }
static inline size_t syrk_flops(size_t k, size_t n) { return k * n * (n + 1); }
static inline size_t trmm_flops(side s, size_t m, size_t n) { return (s == left) ? n * m * m : n * n * m; }
static inline size_t trsm_flops(side s, size_t m, size_t n) { return (s == left) ? n * m * m : n * n * m; }

static inline size_t lauum_flops(size_t n) {
  size_t res = 0;
  for (size_t i = 0; i < n; i++)
    res += (2 * (n - i) - 1) * (i + 1);
  return res + n;
}

static inline size_t trtri_flops(size_t n) {
  size_t res = 0;
  for (size_t i = 0; i < n; i++)
    res += 1 + i * i + i;
  return res;
}

static inline size_t potrf_flops(size_t n) {
//   size_t res = 0;
//   for (size_t i = 0; i <= n; i++)
//     res += i * i;
//   return res;
  return 0.333333 * n * n * n + 0.5 * n * n + 0.166667 * n;
}

static inline size_t potri_flops(size_t n) {
//   size_t res = 0;
//
//   for (size_t i = 0; i < n; i++) {
//     for (size_t j = 0; j < n; j++) {
//       for (size_t k = j; k < i; k++)
//         res += 2;
//       res++;
//     }
//   }
//
//   for (size_t i = 0; i < n; i++) {
//     for (size_t j = 0; j <= i; j++) {
//       for (size_t k = i + 1; k < n; k++)
//         res += 2;
//       res++;
//     }
//   }
//
//   return res;
//   return lauum_flops(n) + trtri_flops(n);
  return 0.666667 * n * n * n + 0.5 * n * n + 0.833333 * n;
}

static inline double transfer_time(direction d, size_t m, size_t n, bool bcc) {
  return  (bcc) ? ((d == htod) ? LATENCY_HTOD : LATENCY_DTOH) + (double)(n * m) / (double)((d == htod) ? BANDWIDTH_HTOD : BANDWIDTH_DTOH) : ((d == htod) ? LATENCY_HTOD : LATENCY_DTOH) * (double)n + (double)m / (double)((d == htod) ? BANDWIDTH_HTOD : BANDWIDTH_DTOH);
}

static inline double gemm_time(size_t m, size_t k, size_t n) { return (double)gemm_flops(m, k, n) / GPU_FLOPS; }
static inline double syrk_time(size_t k, size_t n) { return (double)syrk_flops(k, n) / GPU_FLOPS; }
static inline double trmm_time(side s, size_t m, size_t n) { return (double)trmm_flops(s, m, n) / GPU_FLOPS; }
static inline double trsm_time(side s, size_t m, size_t n) { return (double)trsm_flops(s, m, n) / GPU_FLOPS; }
static inline double lauum_time(size_t n) { return (double)lauum_flops(n) / CPU_FLOPS; }
static inline double trtri_time(size_t n) { return (double)trtri_flops(n) / CPU_FLOPS; }
static inline double potrf_time(size_t n) { return (double)potrf_flops(n) / CPU_FLOPS; }
static inline double potri_time(size_t n) { return (double)potri_flops(n) / CPU_FLOPS; }

int main(int argc, char * argv[]) {
  const size_t NN = 8192, nb = 64;

#ifdef POTRF
  fprintf(stdout, "\"Size\",\"Hybrid Cholesky\",\"Hybrid Cholesky with Block Column Copy\"\n");
  for (size_t N = nb; N <= NN; N += nb) {
    double time0 = 0.0, time1 = 0.0;
    for (size_t i = 0; i < N / nb; i++) {
      double syrk = syrk_time(i * nb, nb);
      double potrf = potrf_time(nb);
      double potrf0 = potrf + transfer_time(htod, nb, nb, false) + transfer_time(dtoh, nb, nb, false);
      double potrf1 = potrf + transfer_time(htod, N, nb, true) + transfer_time(dtoh, N, nb, true);
      double gemm = gemm_time(nb, nb * i, N - nb * i - nb);
      double trsm = trsm_time(left, nb, N - i * nb - nb);
      time0 += syrk + fmax(potrf0, gemm) + trsm;
      time1 += syrk + fmax(potrf1, gemm) + trsm;
    }
    fprintf(stdout, "%zu,%f,%f\n", N, (double)potrf_flops(N) * 1.E-9 / time0, (double)potrf_flops(N) * 1.E-9 / time1);
  }
#elif defined(POTRI)
  fprintf(stdout, "\"Size\",\"Hybrid Inverse\",\"Hybrid Inverse with Block Column Copy\"\n");
  for (size_t N = nb; N <= NN; N += nb) {
    double time0 = 0.0, time1 = 0.0;
    for (size_t i = 0; i < N / nb; i += nb) {
      double trmm0 = trmm_time(left, i * nb, nb);
      double trsm0 = trsm_time(right, i * nb, nb);
      double trtri = trtri_time(nb);
      double trtri0 = trtri + transfer_time(htod, nb, nb, false) + transfer_time(dtoh, nb, nb, false);
      double trtri1 = trtri + transfer_time(htod, N, nb, true) + transfer_time(dtoh, N, nb, true);
      time0 += fmax(trmm0 + trsm0, trtri0);
      time1 += fmax(trmm0 + trsm0, trtri1);

      double trmm1 = trmm_time(right, i * nb, nb);
      double gemm1 = gemm_time(i * nb, N - i * nb - nb, nb);
      double lauum = lauum_time(nb);
      double lauum0 = lauum + transfer_time(htod, nb, nb, false) + transfer_time(dtoh, nb, nb, false);
      double lauum1 = lauum + transfer_time(htod, N, nb, true) + transfer_time(dtoh, N, nb, true);
      double syrk1 = syrk_time(N - i * nb - nb, nb);
      time0 += fmax(trmm1 + gemm1, lauum0) + syrk1;
      time1 += fmax(trmm1 + gemm1, lauum1) + syrk1;
    }
    fprintf(stdout, "%zu,%f,%f\n", N, (double)potri_flops(N) * 1.E-9 / time0, (double)potri_flops(N) * 1.E-9 / time1);
  }
#endif

  return 0;
}
