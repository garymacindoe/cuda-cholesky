#include <stdio.h>
#include <stdbool.h>

#define GPU_FLOPS (705.15E9)
#define CPU_FLOPS (53.08E9)
#define LATENCY_HTOD (0.00001)
#define BANDWIDTH_HTOD (double)(1500 << 20)
#define LATENCY_DTOH (0.000057)
#define BANDWIDTH_DTOH (double)(1447 << 20)

typedef enum __side {left, right} side;
typedef enum __direction {htod, dtoh} direction;

static inline size_t gemm_flops(size_t m, size_t k, size_t n) { return 2 * m * k * n; }
static inline size_t syrk_flops(size_t k, size_t n) { return k * n * (n + 1); }
static inline size_t trmm_flops(side s, size_t m, size_t n) { return (s == left) ? n * m * m : n * n * m; }
static inline size_t trsm_flops(side s, size_t m, size_t n) { return (s == left) ? n * m * m : n * n * m; }

static inline size_t lauum_flops(size_t n) {
  size_t res = 0;
  for (size_t i = 0; i < n; i++)
    res += n - i + n - i - 1 + 2 * (n - i) * i;
  return res + n;
}

static inline size_t trtri_flops(size_t n) {
  size_t res = 0;
  for (size_t i = 0; i < n; i++)
    res += i * i + i;
  return res;
}

static inline size_t potrf_flops(size_t n) {
  size_t res = 0;
  for (size_t i = 0; i <= n; i++)
    res += i * i;
  return res;
}

static inline size_t potri_flops(size_t n) {
  size_t res = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = j; k < i; k++)
        res += 2;
      res++;
    }
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j <= i; j++) {
      for (size_t k = i + 1; k < n; k++)
        res += 2;
      res++;
    }
  }

  return res;
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
  const size_t N = 4096, nb = 64;

#ifdef POTRF
  fprintf(stdout, "\"Iteration\",\"SYRK\",\"POTRF + COPY\",\"POTRF + BCC\",\"GEMM\",\"TRSM\"\n");
  for (size_t i = 0; i < N / nb; i++)
    fprintf(stdout, "%zu,%f,%f,%f,%f,%f\n", i, syrk_time(i * nb, nb), potrf_time(nb) + transfer_time(htod, nb, nb, false) + transfer_time(dtoh, nb, nb, false), potrf_time(nb) + transfer_time(htod, N, nb, true) + transfer_time(dtoh, N, nb, true), gemm_time(nb, nb * i, N - nb * i - nb), trsm_time(left, nb, N - i * nb - nb));
#elif defined(POTRI1)
  fprintf(stdout, "\"Iteration\",\"TRMM\",\"TRSM\",\"TRTRI + COPY\",\"TRTRI + BCC\"\n");
  for (size_t i = 0; i < N / nb; i++)
    fprintf(stdout, "%zu,%f,%f,%f,%f\n", i, trmm_time(left, i * nb, nb), trsm_time(right, i * nb, nb), trtri_time(nb) + transfer_time(htod, nb, nb, false) + transfer_time(dtoh, nb, nb, false), trtri_time(nb) + transfer_time(htod, N, nb, true) + transfer_time(dtoh, N, nb, true));
#elif defined(POTRI2)
  fprintf(stdout, "\"Iteration\",\"TRMM\",\"GEMM\",\"LAUUM + COPY\",\"LAUUM + BCC\",\"SYRK\"\n");
  for (size_t i = 0; i < N / nb; i++)
    fprintf(stdout, "%zu,%f,%f,%f,%f,%f\n", i, trmm_time(right, i * nb, nb), gemm_time(i * nb, N - i * nb - nb, nb), lauum_time(nb) + transfer_time(htod, nb, nb, false) + transfer_time(dtoh, nb, nb, false), lauum_time(nb) + transfer_time(htod, N, nb, true) + transfer_time(dtoh, N, nb, true), syrk_time(N - i * nb - nb, nb));
#endif

  return 0;
}
