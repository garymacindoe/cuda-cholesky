/**
 * Some different types of loop structures to test which the compiler will
 * automatically vectorise/parallelise.
 *
 * Compile with "gcc -ftree-vectorize -ftree-parallelize-loops=8" to activate
 * gcc's vectoriser (using SSE) and paralleliser (using GNU openMP).  Use
 * "icc -parallel" to activate icc's vectoriser (using SSE) and paralleliser
 * (using Intel's openMP).  Intel's openMP uses a fixed size thread pool
 * allocated on startup whereas GNU's openMP uses a fork-join model.  Intel's
 * parallelisation therefore runs a lot faster than GNU's.  Vectorisation is
 * unaffected.
 */

#include <stdlib.h>

// void sgemv(const float a, const float * A, const unsigned int lda, const float * x, const unsigned int incx, const float b, float * y, const unsigned int incy, const unsigned int m, const unsigned int n) {
//   for (unsigned int i = 0; i < m; i++) {
//     float sum = 0.f;
//     for (unsigned int j = 0; j < n; j++)
//       sum += a * A[i * lda + j] * x[j * incx];
//     y[i * incy] = sum + b *  y[i * incy];
//   }
// }

struct array {
  float * data;
  unsigned int n;
};

void fill(struct array * A, float a) {
  const unsigned long n = A->n;
  for (unsigned long i = 0; i < n; i++)
    A->data[i] = a;
}

float sum(const float * array, const unsigned long inc, const unsigned long n) {
  float res = array[0], * e = (float *)calloc(n, sizeof(float));
#pragma ivdep
  for (unsigned long i = 1; i < n; i++) {
    float t = res;
    float y = array[i] + e[i-1];
    res = t + y;
    e[i] = (t - res) + y;
  }
  free(e);
  return res;
}
// #define ARR_SIZE 500 //Define array
// int main()
// {
//   int matrix[ARR_SIZE][ARR_SIZE];
//   int arrA[ARR_SIZE]={10};
//   int arrB[ARR_SIZE]={30};
//   int i, j;
//   for(i=0;i<ARR_SIZE;i++)
//    {
//      for(j=0;j<ARR_SIZE;j++)
//       {
//        matrix[i][j] = arrB[i]*(arrA[i]%2+10);
//       }
//    }
// }
