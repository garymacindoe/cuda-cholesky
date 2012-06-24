template <typename T>
__global__ void all(T * x, size_t xinc, T a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] = a;
}

template <typename T>
__global__ void basis(T * x, size_t xinc, size_t index, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] = (i == index) ? T(1) : T(0);
}

template <typename T>
__global__ void add(T * x, size_t xinc, T * y, size_t yinc, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] += y[i * yinc];
}

template <typename T>
__global__ void addConst(T * x, size_t xinc, T a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] += a;
}

template <typename T>
__global__ void subtract(T * x, size_t xinc, T * y, size_t yinc, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] -= y[i * yinc];
}

template <typename T>
__global__ void subtractConst(T * x, size_t xinc, T a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] -= a;
}

template <typename T>
__global__ void multiply(T * x, size_t xinc, T * y, size_t yinc, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] *= y[i * yinc];
}

template <typename T>
__global__ void multiplyConst(T * x, size_t xinc, T a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] *= a;
}

template <typename T>
__global__ void divide(T * x, size_t xinc, T * y, size_t yinc, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] /= y[i * yinc];
}

template <typename T>
__global__ void divideConst(T * x, size_t xinc, T a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] /= a;
}
