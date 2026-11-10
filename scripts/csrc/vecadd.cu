#include <torch/extension.h>

__global__ void vecadd_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

torch::Tensor vecadd(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    const int n = static_cast<int>(a.numel());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vecadd_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
