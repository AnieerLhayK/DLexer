#include <torch/extension.h>

torch::Tensor vecadd(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vecadd", &vecadd, "CUDA vector addition");
}
