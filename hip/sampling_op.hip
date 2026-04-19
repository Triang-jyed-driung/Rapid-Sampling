#include <torch/extension.h>

at::Tensor batch_sampling_repetition_temperature_topk_topp(
    at::Tensor& logits,
    at::Tensor& penalties,
    at::Tensor& states,
    
    at::Tensor& temperatures,
    at::Tensor& top_ks,
    at::Tensor& top_ps,
    at::Tensor& presence_penalties,
    at::Tensor& repetition_penalties,
    at::Tensor& penalty_decays,
    at::Tensor& eos_mask
);

at::Tensor batch_sampling_temperature_topk_topp(
    at::Tensor& logits,
    at::Tensor& states,

    at::Tensor& temperatures,
    at::Tensor& top_ks,
    at::Tensor& top_ps,
    at::Tensor& eos_mask
);

at::Tensor setup_rand(int64_t seed, int64_t B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("setup_rand", &setup_rand);

    m.def("batch_sampling_repetition_temperature_topk_topp", 
          &batch_sampling_repetition_temperature_topk_topp);

    m.def("batch_sampling_temperature_topk_topp", 
          &batch_sampling_temperature_topk_topp);
}