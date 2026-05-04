from typing import Union, Optional, List
import torch
import os
from torch.utils.cpp_extension import load

ROCm_flag = torch.version.hip is not None
sampler_path = os.path.dirname(__file__)


class Sampler:
    def __init__(
        self, verbose: bool = False, max_batch: int = 1, vocab_size: int = 32000
    ):
        if ROCm_flag:
            sources = ["hip/sampling_op.hip", "hip/sampling.hip"]
            extra_cflags = ["-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics"]
        else:
            sources = ["sampling.cpp", "sampling.cu"]
            extra_cflags = ["-O3", "-res-usage", "-Xptxas", "-O3"]

        self.sample_module = load(
            name="sample",
            sources=[os.path.join(sampler_path, s) for s in sources],
            extra_cuda_cflags=extra_cflags,
            verbose=verbose,
        )

        if vocab_size % 4 != 0:
            raise ValueError(f"vocab_size must be a multiple of 4, got {vocab_size}")
        self.vocab_size = vocab_size
        self.max_batch = max_batch
        self._probs_buffer = None
        if max_batch > 0:
            self._probs_buffer = torch.empty(
                (max_batch, vocab_size), dtype=torch.float32, device="cuda"
            )

    def setup_rand(self, seed: int, batch: int) -> torch.Tensor:
        """
        初始化采样用的随机状态张量。

        Args:
            seed (int): 随机种子。
                Random seed for reproducibility.
            batch (int): 批大小（Batch Size），对应同时采样的样本数量。
                Batch size, number of samples to sample simultaneously.

        Returns:
            torch.Tensor: 包含 CUDA 随机状态结构体的张量，形状与批大小相关。
                Tensor containing CUDA random state structs, shape related to batch size.

        Initializes random state tensor for sampling.
        """
        return self.sample_module.setup_rand(seed, batch)

    @torch.no_grad()
    def raw_sample_fast(
        self,
        logits: torch.Tensor,  # (B, V) float32, contiguous
        states: torch.Tensor,  # from setup_rand (uint8), contiguous
        temperatures: torch.Tensor,  # (B,) float32, contiguous
        top_ks: torch.Tensor,  # (B,) int32, contiguous
        top_ps: torch.Tensor,  # (B,) float32, contiguous
        eos_mask: torch.Tensor,  # (B,) bool, contiguous
    ) -> torch.Tensor:
        """
        极致性能采样：无任何预处理、类型转换或形状检查。
        调用者必须保证所有输入：
        - 位于同一 CUDA 设备
        - 形状正确（logits: (B, V)，其余: (B,)）
        - 数据类型正确（logits: float32, states: uint8, temperatures: float32,
            top_ks: int32, top_ps: float32, eos_mask: bool）
        - 内存连续（contiguous）
        """
        return self.sample_module.batch_sampling_temperature_topk_topp(
            logits, states, temperatures, top_ks, top_ps, eos_mask
        )

    @torch.no_grad()
    def raw_sample_repetition_fast(
        self,
        logits: torch.Tensor,  # (B, V) float32, contiguous
        penalties: torch.Tensor,  # (B, V) float32, contiguous (in-place updated)
        states: torch.Tensor,  # from setup_rand (uint8), contiguous
        temperatures: torch.Tensor,  # (B,) float32, contiguous
        top_ks: torch.Tensor,  # (B,) int32, contiguous
        top_ps: torch.Tensor,  # (B,) float32, contiguous
        presence_penalties: torch.Tensor,  # (B,) float32, contiguous
        repetition_penalties: torch.Tensor,  # (B,) float32, contiguous
        penalty_decays: torch.Tensor,  # (B,) float32, contiguous
        eos_mask: torch.Tensor,  # (B,) bool, contiguous
    ) -> torch.Tensor:
        """
        极致性能采样：无任何预处理、类型转换或形状检查。
        调用者必须保证所有输入：
        - 位于同一 CUDA 设备
        - 形状正确（logits: (B, V)，其余: (B,)）
        - 数据类型正确（logits: float32, states: uint8, temperatures: float32,
            top_ks: int32, top_ps: float32, eos_mask: bool）
        - 内存连续（contiguous）
        """
        return self.sample_module.batch_sampling_repetition_temperature_topk_topp(
            logits,
            penalties,
            states,
            temperatures,
            top_ks,
            top_ps,
            presence_penalties,
            repetition_penalties,
            penalty_decays,
            eos_mask,
        )

    @torch.no_grad()
    def raw_sample(
        self,
        logits: torch.Tensor,  # (B, V) or (1, V) or (B, 1, V) or (B, T, V)
        states: torch.Tensor,  # 来自 setup_rand
        temperatures: torch.Tensor,  # (B,) float32, already clamped to [0.001, 1000]
        top_ks: torch.Tensor,  # (B,) int32, already clamped / -1 replaced by V
        top_ps: torch.Tensor,  # (B,) float32, already clamped to [0.0, 1.0]
        eos_mask: Optional[torch.Tensor] = None,  # (B,) bool, default False
    ) -> torch.Tensor:
        # 形状归一化（只改变视图，不拷贝）
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        batch_size = logits.size(0)

        if eos_mask is None:
            device = logits.device
            eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        return self.sample_module.batch_sampling_temperature_topk_topp(
            logits.float(),
            states.to(torch.int8),
            temperatures.float(),
            top_ks.int(),
            top_ps.float(),
            eos_mask,
        )

    @torch.no_grad()
    def raw_sample_repetition(
        self,
        logits: torch.Tensor,  # (B, V) or (B, T, V)
        penalties: torch.Tensor,  # (B, V) float32, 会被原地更新
        states: torch.Tensor,
        temperatures: torch.Tensor,  # (B,) float32
        top_ks: torch.Tensor,  # (B,) int32
        top_ps: torch.Tensor,  # (B,) float32
        presence_penalties: torch.Tensor,  # (B,) float32
        repetition_penalties: torch.Tensor,  # (B,) float32
        penalty_decays: torch.Tensor,  # (B,) float32
        eos_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        batch_size = logits.size(0)

        if eos_mask is None:
            device = logits.device
            eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        return self.sample_module.batch_sampling_repetition_temperature_topk_topp(
            logits,
            penalties,
            states,
            temperatures,
            top_ks,
            top_ps,
            presence_penalties,
            repetition_penalties,
            penalty_decays,
            eos_mask,
        )

    @torch.no_grad()
    def _to_device_tensor(
        self,
        param: Union[float, int, List[float], List[int], torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if isinstance(param, (int, float)):
            return torch.full((batch_size,), param, dtype=dtype, device=device)
        elif isinstance(param, (list, tuple)):
            if len(param) != batch_size:
                raise ValueError(
                    f"Parameter '{name}' length {len(param)} != batch_size {batch_size}"
                )
            return torch.tensor(param, dtype=dtype, device=device)
        elif isinstance(param, torch.Tensor):
            if param.numel() != batch_size:
                raise ValueError(
                    f"Parameter '{name}' numel {param.numel()} != batch_size {batch_size}"
                )
            return param.to(device=device, dtype=dtype).flatten()
        else:
            raise TypeError(f"Unsupported type for '{name}': {type(param)}")

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        states: torch.Tensor,
        temperatures: Union[float, List[float], torch.Tensor] = 0.2,
        top_ks: Union[int, List[int], torch.Tensor] = 20,
        top_ps: Union[float, List[float], torch.Tensor] = 0.7,
        eos_mask: Union[bool, List[bool], torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        执行带温度缩放、Top-K 和 Top-P（核采样）约束的批量采样（支持向量化参数）。

        Args:
            logits (torch.Tensor): 输入 Logits 张量，支持形状：
                - (V,): 单个样本的词汇表 Logits
                - (B, V): 批大小为 B 的 Logits
                - (B, T, V): 批大小 B、序列长度 T 的 Logits（会被展平为 (B*T, V)）
                要求：词汇表大小 V 必须是 4 的倍数且 ≤ 1048576，数据类型必须是 float32。
            states (torch.Tensor): 来自 `setup_rand` 的随机状态张量。
            temperatures (Union[float, list, torch.Tensor], optional): 采样温度，范围 [0.001, 1000]。
                支持 per-sample 设置（列表/张量长度需匹配批大小）。默认 0.2。
            top_ks (Union[int, list, torch.Tensor], optional): Top-K 参数，-1 表示不限制。
                若 ≤0 或 >V，会被自动设为 V。支持 per-sample 设置。默认 20。
            top_ps (Union[float, list, torch.Tensor], optional): Top-P（核采样）参数，范围 [0.0, 1.0]。
                若为 0，会自动设为 top_k=1、top_p=1.0（即贪婪采样）。支持 per-sample 设置。默认 0.7。

        Returns:
            torch.Tensor: 采样得到的 Token 索引，形状 (BatchSize,)，数据类型 int32，设备 CUDA。

        Raises:
            ValueError: 当 Logits 维度不支持、数据类型非 float32、V 不符合要求或温度超出范围时。

        Performs batch sampling with temperature scaling, Top-K, and Top-P (nucleus) constraints (supports vectorized parameters).
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        batch_size = logits.size(0)
        device = logits.device

        logits = logits.float()
        temp_t = self._to_device_tensor(
            temperatures, batch_size, torch.float32, device, "temperature"
        )
        topk_t = self._to_device_tensor(
            top_ks, batch_size, torch.int32, device, "top_k"
        )
        topp_t = self._to_device_tensor(
            top_ps, batch_size, torch.float32, device, "top_p"
        )
        eos_t = self._to_device_tensor(
            eos_mask, batch_size, torch.bool, device, "eos_mask"
        )

        temp_t.clamp_(0.001, 1000.0)
        topp_t.clamp_(0.0, 1.0)

        return self.raw_sample(logits, states, temp_t, topk_t, topp_t, eos_t)

    @torch.no_grad()
    def sample_repetition(
        self,
        logits: torch.Tensor,
        penalties: torch.Tensor,
        states: torch.Tensor,
        temperatures: Union[float, List[float], torch.Tensor] = 0.2,
        top_ks: Union[int, List[int], torch.Tensor] = 20,
        top_ps: Union[float, List[float], torch.Tensor] = 0.7,
        presence_penalties: Union[float, List[float], torch.Tensor] = 0.2,
        repetition_penalties: Union[float, List[float], torch.Tensor] = 1.3,
        penalty_decays: Union[float, List[float], torch.Tensor] = 0.92,
        eos_mask: Union[bool, List[bool], torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        执行带**存在惩罚**和**重复惩罚**的批量采样（支持向量化参数）。

        Args:
            logits (torch.Tensor): 输入 Logits 张量，形状要求同 `sample`。
            penalties (torch.Tensor): 惩罚状态张量，形状 (BatchSize, VocabSize)，数据类型 float32，设备 CUDA。
                用于记录历史 Token 的惩罚值，**会被原地更新**。
            states (torch.Tensor): 来自 `setup_rand` 的随机状态张量。
            temperatures (Union[float, list, torch.Tensor], optional): 采样温度，同 `sample`。默认 0.2。
            top_ks (Union[int, list, torch.Tensor], optional): Top-K 参数，同 `sample`。默认 20。
            top_ps (Union[float, list, torch.Tensor], optional): Top-P 参数，同 `sample`。默认 0.7。
            presence_penalties (Union[float, list, torch.Tensor], optional): 存在惩罚系数，范围 [0.0, 10.0]。
                若 Token 曾出现过，对其 Logits 减去固定值（鼓励多样性）。支持 per-sample 设置。默认 0.2。
            repetition_penalties (Union[float, list, torch.Tensor], optional): 重复惩罚系数，范围 [1.0, 10.0]。
                若 Token 曾出现过，对其 Logits 除以该值（抑制重复）。支持 per-sample 设置。默认 1.3。
            penalty_decays (Union[float, list, torch.Tensor], optional): 惩罚衰减因子，范围 [0.0, 1.0]。
                每次采样后，历史惩罚值乘以该因子（逐渐遗忘旧 Token）。支持 per-sample 设置。默认 0.92。
            eos_mask (Union[bool, list, torch.Tensor], optional): EOS（结束符）掩码。
                若为 True，对应样本会强制采样 EOS。支持 per-sample 设置。默认 False。

        Returns:
            torch.Tensor: 采样得到的 Token 索引，形状 (BatchSize,)，数据类型 int32，设备 CUDA。

        Raises:
            ValueError: 当 Logits 维度不支持、数据类型非 float32、V 不符合要求或参数超出范围时。

        Note:
            此函数会**原地更新** `penalties` 张量，用于下一次采样的惩罚计算。

        Performs batch sampling with **presence penalty** and **repetition penalty** (supports vectorized parameters).
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        batch_size = logits.size(0)
        device = logits.device

        logits = logits.float()
        temp_t = self._to_device_tensor(
            temperatures, batch_size, torch.float32, device, "temperature"
        )
        topk_t = self._to_device_tensor(
            top_ks, batch_size, torch.int32, device, "top_k"
        )
        topp_t = self._to_device_tensor(
            top_ps, batch_size, torch.float32, device, "top_p"
        )
        pp_t = self._to_device_tensor(
            presence_penalties, batch_size, torch.float32, device, "presence_penalty"
        )
        rp_t = self._to_device_tensor(
            repetition_penalties,
            batch_size,
            torch.float32,
            device,
            "repetition_penalty",
        )
        pd_t = self._to_device_tensor(
            penalty_decays, batch_size, torch.float32, device, "penalty_decay"
        )
        eos_t = self._to_device_tensor(
            eos_mask, batch_size, torch.bool, device, "eos_mask"
        )

        temp_t.clamp_(0.001, 1000.0)
        topp_t.clamp_(0.0, 1.0)
        pp_t.clamp_(0.0, 10.0)
        rp_t.clamp_(1.0, 10.0)
        pd_t.clamp_(0.0, 1.0)

        return self.raw_sample_repetition(
            logits, penalties, states, temp_t, topk_t, topp_t, pp_t, rp_t, pd_t, eos_t
        )
