from typing import Union
import torch
import os
from torch.utils.cpp_extension import load

ROCm_flag = torch.version.hip is not None

class Sampler:
    """
    高性能采样器类，支持温度缩放、Top-K、Top-P 采样及重复惩罚机制。
    High-performance sampler class supporting temperature scaling, Top-K, Top-P sampling, and repetition penalty.
    """
    def __init__(self, verbose: bool = False):
        """
        初始化 Sampler 实例，根据平台（ROCm/CUDA）即时编译（JIT）加载对应的 C++/HIP 扩展。
        
        Args:
            verbose (bool, optional): 是否显示编译详细日志。默认 False。
                Whether to show detailed compilation logs. Defaults to False.
        
        Initializes Sampler instance, JIT-compiles and loads corresponding C++/HIP extension based on platform (ROCm/CUDA).
        """
        if ROCm_flag:
            sources = ["hip/sampling_op.hip", "hip/sampling.hip"]
            extra_cflags = ['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics']
        else:
            sources = ["sampling.cpp", "sampling.cu"]
            extra_cflags = ["-O3", "-res-usage", "-Xptxas", "-O3"]

        self.sample_module = load(
            name="sample",
            sources=sources,
            extra_cuda_cflags=extra_cflags,
            verbose=verbose,
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
    def _validate(
        self,
        param: Union[float, int, list, torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """
        将输入参数标准化为形状为 (BatchSize,) 的 CUDA 张量。
        
        Args:
            param (Union[float, int, list, torch.Tensor]): 输入参数，支持标量、列表或张量。
                Input parameter, supports scalar, list, or tensor.
            batch_size (int): 目标批大小。
                Target batch size.
            dtype (torch.dtype): 目标数据类型。
                Target data type.
            device (torch.device): 目标设备（CUDA）。
                Target device (CUDA).
            name (str): 参数名称，用于错误提示。
                Parameter name for error messaging.
        
        Returns:
            torch.Tensor: 标准化后的一维 CUDA 张量，形状 (BatchSize,)。
                Normalized 1D CUDA tensor with shape (BatchSize,).
        
        Raises:
            ValueError: 列表长度或张量元素数与批大小不匹配。
                If list length or tensor element count does not match batch size.
            TypeError: 输入参数类型不支持。
                If input parameter type is unsupported.
        
        [Internal Helper] Normalizes input parameter to a 1D CUDA tensor of shape (BatchSize,).
        """
        if isinstance(param, (int, float)):
            # 标量广播到批大小
            return torch.empty((batch_size,), dtype=dtype, device=device).fill_(param)
        elif isinstance(param, list):
            if len(param) != batch_size:
                raise ValueError(f"Parameter '{name}' list length {len(param)} must match batch size {batch_size}")
            # 直接在目标设备上创建张量，避免 CPU->GPU 拷贝
            return torch.tensor(param, dtype=dtype, device=device)
        elif isinstance(param, torch.Tensor):
            if param.numel() != batch_size:
                raise ValueError(f"Parameter '{name}' tensor numel {param.numel()} must match batch size {batch_size}")
            # 合并设备/类型转换，确保一维
            return param.to(device=device, dtype=dtype).flatten()
        else:
            raise TypeError(f"Unsupported type for '{name}': {type(param)}")

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        states: torch.Tensor,
        temperatures: Union[float, list[float], torch.Tensor] = 0.2,
        top_ks: Union[int, list[int], torch.Tensor] = 20,
        top_ps: Union[float, list[float], torch.Tensor] = 0.7,
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
        device = logits.device

        if logits.dim() == 1:
            batch_size = 1
            logits = logits.unsqueeze(0)  # (V,) -> (1, V)
        elif logits.dim() == 2:
            batch_size = logits.size(0)
        elif logits.dim() == 3:
            batch_size = logits.size(0) * logits.size(1)
            logits = logits.reshape(batch_size, -1)  # (B, T, V) -> (B*T, V)
        else:
            raise ValueError(f"Unsupported logits dim: {logits.dim()}")

        temp_t = self._validate(temperatures, batch_size, torch.float32, device, "temperature")
        topk_t = self._validate(top_ks, batch_size, torch.int32, device, "top_k")
        topp_t = self._validate(top_ps, batch_size, torch.float32, device, "top_p")

        temp_t.clamp_(0.001, 1000.0)
        topp_t.clamp_(0.0, 1.0)
        
        zero_mask = topp_t == 0.0
        topk_t = torch.where(zero_mask, torch.tensor(1, dtype=torch.int32, device=device), topk_t)
        topp_t = torch.where(zero_mask, torch.tensor(1.0, dtype=torch.float32, device=device), topp_t)

        return self.sample_module.batch_sampling_temperature_topk_topp(
            logits.to(torch.float32, non_blocking=True),
            states, temp_t, topk_t, topp_t
        )
    
    @torch.no_grad()
    def sample_repetition(
        self,
        logits: torch.Tensor,
        penalties: torch.Tensor,
        states: torch.Tensor,
        temperatures: Union[float, list[float], torch.Tensor] = 0.2,
        top_ks: Union[int, list[int], torch.Tensor] = 20,
        top_ps: Union[float, list[float], torch.Tensor] = 0.7,
        presence_penalties: Union[float, list[float], torch.Tensor] = 0.2,
        repetition_penalties: Union[float, list[float], torch.Tensor] = 1.3,
        penalty_decays: Union[float, list[float], torch.Tensor] = 0.92,
        eos_mask: Union[bool, list[bool], torch.Tensor] = False,
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
        device = logits.device

        if logits.dim() == 1:
            batch_size = 1
            logits = logits.unsqueeze(0)
        elif logits.dim() == 2:
            batch_size = logits.size(0)
        elif logits.dim() == 3:
            batch_size = logits.size(0) * logits.size(1)
            logits = logits.reshape(batch_size, -1)
        else:
            raise ValueError(f"Unsupported logits dim: {logits.dim()}")

        temp_t = self._validate(temperatures, batch_size, torch.float32, device, "temperature")
        topk_t = self._validate(top_ks, batch_size, torch.int32, device, "top_k")
        topp_t = self._validate(top_ps, batch_size, torch.float32, device, "top_p")
        pp_t = self._validate(presence_penalties, batch_size, torch.float32, device, "presence_penalty")
        rp_t = self._validate(repetition_penalties, batch_size, torch.float32, device, "repetition_penalty")
        pd_t = self._validate(penalty_decays, batch_size, torch.float32, device, "penalty_decay")
        eos_mask_t = self._validate(eos_mask, batch_size, torch.bool, device, "eos_mask")

        temp_t.clamp_(0.001, 1000.0)
        topp_t.clamp_(0.0, 1.0)
        zero_mask = topp_t == 0.0
        topk_t = torch.where(zero_mask, torch.tensor(1, dtype=torch.int32, device=device), topk_t)
        topp_t = torch.where(zero_mask, torch.tensor(1.0, dtype=torch.float32, device=device), topp_t)

        return self.sample_module.batch_sampling_repetition_temperature_topk_topp(
            logits.to(torch.float32, non_blocking=True),
            penalties, states, temp_t, topk_t, topp_t, pp_t, rp_t, pd_t, eos_mask_t
        )