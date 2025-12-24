import torch

random_ints = torch.randint(0, 0x70000000, (10000, 2048), dtype=torch.int32)
float_tensor = random_ints.view(torch.float32)

def first_func(tensor):
    buffer = tensor.clone()
    n = tensor.size(1)
    for i in range(n.bit_length() - 1, -1, -1):
        offset = 2 ** i
        if offset < n:
            u = buffer[:, :-offset].clone().contiguous()
            buffer[:, offset:] += u
    return buffer

def second_func(tensor):
    buffer = tensor.clone()
    n = tensor.size(1)
    for i in range(n.bit_length()):
        offset = 2 ** i
        if offset < n:
            u = buffer[:, :-offset].clone().contiguous()
            buffer[:, offset:] += u
    return buffer

def is_monotonic(result):
    return torch.all(result[:, 1:] >= result[:, :-1]).item()

first_result = first_func(float_tensor)
is_first_monotonic = is_monotonic(first_result)
second_result = second_func(float_tensor)
is_second_monotonic = is_monotonic(second_result)
print(is_first_monotonic, is_second_monotonic)
