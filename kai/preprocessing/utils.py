import torch
import os
import psutil
def print_gpu_memory():
    # Get the current GPU memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to GB
    cached = torch.cuda.memory_reserved() / 1024**3  # Convert bytes to GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert bytes to GB

    print(f"Allocated memory: {allocated:.2f} GB")
    print(f"Cached memory: {cached:.2f} GB")
    print(f"Total memory: {total:.2f} GB")
    print(f"Unused memory: {total - allocated:.2f} GB")

def free_gpu_memory():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def print_memory_usage():
    # Get memory usage details
    memory_info = psutil.virtual_memory()

    # Total memory in GB
    total_memory = memory_info.total / (1024 ** 3)

    # Available memory in GB
    available_memory = memory_info.available / (1024 ** 3)

    # Used memory in GB
    used_memory = memory_info.used / (1024 ** 3)

    # Memory usage percentage
    memory_usage_percent = memory_info.percent

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB")
    print(f"Memory Usage: {memory_usage_percent:.2f}%")