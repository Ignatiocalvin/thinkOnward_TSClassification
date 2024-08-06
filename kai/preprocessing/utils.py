import torch
def print_gpu_memory():
    # Get the current GPU memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to GB
    cached = torch.cuda.memory_reserved() / 1024**3  # Convert bytes to GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert bytes to GB

    print(f"Allocated memory: {allocated:.2f} GB")
    print(f"Cached memory: {cached:.2f} GB")
    print(f"Total memory: {total:.2f} GB")
    print(f"Unused memory: {total - allocated:.2f} GB")