import torch
import numpy as np
from basics.templates import TransformerLM

VOCAB_SIZE = 50257
CONTEXT_LENGTH = 1024
NUM_LAYERS = 48
D_MODEL = 1600
NUM_HEADS = 25
D_FF = 6400

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires CUDA to measure GPU memory.")

device = 'cuda'
print(f"Using device: {device}")

model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    device=device,
    dtype=torch.float32,
    use_point_wise_ffn=True
)

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
memory_usage_no_grad = []

def compute_linear_regression(batch_sizes, memory_usage):
    """Compute linear regression coefficients for memory usage vs batch size."""
    batch_sizes_array = np.array(batch_sizes, dtype=np.float64)
    memory_usage_array = np.array(memory_usage, dtype=np.float64)
    
    n = len(batch_sizes)
    sum_x = np.sum(batch_sizes_array)
    sum_y = np.sum(memory_usage_array)
    sum_xy = np.sum(batch_sizes_array * memory_usage_array)
    sum_x2 = np.sum(batch_sizes_array ** 2)
    
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    
    return a, b

def print_regression_results(mode, batch_sizes, memory_usage, a, b):
    """Print regression results for a given mode."""
    print(f"\n{'='*60}")
    print(f"Linear regression results ({mode}):")
    print(f"{'='*60}")
    print(f"a = {a:.2f} bytes per batch element")
    print(f"b = {b:.2f} bytes")
    print(f"\nExpression: memory_bytes = {a:.2f} * batch_size + {b:.2f}")
    
    a_gb = a / (1024**3)
    b_gb = b / (1024**3)
    print(f"\nOr in GB: memory_gb = {a_gb:.6f} * batch_size + {b_gb:.6f}")
    
    print(f"\nVerification (predicted vs actual):")
    for i, bs in enumerate(batch_sizes):
        predicted = a * bs + b
        actual = memory_usage[i]
        error = abs(predicted - actual) / actual * 100
        print(f"Batch size {bs:2d}: Predicted = {predicted / (1024**2):.2f} MB, "
              f"Actual = {actual / (1024**2):.2f} MB, Error = {error:.2f}%")
    
    total_memory_gb = 80
    available_memory_bytes = total_memory_gb * (1024**3) - b
    max_batch_size = int(available_memory_bytes / a)
    
    print(f"\nMaximum batch size for 80GB memory ({mode}):")
    print(f"  Total memory: {total_memory_gb} GB")
    print(f"  Constant overhead (b): {b / (1024**3):.2f} GB")
    print(f"  Available for activations: {available_memory_bytes / (1024**3):.2f} GB")
    print(f"  Maximum batch size: {max_batch_size}")

# Test with no_grad (inference)
print("Testing different batch sizes with torch.no_grad() (inference)...")
print("Measuring activation memory using torch.cuda memory tracking...\n")

for batch_size in batch_sizes:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, CONTEXT_LENGTH), device=device)
    
    with torch.no_grad():
        output = model(input_ids)
    
    memory_after = torch.cuda.memory_allocated()
    
    memory_used = memory_after
    
    memory_usage_no_grad.append(memory_used)
    print(f"Batch size {batch_size:2d}: Memory used = {memory_used / (1024**2):.2f} MB")
    
    del input_ids, output
    torch.cuda.empty_cache()

# Compute and print results for no_grad
a_no_grad, b_no_grad = compute_linear_regression(batch_sizes, memory_usage_no_grad)
print_regression_results("no_grad (inference)", batch_sizes, memory_usage_no_grad, a_no_grad, b_no_grad)

# Comparison
print(f"\n{'='*60}")
print("Comparison:")
print(f"{'='*60}")
print(f"Memory per batch element (a):")
print(f"  Inference (no_grad):  {a_no_grad / (1024**3):.6f} GB")
print(f"\nConstant overhead (b):")
print(f"  Inference (no_grad):  {b_no_grad / (1024**3):.2f} GB")
