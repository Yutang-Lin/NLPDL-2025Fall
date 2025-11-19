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
    dtype=torch.float32
)

batch_sizes = [1, 2, 4, 8, 16, 32]
memory_usage = []

print("Testing different batch sizes...")
print("Measuring activation memory using torch.cuda memory tracking...\n")

for batch_size in batch_sizes:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, CONTEXT_LENGTH), device=device)
    
    memory_before = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        output = model(input_ids)
    
    memory_after = torch.cuda.memory_allocated()
    
    memory_used = memory_after - memory_before
    
    memory_usage.append(memory_used)
    print(f"Batch size {batch_size:2d}: Memory used = {memory_used / (1024**2):.2f} MB")
    
    del input_ids, output
    torch.cuda.empty_cache()

batch_sizes_array = np.array(batch_sizes, dtype=np.float64)
memory_usage_array = np.array(memory_usage, dtype=np.float64)

n = len(batch_sizes)
sum_x = np.sum(batch_sizes_array)
sum_y = np.sum(memory_usage_array)
sum_xy = np.sum(batch_sizes_array * memory_usage_array)
sum_x2 = np.sum(batch_sizes_array ** 2)

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - a * sum_x) / n

print(f"\nLinear regression results:")
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

print(f"\nMaximum batch size for 80GB memory:")
print(f"  Total memory: {total_memory_gb} GB")
print(f"  Constant overhead (b): {b / (1024**3):.2f} GB")
print(f"  Available for activations: {available_memory_bytes / (1024**3):.2f} GB")
print(f"  Maximum batch size: {max_batch_size}")
