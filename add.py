import torch
import triton
import triton.language as tl

# Triton kernel for vector addition
@triton.jit
def add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # Get block ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Thread offsets
    mask = offsets < n_elements  # Boundary check
    x = tl.load(x_ptr + offsets, mask=mask)  # Load x
    y = tl.load(y_ptr + offsets, mask=mask)  # Load y
    tl.store(output_ptr + offsets, x + y, mask=mask)  # Store x + y

# Benchmarking function with perf_report decorator
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names for x-axis
        x_vals=[2**i for i in range(12, 28, 1)],  # Sizes from 2^12 to 2^27
        x_log=True,  # Logarithmic x-axis
        line_arg='provider',  # Argument for different lines
        line_vals=['triton', 'torch'],  # Providers to compare
        line_names=['Triton', 'Torch'],  # Labels for lines
        styles=[('blue', '-'), ('green', '-')],  # Line styles
        ylabel='GB/s',  # Y-axis label
        plot_name='vector-add-performance',  # Plot name and filename
        args={},  # No additional args
    )
)

def benchmark(size, provider):
    DEVICE = 'cuda'  # Set device to CUDA
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    output = torch.zeros(size, device=DEVICE, dtype=torch.float32)  # Output tensor
    quantiles = [0.5, 0.2, 0.8]  # Median, 20th, 80th percentiles
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add[grid](x, y, output, size, BLOCK_SIZE=256),
            quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# Run the benchmark
if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)