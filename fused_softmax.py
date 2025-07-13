import torch
import triton
import triton.language as tl



@triton.jit
def fused_softmax(output_ptr,
                  x_ptr,
                  num_cols,
                  num_rows,
                  row_stride,
                  BLOCK_SIZE:tl.constexpr,
                  num_stages:tl.constexpr):
    
    start_row = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    for num_row in tl.range(start_row, num_rows, num_programs, num_stages=num_stages):
        #calculate ptr of row
        row_idx = x_ptr + num_row * row_stride
        offsets = tl.arange(0,BLOCK_SIZE)
        mask = offsets < num_cols
        x = tl.load(offsets + row_idx, mask=mask)
        max = tl.max(x)
        minus_max = x - max
        numerator = tl.exp(minus_max)
        denominator = tl.sum(numerator)
        softmax = numerator / denominator
        output_idx = output_ptr + num_row * row_stride
        tl.store(output_idx + offsets, softmax, mask=mask)


def softmax(x):
    output = torch.empty_like(x).to('cuda')
    row_stride = x.stride(0)
    num_rows, num_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(num_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    fused_softmax[(num_rows, )](output, x, num_cols, num_rows, row_stride, BLOCK_SIZE, 2, num_warps=num_warps)
    return output
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['N'],
        x_vals = [128 * i for i in range(2, 100)],
        line_arg= 'provider',
        line_names=['triton', 'torch'],
        line_vals=['triton', 'torch'],
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N).to('cuda')
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == '__main__':
   benchmark.run(show_plots=True)