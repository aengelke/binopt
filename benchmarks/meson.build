# Force -O3 and disable sanitizers for benchmarks
benchmark_opts = ['-O3', '-fno-sanitize=all', '-D_POSIX_C_SOURCE=199309L', '-march=native', '-mno-avx']
executable('stencil', 'stencil.c', dependencies: [binopt],
           c_args: benchmark_opts + ['-ffast-math'])
