# BinOpt — A Library for Self-guided Runtime Binary Optimization

This library enables explicit optimization of compiled (x86-64) machine code at runtime. The optimization is controlled directly by the application itself (*self-guided*): the application can specify a function to optimize, and a new function with the same interface will be generated, which can be used seamlessly instead of the original function. There are two types of specializations: (1) fixation of function parameters (including pointers), and (2) marking memory regions as constant. An optimizer may honor such configurations to derive more optimized code for the given constraints. Examples of possible optimizations are, constant propagation, dead code removal, and loop unrolling.

### Example

Consider the following example (see [examples/simple.c](examples/simple.c)):

```c
static int func(int a, int b) {
    return a * b;
}

int main(int argc, char** argv) {
    // Create a new handle into the rewriter
    BinoptHandle boh = binopt_init();
    // Create a new function configuration
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    // Configure the parameter types (return value, 2 parameters)
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32);
    // Set parameter 2 to the constant integer 42
    binopt_cfg_set_parami(bcfg, 1, 42);

    int (* new_func)(int, int);
    // Optimize the code; using additional knowledge of the configuration
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    int res = new_func(8, 16); // just for demonstration, always call with 42!
    // If the constant was propagated, this will print 336 instead of 128.
    printf("8 * 16(42) = %d\n", res);

    return 0;
}
```

### Rewriting Approaches

In fact, this repository mostly provides a unified API for applications allowing the actual implementation of the optimizer to be changed transparently. The default implementation just returns the original function without any modifications. Other rewriters perform significantly deeper code transformations. The rewriter can be switched using the `LD_PRELOAD` environment variable — simply preload the rewriter you want to use. The following optimizers with the unified API are currently implemented:

- [DBLL](rewriter/dbll): An LLVM-based binary specializer, which lifts the original function to LLVM-IR using [Rellume](https://github.com/aengelke/rellume), performs optimizations at LLVM-IR level, and generates new code using the LLVM MCJIT compiler. This is the rewriter with the highest instruction coverage supporting most of x86-64, excluding indirect jumps and function calls; and the x87 FPU, MMX, SSE3+, AVX.
- [Drob](https://github.com/davidhildenbrand/drob): A low-level rewriter focusing on lower rewriting times while still doing whole function analyses and optimizations. While functions with unknown instructions are supported, optimization possibilities are limited in such cases.
- [DBrew](https://github.com/caps-tum/dbrew): A tracing binary rewriter with emphasis on compile-time performance while also doing unlimited loop unrolling and inlining. This has the most limited scope and instruction coverage. Also, due to massive unrolling of loops with known bounds, code buffer sizes may be exceeded unintentionally.

### License

This project is originally written and maintained by [Alexis Engelke](https://www.in.tum.de/caps/mitarbeiter/engelke/). All code in this repository is licensed under LGPLv2.1+.
