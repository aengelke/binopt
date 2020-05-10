#include "common.h"
#include <sys/mman.h>

int global = 42;

static int func(int* b) {
    return *b;
}

int main(int argc, char** argv) {
    BinoptHandle boh = test_init(argc, argv);
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 1, BINOPT_TY_INT32, BINOPT_TY_PTR);
    // Note that we don't specify the value itself as constant.
    binopt_cfg_set_parami(bcfg, 0, (uintptr_t) &global);

    int (* new_func)(int*);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    global = 35;
    int param2 = 16;
    // If nothing is propagated, the result is 16. If the pointer was propagated
    // the result is 35. The result must never be 42.
    test_eq_i32(new_func(&param2), param2, global);
    test_fini();
}
