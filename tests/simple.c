#include "common.h"

static int func(int a, int b) {
    return a * b;
}

int main(int argc, char** argv) {
    BinoptHandle boh = test_init(argc, argv);
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32);
    binopt_cfg_set_parami(bcfg, 1, 42);

    int (* new_func)(int, int);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);
    test_eq_i32(new_func(8, 16), 8 * 16, 8 * 42);
    test_fini();
}
