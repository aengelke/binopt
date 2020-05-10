#include "common.h"

static int func(int a, int* b) {
    return a * *b;
}

int main(int argc, char** argv) {
    int param2 = 42;

    BinoptHandle boh = test_init(argc, argv);
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_PTR);
    binopt_cfg_set_paramp(bcfg, 1, &param2, sizeof(param2), BINOPT_MEM_CONST);

    int (* new_func)(int, int*);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    param2 = 16;
    test_eq_i32(new_func(8, &param2), 8 * param2, 8 * 42);
    test_fini();
}
