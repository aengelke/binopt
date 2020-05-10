#include "common.h"

static int func(int edi, int esi, int edx, int ecx, int r8d, int r9d, int sp8,
                int sp16) {
    return sp8 - sp16;
}

int main(int argc, char** argv) {
    BinoptHandle boh = test_init(argc, argv);
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 8, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32,
                    BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32,
                    BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32);
    binopt_cfg_set_parami(bcfg, 7, 42);

    int (* new_func)(int, int, int, int, int, int, int, int);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);
    test_eq_i32(new_func(100, 101, 102, 103, 104, 105, 8, 16), 8 - 16, 8 - 42);
    test_fini();
}
