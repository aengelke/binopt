
#include <binopt.h>

#include <stdio.h>

static int func(int edi, int esi, int edx, int ecx, int r8d, int r9d, int sp8,
                int sp16) {
    return sp8 - sp16;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_set(bcfg, BINOPT_F_LOGLEVEL, 3);
    binopt_cfg_type(bcfg, 8, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32,
                    BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32,
                    BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32);
    binopt_cfg_set_parami(bcfg, 7, 42);

    int (* new_func)(int, int, int, int, int, int, int, int);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    int res = new_func(100, 101, 102, 103, 104, 105, 8, 16);
    printf("8 - 16(42) = %d\n", res);

    return 0;
}
