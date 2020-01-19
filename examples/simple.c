
#include <binopt.h>

#include <stdio.h>

static int func(int a, int b) {
    return a * b;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_INT32);
    binopt_cfg_set_parami(bcfg, 1, 42);

    int (* new_func)(int, int);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    int res = new_func(8, 16);
    printf("8 * 16(42) = %d\n", res);

    return 0;
}
