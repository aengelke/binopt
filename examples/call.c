
#include <binopt.h>

#include <stdio.h>

__attribute__((noinline)) static int nested(int a) {
    return 2 * a;
}

static int func(int a) {
    return nested(a) + a;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 1, BINOPT_TY_INT32, BINOPT_TY_INT32);
    binopt_cfg_set_parami(bcfg, 0, 42);

    int (* new_func)(int);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    int res = new_func(8);
    printf("3 * 8(42) = %d\n", res);

    return 0;
}
