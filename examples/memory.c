
#include <binopt.h>

#include <stdio.h>

int func(int a, int* b) {
    return a * *b;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    int param2 = 42;

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_PTR);
    binopt_cfg_set_paramp(bcfg, 1, &param2, sizeof(param2), BINOPT_MEM_CONST);

    int (* new_func)(int, int*);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    param2 = 16;
    int res = new_func(8, &param2);
    printf("8 * 16(42) = %d\n", res);

    return 0;
}
