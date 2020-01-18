
#include <binopt.h>

#include <stdio.h>

int func(int a, int* b) {
    return a * *b;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT32, BINOPT_TY_INT32, BINOPT_TY_PTR);
    int param2 = 42;
    int* param2_ptr = &param2;
    binopt_cfg_set_param(bcfg, 1, &param2_ptr);
    binopt_cfg_mem(bcfg, param2_ptr, sizeof(param2_ptr), BINOPT_MEM_CONST);

    int (* new_func)(int, int*);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    param2 = 16;
    int res = new_func(8, param2_ptr);
    printf("8 * 16(42) = %d\n", res);

    return 0;
}
