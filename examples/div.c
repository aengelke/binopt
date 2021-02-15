
#include <binopt.h>

#include <stdint.h>
#include <stdio.h>

static uint64_t div(uint64_t dividend, uint64_t divisor) {
    return dividend / divisor;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) div);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_UINT64, BINOPT_TY_UINT64,
                    BINOPT_TY_UINT64);
    binopt_cfg_set_parami(bcfg, 1, 7);

    uint64_t (* new_func)(uint64_t, uint64_t);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    uint64_t res = new_func(60, 10);
    printf("60//10 (//7) = %zu\n", res);

    return 0;
}
