
#include <binopt.h>

#include <stdint.h>
#include <stdio.h>

int64_t func(const int64_t* poly, int64_t x) {
    int64_t res = 0;
    int64_t x_n = 1;
    for (int64_t i = 0; i < *poly; i++) {
        res += x_n * poly[i+1];
        x_n *= x;
    }
    return res;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_INT64, BINOPT_TY_PTR, BINOPT_TY_PTR);
    int64_t poly[] = {3, 2, 1, 1};
    binopt_cfg_set_paramp(bcfg, 0, poly, sizeof(poly), BINOPT_MEM_CONST);

    int64_t (* new_func)(const int64_t*, int64_t);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    poly[0] = 1;
    int64_t res = new_func(poly, 4);
    printf("2(2+1*4^1+1*4^2) = %ld\n", res);

    return 0;
}
