
#include <binopt.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static double func(size_t len, const double poly[len], double x) {
    double res = 0;
    double x_n = 1;
    for (size_t i = 0; i < len; i++) {
        res += x_n * poly[i];
        x_n *= x;
    }
    return res;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 3, BINOPT_TY_DOUBLE, BINOPT_TY_UINT64, BINOPT_TY_PTR,
                    BINOPT_TY_DOUBLE);
    double poly[] = {.5, 1, 2, 0, 0, 0, 3};
    binopt_cfg_set(bcfg, BINOPT_F_FASTMATH, true);
    binopt_cfg_set_parami(bcfg, 0, sizeof poly / sizeof poly[0]);
    binopt_cfg_set_paramp(bcfg, 1, poly, sizeof(poly), BINOPT_MEM_CONST);

    double (* new_func)(size_t, const double*, double);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    double res = new_func(1, poly, 4);
    printf(".5(.5+1*4^1+2*4^2) = %f\n", res);

    return 0;
}
