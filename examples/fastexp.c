
#include <binopt.h>

#include <stdint.h>
#include <stdio.h>

static double fast_exp(uint64_t exp, double val) {
    double ans = 1;
    while (exp) {
        if (exp & 1)
            ans *= val;
        val *= val;
        exp = exp >> 1;
    }
    return ans;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) fast_exp);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_DOUBLE, BINOPT_TY_UINT64,
                    BINOPT_TY_DOUBLE);
    binopt_cfg_set_parami(bcfg, 0, 42);

    double (* new_func)(uint64_t, double);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    double res = new_func(16, 0.99);
    printf("0.99**16 (**42) = %6.4f\n", res);

    return 0;
}
