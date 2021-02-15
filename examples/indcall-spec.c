
#include <binopt.h>

#include <stdio.h>

static float add_one(float a) {
    return a + 1.0f;
}

static float add_two(float a) {
    return a + 2.0f;
}

static float func(float(* fn)(float), float a) {
    return fn(a) * a;
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 2, BINOPT_TY_FLOAT, BINOPT_TY_PTR,
                    BINOPT_TY_FLOAT);
    binopt_cfg_set_parami(bcfg, 0, (size_t) (void*) add_one);

    float (* new_func)(float(*)(float), float);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    float res = new_func(add_two, 5);
    printf("5 * (5 + 2(1)) = %f\n", res);

    return 0;
}
