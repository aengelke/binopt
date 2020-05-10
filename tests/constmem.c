#include "common.h"
#include <sys/mman.h>

static int func(int* b) {
    return *b;
}

int main(int argc, char** argv) {
    int* const_val = mmap(NULL, sizeof(int), PROT_READ|PROT_WRITE,
                          MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    *const_val = 42;
    mprotect(const_val, sizeof(int), PROT_READ);
    // const_val now points to read-only memory.

    BinoptHandle boh = test_init(argc, argv);
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 1, BINOPT_TY_INT32, BINOPT_TY_PTR);
    binopt_cfg_set_parami(bcfg, 0, (uintptr_t) const_val);

    int (* new_func)(int*);
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    mprotect(const_val, sizeof(int), PROT_READ|PROT_WRITE);
    *const_val = 3;

    int param2 = 16;
    // If nothing is propagated, the result is 16. If the pointer was propagated
    // but not the value, the result is 3. If the value from the read-only
    // memory is used, the result is 42.
    test_eq_i32(new_func(&param2), param2, 42);
    test_fini();
}
