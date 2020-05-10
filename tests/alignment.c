
#include <binopt.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void align_test16(void* ptr) {
    if ((uintptr_t) ptr & 15) {
        printf("Test failed (buf): %p\n", ptr);
        abort();
    }
    void* faddr = __builtin_frame_address(0);
    if ((uintptr_t) faddr & 15) {
        printf("Test failed (rsp): %p\n", faddr);
        abort();
    }
}

static void func(void(* align_test_fn)(void*)) {
    unsigned char buf[16] __attribute__((aligned(16)));
    align_test_fn(buf);
    __asm__ volatile("" ::: "memory");
}

int main(int argc, char** argv) {
    printf("Rewriter: %s\n", binopt_driver());

    BinoptHandle boh = binopt_init();
    BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) func);
    binopt_cfg_type(bcfg, 1, BINOPT_TY_VOID, BINOPT_TY_PTR);

    void (* new_func)(void(*)(void*));
    *((BinoptFunc*) &new_func) = binopt_spec_create(bcfg);

    new_func(align_test16);
    printf("Test passed\n");

    return 0;
}
