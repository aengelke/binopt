
#ifndef BINOPT_CONFIG_H
#define BINOPT_CONFIG_H

#include <binopt.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default implementation of configuration API. This is NOT to be used by
 * applications but only by rewriting tools which do not implement their own
 * configuration function. */

struct BinoptCfg {
    BinoptHandle handle;
    BinoptFunc func;
    BinoptType ret_ty;

    // Allow unsafe floating-point optimizations. 0 = none, 1 = all.
    // Upper bits may define individual optimizations in future.
    uint8_t fast_math;
    // Log level verbosity. 0 = none/quiet
    uint8_t log_level;

    size_t param_count;
    size_t param_alloc;
    struct BinoptCfgParam {
        BinoptType ty;
        void* const_val;
    }* params;

    size_t memrange_count;
    size_t memrange_alloc;
    struct BinoptCfgMemrange {
        void* base;
        size_t size;
        BinoptMemFlags flags;
    }* memranges;
};

#ifdef __cplusplus
}
#endif

#endif
