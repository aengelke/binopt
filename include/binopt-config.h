
#ifndef BINOPT_CONFIG_H
#define BINOPT_CONFIG_H

#include <binopt.h>

#include <stddef.h>

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

    size_t param_count;
    size_t param_alloc;
    struct BinoptCfgParam {
        BinoptType ty;
        const void* const_val;
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
