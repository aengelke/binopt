
#include "binopt.h"
#include "binopt-config.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define WEAK __attribute__((weak))

/* Dummy implementation which always returns the original function without
 * modifications. */

WEAK const char* binopt_driver(void) {
    return "Default (no rewriting)";
}
WEAK BinoptHandle binopt_init(void) { return NULL; }
WEAK void binopt_fini(BinoptHandle handle) {}

WEAK BinoptCfgRef binopt_cfg_new(BinoptHandle handle,
                                 BinoptFunc base_func) {
    BinoptCfgRef cfg = calloc(1, sizeof(struct BinoptCfg));
    if (cfg == NULL)
        return NULL;
    cfg->handle = handle;
    cfg->func = base_func;
    return cfg;
}
WEAK BinoptCfgRef binopt_cfg_clone(BinoptCfgRef base_cfg) {
    BinoptCfgRef new_cfg = malloc(sizeof(struct BinoptCfg));
    if (new_cfg == NULL)
        return NULL;
    memcpy(new_cfg, base_cfg, sizeof(struct BinoptCfg));
    if (new_cfg->params != NULL) {
        struct BinoptCfgParam* new_params = malloc(sizeof(struct BinoptCfgParam) * new_cfg->param_alloc);
        if (new_params == NULL) {
            free(new_cfg);
            return NULL;
        }
        memcpy(new_params, base_cfg->params, sizeof(struct BinoptCfgParam) * base_cfg->param_count);
        new_cfg->params = new_params;
    }
    if (base_cfg->memranges != NULL) {
        struct BinoptCfgMemrange* new_memranges = malloc(sizeof(struct BinoptCfgMemrange) * new_cfg->memrange_alloc);
        if (new_memranges == NULL) {
            free(new_cfg->params);
            free(new_cfg);
            return NULL;
        }
        memcpy(new_memranges, base_cfg->memranges, sizeof(struct BinoptCfgMemrange) * base_cfg->memrange_count);
        new_cfg->memranges = new_memranges;
    }
    return new_cfg;
}
// Set function signature.
WEAK void binopt_cfg_type(BinoptCfgRef cfg, unsigned count, BinoptType ret, ...) {
    va_list args;

    cfg->ret_ty = ret;
    if (cfg->params)
        free(cfg->params);
    cfg->params = calloc(count, sizeof(struct BinoptCfgParam));
    if (cfg->params == NULL) {
        cfg->param_count = 0;
        cfg->param_alloc = 0;
        return;
    }

    va_start(args, ret);
    for (unsigned i = 0; i < count; ++i) {
        cfg->params[i].ty = va_arg(args, BinoptType);
    }
    cfg->param_count = count;
    va_end(args);

    return;
}
WEAK void binopt_cfg_set(BinoptCfgRef cfg, BinoptOptFlags flag, size_t val) {}
WEAK void binopt_cfg_set_param(BinoptCfgRef cfg, unsigned idx, const void* val) {
    if (idx >= cfg->param_count)
        return;
    cfg->params[idx].const_val = val;
}
WEAK void binopt_cfg_mem(BinoptCfgRef cfg, void* base, size_t size,
                         BinoptMemFlags flags) {
    if (cfg->memrange_alloc == 0) {
        cfg->memranges = malloc(8 * sizeof(struct BinoptCfgMemrange));
        if (cfg->memranges == NULL)
            return;
        cfg->memrange_alloc = 8;
    }
    if (cfg->memrange_count == cfg->memrange_alloc) {
        size_t new_size = 2 * cfg->memrange_alloc * sizeof(*(cfg->memranges));
        void* new_ranges = realloc(cfg->memranges, new_size);
        if (new_ranges == NULL)
            return; // old range allocation is left untouched
        cfg->memranges = new_ranges;
        cfg->memrange_alloc *= 2;
    }

    size_t range_idx = cfg->memrange_count++;
    cfg->memranges[range_idx].base = base;
    cfg->memranges[range_idx].size = size;
    cfg->memranges[range_idx].flags = flags;
}
WEAK void binopt_cfg_free(BinoptCfgRef cfg) {
    if (cfg->params)
        free(cfg->params);
    if (cfg->memranges)
        free(cfg->memranges);
    free(cfg);
}

WEAK BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    return cfg->func;
}
WEAK void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {}
