
#include "binopt.h"
#include "binopt-config.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
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

static size_t binopt_type_size(BinoptType ty) {
    switch (ty) {
    case BINOPT_TY_VOID: return 0;
    case BINOPT_TY_INT8: return sizeof(int8_t);
    case BINOPT_TY_INT16: return sizeof(int16_t);
    case BINOPT_TY_INT32: return sizeof(int32_t);
    case BINOPT_TY_INT64: return sizeof(int64_t);
    case BINOPT_TY_UINT8: return sizeof(uint8_t);
    case BINOPT_TY_UINT16: return sizeof(uint16_t);
    case BINOPT_TY_UINT32: return sizeof(uint32_t);
    case BINOPT_TY_UINT64: return sizeof(uint64_t);
    case BINOPT_TY_FLOAT: return sizeof(float);
    case BINOPT_TY_DOUBLE: return sizeof(double);
    case BINOPT_TY_PTR: return sizeof(void*);
    case BINOPT_TY_PTR_NOALIAS: return sizeof(void*);
    default: return 0;
    }
}

WEAK BinoptCfgRef binopt_cfg_new(BinoptHandle handle,
                                 BinoptFunc base_func) {
    BinoptCfgRef cfg = calloc(1, sizeof(struct BinoptCfg));
    if (cfg == NULL)
        return NULL;
    cfg->handle = handle;
    cfg->func = base_func;
    cfg->fast_math = 0;
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
        for (size_t i = 0; i < new_cfg->param_count; ++i) {
            struct BinoptCfgParam* base_param = &base_cfg->params[i];
            struct BinoptCfgParam* new_param = &new_params[i];
            size_t const_size = binopt_type_size(base_param->ty);
            new_param->ty = base_param->ty;
            new_param->const_val = malloc(const_size);
            // Silently don't clone constant values if we can't allocate memory
            if (new_param->const_val != NULL)
                memcpy(new_param->const_val, base_param->const_val, const_size);
        }
        new_cfg->params = new_params;
    }
    if (base_cfg->memranges != NULL) {
        struct BinoptCfgMemrange* new_memranges = malloc(sizeof(struct BinoptCfgMemrange) * new_cfg->memrange_alloc);
        if (new_memranges == NULL) {
            if (new_cfg->params)
                for (size_t i = 0; i < new_cfg->param_count; ++i)
                    free(new_cfg->params[i].const_val);
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
WEAK void binopt_cfg_set(BinoptCfgRef cfg, BinoptOptFlags flag, size_t val) {
    switch (flag) {
    case BINOPT_F_FASTMATH: cfg->fast_math = !!val; break;
    default: break; // ignore unknown flags
    }
}
WEAK void binopt_cfg_set_param(BinoptCfgRef cfg, unsigned idx, const void* val) {
    if (idx >= cfg->param_count)
        return;
    size_t const_size = binopt_type_size(cfg->params[idx].ty);
    cfg->params[idx].const_val = malloc(const_size);
    if (!cfg->params[idx].const_val)
        return;
    memcpy(cfg->params[idx].const_val, val, const_size);
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
    if (cfg->params) {
        for (size_t i = 0; i < cfg->param_count; ++i)
            free(cfg->params[i].const_val);
        free(cfg->params);
    }
    if (cfg->memranges)
        free(cfg->memranges);
    free(cfg);
}

WEAK BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    return cfg->func;
}
WEAK void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {}

// Convenience functions
void binopt_cfg_set_parami(BinoptCfgRef cfg, unsigned idx, size_t val) {
    binopt_cfg_set_param(cfg, idx, &val);
}
void binopt_cfg_set_paramp(BinoptCfgRef cfg, unsigned idx, const void* ptr,
                           size_t size, BinoptMemFlags flags) {
    binopt_cfg_set_param(cfg, idx, &ptr);
    binopt_cfg_mem(cfg, ptr, size, flags);
}
