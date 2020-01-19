
#include "binopt.h"
#include "binopt-config.h"

#include <dbrew.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* DBrew bindings, using default configuration API */

const char* binopt_driver(void) {
    return "DBrew";
}
BinoptHandle binopt_init(void) { return NULL; }
void binopt_fini(BinoptHandle handle) {
    (void) handle;
}

BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    // We only support 6 integer-pointer arguments.
    if (cfg->param_count > 6)
        return cfg->func;

    Rewriter* r = dbrew_new();
    dbrew_verbose(r, false, false, false);
    dbrew_optverbose(r, false);
    dbrew_set_function(r, (uintptr_t) cfg->func);
    dbrew_config_parcount(r, cfg->param_count);
    if (cfg->ret_ty == BINOPT_TY_FLOAT || cfg->ret_ty == BINOPT_TY_DOUBLE)
        dbrew_config_returnfp(r);
    void* args[6] = {0};
    for (size_t i = 0; i < cfg->param_count; ++i) {
        if (!cfg->params[i].const_val)
            continue;
        switch (cfg->params[i].ty) {
        case BINOPT_TY_INT8:
        case BINOPT_TY_INT16:
        case BINOPT_TY_INT32:
        case BINOPT_TY_INT64:
        case BINOPT_TY_UINT8:
        case BINOPT_TY_UINT16:
        case BINOPT_TY_UINT32:
        case BINOPT_TY_UINT64:
        case BINOPT_TY_PTR:
        case BINOPT_TY_PTR_NOALIAS:
            dbrew_config_staticpar(r, i);
            memcpy(&args[i], cfg->params[i].const_val, sizeof(args[i]));
            break;
        default:
            dbrew_free(r);
            return cfg->func;
        }
    }
    for (size_t i = 0; i < cfg->memrange_count; ++i) {
        switch (cfg->memranges[i].flags) {
        case BINOPT_MEM_CONST:
        case BINOPT_MEM_NESTED_CONST:
            dbrew_config_set_memrange(r, "<unknown>", false,
                                      (uint64_t) cfg->memranges[i].base,
                                      cfg->memranges[i].size);
            break;
        default:
            // Do nothing.
            break;
        }
    }

    return (BinoptFunc) dbrew_rewrite(r, args[0], args[1], args[2], args[3],
                                      args[4], args[5]);
}

void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {
    // TODO: free rewriter...
    (void) handle;
    (void) spec_func;
}
