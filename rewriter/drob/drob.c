
#include "binopt.h"
#include "binopt-config.h"

#include <drob.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Drob bindings */

const char* binopt_driver(void) {
    return "Drob";
}
BinoptHandle binopt_init(void) {
    drob_setup();
    return NULL;
}
void binopt_fini(BinoptHandle handle) {
    // No teardown, there may be multiple handles.
    (void) handle;
}

static drob_param_type binopt_drob_map_ty(BinoptType ty) {
    switch (ty) {
    case BINOPT_TY_VOID: return DROB_PARAM_TYPE_VOID;
    case BINOPT_TY_INT8: return DROB_PARAM_TYPE_INT8;
    case BINOPT_TY_INT16: return DROB_PARAM_TYPE_INT16;
    case BINOPT_TY_INT32: return DROB_PARAM_TYPE_INT32;
    case BINOPT_TY_INT64: return DROB_PARAM_TYPE_INT64;
    case BINOPT_TY_UINT8: return DROB_PARAM_TYPE_UINT8;
    case BINOPT_TY_UINT16: return DROB_PARAM_TYPE_UINT16;
    case BINOPT_TY_UINT32: return DROB_PARAM_TYPE_UINT32;
    case BINOPT_TY_UINT64: return DROB_PARAM_TYPE_UINT64;
    case BINOPT_TY_FLOAT: return DROB_PARAM_TYPE_FLOAT;
    case BINOPT_TY_DOUBLE: return DROB_PARAM_TYPE_DOUBLE;
    case BINOPT_TY_PTR: return DROB_PARAM_TYPE_PTR;
    case BINOPT_TY_PTR_NOALIAS: return DROB_PARAM_TYPE_PTR;
    default: return DROB_PARAM_TYPE_MAX;
    }
}

BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    if (cfg->param_count > 6)
        return cfg->func;

    if (drob_set_logging(stderr, cfg->log_level))
        fprintf(stderr, "binopt-drob: invalid loglevel %d\n", cfg->log_level);

    drob_param_type dty_ret = binopt_drob_map_ty(cfg->ret_ty);
    drob_param_type dty_args[6] = {0};
    for (size_t i = 0; i < cfg->param_count; i++)
        dty_args[i] = binopt_drob_map_ty(cfg->params[i].ty);

    drob_cfg* dcfg = drob_cfg_new(dty_ret, cfg->param_count, dty_args[0],
                                  dty_args[1], dty_args[2], dty_args[3],
                                  dty_args[4], dty_args[5]);

    if (dcfg == NULL)
        return cfg->func;

    for (size_t i = 0; i < cfg->param_count; ++i) {
        struct BinoptCfgParam* param = &cfg->params[i];
        if (param->ty == BINOPT_TY_PTR_NOALIAS)
            drob_cfg_set_ptr_flag(dcfg, i, DROB_PTR_FLAG_RESTRICT);
        if (!param->const_val)
            continue;
        switch (param->ty) {
        case BINOPT_TY_INT8:
            drob_cfg_set_param_int8(dcfg, i, *(int8_t*) param->const_val);
            break;
        case BINOPT_TY_INT16:
            drob_cfg_set_param_int16(dcfg, i, *(int16_t*) param->const_val);
            break;
        case BINOPT_TY_INT32:
            drob_cfg_set_param_int32(dcfg, i, *(int32_t*) param->const_val);
            break;
        case BINOPT_TY_INT64:
            drob_cfg_set_param_int64(dcfg, i, *(int64_t*) param->const_val);
            break;
        case BINOPT_TY_UINT8:
            drob_cfg_set_param_uint8(dcfg, i, *(uint8_t*) param->const_val);
            break;
        case BINOPT_TY_UINT16:
            drob_cfg_set_param_uint16(dcfg, i, *(uint16_t*) param->const_val);
            break;
        case BINOPT_TY_UINT32:
            drob_cfg_set_param_uint32(dcfg, i, *(uint32_t*) param->const_val);
            break;
        case BINOPT_TY_UINT64:
            drob_cfg_set_param_uint64(dcfg, i, *(uint64_t*) param->const_val);
            break;
        case BINOPT_TY_PTR:
        case BINOPT_TY_PTR_NOALIAS:
            drob_cfg_set_param_ptr(dcfg, i, *(const void**) param->const_val);
            break;
        default:
            return cfg->func;
        }
    }
    for (size_t i = 0; i < cfg->memrange_count; ++i) {
        switch (cfg->memranges[i].flags) {
        case BINOPT_MEM_CONST:
        case BINOPT_MEM_NESTED_CONST:
            drob_cfg_add_const_range(dcfg, cfg->memranges[i].base,
                                     cfg->memranges[i].size);
            break;
        default:
            // Do nothing.
            break;
        }
    }

    drob_cfg_dump(dcfg);
    drob_cfg_set_error_handling(dcfg, DROB_ERROR_HANDLING_RETURN_NULL);
    BinoptFunc res = (BinoptFunc) drob_optimize((drob_f) cfg->func, dcfg);
    return res ? res : cfg->func;
}

void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {
    drob_free((drob_f) spec_func);
    (void) handle;
}
