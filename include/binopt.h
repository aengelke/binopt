
#ifndef BINOPT_H
#define BINOPT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BINOPT_API(name) binopt_ ## name

typedef void* BinoptFunc;

typedef struct BinoptOpaqueHandle* BinoptHandle;
typedef struct BinoptCfg* BinoptCfgRef;

typedef enum {
    BINOPT_MEM_DEFAULT = 0,
    BINOPT_MEM_CONST,
    BINOPT_MEM_NESTED_CONST,
    BINOPT_MEM_DYNAMIC,
} BinoptMemFlags;

typedef enum {
    BINOPT_TY_VOID = 0,
    BINOPT_TY_INT8,
    BINOPT_TY_INT16,
    BINOPT_TY_INT32,
    BINOPT_TY_INT64,
    BINOPT_TY_UINT8,
    BINOPT_TY_UINT16,
    BINOPT_TY_UINT32,
    BINOPT_TY_UINT64,
    BINOPT_TY_FLOAT,
    BINOPT_TY_DOUBLE,
    BINOPT_TY_PTR,
    BINOPT_TY_PTR_NOALIAS,
} BinoptType;

typedef enum {
    BINOPT_F_UNDEF = 0,
    BINOPT_F_STACKSZ,
    BINOPT_F_FASTMATH,
} BinoptOptFlags;

const char* BINOPT_API(driver)(void);
BinoptHandle BINOPT_API(init)(void);
void BINOPT_API(fini)(BinoptHandle handle);

BinoptCfgRef BINOPT_API(cfg_new)(BinoptHandle handle,
                                 BinoptFunc base_func);
BinoptCfgRef BINOPT_API(cfg_clone)(BinoptCfgRef base_cfg);
// Set function signature.
void BINOPT_API(cfg_type)(BinoptCfgRef cfg, unsigned count, BinoptType ret, ...);
void BINOPT_API(cfg_set)(BinoptCfgRef cfg, BinoptOptFlags flag, size_t val);
void BINOPT_API(cfg_set_param)(BinoptCfgRef cfg, unsigned idx, const void* val);
void BINOPT_API(cfg_mem)(BinoptCfgRef cfg, void* base, size_t size,
                         BinoptMemFlags flags);
void BINOPT_API(cfg_free)(BinoptCfgRef cfg);

BinoptFunc BINOPT_API(spec_create)(BinoptCfgRef cfg);
void BINOPT_API(spec_delete)(BinoptHandle handle, BinoptFunc spec_func);

#undef BINOPT_API_NAME
#undef BINOPT_API

#ifdef __cplusplus
}
#endif

#endif
