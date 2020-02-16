
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
    /// The memory region flags depend on the page mapping -- read-only pages
    /// are assumed to be constant, while unmapped and writable regions are
    /// treated as dynamic.
    BINOPT_MEM_DEFAULT = 0,
    /// The memory region is treated as constant. Behavior if the area is
    /// modified between configuration and the last call of the rewritten code
    /// is undefined.
    BINOPT_MEM_CONST,
    /// The memory region and all regions deduced from pointers (recursively)
    /// loaded from that are assumed to be constant. Some rewriters do not
    /// support detecting nested pointers and treat such regions as regular
    /// constant memory.
    BINOPT_MEM_NESTED_CONST,
    /// The memory region is treated as dynamic.
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
    /// Undefined flag value. Do not use.
    BINOPT_F_UNDEF = 0,
    /// Maximum stack size of optimized code.
    BINOPT_F_STACKSZ,
    /// Fast-math optimizations flags.
    BINOPT_F_FASTMATH,
} BinoptOptFlags;

const char* BINOPT_API(driver)(void);
BinoptHandle BINOPT_API(init)(void);
void BINOPT_API(fini)(BinoptHandle handle);

/// Create a new configuration for a given function. Implementations may deduce
/// type information from DWARF or CTF information encoded in the binary, or
/// other sources. If such information is not available, the type must be
/// configured using #binopt_cfg_type.
BinoptCfgRef BINOPT_API(cfg_new)(BinoptHandle handle,
                                 BinoptFunc base_func);
/// Clone a configuration. This creates a deep clone of the previous
/// configuration and inherits all properties. The new configuration is entirely
/// independent of the old configuration. Implementations may use copy-on-write
/// semantics internally.
BinoptCfgRef BINOPT_API(cfg_clone)(BinoptCfgRef base_cfg);
/// Set function signature with a specified number of parameters. Functions with
/// a variable number of arguments are not supported.
void BINOPT_API(cfg_type)(BinoptCfgRef cfg, unsigned count, BinoptType ret, ...);
/// Set a configuration flag. Additional flags may be implementation-defined.
void BINOPT_API(cfg_set)(BinoptCfgRef cfg, BinoptOptFlags flag, size_t val);
/// Set a parameter to a constant value. Note that this API takes a non-captured
/// pointer to the constant value. The size of the dereferenced memory is
/// inferred from the argument type. The value is copied into an internal
/// configuration storage. Behavior for an out-of-range index is undefined.
void BINOPT_API(cfg_set_param)(BinoptCfgRef cfg, unsigned idx, const void* val);
/// Explicitly configure a memory region. Constant memory regions are not
/// copied and must not be modified afterwards.
void BINOPT_API(cfg_mem)(BinoptCfgRef cfg, void* base, size_t size,
                         BinoptMemFlags flags);
/// Free a configuration.
void BINOPT_API(cfg_free)(BinoptCfgRef cfg);

/// Create a specialized implementation for a configuration. The ABI of the
/// returned function is identical to the original function. Implementations are
/// not required to actually return a specialized implementation, they may also
/// return a pointer to the original function. Behavior of the function for
/// incorrect configuration (including subsequent modifications of constant
/// memory) is undefined.
BinoptFunc BINOPT_API(spec_create)(BinoptCfgRef cfg);
/// Delete a specialized implementation for a configuration.
void BINOPT_API(spec_delete)(BinoptHandle handle, BinoptFunc spec_func);


// Convenience functions

/// Configure an integer parameter with a given value.
void BINOPT_API(cfg_set_parami)(BinoptCfgRef cfg, unsigned idx, size_t val);
/// Configure a pointer parameter with a given memory range.
void BINOPT_API(cfg_set_paramp)(BinoptCfgRef cfg, unsigned idx, const void* ptr,
                                size_t size, BinoptMemFlags flags);

#undef BINOPT_API_NAME
#undef BINOPT_API

#ifdef __cplusplus
}
#endif

#endif
