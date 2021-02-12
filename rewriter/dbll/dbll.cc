
#include "binopt.h"
#include "binopt-config.h"

#include "ConstMemProp.h"
#include "LowerNativeCall.h"
#include "PtrToIntFold.h"

#include <rellume/rellume.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AliasAnalysisEvaluator.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/* DBLL based on LLVM+Rellume, using default configuration API */

namespace LogLevel {
enum {
    QUIET = 0,
    WARNING,
    INFO,
    DEBUG,
};
} // namespace LogLevel

const char* binopt_driver(void) {
    return "DBLL";
}


struct DbllHandle {
    llvm::LLVMContext ctx;

    DbllHandle() : ctx() {}
};

BinoptHandle binopt_init(void) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return reinterpret_cast<BinoptHandle>(new DbllHandle());
}
void binopt_fini(BinoptHandle handle) {
    delete reinterpret_cast<DbllHandle*>(handle);
}

static llvm::Type* dbll_map_type(BinoptType type, llvm::LLVMContext& ctx) {
    switch (type) {
    case BINOPT_TY_VOID: return llvm::Type::getVoidTy(ctx);
    case BINOPT_TY_INT8: return llvm::Type::getInt8Ty(ctx);
    case BINOPT_TY_INT16: return llvm::Type::getInt16Ty(ctx);
    case BINOPT_TY_INT32: return llvm::Type::getInt32Ty(ctx);
    case BINOPT_TY_INT64: return llvm::Type::getInt64Ty(ctx);
    case BINOPT_TY_UINT8: return llvm::Type::getInt8Ty(ctx);
    case BINOPT_TY_UINT16: return llvm::Type::getInt16Ty(ctx);
    case BINOPT_TY_UINT32: return llvm::Type::getInt32Ty(ctx);
    case BINOPT_TY_UINT64: return llvm::Type::getInt64Ty(ctx);
    case BINOPT_TY_FLOAT: return llvm::Type::getFloatTy(ctx);
    case BINOPT_TY_DOUBLE: return llvm::Type::getDoubleTy(ctx);
    case BINOPT_TY_PTR: return llvm::Type::getInt8PtrTy(ctx);
    case BINOPT_TY_PTR_NOALIAS: return llvm::Type::getInt8PtrTy(ctx);
    default: return nullptr;
    }
}

static llvm::FunctionType* dbll_map_function_type(BinoptCfgRef cfg) {
    DbllHandle* handle = reinterpret_cast<DbllHandle*>(cfg->handle);
    llvm::Type* ret_ty = dbll_map_type(cfg->ret_ty, handle->ctx);
    if (ret_ty == nullptr)
        return nullptr;

    llvm::SmallVector<llvm::Type*, 8> params;
    for (unsigned i = 0; i < cfg->param_count; ++i) {
        llvm::Type* param_ty = dbll_map_type(cfg->params[i].ty, handle->ctx);
        if (param_ty == nullptr)
            return nullptr;
        params.push_back(param_ty);
    }

    return llvm::FunctionType::get(ret_ty, params, false);
}

static llvm::Type* dbll_get_cpu_type(llvm::LLVMContext& ctx) {
    // TODO: extract from rellume programatically
    llvm::SmallVector<llvm::Type*, 4> cpu_types;
    cpu_types.push_back(llvm::Type::getInt64Ty(ctx)); // instruction pointer
    cpu_types.push_back(llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), 16));
    cpu_types.push_back(llvm::ArrayType::get(llvm::Type::getInt1Ty(ctx), 8));
    cpu_types.push_back(llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), 2));
    cpu_types.push_back(llvm::ArrayType::get(llvm::Type::getIntNTy(ctx, 128), 16));
    return llvm::StructType::get(ctx, cpu_types);
}

static llvm::Value* dbll_gep_helper(llvm::IRBuilder<>& irb, llvm::Value* base,
                                    llvm::ArrayRef<unsigned> idxs) {
    llvm::SmallVector<llvm::Value*, 4> consts;
    for (auto& idx : idxs)
        consts.push_back(irb.getInt32(idx));
    return irb.CreateGEP(base, consts);
}

class StrictSptrAAResult : public llvm::AAResultBase<StrictSptrAAResult> {
    friend llvm::AAResultBase<StrictSptrAAResult>;

    const llvm::DataLayout &DL;
public:
    StrictSptrAAResult(const llvm::DataLayout &DL) : AAResultBase(), DL(DL) {}

    llvm::AliasResult alias(const llvm::MemoryLocation &LocA, const llvm::MemoryLocation &LocB,
                            llvm::AAQueryInfo &AAQI) {
        if (!LocA.Ptr->getType()->isPointerTy() || !LocB.Ptr->getType()->isPointerTy())
            return llvm::NoAlias;

        if (IsSptr(LocA.Ptr) != IsSptr(LocB.Ptr))
            return llvm::NoAlias;

        return AAResultBase::alias(LocA, LocB, AAQI);
    }

private:
    bool IsSptr(const llvm::Value* val) {
        const llvm::Value* underlying = llvm::GetUnderlyingObject(val, DL);
        if (auto* inst = llvm::dyn_cast<llvm::Instruction>(underlying))
            return inst->getMetadata("dbll.sptr") != nullptr;
        if (auto* arg = llvm::dyn_cast<llvm::Argument>(underlying)) {
            if (auto* md = arg->getParent()->getMetadata("dbll.sptr")) {
                llvm::Metadata* idx_md = md->getOperand(0);
                auto* idx_mdc = llvm::cast<llvm::ConstantAsMetadata>(idx_md);
                auto idx = idx_mdc->getValue()->getUniqueInteger().getLimitedValue();
                return arg->getArgNo() == idx;
            }
        }
        return false;
    }
};

class StrictSptrAA : public llvm::AnalysisInfoMixin<StrictSptrAA> {
    friend llvm::AnalysisInfoMixin<StrictSptrAA>;
    static llvm::AnalysisKey Key;
public:
    using Result = StrictSptrAAResult;
    StrictSptrAAResult run(llvm::Function& f, llvm::FunctionAnalysisManager& fam) {
        return StrictSptrAAResult(f.getParent()->getDataLayout());
    }
};

llvm::AnalysisKey StrictSptrAA::Key;

namespace {

class Optimizer {
    BinoptCfgRef cfg;
    std::vector<dbll::ConstMemPropPass::MemRange> const_memranges;

    llvm::LLVMContext& ctx;
    llvm::Module* mod;
    llvm::TargetMachine* tm;

    LLConfig* rlcfg;

    llvm::Function* rl_func_call;
    llvm::Function* rl_func_tail;
    llvm::Function* func_helper_ext;
    llvm::GlobalVariable* llvm_used;

    llvm::DenseMap<BinoptFunc, llvm::Function*> lifted_fns;

    Optimizer(BinoptCfgRef cfg, llvm::Module* mod)
            : cfg(cfg), ctx(mod->getContext()), mod(mod) {}
    ~Optimizer();

    void DebugPrint(int log_level, const char* name);

    bool Init();
    llvm::Function* Lift(BinoptFunc func);
    bool DiscoverAndLift(void);

    llvm::Function* Wrap(llvm::Function* orig_fn);

    static void OptimizeLight(llvm::Function* fn);

    void OptimizeHeavy();
    void PrepareForCodeGen();

public:
    static BinoptFunc OptimizeFromConfig(BinoptCfgRef cfg);
};

Optimizer::~Optimizer() {
    ll_config_free(rlcfg);
}

void Optimizer::DebugPrint(int log_level, const char* name) {
    if (cfg->log_level < log_level)
        return;
    llvm::dbgs() << "==================================================\n"
                 << "== Module dump: " << name << "\n";
    mod->print(llvm::dbgs(), nullptr);
}

bool Optimizer::Init() {
    std::string error;
    std::string triple = llvm::sys::getProcessTriple();
    auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        if (cfg->log_level >= LogLevel::WARNING)
            llvm::errs() << "could not select target: " << error << "\n";

        return false;
    }

    llvm::TargetOptions options;
    options.EnableFastISel = false;
    tm = target->createTargetMachine(triple,
                                     /*CPU=*/llvm::sys::getHostCPUName(),
                                     /*Features=*/"-avx", options,
                                     llvm::None, // llvm::Reloc::Default,
                                     llvm::None, // llvm::CodeModel::JITDefault,
                                     llvm::CodeGenOpt::Aggressive, /*JIT*/true);

    mod->setTargetTriple(triple);

    llvm::Type* i8p = llvm::Type::getInt8PtrTy(ctx);
    llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
    auto helper_ty = llvm::FunctionType::get(void_ty, {i8p}, false);
    auto linkage = llvm::GlobalValue::ExternalLinkage;
    rl_func_call = llvm::Function::Create(helper_ty, linkage, "call_fn", mod);
    rl_func_tail = llvm::Function::Create(helper_ty, linkage, "tail_fn", mod);
    func_helper_ext = dbll::LowerNativeCallPass::CreateNativeCallFn(*mod);

    // Store references to functions in llvm.used to prevent early removal.
    assert(!mod->getGlobalVariable("llvm.used") && "llvm.used already defined");
    llvm::LLVMContext& ctx = mod->getContext();
    llvm::Type* i8p_ty = llvm::Type::getInt8PtrTy(ctx);

    llvm::SmallVector<llvm::Constant*, 4> used_vals;
    used_vals.push_back(llvm::ConstantExpr::getPointerCast(rl_func_call, i8p_ty));
    used_vals.push_back(llvm::ConstantExpr::getPointerCast(rl_func_tail, i8p_ty));
    used_vals.push_back(llvm::ConstantExpr::getPointerCast(func_helper_ext, i8p_ty));

    llvm::ArrayType* used_ty = llvm::ArrayType::get(i8p_ty, used_vals.size());
    llvm_used = new llvm::GlobalVariable(
            *mod, used_ty, /*const=*/false, llvm::GlobalValue::AppendingLinkage,
            llvm::ConstantArray::get(used_ty, used_vals), "llvm.used");
    llvm_used->setSection("llvm.metadata");

    // Create Rellume config
    rlcfg = ll_config_new();
    ll_config_enable_fast_math(rlcfg, !!(cfg->fast_math & 1));
    ll_config_set_call_ret_clobber_flags(rlcfg, true);
    ll_config_set_use_native_segment_base(rlcfg, true);
    ll_config_enable_full_facets(rlcfg, true);
    ll_config_set_tail_func(rlcfg, llvm::wrap(rl_func_tail));
    ll_config_set_call_func(rlcfg, llvm::wrap(rl_func_call));

    for (size_t i = 0; i < cfg->memrange_count; i++) {
        const auto* range = &cfg->memranges[i];
        if (range->flags == BINOPT_MEM_CONST) {
            uintptr_t base = reinterpret_cast<uintptr_t>(range->base);
            const_memranges.push_back({base, range->size});
        }
    }

    std::ifstream proc_maps;
    proc_maps.open("/proc/self/maps");
    if (proc_maps.is_open()) {
        std::string map;
        while (std::getline(proc_maps, map)) {
            char *endptr;
            uintptr_t start = strtoull(map.c_str(), &endptr, 16);
            uintptr_t end = strtoull(++endptr, &endptr, 16);
            bool prot_r = *(++endptr) == 'r';
            bool prot_w = *(++endptr) == 'w';
            // bool prot_x = *(++endptr) == 'x';

            if (prot_r && !prot_w)
                const_memranges.push_back({start, end-start});
        }
    }

    return true;
}

llvm::Function* Optimizer::Lift(BinoptFunc func) {
    const auto& lifted_fns_iter = lifted_fns.find(func);
    if (lifted_fns_iter != lifted_fns.end())
        return lifted_fns_iter->second;

    // Do not lift PLT entries. This is "jmp [rip + <soff32>]".
    if (*((uint8_t*) func) == 0xff && *((uint8_t*) func + 1) == 0x25)
        return nullptr;

    if (cfg->log_level >= LogLevel::DEBUG)
        llvm::dbgs() << "Lifting " << (void*)func << "\n";

    // Note: rl_func_call/rl_func_tail must have no uses before this function.
    assert(rl_func_tail->hasOneUse() && "rl_func_tail has uses (before)");
    assert(rl_func_call->hasOneUse() && "rl_func_call has uses (before)");

    LLFunc* rlfn = ll_func_new(llvm::wrap(mod), rlcfg);
    bool fail = ll_func_decode_cfg(rlfn, reinterpret_cast<uintptr_t>(func),
                                   nullptr, nullptr);
    if (fail) {
        if (cfg->log_level >= LogLevel::DEBUG)
            llvm::dbgs() << "Lifting " << (void*)func << " FAILED (decode).\n";

        ll_func_dispose(rlfn);
        return nullptr;
    }

    llvm::Value* fn_val = llvm::unwrap(ll_func_lift(rlfn));
    ll_func_dispose(rlfn);

    if (!fn_val) {
        if (cfg->log_level >= LogLevel::DEBUG)
            llvm::dbgs() << "Lifting " << (void*)func << " FAILED (lift).\n";
        return nullptr;
    }

    llvm::Function* fn = llvm::cast<llvm::Function>(fn_val);
    fn->setLinkage(llvm::GlobalValue::PrivateLinkage);

    std::stringstream fname;
    fname << "lift_" << std::hex << reinterpret_cast<uintptr_t>(func);
    fn->setName(fname.str());

    llvm::IRBuilder<> irb(fn->getEntryBlock().getFirstNonPHI());

    // Argument index
    llvm::Metadata* sptr_md = llvm::ConstantAsMetadata::get(irb.getInt32(0));
    fn->setMetadata("dbll.sptr", llvm::MDNode::get(ctx, {sptr_md}));

    // fn has the signature void(i8* sptr), and may contain calls to
    // rl_func_call and rl_func_tail. First, we replace these helper functions
    // with other helper functions which additionally contain parameters for RIP
    // and a pointer to the (user) return address, so that constant propagation
    // will eventually give us a constant RIP.
    //
    // The difference between tail_func and call_func is the following: For
    // tail_func, we hook the return address of this function. For call_func we
    // must hook the return address which was just stored on the fake stack.

    llvm::SmallVector<std::pair<llvm::CallInst*, bool>, 8> tmp_insts;
    for (const llvm::Use& use : rl_func_tail->uses()) {
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(use.getUser());
        if (!call || call->getCalledFunction() != rl_func_tail)
            continue;
        tmp_insts.push_back(std::make_pair(call, false));
    }
    for (const llvm::Use& use : rl_func_call->uses()) {
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(use.getUser());
        if (!call || call->getCalledFunction() != rl_func_call)
            continue;
        tmp_insts.push_back(std::make_pair(call, true));
    }

    llvm::Type* i64 = irb.getInt64Ty();
    llvm::Type* i64p = i64->getPointerTo();
    llvm::Value* sptr = &fn->arg_begin()[0];
    llvm::Value* sptr_ip = irb.CreateBitCast(sptr, i64p);
    // Note: we first do the GEP and then cast. EarlyCSE is not very clever in
    // reasoning about where a GEP leads to. For the same reason, use ptrtoint
    // instead of casting to an i64**.
    llvm::Value* sptr_sp_i8p = irb.CreateConstGEP1_64(sptr, 5 * 8);
    llvm::Value* sptr_sp = irb.CreateBitCast(sptr_sp_i8p, i64p);
    llvm::Value* entry_sp = irb.CreateIntToPtr(irb.CreateLoad(i64, sptr_sp), i64p);

    llvm::Value* args[4];
    for (auto [inst, is_call] : tmp_insts) {
        irb.SetInsertPoint(inst);

        assert(inst->getArgOperand(0) == sptr && "multiple sptrs");
        args[0] = sptr;
        if (is_call)
            args[1] = irb.CreateIntToPtr(irb.CreateLoad(i64, sptr_sp), i64p);
        else
            args[1] = entry_sp;
        args[2] = irb.CreateLoad(irb.getInt64Ty(), sptr_ip);
        args[3] = irb.getInt1(is_call ? 1 : 0);

        llvm::Value* return_rip = irb.CreateLoad(irb.getInt64Ty(), args[1]);

        irb.CreateCall(func_helper_ext, args);

        // We have some information about ext_helper regarding RIP/RSP.
        // Set RIP to the address which was just stored on the stack before.
        irb.CreateStore(return_rip, sptr_ip);
        // Set user RSP to stored_rip_ptr + 8
        llvm::Value* new_sp = irb.CreateConstGEP1_64(args[1], 1);
        irb.CreateStore(irb.CreatePtrToInt(new_sp, i64), sptr_sp);

        // Remove call to rl_func_call/rl_func_tail.
        inst->eraseFromParent();
    }
    tmp_insts.clear();

    // Note: rl_func_call/rl_func_tail must have no uses after this function.
    assert(rl_func_tail->hasOneUse() && "rl_func_tail has uses (after)");
    assert(rl_func_call->hasOneUse() && "rl_func_call has uses (after)");

    OptimizeLight(fn);

    // Check if any tail call remains after optimization. If so, don't mark the
    // function as inline.
    bool has_tail_fn = false;
    for (const llvm::Use& use : func_helper_ext->uses()) {
        llvm::CallInst* ci = llvm::dyn_cast<llvm::CallInst>(use.getUser());
        if (!ci || ci->getCalledFunction() != func_helper_ext ||
            ci->getParent()->getParent() != fn)
            continue;
        auto is_call = llvm::dyn_cast<llvm::Constant>(ci->getArgOperand(3));
        if (!is_call || is_call->isZeroValue())
            has_tail_fn = true;
    }
    if (!has_tail_fn) {
        fn->addFnAttr(llvm::Attribute::InlineHint);
        fn->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    lifted_fns[func] = fn;

    return fn;
}

bool Optimizer::DiscoverAndLift() {
    // Try to iteratively discover called functions and lift them as well.
    bool new_code = false;
    bool changed = true;
    llvm::SmallVector<std::pair<uint64_t, llvm::CallInst*>, 8> ext_call_queue;
    while (changed) {
        changed = false;
        for (const llvm::Use& use : func_helper_ext->uses()) {
            llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(use.getUser());
            if (!call || call->getCalledFunction() != func_helper_ext)
                continue;
            if (auto c = llvm::dyn_cast<llvm::Constant>(call->getArgOperand(2))) {
                uint64_t addr = c->getUniqueInteger().getLimitedValue();
                ext_call_queue.push_back(std::make_pair(addr, call));
            }
        }

        for (auto [addr, inst] : ext_call_queue) {
            llvm::Function* fn = Lift(reinterpret_cast<BinoptFunc>(addr));
            if (cfg->log_level >= LogLevel::DEBUG) {
                llvm::errs() << "selecting call: " << (void*) addr << " ";
                inst->print(llvm::errs());
                llvm::errs() << "; got " << fn << "\n";
            }
            if (!fn)
                continue;

            auto is_call = llvm::dyn_cast<llvm::Constant>(inst->getArgOperand(3));
            auto* new_inst = llvm::CallInst::Create(fn, {inst->getArgOperand(0)});
            llvm::ReplaceInstWithInst(inst, new_inst);

            // Directly inline tail functions
            if (is_call && is_call->isZeroValue()) {
                llvm::InlineFunctionInfo ifi;
#if DBLL_LLVM_MAJOR < 11
                llvm::InlineFunction(llvm::CallSite(new_inst), ifi);
#else
                llvm::InlineFunction(*new_inst, ifi);
#endif
            }
            new_code = true;
            changed = true;
        }

        ext_call_queue.clear();
    }

    return new_code;
}

llvm::Function* Optimizer::Wrap(llvm::Function* orig_fn) {
    llvm::FunctionType* fnty = dbll_map_function_type(cfg);
    if (fnty == nullptr) {// if we don't support the type
        DebugPrint(LogLevel::WARNING, "unsupported function type");
        return nullptr;
    }

    // Create new function
    llvm::LLVMContext& ctx = orig_fn->getContext();
    auto linkage = llvm::GlobalValue::ExternalLinkage;
    llvm::Function* fn = llvm::Function::Create(fnty, linkage, "glob",
                                                orig_fn->getParent());
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(ctx, "", fn);
    llvm::IRBuilder<> irb(bb);

    // Allocate CPU struct
    llvm::Type* cpu_type = dbll_get_cpu_type(ctx);
    llvm::AllocaInst* alloca = irb.CreateAlloca(cpu_type, int{0});
    alloca->setMetadata("dbll.sptr", llvm::MDNode::get(ctx, {}));
#if DBLL_LLVM_MAJOR < 10
    alloca->setAlignment(16);
#else
    alloca->setAlignment(llvm::Align(16));
#endif

    // Set direction flag to zero
    irb.CreateStore(irb.getFalse(), dbll_gep_helper(irb, alloca, {0, 2, 6}));

    unsigned gp_regs[6] = { 7, 6, 2, 1, 8, 9 };
    unsigned gpRegOffset = 0;
    unsigned fpRegOffset = 0;
    unsigned stackOffset = 8; // return address
    llvm::SmallVector<std::pair<size_t, llvm::Value*>, 4> stack_slots;
    llvm::Value* target;
    unsigned arg_idx = 0;
    for (auto arg = fn->arg_begin(); arg != fn->arg_end(); ++arg, ++arg_idx) {
        llvm::Type* arg_type = arg->getType();
        llvm::Value* arg_val = arg;

        if (cfg->params[arg_idx].ty == BINOPT_TY_PTR_NOALIAS) {
            fn->addParamAttr(arg_idx, llvm::Attribute::NoAlias);
        }

        // Fix known parameters
        if (const void* const_vptr = cfg->params[arg_idx].const_val) {
            auto const_ptr = reinterpret_cast<const uint64_t*>(const_vptr);
            size_t const_sz = arg_type->getPrimitiveSizeInBits();
            if (arg_type->isPointerTy())
                const_sz = sizeof(void*) * 8;
            llvm::APInt const_val(const_sz, llvm::ArrayRef(const_ptr, const_sz/64));
            arg_val = llvm::ConstantInt::get(ctx, const_val);
        }

        if (arg_type->isIntOrPtrTy()) {
            if (gpRegOffset < 6) {
                if (arg_type->isPointerTy())
                    arg_val = irb.CreatePtrToInt(arg_val, irb.getInt64Ty());
                else // arg_type->isIntegerTy()
                    arg_val = irb.CreateZExtOrTrunc(arg_val, irb.getInt64Ty());

                target = dbll_gep_helper(irb, alloca, {0, 1, gp_regs[gpRegOffset]});
                irb.CreateStore(arg_val, target);
                gpRegOffset++;
            } else {
                stack_slots.push_back(std::make_pair(stackOffset, arg_val));
                stackOffset += 8;
            }
        } else if (arg_type->isFloatTy() || arg_type->isDoubleTy()) {
            if (fpRegOffset < 8) {
                llvm::Type* int_type = irb.getIntNTy(arg_type->getPrimitiveSizeInBits());
                llvm::Value* int_val = irb.CreateBitCast(arg_val, int_type);

                target = dbll_gep_helper(irb, alloca, {0, 4, fpRegOffset});
                llvm::Type* vec_type = target->getType()->getPointerElementType();
                irb.CreateStore(irb.CreateZExt(int_val, vec_type), target);
                fpRegOffset++;
            } else {
                stack_slots.push_back(std::make_pair(stackOffset, arg_val));
                stackOffset += 8;
            }
        } else {
            DebugPrint(LogLevel::WARNING, "unsupported parameter type");
            return nullptr;
        }
    }

    std::size_t stack_frame_size = 4096 - 8;
    std::size_t stack_size = stack_frame_size + stackOffset;

    llvm::Value* stack_sz_val = irb.getInt64(stack_size);
    llvm::AllocaInst* stack = irb.CreateAlloca(irb.getInt8Ty(), stack_sz_val);
#if DBLL_LLVM_MAJOR < 10
    stack->setAlignment(16);
#else
    stack->setAlignment(llvm::Align(16));
#endif
    llvm::Value* sp_ptr = irb.CreateGEP(stack, irb.getInt64(stack_frame_size));
    llvm::Value* sp = irb.CreatePtrToInt(sp_ptr, irb.getInt64Ty());
    irb.CreateStore(sp, dbll_gep_helper(irb, alloca, {0, 1, 4}));

    for (const auto& [offset, value] : stack_slots) {
        llvm::Value* ptr = irb.CreateGEP(sp_ptr, irb.getInt64(offset));
        irb.CreateStore(value, irb.CreateBitCast(ptr, value->getType()->getPointerTo()));
    }

    llvm::Value* call_arg = irb.CreatePointerCast(alloca, irb.getInt8PtrTy());
    llvm::CallInst* call = irb.CreateCall(orig_fn, {call_arg});

    llvm::Type* ret_type = fn->getReturnType();
    switch (ret_type->getTypeID())
    {
        llvm::Value* ret;

        case llvm::Type::TypeID::VoidTyID:
            irb.CreateRetVoid();
            break;
        case llvm::Type::TypeID::IntegerTyID:
            ret = irb.CreateLoad(dbll_gep_helper(irb, alloca, {0, 1, 0}));
            ret = irb.CreateTruncOrBitCast(ret, ret_type);
            irb.CreateRet(ret);
            break;
        case llvm::Type::TypeID::PointerTyID:
            ret = irb.CreateLoad(dbll_gep_helper(irb, alloca, {0, 1, 0}));
            ret = irb.CreateIntToPtr(ret, ret_type);
            irb.CreateRet(ret);
            break;
        case llvm::Type::TypeID::FloatTyID:
        case llvm::Type::TypeID::DoubleTyID:
            ret = irb.CreateLoad(dbll_gep_helper(irb, alloca, {0, 4, 0}));
            ret = irb.CreateTrunc(ret, irb.getIntNTy(ret_type->getPrimitiveSizeInBits()));
            ret = irb.CreateBitCast(ret, ret_type);
            irb.CreateRet(ret);
            break;
        default:
            assert(false);
            break;
    }

    llvm::InlineFunctionInfo ifi;
#if DBLL_LLVM_MAJOR < 11
    llvm::InlineFunction(llvm::CallSite(call), ifi);
#else
    llvm::InlineFunction(*call, ifi);
#endif

    OptimizeLight(fn);

    return fn;
}

void Optimizer::OptimizeLight(llvm::Function* fn) {
    // Do some very simple optimizations, so that calls to ext_helper are
    // simplified that a constant target RIP is propagated and subsequent
    // branches based on the RIP value are eliminated.

    llvm::PassBuilder pb;
    llvm::FunctionPassManager fpm(false);

    llvm::LoopAnalysisManager lam(false);
    llvm::FunctionAnalysisManager fam(false);
    llvm::CGSCCAnalysisManager cgam(false);
    llvm::ModuleAnalysisManager mam(false);

    fam.registerPass([&] {
        llvm::AAManager aa;
        aa.registerFunctionAnalysis<StrictSptrAA>();
        aa.registerFunctionAnalysis<llvm::BasicAA>();
        return aa;
    });
    fam.registerPass([&] { return StrictSptrAA(); });

    // Register analysis passes...
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    fpm.addPass(llvm::EarlyCSEPass(true));
    fpm.addPass(llvm::InstCombinePass(false));
    fpm.addPass(llvm::SimplifyCFGPass());
    fpm.addPass(dbll::PtrToIntFoldPass());
    // fpm.addPass(llvm::AAEvaluator());
    fpm.run(*fn, fam);
}

void Optimizer::OptimizeHeavy() {
    llvm::PassInstrumentationCallbacks pic;
    llvm::StandardInstrumentations si;
    si.registerCallbacks(pic);

    llvm::PassBuilder pb(tm, llvm::PipelineTuningOptions(), llvm::None, &pic);

    llvm::LoopAnalysisManager lam(false);
    llvm::FunctionAnalysisManager fam(false);
    llvm::CGSCCAnalysisManager cgam(false);
    llvm::ModuleAnalysisManager mam(false);

    fam.registerPass([&] {
        llvm::AAManager aa;
        aa.registerFunctionAnalysis<StrictSptrAA>();
        aa.registerFunctionAnalysis<llvm::BasicAA>();
        return aa;
    });
    fam.registerPass([&] { return StrictSptrAA(); });

    // Register analysis passes...
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    pb.registerPeepholeEPCallback([this] (llvm::FunctionPassManager& fpm,
                                       llvm::PassBuilder::OptimizationLevel ol) {
        fpm.addPass(dbll::PtrToIntFoldPass());
        fpm.addPass(dbll::ConstMemPropPass(const_memranges));
    });
    auto ol = llvm::PassBuilder::OptimizationLevel::O3;
    auto mpm = pb.buildPerModuleDefaultPipeline(ol, false);
    // mpm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::AAEvaluator()));

    mpm.run(*mod, mam);
}

void Optimizer::PrepareForCodeGen() {
    // Now we can let the optimizer remove all the function declarations.
    // TODO: this apparently doesn' work.
    // if (llvm_used) {
    //     llvm_used->removeFromParent();
    //     llvm_used = nullptr;
    // }

    // TODO: Don't run a full optimization pipeline here.
    llvm::PassInstrumentationCallbacks pic;
    llvm::StandardInstrumentations si;
    si.registerCallbacks(pic);

    llvm::PassBuilder pb(tm, llvm::PipelineTuningOptions(), llvm::None, &pic);

    llvm::LoopAnalysisManager lam(false);
    llvm::FunctionAnalysisManager fam(false);
    llvm::CGSCCAnalysisManager cgam(false);
    llvm::ModuleAnalysisManager mam(false);

    // Register the AA manager
    // fam.registerPass([&] { return pb.buildDefaultAAPipeline(); });
    fam.registerPass([&] {
        llvm::AAManager aa;
        aa.registerFunctionAnalysis<StrictSptrAA>();
        aa.registerFunctionAnalysis<llvm::BasicAA>();
        return aa;
    });
    fam.registerPass([&] { return StrictSptrAA(); });

    // Register analysis passes...
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    // Lower native calls in the beginning.
    pb.registerPipelineStartEPCallback([] (llvm::ModulePassManager& mpm) {
        mpm.addPass(dbll::LowerNativeCallPass());
    });
    auto ol = llvm::PassBuilder::OptimizationLevel::O3;
    auto mpm = pb.buildPerModuleDefaultPipeline(ol, false);

    mpm.run(*mod, mam);
}

BinoptFunc Optimizer::OptimizeFromConfig(BinoptCfgRef cfg) {
    DbllHandle* handle = reinterpret_cast<DbllHandle*>(cfg->handle);
    auto mod_u = std::make_unique<llvm::Module>("binopt", handle->ctx);
    llvm::Module* mod = mod_u.get();

    Optimizer opt(cfg, mod);
    if (!opt.Init())
        return nullptr;

    llvm::Function* fn = opt.Lift(cfg->func);
    if (!fn)
        return nullptr;

    opt.DebugPrint(LogLevel::DEBUG, "Initially lifted");

    llvm::Function* wrapped_fn = opt.Wrap(fn);
    if (wrapped_fn == nullptr)
        return nullptr;

    opt.DebugPrint(LogLevel::DEBUG, "After ABI wrap");

    opt.DiscoverAndLift();
    bool new_code;
    do {
        opt.DebugPrint(LogLevel::DEBUG, "After discovery iteration");
        opt.OptimizeHeavy();
        new_code = opt.DiscoverAndLift();
    } while (new_code);

    opt.DebugPrint(LogLevel::DEBUG, "After full discovery");

    opt.PrepareForCodeGen();

    opt.DebugPrint(LogLevel::INFO, "Before codegen");

    // This should only scream if our code has a bug.
    if (llvm::verifyFunction(*wrapped_fn, &llvm::errs())) {
        wrapped_fn->eraseFromParent();
        return nullptr;
    }

    llvm::EngineBuilder builder(std::move(mod_u));
    llvm::ExecutionEngine* engine = builder.create(opt.tm);
    if (!engine)
        return cfg->func; // we could not create the JIT engine

    const auto& name = wrapped_fn->getName();
    auto raw_ptr = engine->getFunctionAddress(name.str());

    return reinterpret_cast<BinoptFunc>(raw_ptr);
}

}

BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    if (BinoptFunc new_fn = Optimizer::OptimizeFromConfig(cfg))
        return new_fn;

    if (cfg->log_level >= LogLevel::WARNING)
        llvm::errs() << "warning: returning old function\n";

    return cfg->func;
}

void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {
    // TODO: implement
}

__attribute__((constructor))
static void dbll_support_pass_arguments(void) {
    llvm::cl::ParseEnvironmentOptions("binopt-dbll", "DBLL_OPTS");
}
