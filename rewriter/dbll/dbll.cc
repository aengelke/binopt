
#include "binopt.h"
#include "binopt-config.h"

#include <rellume/rellume.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AliasAnalysisEvaluator.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

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

static llvm::Function* dbll_create_native_helper(llvm::Module* mod) {
    llvm::LLVMContext& ctx = mod->getContext();

    llvm::Type* i8p = llvm::Type::getInt8PtrTy(ctx);
    llvm::Type* i64p = llvm::Type::getInt64Ty(ctx)->getPointerTo();
    llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
    auto fn_ty = llvm::FunctionType::get(void_ty, {i8p, i64p}, false);
    auto linkage = llvm::GlobalValue::PrivateLinkage;
    llvm::Function* fn = llvm::Function::Create(fn_ty, linkage,
                                                "dbll_native_helper", mod);
    fn->addFnAttr(llvm::Attribute::AlwaysInline);

    llvm::BasicBlock* bb = llvm::BasicBlock::Create(ctx, "", fn);
    llvm::IRBuilder<> irb(bb);

    // We need some memory space addressable without any(!) registers except for
    // rip where we can store the stack pointer. As the LLVM MCJIT doesn's
    // implement thread-local storage, we opt for a global variable.
    llvm::GlobalValue* glob_slot = new llvm::GlobalVariable(*mod,
                irb.getInt64Ty(), false, llvm::GlobalValue::PrivateLinkage,
                irb.getInt64(0), "dbll_native_helper_rsp");

    llvm::Value* sptr = irb.CreateBitCast(&fn->arg_begin()[0], i64p);

    llvm::Value* stored_rip_ptr = &fn->arg_begin()[1];

    llvm::Value* sptr_rsp = irb.CreateConstGEP1_64(sptr, 5);

    // Buffer area passed to inline asm.
    // Contains: userrip, cs, userrflags, userrsp, ss
    llvm::Value* asm_buf = irb.CreateAlloca(irb.getInt64Ty(), irb.getInt64(8));
    irb.CreateStore(irb.CreateLoad(sptr), irb.CreateConstGEP1_64(asm_buf, 0));
    irb.CreateStore(irb.getInt64(0x33), irb.CreateConstGEP1_64(asm_buf, 1));
    // TODO: actually compute rflags?
    irb.CreateStore(irb.getInt64(0x202), irb.CreateConstGEP1_64(asm_buf, 2));
    irb.CreateStore(irb.CreateLoad(sptr_rsp), irb.CreateConstGEP1_64(asm_buf, 3));
    irb.CreateStore(irb.getInt64(0x2b), irb.CreateConstGEP1_64(asm_buf, 4));

    llvm::SmallVector<llvm::Value*, 31> sptr_geps;
    llvm::SmallVector<llvm::Value*, 32> asm_args;

    for (unsigned idx = 0; idx < 16; idx++) {
        if (idx == 4)
            continue; // Skip RSP
        llvm::Value* ptr = irb.CreateConstGEP1_64(sptr, 1 + idx);
        sptr_geps.push_back(ptr);
        asm_args.push_back(irb.CreateLoad(irb.getInt64Ty(), ptr));
    }
    llvm::Type* sse_ty = llvm::VectorType::get(irb.getInt64Ty(), 2);
    llvm::Value* sptr128 = irb.CreateBitCast(sptr, sse_ty->getPointerTo());
    for (uint8_t idx = 0; idx < 16; idx++) {
        llvm::Value* ptr = irb.CreateConstGEP1_64(sse_ty, sptr128, 10 + idx, "xmmgep");
        sptr_geps.push_back(ptr);
        asm_args.push_back(irb.CreateLoad(sse_ty, ptr));
    }

    // First construct ty_list for the return type
    llvm::SmallVector<llvm::Type*, 32> ty_list;
    for (llvm::Value* arg : asm_args)
        ty_list.push_back(arg->getType());

    llvm::Type* asm_ret_ty = llvm::StructType::get(ctx, ty_list);

    // Store RAX, RCX and RDX in asm_buf
    irb.CreateStore(asm_args[0], irb.CreateConstGEP1_64(asm_buf, 5));
    irb.CreateStore(asm_args[1], irb.CreateConstGEP1_64(asm_buf, 6));
    irb.CreateStore(asm_args[2], irb.CreateConstGEP1_64(asm_buf, 7));
    // RAX = asm_buf, RCX = stored_rip_ptr, RDX = glob_slot
    asm_args[0] = asm_buf;
    asm_args[1] = stored_rip_ptr;
    asm_args[2] = glob_slot;
    ty_list[0] = i64p;
    ty_list[1] = i64p;
    ty_list[2] = i64p;

    asm_args.push_back(glob_slot);
    ty_list.push_back(i64p);

    auto asm_ty = llvm::FunctionType::get(asm_ret_ty, ty_list, false);
    const auto constraints =
        "={ax},={cx},={dx},={bx},={bp},={si},={di},={r8},={r9},={r10},={r11},"
        "={r12},={r13},={r14},={r15},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},"
        "={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},"
        "={xmm13},={xmm14},={xmm15},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,"
        "17,18,19,20,21,22,23,24,25,26,27,28,29,30,i";
    const auto asm_code =
            "mov %rsp, (%rdx);" // store rsp in glob_slot
            "lea 1f(%rip), %rdx;"
            "mov %rax, %rsp;"
            "mov 0x28(%rsp), %rax;" // setup user rax
            "mov %rdx, (%rcx);" // store return address in stored_rip_ptr
            "mov 0x30(%rsp), %rcx;" // setup user rcx
            "mov 0x38(%rsp), %rdx;" // setup user rdx
            "iretq;" // isn't this cool? -- no, it isn't.
        "1:" // TODO: perhaps check rsp somehow?
            "movabs $62, %rsp;" // throw away user rsp.
            "mov (%rsp), %rsp;";

    auto asm_inst = llvm::InlineAsm::get(asm_ty, asm_code, constraints,
                                         /*hasSideEffects=*/true,
                                         /*alignStack=*/true,
                                         llvm::InlineAsm::AD_ATT);
    llvm::Value* asm_res = irb.CreateCall(asm_inst, asm_args);

    // RIP and RSP are set outside of this function to allow for better
    // optimizations for calls/jumps with known targets.
    for (unsigned i = 0; i < sptr_geps.size(); i++)
        irb.CreateStore(irb.CreateExtractValue(asm_res, i), sptr_geps[i]);

    irb.CreateRetVoid();

    return fn;
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

class ConstMemPropPass : public llvm::PassInfoMixin<ConstMemPropPass> {
    BinoptCfgRef cfg;
public:
    ConstMemPropPass(BinoptCfgRef cfg) : cfg(cfg) {}

private:
    // Returns 0 when the constant could not be folded.
    llvm::APInt ConstantValue(const llvm::Constant* c) {
        // A constant is either undefined...
        if (llvm::isa<llvm::UndefValue>(c))
            return llvm::APInt(64, 0);

        // ... or a constant expression ...
        if (const llvm::ConstantExpr* ce = llvm::dyn_cast<llvm::ConstantExpr>(c)) {
            switch (ce->getOpcode()) {
            case llvm::Instruction::IntToPtr: {
                llvm::APInt ptr_val = ConstantValue(ce->getOperand(0));
                return ptr_val.zextOrTrunc(64);
            }
            default:
                // Can't handle that expression
                return llvm::APInt(64, 0);
            }
        }

        // ... or simple.
        switch (c->getType()->getTypeID()) {
        case llvm::Type::IntegerTyID:
            return llvm::cast<llvm::ConstantInt>(c)->getValue();
        default:
            // Can't handle that constant type
            return llvm::APInt(64, 0);
        }
    }

    std::pair<bool, llvm::APInt> GetConstantMem(uintptr_t addr, size_t size) {
        size_t size_bytes = (size + 7) / 8;
        for (size_t i = 0; i < cfg->memrange_count; ++i) {
            const auto& range = cfg->memranges[i];
            if ((uintptr_t) range.base > addr ||
                (uintptr_t) range.base + range.size < addr + size_bytes)
                continue;
            if (range.flags != BINOPT_MEM_CONST)
                continue;

            auto const_ptr = reinterpret_cast<const uint64_t*>(addr);
            llvm::APInt const_val(size, llvm::ArrayRef(const_ptr, size/64));
            return std::make_pair(true, const_val);
        }
        return std::make_pair(false, llvm::APInt(size, 0));
    }

    llvm::Constant* ConstantFoldLoad(llvm::LoadInst* load) {
        auto addr = llvm::dyn_cast<llvm::Constant>(load->getPointerOperand());
        if (!addr)
            return nullptr;

        uint64_t addr_val = ConstantValue(addr).trunc(64).getLimitedValue();

        if (cfg->log_level >= LogLevel::DEBUG) {
            std::cerr << "folding load: ";
            load->print(llvm::errs());
            std::cerr << " to 0x" << std::hex << addr_val << "\n";
        }

        if (!addr_val)
            return nullptr;

        llvm::Type* target_ty = addr->getType()->getPointerElementType();
        size_t target_bits = target_ty->getPrimitiveSizeInBits();
        if (target_ty->isPointerTy())
            target_bits = 64;

        auto const_mem = GetConstantMem(addr_val, target_bits);
        if (!const_mem.first)
            return nullptr;

        llvm::LLVMContext& ctx = target_ty->getContext();
        llvm::Constant* const_int = llvm::ConstantInt::get(ctx, const_mem.second);
        llvm::Constant* const_val;
        if (target_ty->isPointerTy())
            const_val = llvm::ConstantExpr::getIntToPtr(const_int, target_ty);
        else
            const_val = llvm::ConstantExpr::getBitCast(const_int, target_ty);

        if (cfg->log_level >= LogLevel::DEBUG) {
            std::cerr << "folded to: ";
            const_val->print(llvm::errs());
            std::cerr << "\n";
        }

        return const_val;
    }

public:
    llvm::PreservedAnalyses run(llvm::Function& fn, llvm::FunctionAnalysisManager& fam) {

        const llvm::DataLayout& dl = fn.getParent()->getDataLayout();

        // strategy similar to llvm::ConstantPropagation::runOnFunction.
        llvm::SmallPtrSet<llvm::Instruction*, 16> queue;
        llvm::SmallVector<llvm::Instruction*, 16> queue_vec;
        for (llvm::Instruction& inst : llvm::instructions(fn)) {
            llvm::LoadInst* load = llvm::dyn_cast<llvm::LoadInst>(&inst);
            if (load && llvm::isa<llvm::Constant>(load->getPointerOperand())) {
                queue.insert(&inst);
                queue_vec.push_back(&inst);
            }
        }

        bool changed = false;
        while (!queue.empty()) {
            llvm::SmallVector<llvm::Instruction*, 16> new_queue_vec;
            for (llvm::Instruction* inst : queue_vec) {
                queue.erase(inst);
                llvm::Constant* const_repl = nullptr;
                if (auto load_inst = llvm::dyn_cast<llvm::LoadInst>(inst))
                    const_repl = ConstantFoldLoad(load_inst);
                if (!const_repl)
                    const_repl = llvm::ConstantFoldInstruction(inst, dl, nullptr);
                if (!const_repl)
                    continue;

                for (llvm::User* user : inst->users()) {
                    // If user not in the set, then add it to the vector.
                    if (queue.insert(llvm::cast<llvm::Instruction>(user)).second)
                        new_queue_vec.push_back(llvm::cast<llvm::Instruction>(user));
                }

                inst->replaceAllUsesWith(const_repl);

                if (llvm::isInstructionTriviallyDead(inst, nullptr))
                    inst->eraseFromParent();

                changed = true;
            }

            queue_vec = std::move(new_queue_vec);
        }

        llvm::PreservedAnalyses pa;
        return pa;
    }
};

static void dbll_optimize_new_pm(BinoptCfgRef cfg, llvm::Module* mod,
                                 llvm::TargetMachine* tm) {
    llvm::PassBuilder pb(tm);

    llvm::LoopAnalysisManager lam(false);
    llvm::FunctionAnalysisManager fam(false);
    llvm::CGSCCAnalysisManager cgam(false);
    llvm::ModuleAnalysisManager mam(false);

    // Register the AA manager
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

    pb.registerPeepholeEPCallback([cfg] (llvm::FunctionPassManager& fpm,
                                       llvm::PassBuilder::OptimizationLevel ol) {
        fpm.addPass(ConstMemPropPass(cfg));
    });
    auto mpm = pb.buildPerModuleDefaultPipeline(llvm::PassBuilder::O3, false);
    // mpm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::AAEvaluator()));

    mpm.run(*mod, mam);
    // mpm.run(*mod, mam);
}

namespace {

class Optimizer {
    BinoptCfgRef cfg;

    llvm::LLVMContext& ctx;
    llvm::Module* mod;
    llvm::TargetMachine* tm;

    LLConfig* rlcfg;

    llvm::Function* rl_func_call;
    llvm::Function* rl_func_tail;
    llvm::Function* func_helper_ext;

    Optimizer(BinoptCfgRef cfg, llvm::Module* mod)
            : cfg(cfg), ctx(mod->getContext()), mod(mod) {}
    ~Optimizer();

    bool Init();
    llvm::Function* Lift(BinoptFunc func);
    llvm::Function* Wrap(llvm::Function* orig_fn);
    void LowerExternalCallToNative();

public:
    static BinoptFunc OptimizeFromConfig(BinoptCfgRef cfg);
};

Optimizer::~Optimizer() {
    ll_config_free(rlcfg);
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

    llvm::Type* i1 = llvm::Type::getInt1Ty(ctx);
    llvm::Type* i64 = llvm::Type::getInt64Ty(ctx);
    llvm::Type* i8p = llvm::Type::getInt8PtrTy(ctx);
    llvm::Type* i64p = i64->getPointerTo();
    llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
    auto helper_ty = llvm::FunctionType::get(void_ty, {i8p}, false);
    auto linkage = llvm::GlobalValue::ExternalLinkage;
    rl_func_call = llvm::Function::Create(helper_ty, linkage, "call_fn", mod);
    rl_func_tail = llvm::Function::Create(helper_ty, linkage, "tail_fn", mod);
    // This is void(i8* sptr, i64* retaddr_ptr, i64 rip, i1 is_call)
    auto helper_ext_ty = llvm::FunctionType::get(void_ty, {i8p, i64p, i64, i1}, false);
    func_helper_ext = llvm::Function::Create(helper_ext_ty, linkage,
                                             "ext_helper", mod);

    rlcfg = ll_config_new();
    ll_config_enable_fast_math(rlcfg, !!(cfg->fast_math & 1));
    ll_config_set_tail_func(rlcfg, llvm::wrap(rl_func_tail));
    ll_config_set_call_func(rlcfg, llvm::wrap(rl_func_call));

    return true;
}

llvm::Function* Optimizer::Lift(BinoptFunc func) {
    // Note: rl_func_call/rl_func_tail must have no uses before this function.
    assert(rl_func_tail->user_empty() && "rl_func_tail has uses (before)");
    assert(rl_func_call->user_empty() && "rl_func_call has uses (before)");

    LLFunc* rlfn = ll_func_new(llvm::wrap(mod), rlcfg);
    bool fail = ll_func_decode_cfg(rlfn, reinterpret_cast<uintptr_t>(func),
                                   nullptr, nullptr);
    if (fail) {
        ll_func_dispose(rlfn);
        return nullptr;
    }

    llvm::Value* fn_val = llvm::unwrap(ll_func_lift(rlfn));
    ll_func_dispose(rlfn);

    if (!fn_val)
        return nullptr;

    llvm::Function* fn = llvm::cast<llvm::Function>(fn_val);
    fn->setLinkage(llvm::GlobalValue::PrivateLinkage);

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
        llvm::CallSite cs(use.getUser());
        assert(cs && cs.isCallee(&use) && "strange reference to tail_fn");
        llvm::CallInst* call = llvm::cast<llvm::CallInst>(cs.getInstruction());
        tmp_insts.push_back(std::make_pair(call, false));
    }
    for (const llvm::Use& use : rl_func_call->uses()) {
        llvm::CallSite cs(use.getUser());
        assert(cs && cs.isCallee(&use) && "strange reference to call_fn");
        llvm::CallInst* call = llvm::cast<llvm::CallInst>(cs.getInstruction());
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
    assert(rl_func_tail->user_empty() && "rl_func_tail has uses (after)");
    assert(rl_func_call->user_empty() && "rl_func_call has uses (after)");

    return fn;
}

llvm::Function* Optimizer::Wrap(llvm::Function* orig_fn) {
    llvm::FunctionType* fnty = dbll_map_function_type(cfg);
    if (fnty == nullptr) // if we don't support the type
        return nullptr;

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
            return nullptr;
        }
    }

    std::size_t stack_frame_size = 4096;
    std::size_t stack_size = stack_frame_size + stackOffset;

    llvm::Value* stack_sz_val = irb.getInt64(stack_size);
    llvm::AllocaInst* stack = irb.CreateAlloca(irb.getInt8Ty(), stack_sz_val);
    stack->setAlignment(16);
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
    llvm::InlineFunction(llvm::CallSite(call), ifi);

    return fn;
}

void Optimizer::LowerExternalCallToNative() {
    if (func_helper_ext->user_empty())
        return;

    llvm::Function* native_helper = dbll_create_native_helper(mod);

    llvm::SmallVector<llvm::CallInst*, 8> tmp_insts;
    for (const llvm::Use& use : func_helper_ext->uses()) {
        llvm::CallSite cs(use.getUser());
        assert(cs && cs.isCallee(&use) && "strange reference to helper_ext");
        tmp_insts.push_back(llvm::cast<llvm::CallInst>(cs.getInstruction()));
    }
    for (llvm::CallInst* inst : tmp_insts) {
        llvm::Value* args[2] = {inst->getArgOperand(0), inst->getArgOperand(1)};
        auto* new_inst = llvm::CallInst::Create(native_helper, args);
        llvm::ReplaceInstWithInst(inst, new_inst);

        llvm::InlineFunctionInfo ifi;
        llvm::InlineFunction(llvm::CallSite(new_inst), ifi);
    }

    native_helper->removeFromParent();
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

    if (cfg->log_level >= LogLevel::DEBUG)
        mod->print(llvm::dbgs(), nullptr);

    llvm::Function* wrapped_fn = opt.Wrap(fn);
    if (wrapped_fn == nullptr)
        return nullptr;

    if (cfg->log_level >= LogLevel::DEBUG)
        mod->print(llvm::dbgs(), nullptr);

    opt.LowerExternalCallToNative();

    if (cfg->log_level >= LogLevel::DEBUG)
        mod->print(llvm::dbgs(), nullptr);

    // This should only scream if our code has a bug.
    if (llvm::verifyFunction(*wrapped_fn, &llvm::errs())) {
        wrapped_fn->eraseFromParent();
        return nullptr;
    }

    // dbll_optimize_fast(wrapped_fn);
    dbll_optimize_new_pm(cfg, mod, opt.tm);

    if (cfg->log_level >= LogLevel::INFO)
        mod->print(llvm::dbgs(), nullptr);

    llvm::EngineBuilder builder(std::move(mod_u));
    llvm::ExecutionEngine* engine = builder.create(opt.tm);
    if (!engine)
        return cfg->func; // we could not create the JIT engine

    auto raw_ptr = engine->getFunctionAddress(wrapped_fn->getName());

    return reinterpret_cast<BinoptFunc>(raw_ptr);
}

}

BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    if (BinoptFunc new_fn = Optimizer::OptimizeFromConfig(cfg))
        return new_fn;
    return cfg->func;
}

void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {
    // TODO: implement
}
