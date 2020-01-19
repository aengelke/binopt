
#include "binopt.h"
#include "binopt-config.h"

#include <rellume/rellume.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

/* DBLL based on LLVM+Rellume, using default configuration API */

const char* binopt_driver(void) {
    return "DBLL";
}

struct DbllHandle {
    llvm::LLVMContext ctx;
    llvm::Module mod;
    llvm::ExecutionEngine* jit;

    DbllHandle() : ctx(), mod("binopt", ctx) {
        mod.setTargetTriple(llvm::sys::getProcessTriple());
    }
};

BinoptHandle binopt_init(void) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
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

static llvm::Function* dbll_lift_function(llvm::Module* mod, BinoptCfgRef cfg) {
    LLConfig* rlcfg = ll_config_new();
    ll_config_enable_fast_math(rlcfg, !!(cfg->fast_math & 1));

    LLFunc* rlfn = ll_func_new(llvm::wrap(mod), rlcfg);
    ll_func_decode(rlfn, reinterpret_cast<uintptr_t>(cfg->func));
    llvm::Function* fn = llvm::unwrap<llvm::Function>(ll_func_lift(rlfn));
    ll_func_dispose(rlfn);
    ll_config_free(rlcfg);

    return fn;
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

static llvm::Function* dbll_wrap_function(BinoptCfgRef cfg,
                                          llvm::Function* orig_fn) {
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
    llvm::Value* alloca = irb.CreateAlloca(cpu_type, int{0});

    // Set direction flag to zero
    irb.CreateStore(irb.getFalse(), dbll_gep_helper(irb, alloca, {0, 2, 6}));

    unsigned gp_regs[6] = { 7, 6, 2, 1, 8, 9 };
    unsigned gpRegOffset = 0;
    unsigned fpRegOffset = 0;
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
            if (gpRegOffset >= 6)
                return nullptr;
            target = dbll_gep_helper(irb, alloca, {0, 1, gp_regs[gpRegOffset]});
            llvm::Type* target_ty = target->getType()->getPointerElementType();
            if (arg_type->isPointerTy())
                arg_val = irb.CreatePtrToInt(arg_val, target_ty);
            else // arg_type->isIntegerTy()
                arg_val = irb.CreateZExtOrTrunc(arg_val, target_ty);
            irb.CreateStore(arg_val, target);
            gpRegOffset++;
        } else if (arg_type->isFloatTy() || arg_type->isDoubleTy()) {
            if (fpRegOffset >= 8)
                return nullptr;
            target = dbll_gep_helper(irb, alloca, {0, 4, fpRegOffset});

            llvm::Type* int_type = irb.getIntNTy(arg_type->getPrimitiveSizeInBits());
            llvm::Type* vec_type = target->getType()->getPointerElementType();
            llvm::Value* int_val = irb.CreateBitCast(arg_val, int_type);
            irb.CreateStore(irb.CreateZExt(int_val, vec_type), target);
            fpRegOffset++;
        } else {
            return nullptr;
        }
    }

    std::size_t stack_size = 4096;
    llvm::Value* stack_sz_val = irb.getInt64(stack_size);
    llvm::AllocaInst* stack = irb.CreateAlloca(irb.getInt8Ty(), stack_sz_val);
    stack->setAlignment(16);
    llvm::Value* sp_ptr = irb.CreateGEP(stack, stack_sz_val);
    llvm::Value* sp = irb.CreatePtrToInt(sp_ptr, irb.getInt64Ty());
    irb.CreateStore(sp, dbll_gep_helper(irb, alloca, {0, 1, 4}));

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

static void dbll_optimize_fast(llvm::Function* fn) {
    llvm::legacy::FunctionPassManager pm(fn->getParent());
    pm.doInitialization();

    // replace CPU struct with scalars
    pm.add(llvm::createSROAPass());
    // instrcombine will get rid of lots of bloat from the CPU struct
    pm.add(llvm::createInstructionCombiningPass(false));
    // Simplify CFG, removes some redundant function exits and empty blocks
    pm.add(llvm::createCFGSimplificationPass());
    // Aggressive DCE to remove phi cycles, etc.
    pm.add(llvm::createAggressiveDCEPass());

    pm.run(*fn);
    pm.doFinalization();
}

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

        std::cerr << "folding load: ";
        load->print(llvm::errs());
        std::cerr << " to 0x" << std::hex << addr_val << std::endl;

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
        llvm::Constant* const_val = llvm::ConstantExpr::getBitCast(const_int, target_ty);

        std::cerr << "folded to: ";
        const_val->print(llvm::errs());
        std::cerr << std::endl;

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

static void dbll_optimize_new_pm(BinoptCfgRef cfg, llvm::Function* fn) {
    llvm::PassBuilder pb;

    llvm::LoopAnalysisManager lam(false);
    llvm::FunctionAnalysisManager fam(false);
    llvm::CGSCCAnalysisManager cgam(false);
    llvm::ModuleAnalysisManager mam(false);

    // Register the AA manager
    fam.registerPass([&] { return pb.buildDefaultAAPipeline(); });
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
    auto fpm = pb.buildFunctionSimplificationPipeline(llvm::PassBuilder::O3,
            llvm::PassBuilder::ThinLTOPhase::None, false);

    fpm.run(*fn, fam);
}

BinoptFunc binopt_spec_create(BinoptCfgRef cfg) {
    DbllHandle* handle = reinterpret_cast<DbllHandle*>(cfg->handle);

    llvm::Function* fn = dbll_lift_function(&handle->mod, cfg);
    if (fn == nullptr) // in case something went wrong
        return cfg->func;

    llvm::Function* wrapped_fn = dbll_wrap_function(cfg, fn);
    fn->eraseFromParent(); // delete old function
    fn = nullptr;

    if (wrapped_fn == nullptr)
        return nullptr;

    // This should only scream if our code has a bug.
    if (llvm::verifyFunction(*wrapped_fn, &llvm::errs())) {
        wrapped_fn->eraseFromParent();
        return nullptr;
    }

    // dbll_optimize_fast(wrapped_fn);
    dbll_optimize_new_pm(cfg, wrapped_fn);

    wrapped_fn->print(llvm::dbgs());


    std::unique_ptr<llvm::Module> mod_ptr(&handle->mod);

    llvm::TargetOptions options;
    options.EnableFastISel = false;

    std::string error;
    llvm::EngineBuilder builder(std::move(mod_ptr));
    builder.setEngineKind(llvm::EngineKind::JIT);
    builder.setErrorStr(&error);
    builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
    builder.setTargetOptions(options);

    // Same as "-mcpu=native", but disable AVX for the moment.
    llvm::SmallVector<std::string, 1> MAttrs;
    MAttrs.push_back(std::string("-avx"));
    llvm::Triple triple = llvm::Triple(llvm::sys::getProcessTriple());
    llvm::TargetMachine* target = builder.selectTarget(triple, "x86-64", llvm::sys::getHostCPUName(), MAttrs);

    llvm::ExecutionEngine* engine = builder.create(target);
    if (!engine)
        return cfg->func; // we could not create the JIT engine

    auto raw_ptr = engine->getFunctionAddress(wrapped_fn->getName());

    return reinterpret_cast<BinoptFunc>(raw_ptr);
}

void binopt_spec_delete(BinoptHandle handle, BinoptFunc spec_func) {
    // TODO: implement
}
