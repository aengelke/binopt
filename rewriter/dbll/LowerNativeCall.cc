
#include "LowerNativeCall.h"

#include "Logging.h"

#include "binopt-config.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>


namespace dbll {

static const char* NATIVE_CALL_NAME = "dbll.native_call";

static void LowerNativeTail(llvm::CallInst* call, llvm::GlobalValue* glob) {
    // Create alloca in entry block to avoid RBP-relative addressing due to a
    // variable-size stack frame. RBP is set in the inline-assembly, but LLVM
    // apperently doesn't recognize this and generates the following:
    //
    //        mov    rbp,rcx
    //        ...
    //     => mov    rdi,QWORD PTR [rbp-0x40]
    //        mov    rcx,rdx
    //        mov    r9,rbx
    //        ...
    //        iretq
    //
    // TODO: maybe remove RBP from inline-assembly constraints
    llvm::Function* fn = call->getParent()->getParent();
    llvm::IRBuilder<> irb(fn->getEntryBlock().getFirstNonPHI());
    // TODO: just one alloca per function
    llvm::Value* asm_buf = irb.CreateAlloca(irb.getInt64Ty(), irb.getInt64(8));

    irb.SetInsertPoint(call);

    llvm::Type* i64p = irb.getInt64Ty()->getPointerTo();
    llvm::Value* sptr = irb.CreateBitCast(call->getArgOperand(0), i64p);
    llvm::Value* stored_rip_ptr = call->getArgOperand(1);

    llvm::Value* sptr_rsp = irb.CreateConstGEP1_64(sptr, 5);

    // Buffer area passed to inline asm.
    // Contains: userrip, cs, userrflags, userrsp, ss
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

    llvm::Type* asm_ret_ty = llvm::StructType::get(call->getContext(), ty_list);

    // Store RAX, RCX and RDX in asm_buf
    irb.CreateStore(asm_args[0], irb.CreateConstGEP1_64(asm_buf, 5));
    irb.CreateStore(asm_args[1], irb.CreateConstGEP1_64(asm_buf, 6));
    irb.CreateStore(asm_args[2], irb.CreateConstGEP1_64(asm_buf, 7));
    // RAX = asm_buf, RCX = stored_rip_ptr, RDX = glob
    asm_args[0] = asm_buf;
    asm_args[1] = stored_rip_ptr;
    asm_args[2] = glob;
    ty_list[0] = i64p;
    ty_list[1] = i64p;
    ty_list[2] = i64p;

    asm_args.push_back(glob);
    ty_list.push_back(i64p);

    auto asm_ty = llvm::FunctionType::get(asm_ret_ty, ty_list, false);
    const auto constraints =
        "={ax},={cx},={dx},={bx},={bp},={si},={di},={r8},={r9},={r10},={r11},"
        "={r12},={r13},={r14},={r15},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},"
        "={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},"
        "={xmm13},={xmm14},={xmm15},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,"
        "17,18,19,20,21,22,23,24,25,26,27,28,29,30,i";
    const auto asm_code =
            "mov %rsp, (%rdx);" // store rsp in glob
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
}

static void LowerNativeCall(llvm::CallInst* call, llvm::GlobalValue* glob) {
    // See above for a rationale for having the alloca in the entry block.
    // TODO: maybe remove RBP from inline-assembly constraints
    llvm::Function* fn = call->getParent()->getParent();
    llvm::IRBuilder<> irb(fn->getEntryBlock().getFirstNonPHI());
    // TODO: just one alloca per function
    llvm::Value* asm_buf = irb.CreateAlloca(irb.getInt64Ty(), irb.getInt64(3));

    irb.SetInsertPoint(call);

    llvm::Type* i64p = irb.getInt64Ty()->getPointerTo();
    llvm::Value* sptr = irb.CreateBitCast(call->getArgOperand(0), i64p);

    llvm::Value* sptr_rsp = irb.CreateConstGEP1_64(sptr, 5);
    llvm::Value* userrsp = irb.CreateLoad(irb.getInt64Ty(), sptr_rsp);
    userrsp = irb.CreateAdd(userrsp, irb.getInt64(8)); // we use call ourselves
    llvm::Value* userrsp_ptr = irb.CreateIntToPtr(userrsp, irb.getInt64Ty()->getPointerTo());

    // Store user RIP at right address
    irb.CreateStore(irb.CreateLoad(sptr), irb.CreateConstGEP1_64(userrsp_ptr, -1));

    // Buffer area passed to inline asm.
    // Contains: userrsp, userrdx, userrbx
    irb.CreateStore(userrsp, irb.CreateConstGEP1_64(asm_buf, 0));

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

    llvm::Type* asm_ret_ty = llvm::StructType::get(irb.getContext(), ty_list);

    // Store RBX and RDX in asm_buf
    irb.CreateStore(asm_args[3], irb.CreateConstGEP1_64(asm_buf, 2));
    irb.CreateStore(asm_args[2], irb.CreateConstGEP1_64(asm_buf, 1));
    // RBX = asm_buf, RDX = glob
    asm_args[3] = asm_buf;
    asm_args[2] = glob;
    ty_list[3] = i64p;
    ty_list[2] = i64p;

    asm_args.push_back(glob);
    ty_list.push_back(i64p);

    auto asm_ty = llvm::FunctionType::get(asm_ret_ty, ty_list, false);
    const auto constraints =
        "={ax},={cx},={dx},={bx},={bp},={si},={di},={r8},={r9},={r10},={r11},"
        "={r12},={r13},={r14},={r15},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},"
        "={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},"
        "={xmm13},={xmm14},={xmm15},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,"
        "17,18,19,20,21,22,23,24,25,26,27,28,29,30,i";
    const auto asm_code =
            "mov %rsp, (%rdx);" // store rsp in glob
            "mov 0x0(%rbx), %rsp;" // setup user rsp
            "mov 0x8(%rbx), %rdx;" // setup user rdx
            "mov 0x10(%rbx), %rbx;" // setup user rbx
            "callq *-0x8(%rsp);"
            "movabs $62, %rsp;"
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
}

llvm::PreservedAnalyses LowerNativeCallPass::run(llvm::Module& mod,
                                                 llvm::ModuleAnalysisManager& mam) {
    llvm::Function* native_call_fn = mod.getFunction(NATIVE_CALL_NAME);
    if (!native_call_fn)
        return llvm::PreservedAnalyses::all();

    llvm::SmallVector<llvm::CallInst*, 8> call_insts;
    for (const llvm::Use& use : native_call_fn->uses()) {
        llvm::CallInst* ci = llvm::dyn_cast<llvm::CallInst>(use.getUser());
        if (!ci || ci->getCalledFunction() != native_call_fn)
            continue;
        call_insts.push_back(ci);
    }

    if (call_insts.empty())
        return llvm::PreservedAnalyses::all();

    // Lowering native calls may require some accessible storage space. Since
    // the MCJIT doesn't support thread-local storage, this is not thread-safe.
    llvm::Type* i64 = llvm::Type::getInt64Ty(mod.getContext());
    auto* glob = new llvm::GlobalVariable(mod, i64, false,
                                          llvm::GlobalValue::PrivateLinkage,
                                          llvm::Constant::getNullValue(i64),
                                          "dbll.native_call.glob");

    for (llvm::CallInst* call : call_insts) {
        auto is_call = llvm::dyn_cast<llvm::Constant>(call->getArgOperand(3));
        if (is_call && !is_call->isZeroValue())
            LowerNativeCall(call, glob);
        else
            LowerNativeTail(call, glob);

        call->eraseFromParent();
    }

    llvm::PreservedAnalyses pa;
    return pa;
}

llvm::Function* LowerNativeCallPass::CreateNativeCallFn(llvm::Module& mod) {
    llvm::LLVMContext& ctx = mod.getContext();

    llvm::Type* i1 = llvm::Type::getInt1Ty(ctx);
    llvm::Type* i64 = llvm::Type::getInt64Ty(ctx);
    llvm::Type* i8p = llvm::Type::getInt8PtrTy(ctx);
    llvm::Type* i64p = i64->getPointerTo();
    llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
    // This is void(i8* sptr, i64* retaddr_ptr, i64 rip, i1 is_call)
    auto fn_ty = llvm::FunctionType::get(void_ty, {i8p, i64p, i64, i1}, false);
    auto linkage = llvm::GlobalValue::ExternalLinkage;
    return llvm::Function::Create(fn_ty, linkage, NATIVE_CALL_NAME, &mod);
}

} // namespace dbll
