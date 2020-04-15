
#include "PtrToIntFold.h"

#include "Logging.h"

#include "binopt-config.h"

#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Format.h>
#include <llvm/Transforms/Utils/Local.h>


namespace dbll {

class PtrToIntFold {
    llvm::Function& fn;
public:
    PtrToIntFold(llvm::Function& fn) : fn(fn) {}

private:
    bool PropagatePTI(llvm::PtrToIntInst* pti, llvm::SmallVectorImpl<llvm::PtrToIntInst*>* queue) {
        bool changed = false;
        llvm::SmallVector<llvm::Instruction*, 16> dead_queue;
        for (llvm::User* user : pti->users()) {
            if (auto* ipt = llvm::dyn_cast<llvm::IntToPtrInst>(user)) {
                if (ipt->getAddressSpace() != pti->getPointerAddressSpace())
                    continue;

                llvm::Value* new_val = pti->getPointerOperand();
                if (ipt->getDestTy() != new_val->getType()) {
                    llvm::IRBuilder<> irb(ipt);
                    new_val = irb.CreateBitCast(new_val, ipt->getDestTy());
                }
                ipt->replaceAllUsesWith(new_val);
                dead_queue.push_back(ipt);

                changed = true;
            } else if (auto* binop = llvm::dyn_cast<llvm::BinaryOperator>(user)) {
                if (binop->getOpcode() != llvm::Instruction::Add)
                    continue;

                llvm::IRBuilder<> irb(binop);
                llvm::Value* ptr = pti->getPointerOperand();
                ptr = irb.CreateBitCast(ptr, irb.getInt8PtrTy());

                llvm::Value* other_op;
                if (binop->getOperand(0) == pti)
                    other_op = binop->getOperand(1);
                else if (binop->getOperand(1) == pti)
                    other_op = binop->getOperand(0);
                else
                    assert(false && "binop with invalid user");

                llvm::Value* gep = irb.CreateGEP(ptr, other_op, "ptigep");
                llvm::Value* new_pti = irb.CreatePtrToInt(gep, binop->getType());

                queue->push_back(llvm::cast<llvm::PtrToIntInst>(new_pti));
                binop->replaceAllUsesWith(new_pti);
                dead_queue.push_back(binop);

                changed = true;
            }
        }

        for (llvm::Instruction* inst : dead_queue)
            inst->eraseFromParent();
        if (llvm::isInstructionTriviallyDead(pti, nullptr))
            pti->eraseFromParent();

        return changed;
    }

public:
    bool run() {
        // strategy similar to llvm::ConstantPropagation::runOnFunction.
        llvm::SmallVector<llvm::PtrToIntInst*, 16> queue;
        for (llvm::Instruction& inst : llvm::instructions(fn)) {
            if (auto* pti = llvm::dyn_cast<llvm::PtrToIntInst>(&inst)) {
                queue.push_back(pti);
            }
        }

        bool changed = false;
        while (!queue.empty()) {
            llvm::SmallVector<llvm::PtrToIntInst*, 16> new_queue;
            for (llvm::PtrToIntInst* pti : queue)
                changed |= PropagatePTI(pti, &new_queue);

            queue = std::move(new_queue);
        }

        return changed;
    }
};

llvm::PreservedAnalyses PtrToIntFoldPass::run(llvm::Function& fn,
                                              llvm::FunctionAnalysisManager& fam) {
    PtrToIntFold ptif(fn);

    if (!ptif.run())
        return llvm::PreservedAnalyses::all();

    llvm::PreservedAnalyses pa;
    pa.preserveSet<llvm::CFGAnalyses>();
    return pa;
}

} // namespace dbll
