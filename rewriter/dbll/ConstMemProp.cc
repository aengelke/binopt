
#include "ConstMemProp.h"

#include "Logging.h"

#include "binopt-config.h"

#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Format.h>
#include <llvm/Transforms/Utils/Local.h>


namespace dbll {

class ConstMemProp {
    BinoptCfgRef cfg;
public:
    ConstMemProp(BinoptCfgRef cfg) : cfg(cfg) {}

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
            llvm::dbgs() << "folding load ";
            load->print(llvm::dbgs());
            llvm::dbgs() << " from " << llvm::format("%p", addr_val) << " to ";
            const_val->print(llvm::dbgs());
            llvm::dbgs() << "\n";
        }

        return const_val;
    }

public:
    bool run(llvm::Function& fn) {
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

        return changed;
    }
};

llvm::PreservedAnalyses ConstMemPropPass::run(llvm::Function& fn,
                                              llvm::FunctionAnalysisManager& fam) {
  ConstMemProp cmp(cfg);

  if (!cmp.run(fn))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses pa;
  pa.preserveSet<llvm::CFGAnalyses>();
  return pa;
}

} // namespace dbll
