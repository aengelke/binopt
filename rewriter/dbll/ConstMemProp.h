
#ifndef DBLL_CONST_MEM_PROP
#define DBLL_CONST_MEM_PROP

#include "binopt-config.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace dbll {

class ConstMemPropPass : public llvm::PassInfoMixin<ConstMemPropPass> {
public:
    using MemRange = std::pair<uintptr_t, size_t>;

private:
    llvm::ArrayRef<MemRange> memranges;

public:
    ConstMemPropPass(llvm::ArrayRef<MemRange> mr) : memranges(mr) {}

    llvm::PreservedAnalyses run(llvm::Function& fn, llvm::FunctionAnalysisManager& fam);
};

} // namespace dbll

#endif
