
#ifndef DBLL_PTR_TO_INT_FOLD
#define DBLL_PTR_TO_INT_FOLD

#include "binopt-config.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace dbll {

class PtrToIntFoldPass : public llvm::PassInfoMixin<PtrToIntFoldPass> {
public:
    PtrToIntFoldPass() {}

    llvm::PreservedAnalyses run(llvm::Function& fn, llvm::FunctionAnalysisManager& fam);
};

} // namespace dbll

#endif
