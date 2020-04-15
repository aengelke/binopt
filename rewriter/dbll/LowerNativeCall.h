
#ifndef DBLL_LOWER_NATIVE_CALL
#define DBLL_LOWER_NATIVE_CALL

#include "binopt-config.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

namespace dbll {

class LowerNativeCallPass : public llvm::PassInfoMixin<LowerNativeCallPass> {
public:
    LowerNativeCallPass() {}

    llvm::PreservedAnalyses run(llvm::Module& mod, llvm::ModuleAnalysisManager& mam);

    static llvm::Function* CreateNativeCallFn(llvm::Module& mod);
};

} // namespace dbll

#endif
