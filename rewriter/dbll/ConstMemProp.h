
#ifndef DBLL_CONST_MEM_PROP
#define DBLL_CONST_MEM_PROP

#include "binopt-config.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace dbll {

class ConstMemPropPass : public llvm::PassInfoMixin<ConstMemPropPass> {
    BinoptCfgRef cfg;
public:
    ConstMemPropPass(BinoptCfgRef cfg) : cfg(cfg) {}

    llvm::PreservedAnalyses run(llvm::Function& fn, llvm::FunctionAnalysisManager& fam);
};

} // namespace dbll

#endif
