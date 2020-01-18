dbrew_subproject = subproject('dbrew')
dbreworiginllvm = dbrew_subproject.get_variable('dbreworiginllvm')

sources = ['dbrew-llvm.c']

binopt_dbrew_llvm_lib = shared_library('binopt-dbrew-llvm', sources,
                                  dependencies: [binopt, dbreworiginllvm],
                                  install: true)
