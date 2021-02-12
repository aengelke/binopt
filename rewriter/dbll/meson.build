libllvm = dependency('llvm', version: ['>=8', '<12'])
cpp_args = ['-DDBLL_LLVM_MAJOR='+libllvm.version().split('.')[0]]

rellume = subproject('rellume', default_options: ['with_rv64=false'])
librellume = rellume.get_variable('librellume')

sources = ['dbll.cc', 'ConstMemProp.cc', 'LowerNativeCall.cc', 'PtrToIntFold.cc']

binopt_dbll_lib = shared_library('binopt-dbll', sources,
                                 dependencies: [binopt, libllvm, librellume],
                                 cpp_args: cpp_args,
                                 install: true)
