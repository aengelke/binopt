libllvm = dependency('llvm', version: ['>=8', '<=9'])
cpp_args = ['-DDBLL_LLVM_MAJOR='+libllvm.version().split('.')[0]]

rellume = subproject('rellume')
librellume = rellume.get_variable('librellume')

sources = ['dbll.cc']

binopt_dbll_lib = shared_library('binopt-dbll', sources,
                                 dependencies: [binopt, libllvm, librellume],
                                 install: true)
