
tests = [
  # test binary; expected results for [Default, DBrew, DBLL, Drob]
  # 0 means no optimization, 1 means optimized with values propagated
  [executable('test_simple', 'simple.c', dependencies: [binopt]), 0, 1, 1, 1],
  [executable('test_manyparams', 'manyparams.c', dependencies: [binopt]), 0, 0, 1, 0],
  [executable('test_memory', 'memory.c', dependencies: [binopt]), 0, 1, 1, 1],
  [executable('test_constmem', 'constmem.c', dependencies: [binopt]), 0, 0, 1, 1],
  [executable('test_writablemem', 'writablemem.c', dependencies: [binopt]), 0, 1, 1, 1],
  [executable('test_alignment', 'alignment.c', dependencies: [binopt]), 0, 1, 1, 1],
  [executable('test_call', 'call.c', dependencies: [binopt]), 0, 1, 1, 0],
  [executable('test_indjmp-static', 'indjmp-static.c', dependencies: [binopt]), 0, 1, 1, 0],
  [executable('test_indjmp-dynamic', 'indjmp-dynamic.c', dependencies: [binopt]), 0, 1, 1, 0],
]

rewriters = [[binopt_lib, 'Default (no rewriting)', 1]]

if get_option('build_dbrew')
  rewriters += [[binopt_dbrew_lib, 'DBrew', 2]]
endif
if get_option('build_dbll')
  rewriters += [[binopt_dbll_lib, 'DBLL', 3]]
endif
if get_option('build_drob')
  rewriters += [[binopt_drob_lib, 'Drob', 4]]
endif

foreach case : tests
  foreach rewriter : rewriters
    env = environment()
    env.set('LD_PRELOAD', rewriter[0].full_path())
    name = '@0@ @1@'.format(case[0].full_path().split('/')[-1], rewriter[1])
    test(name, case[0], args: [rewriter[1], '@0@'.format(case[rewriter[2]])], env: env, depends: [rewriter[0]])
  endforeach
endforeach
