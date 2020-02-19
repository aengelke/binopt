drob = subproject('drob')
libdrob = drob.get_variable('drob')

sources = ['drob.c']

binopt_drob_lib = shared_library('binopt-drob', sources,
                                 dependencies: [binopt, libdrob],
                                 install: true)
