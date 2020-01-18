dbrew_subproject = subproject('dbrew')
dbrew = dbrew_subproject.get_variable('dbrew')

sources = ['dbrew.c']

binopt_dbrew_lib = shared_library('binopt-dbrew', sources,
                                  dependencies: [binopt, dbrew],
                                  install: true)
