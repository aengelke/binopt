project('binopt', ['c', 'cpp'],
        default_options: ['buildtype=debugoptimized',
                          'warning_level=3',
                          'c_std=c99',
                          'cpp_std=c++17'])

if get_option('warning_level').to_int() >= 3
  add_project_arguments(['-Wmissing-field-initializers',
                         '-Wold-style-definition',
                         '-Wmissing-declarations',
                         '-Wmissing-prototypes',
                         '-Wredundant-decls',
                         '-Wmissing-noreturn',
                         '-Wshadow',
                         '-Wpointer-arith',
                         '-Wwrite-strings',
                         '-Winline',
                         '-Wformat-nonliteral',
                         '-Wformat-security',
                         '-Wswitch-default',
                         '-Winit-self',
                         '-Wnested-externs',
                         '-Wstrict-prototypes',
                         '-Wmissing-include-dirs',
                         '-Wundef',
                         '-Wmissing-format-attribute'
                        ], language: 'c')
endif

add_project_arguments(['-Wno-unused-parameter'], language: 'c')
add_project_arguments(['-Wno-unused-parameter'], language: 'cpp')

binopt_inc = include_directories('include')
subdir('include')
subdir('src')
binopt = declare_dependency(include_directories: binopt_inc,
                            link_with: binopt_lib)

subdir('rewriter')

subdir('examples')
