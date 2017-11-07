
def removefile(f):
  try:
    os.remove(f)
  except OSError:
    pass


import platform
  
# this should cover Linux and Mac
if platform.system() in ['Linux', 'Darwin']:

  import os
  import distutils.sysconfig
  import distutils.ccompiler

  compiler = distutils.ccompiler.new_compiler()
  distutils.sysconfig.customize_compiler(compiler)

  #
  # Getting names compiler reported by the compiler
  # should work for clang, icpc, gcc
  #
  
  name_python = compiler.compiler_so[0]

  # --version
  name = os.popen(name_python + " --version").read().split(' ')[0]
  #print "*", name, "*"

  # -dumpversion is present in clang, icpc, gcc
  ver = os.popen(name_python + " -dumpversion").read().strip()
  #print "*", ver, "*"

  from distutils.version import LooseVersion, StrictVersion

  import re

  compiler_found = False;
  ok = False;

  #GCC compiler
  if name in ['gcc' , 'g++'] or re.search(r'gcc', name):
    ok = LooseVersion(ver) > LooseVersion("5.0")
    compiler_found = True

  #LLVm clang compiler
  if name == 'clang':
    ok = LooseVersion(ver) > LooseVersion("3.3")
    compiler_found = True

  #Intel compilers
  if name in ['icc', 'icpc']:
    ok = LooseVersion(ver) > LooseVersion("16.0.0")
    compiler_found = True

  #GCC masquerading under different name
  #test: 
  #  ln -s `which gcc` a
  #  CC=`pwd`/a python check_compiler.py
  
  if not compiler_found:
       
    import tempfile
    tempdir = tempfile.gettempdir();
    
    src = '_gnu_check.c'
    exe = '_gnu_check.exe'
    obj = '_gnu_check.o'
    
    with open(tempdir + '/' + src, 'w') as tmp:    
      tmp.writelines(
        ['#include <stdio.h>\n',
         'int main(int argc, char *argv[]) {\n',
          '#if defined(__GNUC__)\n',
          '  printf("gcc %d.%d.%d\\n",__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);\n',
          '#else\n',
          '  printf("not_gcc");\n',
          '#endif\n',
          'return 0;\n',
          '}\n'
        ])
          
    objects = compiler.compile([tempdir+'/'+ src], output_dir='/')
    compiler.link_executable(objects, exe, output_dir = tempdir)
    
    out = os.popen(tempdir+'/'+ exe).read()
    
    if len(out) != 0:
      name, ver = out.split(' ')
      ok = LooseVersion(ver) > LooseVersion("5.0")
      compiler_found = True
    
    removefile(tempdir+'/'+ src)
    removefile(tempdir+'/'+ exe)
    removefile(tempdir+'/'+ obj)
    
  if compiler_found:  
    if ok:
      print "Ready to go."
    else:
      print "Buddy, use a newer compiler."
  else:
    print "Did not recognize compiler %s" % (name_python)
