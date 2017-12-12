from distutils.core import setup, Extension
import platform
import os

#
# Auxiliary functions 
#

def removefile(f):
  try:
    os.remove(f)
  except OSError:
    pass

#
# Check the platform and C++ compiler (g++ > 5.0)
#

status = False

if platform.system() == 'Windows':
  os.environ['VS90COMNTOOLS'] = os.environ['VS140COMNTOOLS']
  status = True
  
# this should cover Linux and Mac
if platform.system() in ['Linux', 'Darwin']:

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
  version_ok = False;

  # GCC compiler
  if name in ['gcc' , 'g++'] or re.search(r'gcc', name):
    version_ok = LooseVersion(ver) > LooseVersion("5.0")
    compiler_found = True

  # LLVm clang compiler
  if name == 'clang':
    version_ok = LooseVersion(ver) > LooseVersion("3.3")
    compiler_found = True

  # Intel compilers
  if name in ['icc', 'icpc']:
    version_ok = LooseVersion(ver) > LooseVersion("16.0.0")
    compiler_found = True

  # GCC could be masquerading under different name
  # check this out: 
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
      version_ok = LooseVersion(ver) > LooseVersion("5.0")
      compiler_found = True
    
    removefile(tempdir+'/'+ src)
    removefile(tempdir+'/'+ exe)
    removefile(tempdir+'/'+ obj)
    
  if compiler_found:  
    if version_ok:
      print("Ready to compile.")
      status = True
    else:
      print("Compiler is too old.")
      status = False
  else:
    print("Did not recognize compiler %s" % (name_python))
    status = False


if not status:
  print("Quitting setup.py")
  quit()

#
# Setting up the external modules
#

ext_modules = [
    Extension('libphoebe',
      sources = ['./phoebe/lib/libphoebe.cpp'],
      extra_compile_args = ["-std=c++11"]),

    Extension('phoebe.algorithms.ceclipse',
              sources = ['phoebe/algorithms/ceclipse.cpp']),
]

#
# Main setup
#
setup (name = 'phoebe',
       version = 'devel',
       description = 'PHOEBE devel',
       author = 'PHOEBE development team',
       author_email = 'phoebe-devel@lists.sourceforge.net',
       url = 'http://github.com/phoebe-project/phoebe2',
       download_url = 'https://github.com/phoebe-project/phoebe2/tarball/2.0.3',
       packages = ['phoebe', 'phoebe.constants', 'phoebe.parameters', 'phoebe.frontend', 'phoebe.constraints', 'phoebe.dynamics', 'phoebe.distortions', 'phoebe.algorithms', 'phoebe.atmospheres', 'phoebe.backend', 'phoebe.utils'],
       install_requires=['numpy>=1.10','scipy>=0.18','astropy>=1.0'],
       package_data={'phoebe.atmospheres':['tables/wd/*', 'tables/passbands/*'],
                    },
       ext_modules = ext_modules)
