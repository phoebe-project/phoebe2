import sys

try:
  import numpy
except ImportError:
  print "Numpy is needed for running and building of PHOEBE"
  sys.exit(1)

from numpy.distutils.core import setup, Extension
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.command.build_py import build_py

from distutils.version import LooseVersion, StrictVersion
from distutils.cmd import Command

import platform
import os
import re


#
# Setup for MS Windows
#

if platform.system() == 'Windows':
  os.environ['VS90COMNTOOLS'] = os.environ['VS140COMNTOOLS']

#
# Auxiliary functions
#

def removefile(f):
  try:
    os.remove(f)
  except OSError:
    pass


def find_version_gcc(s):
  return s.split()[-2]


def __find_version_clang(s):
  ver = ''
  sp = s.split()

  for i, w in enumerate(sp):
    if w == 'version':
      if i < len(sp): ver = sp[i+1]
      break
  return ver

def find_version_clang(s):
  if re.search(r'LLVM version', s):
    ver = ("llvm", __find_version_clang(s))
  else:
    ver = ("clang", __find_version_clang(s))
  return ver

def find_version_intel(s):
  return s.split()[-2]

#
# Check the platform and C++ compiler (g++ > 5.0)
#
def check_compiler(compiler, extensions, compiler_name):

  status = False

  plat = platform.system()
  plat_ver = platform.release();

  print("OS: %s" %(plat))
  print("OS version: %s" %(plat_ver))

  if plat == 'Windows':

    # we don't do checks, as we don't know to make them
    status = True

  # this should cover Linux and Mac
  elif plat in ['Linux', 'Darwin']:

    s = os.popen(compiler_name + " --version").readline().strip()

    # debug output
    print("Compiler: %s"%(compiler_name))
    print("Compiler version: %s"%(s))

    compiler_found = False;
    version_ok = False;

    # GCC compiler
    if re.search(r'gcc', compiler_name) or re.search(r'^g\+\+', compiler_name):
      name = 'gcc'
      compiler_found = True
      ver = find_version_gcc(s)
      if ver != '': version_ok = LooseVersion(ver) >= LooseVersion("5.0")

    # LLVm clang compiler
    elif re.search(r'^clang', compiler_name):
      name = 'clang'
      compiler_found = True

      # https://stackoverflow.com/questions/19774778/when-is-it-necessary-to-use-use-the-flag-stdlib-libstdc
      if plat == 'Darwin':
        opt ="-stdlib=libc++"
        if LooseVersion(plat_ver) < LooseVersion("13.0"): #OS X Mavericks
          for e in extensions:
            if not (opt in e.extra_compile_args):
              e.extra_compile_args.append(opt)

      ver = find_version_clang(s)

      if ver != '':
        if ver[0] == 'clang': # CLANG version
          version_ok = LooseVersion(ver[1]) >= LooseVersion("3.3")
        else:                 # LLVM version
          version_ok = LooseVersion(ver[1]) >= LooseVersion("7.0")

    # Intel compilers
    elif re.search(r'^icc', compiler_name) or re.search(r'^icpc', compiler_name):
      name = 'icc'
      compiler_found = True

      ver = find_version_intel(s)
      version_ok = LooseVersion(ver) >= LooseVersion("16")

    # compiler could be masquerading under different name
    # check this out:
    #  ln -s `which gcc` a
    #  CC=`pwd`/a python check_compiler.py

    if not compiler_found:

      import tempfile
      import random
      import string

      tempdir = tempfile.gettempdir();

      pat = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
      src = pat+'_compiler_check.c'
      exe = pat+'_compiler_check.exe'
      obj = pat+'_compiler_check.o'

      with open(tempdir + '/' + src, 'w') as tmp:
        tmp.writelines(
          ['#include <stdio.h>\n',
           'int main(int argc, char *argv[]) {\n',
            '#if defined (__INTEL_COMPILER)\n',
            '  printf("icc %d.%d", __INTEL_COMPILER, __INTEL_COMPILER_UPDATE);\n',
            '#elif defined(__clang__)\n',
            '  printf("clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);\n',
            '#elif defined(__GNUC__)\n',
            '  printf("gcc %d.%d.%d\\n",__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);\n',
            '#else\n',
            '  printf("not_gcc");\n',
            '#endif\n',
            'return 0;\n',
            '}\n'
          ])

      try:
        objects = compiler.compile([tempdir+'/'+ src], output_dir='/')
        compiler.link_executable(objects, exe, output_dir = tempdir)

        out = os.popen(tempdir+'/'+ exe).read()

        if len(out) != 0:
          name, ver = out.split(' ')

          if name == 'gcc':
            version_ok = LooseVersion(ver) >= LooseVersion("5.0")
            compiler_found = True

          if name == 'clang':
            version_ok = LooseVersion(ver) >= LooseVersion("3.3") # not LLVM version !!!
            compiler_found = True

          if name == 'icc':
            version_ok = LooseVersion(ver) >= LooseVersion("1600")
            compiler_found = True
      except:
        print("Unable to build a test program to determine the compiler.")
        status = False

      # Cleanup
      removefile(tempdir+'/'+ src)
      removefile(tempdir+'/'+ exe)
      removefile(tempdir+'/'+ obj)

    if compiler_found:
      if version_ok:
        print("Ready to compile with %s %s." % (name, ver))
        status = True
      else:
        print("Compiler is too old. PHOEBE requires gcc 5.0, clang 3.3, or icc 1600 or above.\nThe found compiler is %s %s." % (name, ver))
        status = False
    else:
      print("Did not recognize the compiler %s." % (compiler_name))
      status = False

  else:
    print("Unknown architecture, so no pre-checks done. Please report in the case of build failure.")
    status = True

  return status

#
# Hooking the building of extentions
#
class build_check(build_ext):
  def build_extensions(self):
    if (
        check_compiler(self.compiler, self.extensions, self.compiler.compiler_cxx[0]) and
        check_compiler(self.compiler, self.extensions, self.compiler.compiler_so[0])
       ):

      for e in self.extensions:
        print("  extra_args=%s"%(e.extra_compile_args))

      build_ext.build_extensions(self)
    else:
      print("Cannot build phoebe2. Please check the dependencies and try again.")
      sys.exit(1)
#
# Setting up the external modules
#

class import_check(Command):
  description = "Checks python modules needed to successfully import phoebe"
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    required, optional = [], []
    try:
      import astropy
      astropy_version = astropy.__version__
      if LooseVersion(astropy_version) < LooseVersion('1.0'):
        required.append('astropy 1.0+')
    except:
      required.append('astropy')
    try:
      import scipy
      scipy_version = scipy.__version__
      if LooseVersion(scipy_version) < LooseVersion('0.1'):
        required.append('scipy 0.1+')
    except:
      required.append('scipy')
    try:
      import matplotlib
      mpl_version = matplotlib.__version__
      if LooseVersion(mpl_version) < LooseVersion('1.4.3'):
        optional.append('matplotlib 1.4.3+')
    except:
      optional.append('matplotlib')
    try:
      import sympy
      sympy_version = sympy.__version__
      if LooseVersion(sympy_version) < StrictVersion('1.0'):
        optional.append('sympy 1.0+')
    except:
      optional.append('sympy')

    if required == []:
      print('All required import dependencies satisfied.')
    else:
      print('NOTE: while all the build dependencies are satisfied, the following import dependencies')
      print('      are still missing: %s.' % required)
      print('      You will not be able to import phoebe before you install those dependencies.')

    if optional == []:
      print('All optional import dependencies satisfied.')
    else:
      print('NOTE: while all the build dependencies are satisfied, the following optional dependencies')
      print('      are still missing: %s.' % optional)
      print('      Some of the core phoebe functionality will be missing until you install those dependencies.')

class PhoebeBuildCommand(build_py):
  def run(self):
    build_py.run(self)
    self.run_command('build_ext')
    self.run_command('check_imports')

ext_modules = [
    Extension('libphoebe',
      sources = ['./phoebe/lib/libphoebe.cpp'],
      language='c++',
      extra_compile_args = ["-std=c++11"],
      include_dirs=[numpy.get_include()]
      ),

    Extension('phoebe.algorithms.ceclipse',
      language='c++',
      sources = ['phoebe/algorithms/ceclipse.cpp'],
      include_dirs=[numpy.get_include()]
      ),
]

#
# Main setup
#
setup (name = 'phoebe',
       version = '2.1.1',
       description = 'PHOEBE 2.1.1',
       author = 'PHOEBE development team',
       author_email = 'phoebe-devel@lists.sourceforge.net',
       url = 'http://github.com/phoebe-project/phoebe2',
       download_url = 'https://github.com/phoebe-project/phoebe2/tarball/2.1.1',
       packages = ['phoebe', 'phoebe.parameters', 'phoebe.frontend', 'phoebe.constraints', 'phoebe.dynamics', 'phoebe.distortions', 'phoebe.algorithms', 'phoebe.atmospheres', 'phoebe.backend', 'phoebe.utils', 'phoebe.dependencies', 'phoebe.dependencies.autofig', 'phoebe.dependencies.nparray', 'phoebe.dependencies.unitsiau2015'],
       install_requires=['numpy>=1.10','scipy>=0.17','astropy>=1.0,<3.0'],
       package_data={'phoebe.atmospheres':['tables/wd/*', 'tables/passbands/*'],
                    },
       ext_modules = ext_modules,
       cmdclass = {
         'build_ext': build_check,
         'check_imports': import_check,
         'build_py': PhoebeBuildCommand
        }
      )
