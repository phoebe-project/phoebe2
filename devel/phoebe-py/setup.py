from distutils.core import setup, Extension

module1 = Extension('phoebeBackend',
                    sources = ['phoebe_backend.c'],
		    libraries = ['phoebe'])

setup (name = 'PHOEBE backend',
       version = '0.40',
       description = 'PHOEBE python package',
       ext_modules = [module1])

