from setuptools import setup

with open("README.rst", 'r') as f:
    long_description = f.read()
    
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='ProcessMCRaT',
   version='0.2.0',
   description='The ProcessMCRaT library is a collection of scripts that can be used to process the output of the MCRaT code.',
   license="MIT",
   long_description=long_description,
   author='Tyler Parsotan',
   author_email='parsotat@oregonstate.edu',
   url="https://github.com/parsotat/ProcessMCRaT",
   classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
   ],
   keywords='astronomy radiation-transfer hydrodynamics',
   packages=['processmcrat','processmcrat.test'],  #same as name
   #install_requires=['numpy', 'scipy', 'matplotlib', 'h5py>=3.2.1', 'tables', 'astropy'], #external packages as dependencies
   install_requires=required,
   python_requires='>=3.3',
)
