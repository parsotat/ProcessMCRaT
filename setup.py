from setuptools import setup

try:
    with open("README.rst", 'r') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''
    
with open('requirements.txt') as f:
    required = f.read().splitlines()
#update version in init as well
setup(
   name= 'processmcrat', #'ProcessMCRaT',
   version='2.0.1',
   description='The ProcessMCRaT library is a python package that can be used to process the output of the MCRaT code.',
   license="MIT",
   long_description=long_description,
   author='Tyler Parsotan',
   author_email='parsotat@umbc.edu',
   url="https://github.com/parsotat/ProcessMCRaT",
   classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
   ],
   keywords='astronomy radiation-transfer hydrodynamics',
   packages=['processmcrat','processmcrat.test'],  #same as name
   install_requires=required,
   python_requires='>=3.3',
   data_files=[
        ('Data_files',['processmcrat/Data_files/Dataset_lundman_1.csv']),
        ('Data_files',['processmcrat/Data_files/Dataset_lundman_2.csv']),
        ('Data_files',['processmcrat/Data_files/lundman_p_4_thetaj_0.1.csv']),
        ('Data_files',['processmcrat/Data_files/GRB_list.dat']),
        ('Data_files',['processmcrat/Data_files/FERMI_BEST_GRB.dat']),
    ],
    include_package_data=True,
)
