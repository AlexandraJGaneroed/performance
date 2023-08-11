from setuptools import setup

setup(
    name='performance_diagram',
    version='0.1.0',    
    description='A library to plot performance diagrams',
    url='https://gitlab.fbk.eu/dsip/dsip_meteo/performance-diagram',
    author='Gabriele Franch',
    author_email='franch@fbk.eu',
    license='MIT License',
    packages=['performance_diagram'],
    install_requires=['matplotlib',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)