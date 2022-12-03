from setuptools import setup

name = "hypertiling"
package_name = name
version = '1.1.3'

try:
    with open('README.md', 'r') as f:
        long_desc = f.read()
except:
    logger.warning('Could not open README.md.  long_description will be set to None.')
    long_desc = None

setup(
    name=package_name,
    version=version,
    description='A high-performance Python 3 library for the generation and visualization of hyperbolic tilings',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    author='Manuel Schrauth, Felix Dusel, Yanick Thurn, Florian Goth, Dietmar Herdt, Jefferson S. E. Portela',
    author_email='manuel.schrauth@uni-wuerzburg.de',
    url='http://www.hypertiling.de/',
    license='MIT',
    keywords='hyperbolic tessellation tiling curvature poincare geometry',
    packages=[package_name,
              f"{package_name}.generative",
              f"{package_name}.static",
              f"{package_name}.experimental",
              f"{package_name}.graphics"],
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires=['numpy'],
)
