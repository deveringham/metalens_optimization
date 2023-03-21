# Metalens Optimization
Methods for metamaterial lens shape discovery and optimization, for the design of COPILOT optical instruments.

MSc thesis work of Dylan Everingham as part of the Computer Simulations for Science and Engineering (COSSE) joint programme at the Delft University of Technology and Technical University of Berlin, 2022. In depth-explanation of the problem and solution implementation can be found in the completed thesis here: https://repository.tudelft.nl/islandora/object/uuid%3A3bbed872-bbef-453b-b2e7-ac30d2864e9c

### Overview
This reposity contains a working Python software tool for development of metamaterial lenses proposed for flight on the COPILOT high-altitude astrophysics balloon experiment. Geometric parameters of an optical metasurface are optimized via backpropagation by leveraging a machine learning framework for automatic differentiation and a compatible implementation of RCWA (Rigorous Coupled-Wave Analysis), a semi-analytical solver method for Maxwell's equations in periodic dielectric structures. The tool is optimized to optionally take advantage of GPU execution for performance gains in order to simlulate full-scale problems.

The method as implemented here applies specifically to optimizing optical devices under the design constraints of COPILOT, which are primarily characterized by descretized topology as described in the thesis paper. However the tool is flexible and allows user configuration of many parameters including:

- Optimization objective function
- Wavelength, angle, and polarization of light sources
- Device dimensions and materials

Both an implementation of the tool using TensorFlow as well as one using PyTorch are provided. I began with the TensorFlow implementation, but eventually found certain performance and memory utilization adavantages to using PyTorch, so both are available.

### Structure
Jupyter notebooks containing examples of the tool's usage can be found in the root directory.

Implementation of core methods for the tool can be found in ./src.

Python scripts used to run the tool and produce plots for the thesis paper can be found in ./scripts.

Log files which record previous runs of the tool can be found in ./results.

Differentiable implementation of RCWA using PyTorch can be found in ./rcwa_pt.

The rcwa_tf library, developed by Shane Colburn at University of Washington, used in the TensorFlow implementation, is included and can be found in ./rcwa_tf. The original library can be found at https://github.com/scolburn54/rcwa_tf/.

### Contact

Feel free to contact me any time with questions about the metalens optimization tool at dceveringham(at)gmail.com.