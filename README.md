# Metalens Optimization
Methods for metamaterial lens shape discovery and optimization, for the design of COPILOT optical instruments.

### Overview
This repository holds experiments being run to develop and test methods for metamaterial lens design, using TensorFlow/Keras. Geometric parameters of an optical metasurface are optimized via backpropagation by leveraging TensorFlow's framework for automatic differentiation and a compatible implementation of RCWA, a semi-analytical solver method for Maxwell's equations in periodic dielectric structures. This work forms a part of my MSc thesis project being completed at TU Delft as part of the Computer Simulations for Science and Engineering (COSSE) joint programme of TU Delft, TU Berlin, and KTH.

### Structure
Jupyter notebooks containing experiments are found in the root directory. The rcwa_tf library, developed by Shane Colburn at University of Washington, is used and modified by this project and can be found in ./rcwa_tf. The original library can be found at https://github.com/scolburn54/rcwa_tf/.
