# djlib
A library of functions for manipulating and fitting cluster expansions.

Generally organized as:

djlib/djlib: Functions and classes that are very general purpose and do not have an obvious place in other submodules.

djlib/clex: Functions associated with fitting cluster expansions 

djlib/mc: Functions that are associated with managing grand canonical monte carlo calculations within CASM. 

djlib/plotting: Functions that are used for plotting relevant figures. All plotting functions should be placed here. 

djlib/propagation: Still in its early stage of development. Currently includes all functions associated with propagating a Bayesian ECI posterior through grand canonical monte carlo. 

djlib/casmcalls: Infrequently used; rudimentary python wrapper to command line calls for casm. 

djlib/vasputils: Rudimentary functions used for writing and parsing vasp files. 

djlib/templates: Files for frequently used tasks such as settings files for VASP, CASM monte carlo and slurm job submission. Made to be loaded and templated.

examples/: Collection of example jupyter notebooks that demonstrate standard use cases of djlib code. 


