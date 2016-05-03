# JINGLE_SCUBA_MCMC

This respository contains Python scripts which enable a user to perform a single MBB fit to Herschel+SCUBA photometric measurments using a Bayesian Inference MCMC method.

We have uploaded the script 'Jingle_mcmc_parallel.py' which contains the functions, objects and classes needed to perform Bayesian MCMC MBB fitting. This file should not be changed.

An example of the codes' capabilities can be found in 'Jingle_reduction_parallel.py' where MBB fitting was performed on object J130713.20+280249.0. The code outputs an array containing the dust mass, temperature and best fitting beta value. It also outputs triangular posterior and BB SEDs; examples are given in 'sed_fit_J130713.20+280249.0_all.pdf' and 'triangle_three_dims_J130713.20+280249.0_all.png'

For further enquiries please contact gioacchino.accurso.13@ucl.ac.uk
