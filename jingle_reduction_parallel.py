import os, sys
import numpy as np
lib_path = os.path.abspath('/export/zupcx7/gaccurso/JCMT/JINGLE_MCMC')
sys.path.append(lib_path)
import Jingle_mcmc_parallel

x = np.array([100, 160, 250, 350, 450, 500, 850])*1E-06
y = np.array([0.360, 0.549, 0.3075930, 0.1441790, 0.02848, 0.0665810, 0.01008])
y_error = np.array([0.072,0.1098,0.0615186,  0.0288358, 0.008544, 0.0133162, 0.003024])
distance = (1.10394E+08)*(3.086e+16)
name = 'J130713.20+280249.0_all'
nwalkers = 150
ndim = 3
cores = 8
nburn= 5000
nsteps = 12000

#fitting = Jingle_mcmc_parallel.JINGLE_mcmc(x, y, y_error, distance, nwalkers, ndim, cores, nburn, nsteps, name)

#fitting.perform_mbb_fit()

fitting = Jingle_mcmc_parallel.JINGLE_mcmc(x, y, y_error, distance, nwalkers, ndim, cores, nburn, nsteps, name)

new = fitting.perform_mbb_fit()
print(new)
# In[ ]:



