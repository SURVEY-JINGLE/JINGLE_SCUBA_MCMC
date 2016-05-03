import pickle 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import sys
import emcee
from multiprocessing import Pool
import triangle
import pylab
import os
import itertools
from num2words import num2words
import scipy.constants as const

###########Defining Constants###############
pi = const.pi 
c = const.c
h = const.h 
k = const.k 
#############################################

def modified_black_body_model_with_beta(theta, x, distance):
	#Single MBB fit
	#Theta is an array with the free parameters [Mdust, Temp, beta] units of [kg, Kelvin, unitless]
	#x is the variable length array of wavelengths in meters
	#distance is in meters also
	numerator = 2.0*h*c/x**3
	denominator = np.exp(h*c/(k*theta[1]*x))-1.0
	planck_function = (1E+26)*(numerator/denominator)
	emissivity = 4.5*((100E-06/x)**theta[2])
	flux_density = ((((10**theta[0])*(1.989E+30))*(emissivity)*planck_function)/(distance**2))					
	return flux_density	

def log_pr_with_beta(theta):
	if any(t < 0 for t in theta):
		return -np.inf
	if theta[0] < 4:
		return -np.inf
	if theta[0] > 10:
		return -np.inf
	if theta[1] > 50.0:
		return -np.inf
	if theta[2] > 5.0:
		return -np.inf
	return 0

def log_likelihood_simple_mbb_with_beta(theta, x, y, y_error, distance):
	y_model = modified_black_body_model_with_beta(theta, x, distance)
	return -2.0*np.sum(((y - y_model) ** 2.0)/((2.0*y_error) ** 2.0)) - (len(x)/2.0)*(1.837877) - np.sum(np.log(y_error))

def log_post_one_with_beta(theta, x, y, y_error, distance):
	return log_pr_with_beta(theta) + log_likelihood_simple_mbb_with_beta(theta, x, y, y_error, distance)
	
def bayesian_fitting(x, y, y_error, distance, nwalkers, ndim, cores, nburn, nsteps, galaxy_name):
	np.random.seed(0)
	print('Number of free parameters in our model =', ndim)
	starting_guesses = np.random.rand(nwalkers, ndim)
	starting_guesses[:,0] *= 5
	starting_guesses[:,1] *= 25
	starting_guesses[:,2] *= 2.5
	pool = Pool(cores)         
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_one_with_beta, args=[x, y, y_error, distance], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)     
	figure = triangle.corner(sample, labels=['Log(Dust Mass)', 'Temp', 'Beta'], truths=answers)
	figure.savefig("triangle_three_dims_"+galaxy_name+".png", format='png')
	sampler.pool.terminate()
	plt.clf()
	plt.close()
	return answers

	
class JINGLE_mcmc:

	def __init__(self, x, y, y_error, distance, nwalkers, ndim, cores, nburn, nsteps, galaxy_name):
		self.wavelengths = x
		self.flux = y
		self.fluxerror = y_error
		self.distance = distance
		self.nwalkers = nwalkers
		self.ndim = ndim
		self.cores = cores
		self.nburn = nburn
		self.nsteps = nsteps
		self.samplesize = len(x)
		self.galaxy_id = galaxy_name  	

	def modified_black_body_model_with_beta_for_plot(self, wavelength, distance_to_object, theta):
		numerator = 2.0*h*c/wavelength**3
		denominator = np.exp(h*c/(k*theta[1]*wavelength))-1.0
		planck_function = (1E+26)*(numerator/denominator)
		emissivity = 4.5*((100E-06/wavelength)**theta[2])
		flux_density = ((((10**theta[0])*(1.989E+30))*(emissivity)*planck_function)/(distance_to_object**2))					
		return flux_density	
		
	def perform_mbb_fit(self):
		answers = bayesian_fitting(self.wavelengths, self.flux, self.fluxerror, self.distance, self.nwalkers, self.ndim, self.cores, self.nburn, self.nsteps, self.galaxy_id)	
		plt.scatter((1E+06)*self.wavelengths, self.flux)
		plt.plot(np.logspace(1,3,100), self.modified_black_body_model_with_beta_for_plot(np.logspace(1,3,100)*1E-06, self.distance, answers))
		plt.errorbar((1E+06)*self.wavelengths, self.flux, yerr=self.fluxerror, fmt='o')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim([10, 1000])
		plt.ylim([0.001, 10.0])
		plt.xlabel('Wavelength [$\mu$m]')
		plt.ylabel('F$_{\\nu}$ [Jy]')
		plt.title(self.galaxy_id)
		plt.savefig("sed_fit_"+self.galaxy_id+".pdf")
		plt.close()
		return answers
    

    
