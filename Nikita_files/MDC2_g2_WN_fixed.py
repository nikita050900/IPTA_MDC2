from __future__ import division

import numpy as np
import glob, sys, os, json, pickle
import matplotlib.pyplot as plt
import scipy.linalg as sl

import libstempo as libs
import libstempo.plot as libsplt

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise.signals import signal_base

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

#NEED TO CHANGE FILE ON DIFFERENT RUNS (ie full_run_1 -> full_run_2)
jeff = False
runnum = '1'
dataset = 'dataset_2b'
group = 'group1'
# if group == 'group1':
#     closed = False
runname = 'fixed_WN_run_' + runnum
# elif group == 'group2':
#     closed = True
#     runname = 'fixed_WN_run_' + runnum
# else:
#     print('Invalid group name')
#     sys.exit(0)
refit = False    # change to true to refit par files. May cause errors.

topdir = os.getcwd()
#Where the original data is
origdatadir = topdir + '/mdc2/' + group + '/' + dataset + '/'
#Which noise directory to use
noisedir = topdir + '/' + dataset + '/noise3'
#Where the json noise file is (use Jeff's file or result of noise run)
if jeff:
    updatednoisefile = topdir + '/mdc2/' + group + '/group1_psr_noise.json'
else:
    updatednoisefile = noisedir + '/fit_psr_noise.json'
#Where the dataset files are located
datadir = topdir + '/' + dataset + '/'
#Where the refit par files are
pardir = datadir + '/newpars/' + dataset[-2:] + '/'
#Where the chains should be saved to
chaindir = datadir + 'chains/'
#Where the everything should be saved to (chains, cornerplts, histograms, etc.)
outdir = datadir + runname + '/'
#Where we save figures n stuff
figdir = datadir + '/Cornerplts/'
#The pickled pulsars
psr_obj_file = topdir + '/' + dataset + '/psr_objects.pickle'

if os.path.exists(datadir) == False:
    os.mkdir(datadir)
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

def Refit_pars(origdir,newdir):
    orig_parfiles = sorted(glob.glob(origdir + '/*.par'))
    orig_timfiles = sorted(glob.glob(origdir + '/*.tim'))
    #Load all of the Pulsars into libstempo
    orig_libs_psrs = []
    for p, t in zip(orig_parfiles, orig_timfiles):
        orig_libs_psr = libs.tempopulsar(p, t)
        orig_libs_psrs.append(orig_libs_psr)

    #Fit the par files again
    #Save them to new directory (Overwrites ones currently used in newdatadir)
    if os.path.exists(newdir) == False:
        os.mkdir(newdir)
    for new_libs_psr in orig_libs_psrs:
        new_libs_psr['DM'].fit = False
        new_libs_psr['DM1'].fit = False
        new_libs_psr['DM2'].fit = False
        try:
            new_libs_psr.fit(iters=3)
        except:
            continue
        new_libs_psr.savepar(newdir + new_libs_psr.name + '.par')

if refit:
    #Refitting par files using libstempo
    Refit_pars(origdatadir,pardir)
    
    #Loading par and tim files into enterprise Pulsar class
    parfiles = sorted(glob.glob(pardir + '/*.par'))
    timfiles = sorted(glob.glob(origdatadir + '/*.tim'))

    #Load all the pulsars if no pickle file
    try:    #Load pulsars from pickle file
        with open(psr_obj_file,'rb') as psrfile:
            psrs = pickle.load(psrfile)
            psrfile.close()
    except:   #If no pickle file, load and save pulsars
        psrs = []
        for p, t in zip(parfiles,timfiles):
            psr = Pulsar(p, t)
            psrs.append(psr)
        #Save 9yr pulsars to a pickle file
        with open(psr_obj_file,'wb') as psrfile:
            pickle.dump(psrs,psrfile)
            psrfile.close()
else:
    parfiles = sorted(glob.glob(origdatadir + '/*.par'))
    timfiles = sorted(glob.glob(origdatadir + '/*.tim'))
    psrs = []
    for p, t in zip(parfiles,timfiles):
        psr = Pulsar(p, t)
        psrs.append(psr)

# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

#Get true noise values for pulsars
params = {}
with open(updatednoisefile, 'r') as nf:
    params_dict = json.load(nf)
    nf.close()
if 'group1_psr_noise.json' in updatednoisefile:
    for psr, psr_params in params_dict.items():
        print(psr_params)
        for param_name, param_val in psr_params.items():
            if param_name == 'efac':
                updated_param_name = psr + '_efac'
            elif param_name == 'equad':
                updated_param_name = psr + '_log10_equad'
            elif param_name == 'ecorr':
                updated_param_name = psr + '_log10_ecorr'
            elif param_name == 'rn_log10_A':
                updated_param_name = psr + '_red_noise_log10_A'
            elif param_name == 'rn_spec_ind':
                updated_param_name = psr + '_red_noise_gamma'
            else:
                updated_param_name = param_name
            params[updated_param_name] = param_val
elif 'fit_psr_noise.json' in updatednoisefile:
    for psr, psr_params in params_dict.items():
        params.update(psr_params)
else:
    print('Unrecognized noise file')
    sys.exit()

##### parameters and priors #####

# white noise parameters
efac = parameter.Constant()
log10_equad = parameter.Constant()

# red noise parameters
red_noise_log10_A = parameter.Uniform(-20,-12)
red_noise_gamma = parameter.Uniform(0,7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.Uniform(-18,-13)('zlog10_A_gw')
gamma_gw = parameter.Constant(13/3)('zgamma_gw')

##### Set up signals #####

# timing model
tm = gp_signals.TimingModel()

# white noise
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad = log10_equad)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=red_noise_log10_A, gamma=red_noise_gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
# Hellings and Downs ORF
orf = utils.hd_orf()

#Common red noise process with no correlations
#crn = gp_signals.FourierBasisGP(spectrum = cpl, components=30, Tspan=Tspan, name = 'gw')

# gwb with Hellings and Downs correlations
gwb = gp_signals.FourierBasisCommonGP(cpl, orf, components=30, name='gw', Tspan=Tspan)

# full model is sum of components
model = ef + eq + rn + tm + gwb

# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])

pta.set_default_params(params)

#make dictionary of pulsar parameters from these runs
param_dict = {}
for psr in pta.pulsars:
    param_dict[psr] = {}
    for param, idx in zip(pta.param_names,range(len(pta.param_names))):
        if param.startswith(psr):
            param_dict[psr][param] = idx
print(pta.param_names)
#Save to json file
with open(outdir + '/Search_params.json','w') as paramfile:
    json.dump(param_dict,paramfile,sort_keys = True,indent = 4)
    paramfile.close()

#Pick random initial sampling
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(pta.param_names)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups
groups  = [range(0, ndim)]
groups.extend(map(list, zip(range(0,ndim,2), range(1,ndim,2))))
groups.extend([[ndim-1]])

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir)
#Use this one if you want to do more runs/ RESUME is on. Be careful about saving different runs
#sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir, resume = True)

# sampler for N steps
N = 2000000
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
