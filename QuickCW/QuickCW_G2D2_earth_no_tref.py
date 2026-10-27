"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""

from time import perf_counter
import json
import pickle
import numpy as np
from numba import config

config.THREADING_LAYER = 'omp'
print("Number of cores used for parallel running: ", config.NUMBA_NUM_THREADS)

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils, signal_base, selections, white_signals, gp_signals
from enterprise_extensions import deterministic

import QuickCW.const_mcmc as cm
from QuickCW.QuickMCMCUtils import MCMCChain, ChainParams
from QuickCW.PulsarDistPriors import DMDistParameter, PXDistParameter
import inspect

# ======== SET CORRECT REFERENCE EPOCH (tref) ========= #
#cm.tref = int(round(55443.93364609394 * 86400))  # seconds
#print("QuickCW tref set to", cm.tref, "seconds (MJD =", cm.tref / 86400.0, ")")
# ===================================================== #


def QuickCW(chain_params, psrs, noise_json=None, use_legacy_equad=True,
            include_ecorr=False, amplitude_prior='detection',
            gwb_gamma_prior=None, psr_distance_file=None, backend_selection=False):
    """Set up all essential objects for QuickCW (Earth-term only)."""
    print("Began Main Loop")

    ti = perf_counter()

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    if backend_selection:
        selection = selections.Selection(selections.by_backend)
    else:
        selection = selections.Selection(selections.no_selection)

    if use_legacy_equad:
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
    else:
        efq = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)

    if include_ecorr:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    log10_Agw = parameter.Uniform(-20, -11)('gwb_log10_A')
    if gwb_gamma_prior is None:
        gwb_gamma_prior = np.array([0, 7])
    gamma_gw = parameter.Uniform(gwb_gamma_prior[0], gwb_gamma_prior[1])('gwb_gamma')
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=chain_params.gwb_comps, Tspan=Tspan, name='gw')
    tm = gp_signals.TimingModel()

    if include_ecorr:
        s_base = ef + eq + ec + rn + crn + tm if use_legacy_equad else efq + ec + rn + crn + tm
    else:
        s_base = ef + eq + rn + crn + tm if use_legacy_equad else efq + rn + crn + tm

    cos_gwtheta = parameter.Uniform(chain_params.cos_gwtheta_bounds[0],
                                    chain_params.cos_gwtheta_bounds[1])('0_cos_gwtheta')
    gwphi = parameter.Uniform(chain_params.gwphi_bounds[0],
                              chain_params.gwphi_bounds[1])('0_gwphi')

    if np.isnan(chain_params.freq_bounds[0]):
        chain_params.freq_bounds[0] = 1 / Tspan
    log10_fgw = parameter.Uniform(np.log10(chain_params.freq_bounds[0]),
                                  np.log10(chain_params.freq_bounds[1]))('0_log10_fgw')

    m_min, m_max = 7.03, 11
    phase0 = parameter.Uniform(0, 2*np.pi)('0_phase0')
    psi = parameter.Uniform(0, np.pi)('0_psi')
    cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')
    p_phase = parameter.Uniform(0, 2*np.pi)
    log10_h = parameter.Uniform(-18, -11)('0_log10_h')

    if amplitude_prior == 'detection':
        log10_mc = parameter.Uniform(m_min, m_max)('0_log10_mc')
    elif amplitude_prior == 'UL':
        log10_mc = parameter.LinearExp(m_min, m_max)('0_log10_mc')
    else:
        raise NotImplementedError("amplitude_prior must be 'detection' or 'UL'")

    if psr_distance_file is None:
        if np.any(np.array([psr.pdist[0] for psr in psrs]) == 0):
            raise ValueError("Some pulsar distances are zero — provide psr_distance_file or nonzero pdist.")
        p_dist = parameter.Normal(0, 1)

        # EARTH TERM ONLY
        cw_wf = deterministic.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi,
                                       log10_mc=log10_mc, log10_h=log10_h,
                                       log10_fgw=log10_fgw, phase0=phase0,
                                       psrTerm=False,  # <- changed
                                       p_phase=p_phase, p_dist=p_dist,
                                       evolve=True, psi=psi, cos_inc=cos_inc,
                                       tref=cm.tref)

        cw = deterministic.CWSignal(cw_wf, psrTerm=False, name='cw0')  # <- changed
        s = s_base + cw
        models = [s(psr) for psr in psrs]

    else:
        with open(psr_distance_file, 'rb') as fp:
            pulsar_distances = pickle.load(fp)
        models = []
        for psr in psrs:
            cw_delay_args = dict(cos_gwtheta=cos_gwtheta, gwphi=gwphi,
                                 log10_mc=log10_mc, log10_h=log10_h,
                                 log10_fgw=log10_fgw, phase0=phase0,
                                 psrTerm=False,  # <- changed
                                 p_phase=p_phase, evolve=True,
                                 psi=psi, cos_inc=cos_inc, tref=cm.tref)
            CWSignal_args = dict(psrTerm=False, name='cw0')  # <- changed
            cw = per_pulsar_prior(psr, pulsar_distances, cw_delay_args, CWSignal_args)
            s = s_base + cw
            models.append(s(psr))

    t1 = perf_counter()
    print("Begin Loading PTA from Enterprise at %8.3fs" % (t1 - ti))
    pta = signal_base.PTA(models)
    t1 = perf_counter()
    print("Finished Loading PTA from Enterprise at %8.3fs" % (t1 - ti))

    with open(noise_json, 'r') as fp:
        noisedict = json.load(fp)
    pta.set_default_params(noisedict)
    if chain_params.verbosity > 0:
        print("Model parameters and priors:")
        print(pta.params)

    max_toa = np.max([p.toas.max() for p in psrs])
    mcc = MCMCChain(chain_params, psrs, pta, max_toa, noisedict, ti)
    return pta, mcc


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def per_pulsar_prior(enterprise_pulsar: Pulsar, pulsar_distances: dict,
                     cw_delay_args: dict = None, CWSignal_args: dict = None):
    if cw_delay_args is None:
        cw_delay_args = get_default_args(deterministic.cw_delay)
    if CWSignal_args is None:
        CWSignal_args = get_default_args(deterministic.CWSignal)

    if 'DM' in pulsar_distances[enterprise_pulsar.name]:
        p_dist = DMDistParameter(*pulsar_distances[enterprise_pulsar.name])
    elif 'PX' in pulsar_distances[enterprise_pulsar.name]:
        p_dist = PXDistParameter(*pulsar_distances[enterprise_pulsar.name])

    cw_wf = deterministic.cw_delay(p_dist=p_dist, **cw_delay_args)
    cw = deterministic.CWSignal(cw_wf, **CWSignal_args)
    return cw
