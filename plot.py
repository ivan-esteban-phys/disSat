import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 17})
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams.update({'text.usetex': True})
import matplotlib.pyplot as plt
from .observations.vcorrect import *


def plot_observed_velocity_function(label=False, axes=None):
    """
    Plots the observed velocity function (no completeness corrections).
    """
    if axes==None: axes = plt.gca()
    obssigmas = np.array([ dwarf['sigma'] for dwarf_name,dwarf in dwarfs.items() if dwarf['sigma'] != None ])
    obssigmas.sort()
    obsvfxn = np.array([ sum(obssigmas>=sigma) for sigma in obssigmas ])
    axes.plot(obssigmas,obsvfxn,color='0.5',label='observed' if label==True else '') # 'k' if plotband else 'C5' # obs

    
def plot_corrected_velocity_function(label=False, uncertainty='fiducial', color='k', axes=None):
    """
    Completeness corrects the MW satellite velocity function for dwarfs
    discovered through SDSS, and plots both the raw observed and
    corrected velocity function.

    label = whether or not to add legend items for observed VFs.

    uncertainty = 'fiducial' to include just anisotropy and measurement errors
                  'sigdiv2'  to divide all uncertainties by a factor of 2
                  'BooIIAM'  to use the McConnachie+ 2012 values/uncertainties for Boo II
                  'strip'    to use 2*sig for unobserved analogs of likely stripped dwarfs.

    color = 'k' used for fiducial uncertainties and 'C5' for other uncertainties
        in Kim & Peter 2022.
    """

    if axes==None: axes = plt.gca()
    
    if label==True:
        if   uncertainty=='fiducial': the_label = 'corrected'
        elif uncertainty=='sigdiv2' : the_label = 'uncertainties/2'
        elif uncertainty=='BooIIAM' : the_label = r'$\sigma_*^{\rm BooII}$ mod'
        elif uncertainty=='strip'   : the_label = r'2$\sigma^*_{\rm los}$ stripped'
    else: the_label = ''

    profiles = ['nfw','h14e,gk17']
    sigmas, rdist_names,nboot = vcorrect(profiles,bootstrap=True,obs_uncertainty=uncertainty)
    obssigmas = np.array([ dwarf['sigma'] for dwarf_name,dwarf in dwarfs.items() if dwarf['sigma'] != None ])
    obssigmas.sort()
    vfxn   = np.array([ np.array([median    ([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)]   ) for ip in range(len(profiles)) ]) for sigma in obssigmas ])
    vfxn10 = np.array([ np.array([percentile([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)],10) for ip in range(len(profiles)) ]) for sigma in obssigmas ])
    vfxn90 = np.array([ np.array([percentile([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)],90) for ip in range(len(profiles)) ]) for sigma in obssigmas ])

    classicals = np.array([ dwarf['sigma'] for dwarf_name,dwarf in dwarfs.items() if dwarf['type']=='classical' ])
    cvfxn = np.array([ sum(classicals>=sigma) for sigma in obssigmas ])

    axes.fill_between(obssigmas,vfxn[:,0],vfxn[:,1],color=color,alpha=0.2) # bounded by median prediction for NFW and GK14 disk stripping distributions
    axes.fill_between(obssigmas,vfxn10[:,0],vfxn90[:,1],color=color,alpha=0.2,label=the_label) # 2-sigmas


def plot_theoretical_velocity_fuction(satpops, median_only=False, alpha=1, label='', color='C0', linestyle='-', axes=None):
    """
    Must give an array of SatellitePopulation objects.
    """

    if axes==None: axes = plt.gca()
    
    plotsigs = np.logspace(np.log10(2),np.log10(30))

    # for a single satellite population
    #vfxn = [ sum(satellites.properties['sigLOS']>s) for s in plotsigs ]
    #plt.plot(plotsigs,vfxn)
    
    # for an array of satellite populations
    vfxns = np.array([ satpop.properties['sigLOS'] for satpop in satpops ],dtype=object)
    p2sm = np.array([ np.percentile([sum(sigs>s) for sigs in vfxns], 2.3) for s in plotsigs])
    p1sm = np.array([ np.percentile([sum(sigs>s) for sigs in vfxns],15.9) for s in plotsigs])
    vfxn = np.array([ np.percentile([sum(sigs>s) for sigs in vfxns],50  ) for s in plotsigs])
    p1sp = np.array([ np.percentile([sum(sigs>s) for sigs in vfxns],84.1) for s in plotsigs])
    p2sp = np.array([ np.percentile([sum(sigs>s) for sigs in vfxns],97.7) for s in plotsigs])

    if not median_only:
        axes.fill_between(plotsigs,p2sm,p2sp,alpha=0.1,color=color)
        axes.fill_between(plotsigs,p1sm,p1sp,alpha=0.2,color=color)
    return axes.plot(plotsigs,vfxn,label=label,color=color,linestyle=linestyle,alpha=alpha)
    

def finalize(legend=False,legend2=None, axes=None, xlabel=True):

    if axes==None:
        axes = plt.gca()
        plt.figure(num=plt.gcf().number, figsize=(6.4,4.8))
    
    #plt.text(5,350,'coreNFW > 5e8',fontsize=14) # 15
    #plt.text(7,350,'NFW',fontsize=14)
    ##plt.text(15,200,"('no' scatter)",fontsize=14)
    #plt.text(15,200,'minus rhalf',fontsize=14,fontweight='bold')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlim([2.5,30])
    axes.set_ylim([1,1e3])
    if xlabel: axes.set_xlabel(r'$\sigma^*_{\rm los}$ (km/s)')
    axes.set_ylabel(r'N($>\sigma^*_{\rm los}$)')
    if legend: axes.legend(loc='best',fontsize=12,frameon=False)
    if legend2 != None: axes.add_artist(legend2)
