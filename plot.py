import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 17})
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams.update({'text.usetex': True})
import matplotlib.pyplot as plt
from vcorrect import *


def plot_observed_velocity_function():

    profiles = ['nfw','h14e,gk17']
    sigmas, rdist_names,nboot = vcorrect(profiles,bootstrap=True)
    obssigmas = np.array([ dwarf['sigma'] for dwarf_name,dwarf in dwarfs.items() if dwarf['sigma'] != None ])
    obssigmas.sort()
    vfxn   = np.array([ array([median    ([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)]   ) for ip in range(len(profiles)) ]) for sigma in obssigmas ])
    vfxn10 = np.array([ array([percentile([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)],10) for ip in range(len(profiles)) ]) for sigma in obssigmas ])
    vfxn90 = np.array([ array([percentile([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)],90) for ip in range(len(profiles)) ]) for sigma in obssigmas ])

    obsvfxn   = np.array([ sum(obssigmas>=sigma) for sigma in obssigmas ])
    classicals = np.array([ dwarf['sigma'] for dwarf_name,dwarf in dwarfs.items() if dwarf['type']=='classical' ])
    cvfxn = np.array([ sum(classicals>=sigma) for sigma in obssigmas ])

    plt.fill_between(obssigmas,vfxn[:,0],vfxn[:,1],color='k',alpha=0.2) # bounded by median prediction for NFW and GK14 disk stripping distributions
    plt.fill_between(obssigmas,vfxn10[:,0],vfxn90[:,1],color='k',alpha=0.2) # 2-sigmas
    plt.plot(obssigmas,obsvfxn,color='0.5') # 'k' if plotband else 'C5' # obs


def plot_theoretical_velocity_fuction(satpops, median_only=False, alpha=1, label='', color='C0', linestyle='-'):
    """
    Must give an array of SatellitePopulation objects.
    """

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
        plt.fill_between(plotsigs,p2sm,p2sp,alpha=0.1,color=color)
        plt.fill_between(plotsigs,p1sm,p1sp,alpha=0.2,color=color)
    return plt.plot(plotsigs,vfxn,label=label,color=color,linestyle=linestyle,alpha=alpha)
    

def finalize_and_save_plot(figname='vfxn.pdf',legend=False,legend2=None):

    plt.figure(num=plt.gcf().number, figsize=(6.4,4.8))
    #plt.text(5,350,'coreNFW > 5e8',fontsize=14) # 15
    #plt.text(7,350,'NFW',fontsize=14)
    ##plt.text(15,200,"('no' scatter)",fontsize=14)
    #plt.text(15,200,'minus rhalf',fontsize=14,fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([2.5,30])
    plt.ylim([1,1e3])
    plt.xlabel(r'$\sigma^*_{\rm los}$ (km/s)')
    plt.ylabel(r'N($>\sigma^*_{\rm los}$)')
    if legend: plt.legend(loc='best',fontsize=12,frameon=False)
    if legend2 != None: plt.gca().add_artist(legend2)
    plt.savefig(figname)
    print('wrote',figname)
