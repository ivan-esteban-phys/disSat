from sys import *
import numpy as np
from numpy import *
import scipy
from numpy.random import normal
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import tidal_stripping
from .. import vutils
from ..genutils import *

G = 4.30092e-6 # kpc km^2 / MSUN / s^2



def rvir(mhalo, massdef='200c', z=0, cNFW_method='d15'):
    """
    Returns the radius of the halo, accounting for tidal stripping if relevant.
    """

    if massdef=='vir':
        raise NotImplementedError('Computing NFW radii with virial halo mass definition not yet supported.')
    else:
        delta = float(massdef[:-1])
        rho = (rhoc(z,method=cNFW_method) * KPC**3/MSUN) if massdef[-1]=='c' else rho_bar(z,method=cNFW_method)
       
    return ( mhalo / (4*pi/3*delta*rho) )**(1./3)

    
def rs(mhalo, chalo=None, massdef='200c', z=0, cNFW_method='d15'):
    """
    Returns the scale radius of the halo, accounting for tidal stripping if relevant.
    """

    if (not hasattr(chalo,'__iter__')) and chalo==None:
        chalo = cNFW(mhalo,z=z,massdef=massdef,method=cNFW_method)
        
    rhalo = rvir(mhalo, massdef=massdef, z=z, cNFW_method=cNFW_method)
    return rhalo/chalo


def rmax(mhalo, massdef='200c', mleft=1.0, density_profile='nfw', slope=1.0,
         chalo=None, cNFW_method='d15', zin=0., tSF=None,
         mstar=None, smhm='m13', reff='r17', Re0=None, nostripRe=False,
         mcore_thres=None, wdm=False,mWDM=5., sigmaSI=None,fudge=1.3,stretch=0.):
    """
    Returns the rmax of the halo, accounting for tidal stripping if relevant.
    Assumes mhalo given in msun, and returns rmax in kpc.
        slope = negative of density profile slope at small radii, i.e. rho propto r^-slope.
            For example, would be 1 for NFW, 0 for core.
    """

    onesub = not hasattr(mleft,'__iter__') and not hasattr(mhalo,'__iter__')

    # not tidally stripped
    if (onesub and mleft==1.) or (hasattr(mleft,'__iter__') and np.all(mleft==np.ones(len(mleft)))):
        rs0 = rs(mhalo, massdef=massdef, z=zin, cNFW_method=cNFW_method)
        if   density_profile=='nfw':   return 2.16258*rs0
        elif density_profile=='cored': return 4.4247 *rs0
        else:  mleft100 = mleft

    # tidally stripped
    else:
        if massdef != '100c':
            mleft100 = tidal_stripping.convert_mleft_to_mleft100(mleft, mhalo, massdef=massdef, chalo0=chalo,
                                                                 density_profile=density_profile, zin=zin,
                                                                 mcore_thres=mcore_thres, wdm=wdm,mWDM=mWDM,
                                                                 sigmaSI=sigmaSI,fudge=fudge,stretch=stretch)
        else:
            mleft100 = mleft
        print('computed mleft100',mleft100)

        rmu,reta,vmu,veta = tidal_stripping.get_stripping_coefficients(slope, mleft100)
        
        if density_profile=='nfw' or density_profile=='cored':
            rs0 = rs(mhalo, massdef=massdef, z=zin, cNFW_method=cNFW_method)
            rmax0 = (2.16258 if density_profile=='nfw' else 4.4247) * rs0
            return 2**rmu * mleft100**reta / (1+mleft100)**rmu * rmax0

    # if not NFW or cored, then solve for vmax
    rhalo = rvir(mhalo, massdef=massdef, z=zin, cNFW_method=cNFW_method)
    r = np.logspace(-1,0.5,num=100)*rhalo
    m = vutils.menc(r, mhalo, density_profile, massdef=massdef, c200=chalo, cNFW_method=cNFW_method,
                    mleft100=mleft100, zin=zin, tSF=tSF,
                    mstar=mstar, smhm=smhm, reff=reff, Re0=Re0, nostripRe=nostripRe,
                    mcore_thres=mcore_thres, wdm=wdm,mWDM=mWDM, sigmaSI=sigmaSI,fudge=fudge,stretch=stretch)
    vcirc = np.sqrt(G*m/r)
    f = scipy.interpolate.CubicSpline(r,vcirc)
    rmax = scipy.optimize.fmin(lambda x: -f(x), r[0]*2, disp=False)
    return rmax


def vmax(mhalo, massdef='200c', mleft=1.0, density_profile='nfw', slope=1.0,
         chalo=None, cNFW_method='d15', zin=0., tSF=None,
         mstar=None, smhm='m13', reff='r17', Re0=None, nostripRe=False,
         mcore_thres=None, wdm=False,mWDM=5., sigmaSI=None,fudge=1.3,stretch=0.):
    """
    Returns the vmax of the halo, accounting for tidal stripping if relevant.
    Assumes mhalo given in msun, and returns rmax in kpc.
        slope = negative of density profile slope at small radii, i.e. rho propto r^-slope.
            For example, would be 1 for NFW, 0 for core.
    """

    onesub = not hasattr(mleft,'__iter__') and not hasattr(mhalo,'__iter__')

    # not tidally stripped
    if (onesub and mleft==1.) or (hasattr(mleft,'__iter__') and np.all(mleft==np.ones(len(mleft)))):
        rs0 = rs(mhalo, massdef=massdef, z=zin, cNFW_method=cNFW_method)
        if   density_profile=='nfw':
            if not hasattr(chalo,'__iter__') and chalo==None:
                chalo = cNFW(mhalo,z=zin,virial=False,massdef=massdef,method=cNFW_method,wdm=wdm,mWDM=mWDM)
            rhos = mhalo / (4*pi*rs0**3*(np.log(1+chalo) - chalo/(1+chalo)))
            return 1.64*rs0*np.sqrt(G*rhos)
        elif density_profile=='cored': NotImplementedError()
        else:  mleft100 = mleft

    # tidally stripped
    else:
        if massdef != '100c':
            mleft100 = tidal_stripping.convert_mleft_to_mleft100(mleft, mhalo, massdef=massdef, chalo0=chalo,
                                                                 density_profile=density_profile, zin=zin,
                                                                 mcore_thres=mcore_thres, wdm=wdm,mWDM=mWDM,
                                                                 sigmaSI=sigmaSI,fudge=fudge,stretch=stretch)
        else:
            mleft100 = mleft
        print('computed mleft100',mleft100)

        rmu,reta,vmu,veta = tidal_stripping.get_stripping_coefficients(slope, mleft100)
        
        if density_profile=='nfw' or density_profile=='cored':
            rs0 = rs(mhalo, massdef=massdef, z=zin, cNFW_method=cNFW_method)
            vmax0 = vmax(mhalo, massdef=massdef, mleft=1.0, density_profile=density_profile, slope=slope,
                         chalo=chalo, cNFW_method=cNFW_method, zin=zin, tSF=tSF,
                         mstar=mstar, smhm=smhm, reff=reff, Re0=Re0, nostripRe=nostripRe,
                         mcore_thres=mcore_thres, wdm=wdm, mWDM=mWDM, sigmaSI=sigmaSI,fudge=fudge,stretch=stretch)
            return 2**vmu * mleft100**veta / (1+mleft100)**vmu * vmax0

    # if not NFW or cored, then solve for vmax
    rhalo = rvir(mhalo, massdef=massdef, z=zin, cNFW_method=cNFW_method)
    r = np.logspace(-1,0.5,num=100)*rhalo
    m = vutils.menc(r, mhalo, density_profile, massdef=massdef, c200=chalo, cNFW_method=cNFW_method,
                    mleft100=mleft100, zin=zin, tSF=tSF,
                    mstar=mstar, smhm=smhm, reff=reff, Re0=Re0, nostripRe=nostripRe,
                    mcore_thres=mcore_thres, wdm=wdm,mWDM=mWDM, sigmaSI=sigmaSI,fudge=fudge,stretch=stretch)
    vcirc = np.sqrt(G*m/r)
    f = scipy.interpolate.CubicSpline(r,vcirc)
    rmax = scipy.optimize.fmin(lambda x: -f(x), r[0]*2, disp=False)
    return f(rmax)

