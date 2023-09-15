from sys import *
import numpy as np
import scipy

from .dark_matter import halos
from .genutils import *  # NFW definitions imported from here



####################################################################################################
# PENARRUBIA+ 2010

def convert_mleft_to_mleft100(mleft, mhalo0, chalo0=None, massdef='200c', density_profile='nfw',
                              cNFW_method='d15', zin=0., mcore_thres=None, 
                              wdm=False,mWDM=5., sigmaSI=None,fudge=1.3,stretch=0):
    
    """
    Convert fraction of total mass left from one mass definition to m100c for use
    with Penarrubia+ 2010 tidal stripping formulae.
       mleft   = fraction of initial mass still bound to halo
       mhalo0  = original mass of halo
       massdef = mass definition of mhalo0, in COLOSSUS format
          ('vir' or 'delta'+'c'/'m' for multiple of critical or matter density)
    """

    if massdef=='100c':  return mleft

    
    # if no mass lost, then set mleft100 = 1 or ones(N)
    if hasattr(mleft,'__iter__') and all(mleft==ones(len(mleft))):
        return ones(len(mleft))
    elif mleft==1:
        return 1.

    
    # else solve for mass left for 100c massdef
    base_profile = 'nfw' if (density_profile=='coreNFW' or density_profile=='sidm') else density_profile
    h0 = h(0,method=cNFW_method)

    if (not hasattr(chalo0,'__iter__')) and chalo0==None:  # compute mleft = mass in rhalo0
        chalo0 = cNFW(mhalo0,z=zin,massdef=massdef,method=cNFW_method, wdm=wdm,mWDM=mWDM)
    rs,rhalo0 = nfw_r(mhalo0,c=chalo0,z=zin,cNFW_method=cNFW_method,massdef=massdef) # rs indepedent of massdef
    if massdef != '200c':  # menc() as written needs halo properties in '200c' units
        m200_times_h, r200_times_h, c200_0 = changeMassDefinition(mhalo0*h0, chalo, zin, massdef, '200c')
        m200_0, r200_0 = m200_times_h / h0, r200_times_h / h0
    else:
        m200_0, r200_0, c200_0 = mhalo0, rhalo0, chalo0

    m100_times_h, r100_times_h, c100 = changeMassDefinition(mhalo0*h0, chalo0, zin, massdef, '100c')
    m100, r100 = m100_times_h / h0, r100_times_h / h0
    print('m100',m100,'r100',r100)
        
    def f(ml100, ml,m200,c200,r200):
        return menc(rhalo, m200, base_profile, c200=c200, mleft100=ml100,
                    cNFW_method=cNFW_method, zin=zin, mcore_thres=mcore_thres,
                    sigmaSI=sigmaSI,fudge=fudge,stretch=stretch, wdm=wdm,mWDM=mWDM)/m200 - ml

    
    if (hasattr(mleft,'__iter__') and len(mleft) > 1) or (hasattr(mhalo0,'__iter__') and len(mhalo0) > 1):

        x0 = mleft if (hasattr(mleft,'__iter__') and len(mleft) > 1) else mleft*ones(len(mhalo0))
        mleft100 = array([ scipy.optimize.root(f,x0=x,args=(ml,m,c,r)).x[0]  \
                           for ml,m,c,r,x in zip(mleft,m200_0,c200_0,rhalo0,x0) ])
        mleft100[mleft100 > 0.9] = 1.
        return mleft100
    
    else:
        
        print(mleft,m200_0,c200_0,rhalo0)
        mleft100 = scipy.optimize.root(f,x0=mleft,args=(mleft,m200_0,c200_0,r200_0)).x[0]
        print(mleft100)
        if mleft100 > 0.9: mleft100 = 1.
        print(mleft100)
        return mleft100


    
gamma = [ 1.5 ,  1.0 ,  0.5 ,  0.   ]
r_mu  = [ 0.00, -0.30, -0.40, -1.30 ]
r_eta = [ 0.48,  0.40,  0.27,  0.05 ]
v_mu  = [ 0.40,  0.40,  0.40,  0.40 ]
v_eta = [ 0.24,  0.30,  0.35,  0.37 ]

def get_stripping_coefficients(slope, mleft100):
    """
    Return's Penarrubia+ 2010's fit to the parameters in the fitting functions they give for
    how the rmax and vmax change with tidal stripping.  Interpolates when necessary.
    """
    rmu  = np.interp(slope, gamma,r_mu )
    reta = np.interp(slope, gamma,r_eta)
    vmu  = np.interp(slope, gamma,v_mu )
    veta = np.interp(slope, gamma,v_eta)
    return rmu,reta, vmu,veta


def mass_enclosed_penarrubia10(renc, m100, slope, mleft100, c100=None, cNFW_method='d15', z=0.):
    """
    Calculates the stripped mass of a halo with a given inner denesity slope in completely m100c units.
    """
    if   slope==1:  density_profile = 'nfw'
    elif slope==0:  density_profile = 'core'
    else:
        print('Warning:  mass_enclosed_penarrubia10() with density profiles != NFW or cored not fully tested!')
        density_profile = None
    
    rmu,reta, vmu,veta = get_stripping_coefficients(slope, mleft100)
    
    rmax0 = halos.rmax(m100, massdef='100c', mleft=1., slope=slope, density_profile=density_profile)
    vmax0 = halos.vmax(m100, massdef='100c', mleft=1., slope=slope, denesity_profile=density_profile) # IMPLEMENT THIS
    
    rmax_new = 2**rmu * mleft100**R_ETA / (1+mleft100)**R_MU * rmax0 # in kpc
    vmax_new = 2**vmu * mleft100**V_ETA / (1+mleft100)**V_MU * vmax0 # in km/s
    rs_new = None # IMPLEMENT THIS

    return rmax_new * vmax_new**2 / 4.30092e-6 * fNFWstrip(renc*KPC/rs_new)/fNFWstrip(rmax_new/rs_new) # IMPLEMENT THIS  # G in units of kpc km^2 / MSUN / s^2    

