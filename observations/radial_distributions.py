import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from .. import DISDIR


def load_radial_distribution_function(model,rvir=300.):
    datfn = DISDIR+'/data/radial_distributions/'+model+'.dat'
    rr,mm = np.loadtxt(datfn,unpack=True)
    rr = rr/rr[-1] * rvir
    return interp1d(rr,mm/mm[-1], fill_value='extrapolate')



class RadialDistribution():

    name = 'RadialDistribution'

    def __init__(self, c=9., rvir=300.):  ### HERE: could be better
        self.parameters = {'c': c, 'rvir': rvir}
        self.c = c
        self.rvir = rvir
        self.rs = rvir/c
        self._init_menc()

    def _init_menc(self):
        raise NotImplementedError('This is an abstract class.')
    
    def menc(self):
        raise NotImplementedError('This is an abstract class.')
    
    def __call__(self, r):
        return self.menc(r)


    
############################################################


class NFW(RadialDistribution):

    name = 'NFW'

    def _init_menc(self):
        self.menc = lambda r: (np.log(1.+r/self.rs)-1./(self.rs/r+1.)) / (np.log(1+self.c) - 1./(1./self.c + 1))
        
    
          
class SIS(RadialDistribution):

    name = 'SIS'
    
    def _init_menc(self):
        self.menc = lambda r: r/self.rvir


class Hernquist(RadialDistribution):

    name = 'Hernquist'

    def _init_menc(self):
        self.menc = lambda r: ( (1./self.c+1) / (self.rs/r+1) )**2


class Einasto(RadialDistribution):
        
    name = 'Einasto'

    def __init__(self, c=4.9, rvir=300., alpha=0.24):
        self.parameters = {'c': c, 'rvir': rvir, 'alpha': alpha}
        self.c = c
        self.rvir = rvir
        self.alpha = alpha
        self._init_menc()

    def _init_menc(self):
        r = np.logspace(-5, 0) * self.rvir
        rho = lambda rr: np.exp(-2./self.alpha * (self.c*rr)**self.alpha)
        menc = np.array([ quad(lambda rr: rho(rr)*rr**2, 0,rend)[0] for rend in r/self.rvir ])
        menc /= menc[-1]
        menc = np.concatenate([[0],menc])
        rr = np.concatenate([[0],r])
        self.menc = interp1d(rr,menc) #,fill_value=[0,1],bounds_error=False) ]


class Dooley17(RadialDistribution):

    name = 'Dooley17'

    def _init_menc(self):
        self.menc = load_radial_distribution_function('d17-menc', rvir=self.rvir)


class Hargis14Early(RadialDistribution):

    name = 'Hargis14Early'

    def _init_menc(self):
        self.menc = load_radial_distribution_function('h14e-menc', rvir=self.rvir)
    
        
class Hargis14Massive(RadialDistribution):

    name = 'Hargis14Massive'

    def _init_menc(self):
        self.menc = load_radial_distribution_function('h14m-menc', rvir=self.rvir)


class Hargis14Relics(RadialDistribution):

    name = 'Hargis14Relics'

    def _init_menc(self):
        self.menc = load_radial_distribution_function('h14r-menc', rvir=self.rvir)


# there is data for these, but I haven't implemented them
#class GarrisonKimmel17dm
#class GarrisonKimmel17
#class SIDM
#class Tidal_all
#class Tidal_early
#class Tidal_massive
#class Tidal_relics


############################################################


class RadialDistributionMod():

    name = 'RadialDistributionMod'

    def __init__(self, rvir=300.):
        self.parameters = {'rvir': rvir}
        self.rvir = rvir
        self._init_mod()
    
    def __call__(self, r):
        return self.mod(r)

    def _init_mod(self):
        raise NotImplementedError('This is an abstract class.')


class GarrisonKimmel17(RadialDistributionMod):

    def _init_mod(self):
        self.mod = load_radial_distribution_function('gk17-mod', rvir=self.rvir)


############################################################
# Wrapper for all distribution functions

def radial_distribution(r, model, c=9.0, rvir=300., alpha=0.24):

    """
    Radial distributions are coded as cumulative distribution functions,
    normalized so that at rvir, the distributions = 1.

    Notes on Inputs:

    r = the radii at which to compute the distribution
    
    model = the profile to assume.  Accepted values include:

        'nfw' = Navarro-Frenk-White profile
        'sis' = singular isothermal sphere
        'hernquist' = Hernquist 1990 profile
        'einasto' = Einasto profile
        'classicals' = profile matching the MW classical dwarfs

        The following are based on Hargis+ 2014, where they assumed galaxies
        were hosted by subhalos in the ELVIS dark matter only simulations
        of the Milky Way that

        'h14e' = fell into the MW the earliest
        'h14m' = were the most massive at infall
        'h14r' = were formed before reionization

        Last but not least, you can modify any of these distributions to
        include the effect of tidal destruction of satellites by the Milky 
        Way's disk, which removes satellites close to the center.  This is
        based on work by Garrison-Kimmel+ 2017.  To do this, you can append 

        '+gk17' = Garrison-Kimmel+ 2017 tidal destruction of close-in satellites

        to the end of the distribution name, e..g 'nfw+gk17' to model an
        NFW distribution with GK17-level tidal destruction of satellites.

    c (optional) = the concentration of the halo.  Required by all distributions.
        Assumed to be 9.0 if not supplied, appropriate for the Milky Way for all
        distributions except Einasto.  For Einasto, the Milky Way has c=4.9.

    rvir (optional) = the outermost extent of the halo (agnostic to the mass
        definition used, e..g can represent r200, r100, or even the true virial
        radius.  Required by all distributions.  Assumed to be 300 kpc by default,
        appropriate for the Milky Way.

    alpha (optional) = Einasto profile parameter, required only by the Einastro
        profile.  Assumed to be 0.24 by default, appropriate for the Milky Way.

    """

    if '+' not in model:
        base,mod = model,None
    else:
        base,mod = model.split('+')
    
    if   'nfw'        in model: profile = NFW(c=c, rvir=rvir)
    elif 'sis'        in model: profile = SIS(c=c, rvir=rvir)
    elif 'hernquist'  in model: profile = Hernquist(c=c, rvir=rvir)
    elif 'einasto'    in model: profile = Einasto(c=c, rvir=rvir, alpha=alplha)
    elif 'classicals' in model: profile = Dooley17(c=c, rvir=rvir)
    elif 'h14e'       in model: profile = Hargis14Early(c=c,rvir=rvir)
    elif 'h14m'       in model: profile = Hargis14Massive(c=c,rvir=rvir)
    elif 'h14r'       in model: profile = Hargis14Relics(c=c,rvir=rvir)
    else:
        raise ValueError("No support for radial distribution "+model+'!')
    
    if mod==None:     modification = lambda rr: 1. if np.array(rr).size==1 else np.ones(len(rr)) 
    elif mod=='gk17': modification = GarrisonKimmel17(rvir=rvir)
    
    return profile(np.array(r)) * modification(np.array(r))
        
