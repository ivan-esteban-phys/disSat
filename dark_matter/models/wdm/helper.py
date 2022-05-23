import numpy as np

import colossus
cosmo = colossus.cosmology.cosmology.setCosmology('planck18')


def half_mode_mass(mWDM):
    """Returns the WDM half-mode mass in msun."""
    MU = 1.12 ### need to better integrate this, using value from transfer function
    h0        = cosmo.Hz(0)/100.
    rho_bar0  = cosmo.rho_m(0) / (cosmo.Hz(0)/100.)**2
    OMEGA_WDM = cosmo.Om(0)
    
    alpha_hm  = 49.0 * mWDM**-1.11 * (OMEGA_WDM/0.25)**0.11 * (h0/0.7)**1.22 / h0 # kpc
    lambda_hm = 2*np.pi* alpha_hm * (2**(MU/5.) - 1)**(-1./2/MU)
    return  4*np.pi/3 * rho_bar0 * (lambda_hm/2)**3
