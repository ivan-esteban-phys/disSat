import numpy as np
import numpy.random as random

import colossus
cosmo = colossus.cosmology.cosmology.setCosmology('planck18')

from ....relations import Relation
from ... import concentrations, subhaloMF
from ..skeleton import ModifiedConcentration
from .helper import half_mode_mass
from . import transfer_functions


class WDM:

    name = 'WDM'

    def __init__(self, mWDM=3.):
        self.mWDM = mWDM
        self.parameters = {'mWDM': self.mWDM}
        self.concentration = concentrations.Diemer19(scatter=True)
        self.modified_concentration = Schneider12()
        self.transfer_function = transfer_functions.Lovell20()



############################################################
# WDM concentration
        
class Schneider12(ModifiedConcentration):
    """
    Based on Schneider+ 2012 (parameters to convert cCDM
    to cWDM via their eq. 39).  Assumes masses are in 200c units.
    """

    name = 'Schneider12'
    
    def __call__(self, mass, cCDM, dark_matter_model, z=0.):

        mWDM = dark_matter_model.mWDM

        GAMMA1 = 15.
        GAMMA2 = 0.3
        h0 = cosmo.Hz(z)/100.
        
        # Schneider+ 2012 uses rho_bar in mass def, so must convert
        m1m_div_h, r1m_div_h, c1m = colossus.halo.mass_defs.changeMassDefinition(mass/h0, cCDM, z, '200c', '1m')
        m1m = m1m_div_h * h0

        return cCDM * (1 + GAMMA1*half_mode_mass(mWDM)/m1m)**(-GAMMA2)

    
    
############################################################
# decorators to transform CDM to WDM

def WDMify_MF(cls, mWDM, transfer_func):

    class WDMSubhaloMassFunction:

        def __init__(self, CDM_MF):
            self.wrap = CDM_MF()
        
    return WDMSubhaloMassFunction
        

def WDMify_concentration(cls):

    class WDMConcentration:

        def __init__(self, cCDM):
            self.wrap = cCDM()

    return WDMConcentration
