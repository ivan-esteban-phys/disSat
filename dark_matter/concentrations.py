import warnings
import numpy.random as random
import numpy as np
from ..relations import Relation
import colossus
cosmoWMAP5 = colossus.cosmology.cosmology.setCosmology('WMAP5')
cosmoP13 = colossus.cosmology.cosmology.setCosmology('planck13')
cosmoP18 = colossus.cosmology.cosmology.setCosmology('planck18')


"""
For now, this only support concentrations in the 200c definition.
"""


class MassConcentration(Relation):

    name = 'MassConcentration'
    
    def __init__(self, scatter=True, z=None, M_min=1e6, M_max=1e12):
        """
          If z is a number, the redshift is fixed at construction time,
        the concentration-mass relation is computed once, and the result
        is interpolated for halo masses between M_min and M_max.
        This can make sampling orders of magnitude faster.
        """
        self.parameters = {}
        self.sample_scatter = scatter
        if z == None:
            self.fixed_z = False
        else:
            self.fixed_z = True
            self.z = z

    def central_value(self, mass, z=None):
        raise NotImplementedError('This is an abstract class.')

    @staticmethod
    def scatter(self):
        """Lognormal scatter."""
        raise NotImplementedError('This is an abstract class.')

    def __call__(self, mass, z):
        median = self.central_value(mass, z)
        if self.sample_scatter:
            return median * 10**random.normal(loc=0,scale=self.scatter(),size=len(mass))
        else:
            return median



############################################################


class Diemer19(MassConcentration):

    name = 'Diemer19'

    def __init__(self, scatter=True, z=None, M_min=1e6, M_max=1e12):
        super().__init__(scatter, z, M_min, M_max)
        if self.fixed_z:
            colossus.cosmology.cosmology.setCurrent(cosmoP18)
            h0 = cosmoP18.Hz(0)/100.
            
            self.mass_interp_list = np.geomspace(M_min, M_max, 600)
            self.conc_interp_list = colossus.halo.concentration.concentration(self.mass_interp_list/h0, '200c', self.z, model='diemer19')

    def central_value(self, mass, z=None):
        if self.fixed_z:
            c = np.interp(np.log10(mass), np.log10(self.mass_interp_list), self.conc_interp_list)            
        else:
            colossus.cosmology.cosmology.setCurrent(cosmoP18)
            h0 = cosmoP18.Hz(0)/100.
            c = colossus.halo.concentration.concentration(mass/h0, '200c', z, model='diemer19')
            
        return c
    
    @staticmethod
    def scatter():
        """Lognormal scatter."""
        return 0.16


class Duffy08(MassConcentration):

    name = 'Duffy08'

    def __init__(self, scatter=True, z=None, M_min=1e6, M_max=1e12):
        super().__init__(scatter, z, M_min, M_max)
        if self.fixed_z:
            colossus.cosmology.cosmology.setCurrent(cosmoWMAP5)
            h0 = cosmoWMAP5.Hz(0)/100.
            
            self.mass_interp_list = np.geomspace(M_min, M_max, 600)
            self.conc_interp_list = colossus.halo.concentration.concentration(self.mass_interp_list/h0, '200c', self.z, model='duffy08')
            colossus.cosmology.cosmology.setCurrent(cosmoP18)  # set back to Planck 18            

    def central_value(self, mass, z=None):
        if self.fixed_z:
            c = np.interp(np.log10(mass), np.log10(self.mass_interp_list), self.conc_interp_list)            
        else:
            colossus.cosmology.cosmology.setCurrent(cosmoWMAP5)
            h0 = cosmoWMAP5.Hz(0)/100.
            c = colossus.halo.concentration.concentration(mass/h0, '200c', z, model='duffy08')
            colossus.cosmology.cosmology.setCurrent(cosmoP18)  # set back to Planck 18            
            
        return c            

    @staticmethod
    def scatter():
        return 0.15  # 0.11 for just relaxed halos



class Dutton14(MassConcentration):

    name = 'Dutton14'

    def __init__(self, scatter=True, z=None, M_min=1e6, M_max=1e12):
        super().__init__(scatter, z, M_min, M_max)
        if self.fixed_z:
            colossus.cosmology.cosmology.setCurrent(cosmoP13)
            h0 = cosmoP13.Hz(0)/100.
            
            self.mass_interp_list = np.geomspace(M_min, M_max, 600)
            self.conc_interp_list = colossus.halo.concentration.concentration(self.mass_interp_list/h0, '200c', self.z, model='dutton14')
            colossus.cosmology.cosmology.setCurrent(cosmoP18)  # set back to Planck 18

    def central_value(self, mass, z=None):
        if self.fixed_z:
            c = np.interp(np.log10(mass), np.log10(self.mass_interp_list), self.conc_interp_list)            
        else:
            colossus.cosmology.cosmology.setCurrent(cosmoP13)
            h0 = cosmoP13.Hz(0)/100.
            c = colossus.halo.concentration.concentration(mass/h0, '200c', z, model='dutton14')
            colossus.cosmology.cosmology.setCurrent(cosmoP18)  # set back to Planck 18
            
        return c

    @staticmethod
    def scatter():
        return 0.11  # for relaxed halos...
