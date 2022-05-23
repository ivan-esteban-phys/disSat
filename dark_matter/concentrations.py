import numpy.random as random
from ..relations import Relation
import colossus
cosmo = colossus.cosmology.cosmology.setCosmology('planck18')


"""
For now, this only support concentrations in the 200c definition.
"""


class MassConcentration(Relation):
    
    def __init__(self, scatter=True):
        self.parameters = {}
        self.sample_scatter = scatter
        
    @classmethod
    def central_value(cls, mass, z):
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

    @classmethod
    def central_value(cls, mass, z):
        h0 = cosmo.Hz(0)/100.
        return colossus.halo.concentration.concentration(mass/h0, '200c', z, model='diemer19')
    
    @staticmethod
    def scatter():
        """Lognormal scatter."""
        return 0.16
