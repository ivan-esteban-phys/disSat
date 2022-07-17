import warnings
import numpy.random as random
from ..relations import Relation
import colossus
cosmoWMAP5 = colossus.cosmology.cosmology.setCosmology('WMAP5')
cosmoP13 = colossus.cosmology.cosmology.setCosmology('planck13')
cosmoP18 = colossus.cosmology.cosmology.setCosmology('planck18')


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
        colossus.cosmology.cosmology.setCurrent(cosmoP18)
        h0 = cosmoP18.Hz(0)/100.
        return colossus.halo.concentration.concentration(mass/h0, '200c', z, model='diemer19')
    
    @staticmethod
    def scatter():
        """Lognormal scatter."""
        return 0.16


class Duffy08(MassConcentration):

    name = 'Duffy08'

    @classmethod
    def central_value(cls, mass, z):
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

    @classmethod
    def central_value(cls, mass, z):
        colossus.cosmology.cosmology.setCurrent(cosmoP13)
        h0 = cosmoP13.Hz(0)/100.
        c = colossus.halo.concentration.concentration(mass/h0, '200c', z, model='dutton14')
        colossus.cosmology.cosmology.setCurrent(cosmoP18)  # set back to Planck 18
        return c

    @staticmethod
    def scatter():
        return 0.11  # for relaxed halos...
