import numpy as np
import numpy.random as random
from ..relations import Relation


class GalaxySize(Relation):

    def __init__(self, scatter=True):
        self.parameters = {}
        self.sample_scatter = scatter

    @classmethod
    def central_value(cls, mstar):
        raise NotImplementedError('This is an abstract class.')

    @staticmethod
    def scatter(cls):
        """Lognormal scatter"""
        raise NotImplementedError('This is an abstract class.')

    def __call__(self, mstar):
        median = self.central_value(mstar)
        if self.sample_scatter:
            return median * 10**random.normal(loc=0,scale=self.scatter(),size=len(mstar))
        else:
            return median



############################################################

class Read17(GalaxySize):
    """
    Fit to 2D sizes of isolated dwarfs from Read+ 2017 and McConnachie+ 2012,
    taking out repeats from the latter, and no Leo T.
    """

    name = 'Read17'

    @classmethod
    def central_value(cls, mstar):
        return 10**(0.268*np.log10(mstar)-2.11)

    @staticmethod
    def scatter():
        return 0.234

    
class Danieli18(GalaxySize):
    """
    From Danieli+ 2018, which fit a relation to observed 2D sizes of MW, M31,
    and LG dwarfs, assuming a V-band mass-to-light ratio = 2.0.
    """

    name = 'Danieli18'

    @classmethod
    def central_value(cls, mstar):
        return 10**(0.23*np.log10(mstar)-1.93)

    @staticmethod
    def scatter():
        return 0.29


class Jiang19(GalaxySize):
    """
    Jiang+ 2018's 3D half-light radius to dark matter virial radius
    scaling relation, converted into 2D sizes via Wolf+ 2010's 
    conversion factor.
    """

    name = 'Jiang19'
    
    @classmethod
    def central_value(cls, rvir, c200):
        rhalf3D = 0.02 * (c200/10.)**-0.7 * rvir
        return 0.75*rhalf3D # convert to 2D

    @staticmethod
    def scatter():
        return 0.12
