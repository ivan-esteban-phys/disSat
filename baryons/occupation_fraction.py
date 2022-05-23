import numpy as np
import numpy.random as random
from ..relations import Relation

DISDIR = '/home/stacykim/research/projects/vfxn/orig/disPy'



class OccupationFraction(Relation):
        
    @classmethod
    def central_value(self, mass):
        raise NotImplementedError('This is an abstract class.')
        
    def __call__(self, mass):
        return self.central_value(mass)

    def mask_dark_galaxies(self, mass):
        return np.less_equal(random.random(size=len(mass)), self.__call__(mass))



class Dooley17(OccupationFraction):

    name = 'Dooley17'

    # load data
    fnflum = DISDIR+'/data/focc/d17-flum93.dat'
    minfall93,flum_dat93 = np.loadtxt(fnflum,unpack=True)
    fnflum = DISDIR+'/data/focc/d17-flum113.dat'
    minfall113,flum_dat113 = np.loadtxt(fnflum,unpack=True)
    fnflum = DISDIR+'/data/focc/d17-flum144.dat'
    minfall144,flum_dat144 = np.loadtxt(fnflum,unpack=True)

    minfall,flum_dat = None,None

    
    def __init__(self, zreion=9.3):

        self.parameters = {'reionization_redshift': zreion}

        if zreion == 9.3:
            self.__class__.minfall  = self.__class__.minfall93
            self.__class__.flum_dat = self.__class__.flum_dat93
        elif zreion == 11.3:
            self.__class__.minfall  = self.__class__.minfall113
            self.__class__.flum_dat = self.__class__.flum_dat113
        elif zreion == 14.4:
            self.__class__.minfall  = self.__class__.minfall144
            self.__class__.flum_dat = self.__class__.flum_dat144
        else:
            raise ValueError('no support for reionization redshift '+str(zreion)+'in Dooley+ 2017 occupation fraction')

    @classmethod
    def central_value(cls, mass):
        return np.interp(np.log(mass),np.log(cls.minfall),cls.flum_dat,left=0,right=1)

        

class AllLuminous(OccupationFraction):

    name = 'AllLuminous'

    def __init__(self):
        self.parameters = {}

    @classmethod
    def central_value(cls, mass):
        if hasattr(mass,'__iter__'):
            return np.ones(len(mass))
        else:
            return 1.
