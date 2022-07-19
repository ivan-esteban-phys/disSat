import numpy as np

from . import helper
from ....relations import Relation


class TransferFunction(Relation):

    name = 'TransferFunction'

    def __init__(self):
        self.parameters = {'mu': self.mu(),
                           'beta': self.beta(),
                           'gamma': self.gamma()}

    @staticmethod
    def mu():
        return 1.12  # exponent in transfer function in Schneider+ 2012

    @staticmethod
    def beta():
        raise NotImplementedError('This is an abstract class.')

    @staticmethod
    def gamma():
        raise NotImplementedError('This is an abstract class.')

    @staticmethod
    def __call__(self, mass, mWDM):
        raise NotImplementedError('Need to look up and code in (use Lovell20 instead for now)')
    

class Schneider12(TransferFunction):

    name = 'Schneider12'
    
    @staticmethod
    def beta():
        return 1.16

    @staticmethod
    def gamma():
        return 1.0


class Lovell14(TransferFunction):

    name = 'Lovell14'

    @staticmethod
    def beta():
        return 0.99

    @staticmethod
    def gamma():
        return 2.7

    
class Lovell20(TransferFunction):

    name = 'Lovell20'
    
    @staticmethod
    def alpha(): return 0

    @staticmethod
    def beta (): return 0

    @staticmethod
    def gamma(): return 0    
        
    @staticmethod
    def __call__(mass, mWDM):
        mhalf = helper.half_mode_mass(mWDM)
        return ( 1 + (4.2*mhalf/mass)**2.5 )**-0.2
