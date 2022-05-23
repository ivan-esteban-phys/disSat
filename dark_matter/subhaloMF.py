import numpy.random as random
from ..relations import Relation


class SubhaloMassFunction(Relation):
    
    def __init__(self, baryon_reduction=0.8, scatter=True):
        self.parameters = {'alpha': self.alpha(),
                           'normalization': self.normalization(),
                           'baryon_reduction': baryon_reduction}
        self.sample_scatter = scatter

    @staticmethod
    def alpha():
        return NotImplementedError('This is an abstract class.')

    @staticmethod
    def normalization():
        return NotImplementedError('This is an abstract class.')
    
    def number_of_subhalos(self, mhost, min_mass):
        a,K0,baryon_reduction = self.alpha(), self.normalization(), self.parameters['baryon_reduction']
        median = K0 * mhost / (a-1) * (min_mass**(1-a) - mhost**(1-a))
        if self.sample_scatter:
            return int(round(random.poisson(lam=median) * baryon_reduction))
        else:
            return median

    @classmethod
    def __call__(cls, min_mass, max_mass, nsubhalos):
        return cls._sample_negative_power(alpha=-cls.alpha(),
                                           low=min_mass, high=max_mass,
                                           size=nsubhalos)
        
    # numpy/scipy doesn't allow for negative power law exponents
    def _sample_negative_power(alpha=-2, low=1e7, high=1e12, size=1):
        """
        This is written differently from most power law samplers, in that
        it takes the power law exponent, alpha<0, instead of alpha+1.
        
        Implementation from https://stackoverflow.com/questions/31114330/
        python-generating-random-numbers-from-a-power-law-distribution/31117560
        """
        aplus1 = alpha + 1
        r = random.random(size=size)
        return (low**aplus1 + (high**aplus1 - low**aplus1)*r)**(1./aplus1)



    
############################################################

class Moster13(SubhaloMassFunction):

    name = 'Moster13'

    @staticmethod
    def alpha():
        return 1.84

    @staticmethod
    def normalization():
        return 0.000854
    
