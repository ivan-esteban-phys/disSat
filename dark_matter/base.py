import numpy.random as random

from ..relations import Relation


class DarkMatterModel():

    class SubhaloMassFunction(Relation):

        name = 'subhalo_mass_function'

        def central_value(self, mhost, min_mass):
            raise NotImplementedError('This is an abstract class.')
            
        def sample(self, mhost, min_mass, baryon_reduction):

            mean_nsub = self.central_value(mhost, min_mass)

            if self.sample_scatter:
                return int(round(random.poisson(lam=mean_nsub) * baryon_reduction))
            else:
                return mean_nsub
            

    class MassConcentration(Relation):

        name = 'mass-concentration'

        def __init__(self, model='diemer19', scatter=True):
            self.parameters = {'model': model}
            self.sample_scatter = scatter
        
        def central_value(self, mass, z):
            raise NotImplementedError('This is an abstract class.')
    
        def scatter(self):
            """Log-normal scatter."""
            raise NotImplementedError('This is an abstract class')

        def sample(self, mass, z):

            median = self.central_value(mass, massdef, z)

            if self.sample_scatter:
                return median * 10**random.normal(loc=0,scale=self.scatter(),size=len(mass))
            else:
                return median

