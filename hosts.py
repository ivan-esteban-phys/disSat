from . import relations, dark_matter


class Host:

    def __init__(self, cosmology=dark_matter.models.CDM()):
        self.dark_matter = cosmology
        self.subhalo_mass_function = dark_matter.subhaloMF.Moster13(scatter=True)
        
    @staticmethod
    def mass():
        raise NotImplementedError('This is an abstract class')
            
    def set_number_of_subhalos(self, min_mass):
        self.subhalo_min_mass = min_mass
        self.number_of_subhalos = self.subhalo_mass_function.number_of_subhalos(self.mass(), min_mass)

    
class MilkyWay(Host):
    
    @staticmethod
    def mass():
        return 1e12 # msun
        
