import numpy as np
import numpy.random as random
from vutils import mvir2sigLOS
from . import hosts, baryons, dark_matter
from .relations import Relation



class SatellitePopulation:

    name = 'Satellites'
    
    def generate_population(self):
        """
        Computes satellite properties based on chosen relations.

        n = number of realizations to generate
        """
        self.properties['mass'] = self._sample_subhalo_masses()
        self.properties['c200'] = self._sample_concentrations()
        self.properties['mass_stars'] = self._sample_stellar_masses()
        self.properties['rhalf2D'] = self._sample_galaxy_sizes()
        
        if self.profile != 'mix':
            self.properties['profile'] = np.repeat(self.profile, self.host.number_of_subhalos)
        else:
            self.properties['profile'] = np.repeat('coreNFW', self.host.number_of_subhalos)
            self.properties['profile'][self.properties['mass'] <  self.mswitch_profile] = 'nfw'

        self.properties['sigLOS'] = self._sample_velocity_dispersions()

    def get_relations(self):
        relations = {}
        for name,attribute in self.__dict__.items():
            if isinstance(attribute, Relation):
                relations[name] = attribute
        return relations
        
    def print_relations(self):
        relations = self.get_relations()
        for key,rel in relations.items():
            print(key,':',rel.name,'' if len(rel.parameters.keys())==0 else rel.parameters)

    def sources_of_scatter(self):
        relations = self.get_relations()
        sources = {}
        for name,relation in relations.items():
            if 'sample_scatter' in dir(relation):
                sources[name] = relation.sample_scatter
        return sources

    def set_scatter(self, sources):
        """
        Requires as input a dictionary with name of the relations and
        whether to turn the scatter on/off (True/False).
        """
        relations = self.get_relations()
        for name,scatter in sources.items():
            self.__dict__[name].sample_scatter = scatter
            

    def _sample_subhalo_masses(self):
        m = self.host.subhalo_mass_function(self.min_mass, self.host.mass(),
                                            self.host.number_of_subhalos)
        if isinstance(self.dark_matter, dark_matter.models.WDM):
            tf = self.dark_matter.transfer_function(m, self.dark_matter.mWDM)
            mask = np.less_equal(random.random(size=len(m)), tf)
            self.host.number_of_subhalos = sum(mask)
            return m[mask]
        else:
            return m
        
    def _sample_concentrations(self):
        c = self.concentration(self.properties['mass'], self.z_infall)
        relations = self.get_relations()
        for name,relation in relations.items():
            if isinstance(relation, dark_matter.models.ModifiedConcentration):
                cmod = relation(self.properties['mass'], c, self.dark_matter, z=self.z_infall)
                return cmod
        return c

    def _sample_stellar_masses(self):
        mstar = self.smhm(self.properties['mass'], z=self.z_infall)
        mstar *= self.occupation_fraction.mask_dark_galaxies(self.properties['mass'])
        return mstar

    def _sample_galaxy_sizes(self):
        rhalf = np.zeros(self.host.number_of_subhalos)
        mstar = self.properties['mass_stars']
        rhalf[mstar>0] = self.rhalf_2D(mstar[mstar>0])
        return rhalf
    
    def _sample_velocity_dispersions(self):

        c_model = self.concentration.name
        c_model_short = c_model[0].lower()+c_model[-2:]
        if c_model_short=='d19': c_model_short = 'd15'

        sigLOS = np.zeros(self.host.number_of_subhalos)
        mstar = self.properties['mass_stars']
        is_galaxy = mstar>0
        
        if self.profile != 'mix':
            sigLOS = mvir2sigLOS(self.properties['mass'],
                                 self.profile,
                                 mleft=self.mleft,
                                 zin=self.z_infall,
                                 mstar=self.properties['mass_stars'],
                                 Re0=self.properties['rhalf2D'],
                                 c200=self.properties['c200'],
                                 cNFW_method=c_model_short,
                                 sigmaSI=self.dark_matter.sigSI if isinstance(self.dark_matter, dark_matter.models.SIDM) else None)
        else:
            sigLOS = np.zeros(self.host.number_of_subhalos)
            icore = np.where(is_galaxy * (self.properties['profile']=='coreNFW'))[0]
            sigLOS[icore] = mvir2sigLOS(self.properties['mass'][icore],
                                        'coreNFW',
                                        mleft=self.mleft,
                                        zin=self.z_infall,
                                        mstar=self.properties['mass_stars'][icore],
                                        Re0=self.properties['rhalf2D'][icore],
                                        c200=self.properties['c200'][icore],
                                        cNFW_method=c_model_short)
            icusp = np.where(is_galaxy * (self.properties['profile'] == 'nfw'))[0]
            sigLOS[icusp] = mvir2sigLOS(self.properties['mass'][icusp],
                                        'nfw',
                                        mleft=self.mleft,
                                        zin=self.z_infall,
                                        mstar=self.properties['mass_stars'][icusp],
                                        Re0=self.properties['rhalf2D'][icusp],
                                        c200=self.properties['c200'][icusp],
                                        cNFW_method=c_model_short)

        return sigLOS
        

        
class MilkyWaySatellites(SatellitePopulation):

    name = 'MilkyWaySatellites'

    def __init__(self, min_mass=1e7, profile='mix', mleft=1., cosmology=dark_matter.models.CDM()):

        self.z_infall = 1.
        self.min_mass = min_mass
        self.massdef = '200c'
        self.profile = profile
        self.mswitch_profile = 5e8 # msun
        self.mleft = mleft

        self.dark_matter = cosmology
        
        self.host = hosts.MilkyWay(cosmology=cosmology)
        self.host.set_number_of_subhalos(min_mass)
        
        self.properties = {}

        # choose relations
        self.concentration = dark_matter.concentrations.Diemer19(scatter=True)
        self.smhm = baryons.smhm.Moster13(scatter=True)
        self.occupation_fraction = baryons.occupation_fraction.Dooley17(zreion=9.3)
        self.rhalf_2D = baryons.galaxy_size.Read17(scatter=True)

        for name, attribute in cosmology.__dict__.items():
            if isinstance(attribute, Relation):
                self.__dict__[name] = attribute

        for name, attribute in self.host.__dict__.items():
            if isinstance(attribute, Relation):
                self.__dict__[name] = attribute
                
