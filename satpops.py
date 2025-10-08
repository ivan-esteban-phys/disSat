import numpy as np
import numpy.random as random
from .vutils import mvir2sigLOS, mvir2sigLOS_new
from . import hosts, baryons, dark_matter
from .relations import Relation
from . import genutils

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
        
        if self.density_profile != 'mix':
            self.properties['density_profile'] = np.repeat(self.density_profile, self.host.number_of_subhalos*self.number_of_instances)
        else:
            self.properties['density_profile'] = np.repeat('coreNFW', self.host.number_of_subhalos*self.number_of_instances)
            self.properties['density_profile'][self.properties['mass'] <  self.mswitch_profile] = 'nfw'

        self.properties['sigLOS'] = self.sample_velocity_dispersions()
        
        is_galaxy = self.properties['mass_stars'] > 0        
        
    def get_input_parameters(self):
        parameters = {}
        for key,val in self.__dict__.items():
            if   key=='properties': continue
            elif key=='dark_matter': parameters[key] = val
            elif key=='host': parameters[key] = val
            elif callable(val):
                parameters[key] = val
            else:
                parameters[key] = val
        return parameters
        
    def print_input_parameters(self):
        for key,val in self.__dict__.items():
            if   key=='properties' or key=='parameters': continue
            elif key=='dark_matter': print(key,':',self.dark_matter.name,'(class Host)')
            elif key=='host': print(key,':',self.host.name,'(class DarkMatterModel)')
            elif callable(val):
                print(key,':',val.name,'(class',val.__class__.__base__.__name__+')')
            else:
                print(key,':',val,'(value)')

                
    def get_relations(self):

        # check if host or dark matter model has changed
        if not self.parameters['dark_matter'] == self.dark_matter:
            for name, attribute in self.dark_matter.__dict__.items():
                if isinstance(attribute, Relation):
                    self.__dict__[name] = attribute
            self.parameters['dark_matter'] = self.dark_matter

        if not self.parameters['host'] == self.host:
            for name, attribute in self.host.__dict__.items():
                if isinstance(attribute, Relation):
                    self.__dict__[name] = attribute
            self.parameters['host'] == self.host
        
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
                                            self.host.number_of_subhalos*self.number_of_instances)
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
        rhalf = np.zeros(self.host.number_of_subhalos*self.number_of_instances)
        mstar = self.properties['mass_stars']
        rhalf[mstar>0] = self.rhalf_2D(mstar[mstar>0])
        return rhalf
    
    def sample_velocity_dispersions(self):

        c_model = self.concentration.name
        c_model_short = c_model[0].lower()+c_model[-2:]
        if c_model_short=='d19': c_model_short = 'd15'

        sigLOS = np.zeros(self.host.number_of_subhalos*self.number_of_instances)
        mstar = self.properties['mass_stars']
        is_galaxy = mstar>0
        
        if self.density_profile != 'mix':
            sigLOS = mvir2sigLOS_new(mvir=self.properties['mass'],
                                     c200=self.properties['c200'],
                                     r200=(self.properties['mass'][is_galaxy] / (200*self.rho_c * genutils.KPC**3/genutils.MSUN * 4./3.*np.pi))**(1./3.),
                                     profile=self.density_profile,
                                     mstar=self.properties['mass_stars'],
                                     Re0=self.properties['rhalf2D'],
                                     zin=self.z_infall,
                                     h0=self.h0,
                                     tin=self.t_infall,
                                     mleft=self.mleft,
                                     sigmaSI=self.dark_matter.sigSI if isinstance(self.dark_matter, dark_matter.models.SIDM) else None,
                                     wdm=isinstance(self.dark_matter, dark_matter.models.WDM),
                                     mWDM=self.dark_matter.mWDM if isinstance(self.dark_matter, dark_matter.models.WDM) else 5.0)
        else:
            sigLOS = np.zeros(self.host.number_of_subhalos*self.number_of_instances)
            icore = np.where(is_galaxy * (self.properties['density_profile']=='coreNFW'))[0]
            icusp = np.where(is_galaxy * (self.properties['density_profile'] == 'nfw'))[0]
            sigLOS[icore] = mvir2sigLOS_new(mvir=self.properties['mass'][icore],
                                            c200=self.properties['c200'][icore],
                                            r200=(self.properties['mass'][icore] / (200*self.rho_c * genutils.KPC**3/genutils.MSUN * 4./3.*np.pi))**(1./3.),
                                            profile='coreNFW',
                                            mstar=self.properties['mass_stars'][icore],
                                            Re0=self.properties['rhalf2D'][icore],
                                            zin=self.z_infall,
                                            h0=self.h0,
                                            tin=self.t_infall,
                                            mleft=self.mleft)
            sigLOS[icusp] = mvir2sigLOS_new(mvir=self.properties['mass'][icusp],
                                            c200=self.properties['c200'][icusp],
                                            r200=(self.properties['mass'][icusp] / (200*self.rho_c * genutils.KPC**3/genutils.MSUN * 4./3.*np.pi))**(1./3.),
                                            profile='nfw',
                                            mstar=self.properties['mass_stars'][icusp],
                                            Re0=self.properties['rhalf2D'][icusp],
                                            zin=self.z_infall,
                                            h0=self.h0,
                                            tin=self.t_infall,
                                            mleft=self.mleft)            

        return sigLOS
    
    def _finish_setup(self):

        self.properties = {}
    
        for name, attribute in self.dark_matter.__dict__.items():
            if isinstance(attribute, Relation):
                self.__dict__[name] = attribute

        for name, attribute in self.host.__dict__.items():
            if isinstance(attribute, Relation):
                self.__dict__[name] = attribute
                
        self.parameters = self.get_input_parameters()
        
        
class MilkyWaySatellites(SatellitePopulation):

    name = 'MilkyWaySatellites'

    def __init__(self, min_mass=1e7, density_profile='mix', mleft=1.,
                 cosmology=dark_matter.models.CDM(), number_of_instances=1):

        # set parameters
        self.z_infall = 1.
        self.rho_c = genutils.rhoc(self.z_infall)
        self.h0 = genutils.h(0)
        self.t_infall = genutils.age(self.z_infall)
        self.min_mass = min_mass
        self.massdef = '200c'
        self.density_profile = density_profile
        self.mswitch_profile = 5e8 # msun
        self.mleft = mleft

        # set dark matter model and host
        self.dark_matter = cosmology        
        self.host = hosts.MilkyWay(cosmology=cosmology)
        self.host.set_number_of_subhalos(min_mass)
        self.number_of_instances = number_of_instances
        
        # choose relations
        self.concentration = dark_matter.concentrations.Diemer19(scatter=True, z=self.z_infall, M_min = min_mass, M_max = self.host.mass())
        self.smhm = baryons.smhm.Moster13(scatter=True)
        self.occupation_fraction = baryons.occupation_fraction.Dooley17(reionization_redshift=9.3)
        self.rhalf_2D = baryons.galaxy_size.Read17(scatter=True)

        # finish setup
        self._finish_setup()
        
