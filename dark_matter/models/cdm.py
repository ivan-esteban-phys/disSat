import numpy as np
import numpy.random as random

from .. import subhaloMF, concentrations



class CDM:

    name = 'CDM'

    def __init__(self):
        self.parameters = {}
        self.relations = {}
        self.relations['mass-concentration'] = concentrations.Diemer19(scatter=True)

        self.concentration = concentrations.Diemer19(scatter=True)

