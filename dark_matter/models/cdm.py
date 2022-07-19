import numpy as np
import numpy.random as random

from .. import subhaloMF, concentrations
from .skeleton import DarkMatterModel


class CDM(DarkMatterModel):

    name = 'CDM'

    def __init__(self):
        self.parameters = {}

