import numpy as np
import numpy.random as random


class Relation:

    name = 'Relation'
    
    def __init__(self):
        self.parameters = {}
        self.sample_scatter = True
        
    def central_value():
        raise NotImplementedError('This is an abstract class.')

    def scatter():
        raise NotImplementedError('This is an abstract class.')
    
    def __call__():
        raise NotImplementedError('This is an abstract class.')
