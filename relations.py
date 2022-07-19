import sys, inspect
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



################################################################################
# routines to figure out what alternatives are implemented for a given class

def get_alternatives_of_type(the_class):

    all_dis_objects = [ module for module in sys.modules.keys() ]
    objects = []
    
    for module_name in all_dis_objects:
        
        new_objects = inspect.getmembers(sys.modules[module_name], lambda member, the_class=the_class: inspect.isclass(member) and issubclass(member,the_class) and member.__name__ != the_class.__name__)
        
        for i in range(len(new_objects)):
            if new_objects[i] not in objects:
                objects += [ new_objects[i] ]
                
    return objects


def list_alternatives(the_object):

    the_class = the_object.__class__.__base__
    objects = get_alternatives_of_type(the_class)
    print( [an_object[0] for an_object in objects] )


def get_alternative(old_object, new_object, **kwargs):
    """
    Initalizes an instance of the new relation.
    """
    
    the_class = old_object.__class__.__base__
    objects = dict(get_alternatives_of_type(the_class))
    return objects[new_object](**kwargs)
