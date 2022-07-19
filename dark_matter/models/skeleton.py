from ...relations import Relation


class DarkMatterModel:

    name = 'DarkMatterModel'

        

class ModifiedConcentration(Relation):

    name = 'ModifiedConcentration'
    
    def __call__(self):
        raise NotImplementedError('This is an abstract class.')
