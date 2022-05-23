from ...relations import Relation

class ModifiedConcentration(Relation):

    def __call__(self):
        raise NotImplementedError('This is an abstract class.')
