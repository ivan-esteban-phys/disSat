from .skeleton import DarkMatterModel

class SIDM(DarkMatterModel):
    
    name = 'SIDM'

    def __init__(self, sigSI=1.):
        self.sigSI = sigSI
        self.parameters = {'sigSI': self.sigSI}
