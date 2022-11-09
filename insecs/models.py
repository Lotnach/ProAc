from django.db import models

import insecs

# modelo de indice sectorial.

class Insec:
    def __init__(self,llave,nemo):
        self.llave=llave
        self.nemo=nemo

class InsecFactory:
    def __init__(self):
        self.insecs=[]
        self.insecs.append(Insec("ISC001","SPLXENERGY"))
        
    
    def obtenerInsecs(self):
        return self.insecs

    def getInsec(self,llave):  #detalle indice sectorial
        for insec in self.insecs:
            if insec.llave==llave:
                return insec
        return None