
class Material():
    """Generic class for all solid materials, place common functions in this class."""

    name = "Generic"
    density = 1000
    color = (1.0, 0, 1.0)



class Aluminium(Material):

    name = "Aluminium"
    density = 2700
    color = (0.5, 0.5, 0.5)

    
    
class StainlessSteel(Material):

    name = "Stainless Steel"
    density = 7800
    color = (0.4, 0.4, 0.4)



class CFRP(Material):

    name = "Carbon Fibre"
    density = 1800
    color = (0, 0, 0)