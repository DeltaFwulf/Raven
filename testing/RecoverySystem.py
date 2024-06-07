# TODO: Calculate packed volume and parachute mass
class Parachute:

    def __init__(self, diameter, lineLength, shape, material):

        self.diameter = diameter
        self.lineLength = lineLength # This is used for line-snatch calculations, and so assumes the canopy is closed
        self.Cx = self.findCx(self, shape)
        self.material = material # use to find mass

    # Input different cX etc here depending on shape
    def findCx(self, shape):
        
        return {
            'elliptical':0,
            'toroidal':0,
            'conical':0
        }.get(shape, 0) # if you get a 0, then tell the operator about it (throw an 'exception')




class Riser:

    def __init__(self, length, material):

        self.length = length
        self.unitStiffness = self.getStiffness(self, material)


    def getStiffness(self, material):

        return {
            'nylon':0,
            'kevlar':0,
            'polyester':0
        }.get(material, 0)





class RecoverySystem:

    def __init__(self, hasReserve, drogue, main, reserve, drogueRiser, mainRiser, reserveRiser):
        
        self.main = main
        self.drogue = drogue

        self.hasReserve = hasReserve

        if(hasReserve):    
            self.reserve = reserve

        self.drogueRiser = drogueRiser
        self.mainRiser = mainRiser
        self.reserveRiser = reserveRiser