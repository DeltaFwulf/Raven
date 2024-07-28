from numpy import pi

class Fin:

    # for now, assume fins to be clipped delta (expand this later on with a generalised solver)
    def __init__(self, number, rootChord, tipChord, span, sweepAngle):

        self.number = number

        """Fin Geometry"""
        self.rootChord = rootChord
        self.tipChord = tipChord
        self.span = span
        self.sweepAngle = sweepAngle
        
        self.taperRatio = self.tipChord / self.rootChord
        self.meanX = 0
        self.meanY = 0 
        self.meanChord = self.findMeanChord(self.rootChord, self.tipChord, self.taperRatio)

        self.ClAlpha = 2 * pi
        self.alpha0 = 0 # symmetric aerofoils

    def findMeanChord(self, rootChord, tipChord, taperRatio):
        
        meanChord = (2 / 3) * ((rootChord ** 2) / (rootChord + tipChord)) * ((taperRatio**2) + taperRatio + 1)
        return meanChord
    



class Nosecone:

    def __init__(self, shape, diameter, finenessRatio, surfaceFinish):
        
        self.shape = shape
        self.diameter = diameter
        self.finenessRatio = finenessRatio
        self.length = diameter * finenessRatio / 2
        self.surfaceFinish = surfaceFinish


    def getDragCoefficient(Nosecone, M, airDensity):
        
        # Calculate the estimate Cd of the nosecone at a specific timestep
        pass