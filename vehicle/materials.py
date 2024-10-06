# definitions of all materials
# materials have (for now) only densities, but we can work in melting points, strengths, etc. 

class Materials():

    densities = {
        "generic":1000,
        "aluminium":2700,
        "steel":7800,
        "cfrp":1800
    }

    maxTemp = {
        "generic":273,
        "aluminium":0,
        "steel":0,
        "cfrp":0
    }

    