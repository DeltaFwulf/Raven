"""Contains primitives defined as triangulated meshes."""



class Conic():
    """Generalised conic shape.

    Represents any shape with some annular (to circular) cross section, revolved
    by some angle (0 -> 360 deg) about a central axis."""

    def __init__(od_root:float, od_end:float, length:float, material:dict, **kwargs):

        id_root = 0.0 if kwargs.get('id_root') is None else kwargs.get('id_root')
        id_end = 0.0 if kwargs.get('id_root') is None else kwargs.get('id_end')

        pts = []
        tris = []

        # root face:
        if id_root == 0.0 and od_root > 0.0:
            pass
        elif id_root > 0.0 and id_root == od_root:
            # AA singular
            pass
        elif id_root > 0.0 and od_root > 0.0:
            # annular
            pass
        else:
            # point


        # build the mesh

            # root outer

            # end outer

            # root inner

            # end inner
        
        # calculate properties
