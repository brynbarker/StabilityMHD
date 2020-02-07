import numpy as np
from element1d import element


class mesh():
    """
    Creates a mesh object and initializes nodes
    and elements that make up the mesh

    Attributes:
        geom     (geom): the geometry of the mesh
        dim       (int): dimension of the mesh
        n         (int): number of elements
        nodes    (dict): dictionary mapping the dimension to an
                         array of the nodes in that dimension
        elements (list): list of elements in the mesh

    Functions:
        _defineNodes: defines and stores nodes
        _initElements: create and store elements
    """
    def __init__(self,n):
        """
        Initialized mesh by defining nodes and creating
        element objects

        Arguments:
            geom (geom): the geometry of the mesh
            n     (int): number of elements
        """
        self.n = n
        self._defineNodes()
        self._initElements()

    def _defineNodes(self):
        """ Define and store nodes """
        self.nodes = np.linspace(0,1,self.n+1)
        return

    def _initElements(self):
        """ Create and store elements """
        self.elements = []

        # iterate over each dimension
        for i in range(self.n):
            n1 = self.nodes[i]
            n2 = self.nodes[i+1]
            self.elements.append(element(i,n1,n2))
        return
