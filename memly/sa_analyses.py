"""
sa_analyses.py
Contains the methods for surface area-based analysis metrics: the surface area per lipid and the area compressibility
modulus.
"""


from memly.metrics import Metric


class SurfaceArea(Metric):
    def __init__(self, membrane, title="Surface area", units="angstrom^2"):
        """
        Measures the surface area of each leaflet in the simulation.

        :param membrane: memly.membrane.Membrane object.
        :param title: str, optional. The title of the metric. Default is "Surface area".
        :param units: str, optional. Units of the metric. Default is "angstrom^2".
        """


