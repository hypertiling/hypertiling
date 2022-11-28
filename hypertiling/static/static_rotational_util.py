import numpy as np

class DuplicateContainerSimple:
    # since set is a hashed type, we need to round

    def __init__(self, digits):
        self.digits = digits
        self.elements = set()

    def add(self, element):
        self.elements.add(np.round(element, self.digits))

    def is_duplicate(self, element):
        element = np.round(element, self.digits)
        return (element in self.elements)