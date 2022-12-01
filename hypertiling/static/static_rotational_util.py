import numpy as np

class DuplicateContainerSimple:
    # since set is a hashed type, we need to round
    # warning: due to the rounding this duplicate container 
    # is not 100 percent reliable; if can happen that two duplicates
    # end up being rounded to different numbers even though they are 
    # only eps apart; in this case they are undetected as duplicates

    def __init__(self, digits):
        self.digits = digits
        self.elements = set()

    def add(self, element):
        self.elements.add(np.round(element, self.digits))

    def is_duplicate(self, element):
        element = np.round(element, self.digits)
        return (element in self.elements)