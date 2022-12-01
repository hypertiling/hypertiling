import numpy as np

class DuplicateContainerSlow:

    def __init__(self):
        self.elements = []

    def add(self, element, idx):
        self.elements.append((element,idx))

    def is_duplicate(self, element):
        for elm in self.elements:
            if np.abs(elm[0]-element) < 1e-12:
                return True, elm[1]

        return False, -1