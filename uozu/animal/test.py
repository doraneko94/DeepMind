import numpy as np

class A:
    def __init__(self):
        self.a = 1

L = [A(), A(), A()]
L[0].a = 2
print(L[0].a)