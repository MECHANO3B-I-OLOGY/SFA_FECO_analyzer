import math

class distance:

    def __init__(self, lambdaOdd, lambdaEven, lambdaD, muMica=1.582, mu = 1.333, n = 3):
        self.lambdaOdd = lambdaOdd
        self.lambdaEven = lambdaEven
        self.lambdaD = lambdaD
        self.muMica = muMica
        self.mu = mu
        self.n = n

    def calcFn(self):
        return self.lambdaEven/(self.lambdaEven - self.lambdaOdd)

    def saaDCalc(self, nOdd):
        Fn = calcFn()
        top = self.Fn*(self.lambdaD - self.lambdaOdd)
        oddD = nOdd*top/(2*self.muMica)
        evenD = (nOdd-1)*top*self.muMica/(2*self.mu**2)
        
        return (oddD, evenD)
