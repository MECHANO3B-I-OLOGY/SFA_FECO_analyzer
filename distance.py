import math

class distance:

    def __init__(self, lambdaOdd, lambdaEven, lambdaD, muMica=1.582, mu = 1.333, n = 3):
        self.lambdaOdd = lambdaOdd
        self.lambdaEven = lambdaEven
        self.lambdaD = lambdaD
        self.muMica = muMica
        self.mu = mu
        self.n = n

        self.muBar = self.calcMuBar()

    def calcMuBar(self):
        return self.muMica/self.mu

    def calcFn(self):
        return self.lambdaEven/(self.lambdaEven - self.lambdaOdd)

    def saaDCalc(self, nOdd):
        Fn = self.calcFn()
        top = Fn*(self.lambdaD - self.lambdaOdd)
        oddD = nOdd*top/(2*self.muMica)
        evenD = (nOdd-1)*top*self.muMica/(2*self.mu**2)
        
        return (oddD, evenD)

    def realDCalc(self):
        def rightHandSide():
            upperInner = 1-(self.lambdaOdd/self.lambdaD)
            lowerInner = 1-(self.lambdaOdd/self.lambdaEven)
            innerTerm = (upperInner/lowerInner)*math.pi

            upperTerm = 2*self.muBar * math.sin(innerTerm)
            lowerTerm = (1+self.muBar**2) * math.cos(innerTerm)
            pm = self.muBar**2 - 1

            plus = upperTerm/(lowerTerm + pm)
            minus = upperTerm/(lowerTerm - pm)

            return (plus, minus)

        ret = []
        for right in rightHandSide():
            afterTan = math.atan(right)
            ret += [(afterTan * self.lambdaD)/(2*math.pi * self.mu)]

        return (ret[0], ret[1])

if __name__ == "__main__":
    dist = distance(560.1765, 572.9366, 560.720, muMica=1.5971, mu = 1.34,n=43)
    print(dist.saaDCalc(dist.n))
    print(dist.realDCalc())

