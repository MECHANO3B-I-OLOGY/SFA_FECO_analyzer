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

            rhs = upperTerm/(lowerTerm + (-1)**(n+1)*pm)
            #plus = upperTerm/(lowerTerm + pm)
            #minus = upperTerm/(lowerTerm - pm)

            return rhs

        afterTan = math.atan(rightHandSide())
        ret = (afterTan * self.lambdaD)/(2*math.pi * self.mu)

        return ret

if __name__ == "__main__":
    #dist = distance(560.1765, 572.9366, 560.720, muMica=1.5971, mu = 1.34,n=43)
    
    lambda_odd = float(input("Enter Lambda odd (nm): "))
    lambda_even = float(input("Enter Lambda even (nm): "))
    lambda_D = float(input("Enter Lambda D (nm): "))

    # Prompt for optional inputs with defaults
    muMica_input = input("Enter muMica (default 1.5971): ").strip()
    mu_input = input("Enter mu (default 1.34): ").strip()
    n_input = input("Enter n (default 3): ").strip()

    # Use defaults if no input is given
    muMica = float(muMica_input) if muMica_input else 1.5971
    mu = float(mu_input) if mu_input else 1.34
    n = float(n_input) if n_input else 3

    # Create distance object using entered values
    dist = distance(lambda_odd, lambda_even, lambda_D, muMica=muMica, mu=mu, n=n)

    # Print results
    print(f"The distance is {dist.realDCalc()} nm")

