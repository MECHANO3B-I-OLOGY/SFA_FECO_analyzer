import math

class distance:

    def __init__(self, lambdaOdd, lambdaEven, lambdaD = None, mu = 1.333, n = 3):
        self.lambdaOdd = lambdaOdd
        self.lambdaEven = lambdaEven
        self.lambdaD = lambdaD
        self.mu = mu
        self.n = n

    def calcMuMica(self, lambda2):
        return 1.5820 + (4.76e3/(lambda2**2))

    def calcMuBar(self, muMica):
        return muMica/self.mu

    def realDCalc(self):
        def rightHandSide():
            muBar = self.calcMuBar(self.calcMuMica(self.lambdaD))

            upperInner = 1-(self.lambdaOdd/self.lambdaD)
            lowerInner = 1-(self.lambdaOdd/self.lambdaEven)
            innerTerm = (upperInner/lowerInner)*math.pi

            upperTerm = 2*muBar * math.sin(innerTerm)
            lowerTerm = (1+muBar**2) * math.cos(innerTerm)
            pm = muBar**2 - 1

            rhs = upperTerm/(lowerTerm + (-1)**(self.n+1)*pm)
            #plus = upperTerm/(lowerTerm + pm)
            #minus = upperTerm/(lowerTerm - pm)

            return rhs

        afterTan = math.atan(rightHandSide())
        ret = (afterTan * self.lambdaD)/(2*math.pi * self.mu)

        return ret

    def asymMicaAu(self):
        def rightHandSide():
            muMica = self.calcMuMica(self.lambdaD)

            innerTerm = (self.muMica - self.mu)/(self.muMica + self.mu)
            upperTerm = (1- innerTerm**2)*math.sin(2*math.pi*self.lambdaOdd/self.lambdaD)
            lowerTerm = -2*(innerTerm) + (1+ (innerTerm**2))*math.cos(2*math.pi*self.lambdaOdd/self.lambdaD)

            return upperTerm / lowerTerm

        afterTan = math.atan(rightHandSide())
        ret = (afterTan * self.lambdaD)/(4*math.pi * self.mu)

        return ret

    def arrayDistance(self, lambdaDArray, func):
        if not hasattr(self, func):
            raise ValueError(f"Unknown function '{func}'")

        func = getattr(self, func)
        results = []
        for val in lambdaDArray:
            self.lambdaD = val
            results.append(func())
        return results

if __name__ == "__main__":
    #dist = distance(560.1765, 572.9366, 560.720, muMica=1.5971, mu = 1.34,n=43)
    
    lambda_odd = float(input("Enter Lambda odd (nm): "))
    lambda_even = float(input("Enter Lambda even (nm): "))

    # Prompt for optional inputs with defaults
    mu_input = input("Enter mu (default 1.34): ").strip()
    n_input = input("Enter n (default 3): ").strip()

    # Use defaults if no input is given
    mu = float(mu_input) if mu_input else 1.34
    n = float(n_input) if n_input else 3

    # Create distance object using entered values
    dist = distance(lambda_odd, lambda_even, mu=mu, n=n)

    result = dist.arrayDistance([560.720, 560.717, 560.718, 560.724, 560.727, 560.723, 560.723, 560.717, 560.717, 560.724, 560.714, 560.725, 560.720], "realDCalc")

    # Print results
    for i in result:
        print(f"The distance is {i} nm")


