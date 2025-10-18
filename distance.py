import math

class distance:

    def __init__(self, lambdaOdd, lambdaEven, lambdaD, mu = 1.333, n = 3):
        self.lambdaOdd = lambdaOdd
        self.lambdaEven = lambdaEven
        self.lambdaD = lambdaD
        self.mu = mu
        self.n = n

        self.muBar = self.calcMuBar()

    def calcMuMica(self, lambda2):
        return 1.5820 + (4.76e3/(lambda2**2))

    def calcMuBar(self, muMica):
        return muMica/self.mu

    def calcFn(self):
        return self.lambdaEven/(self.lambdaEven - self.lambdaOdd)

    def realDCalc(self):
        def rightHandSide():
            muBar = calcMuBar(calcMuMica(self.lambda_D))

            upperInner = 1-(self.lambdaOdd/self.lambdaD)
            lowerInner = 1-(self.lambdaOdd/self.lambdaEven)
            innerTerm = (upperInner/lowerInner)*math.pi

            upperTerm = 2*muBar * math.sin(innerTerm)
            lowerTerm = (1+muBar**2) * math.cos(innerTerm)
            pm = muBar**2 - 1

            rhs = upperTerm/(lowerTerm + (-1)**(n+1)*pm)
            #plus = upperTerm/(lowerTerm + pm)
            #minus = upperTerm/(lowerTerm - pm)

            return rhs

        afterTan = math.atan(rightHandSide())
        ret = (afterTan * self.lambdaD)/(2*math.pi * self.mu)

        return ret

    def asymMicaAu(self):
        def rightHandSide():
            muMica = calcMuMica(self.lambda_D)

            innerTerm = (self.muMica - self.mu)/(self.muMica + self.mu)
            upperTerm = (1- innerTerm**2)*math.sin(2*math.pi*self.lambda_odd/self.lambda_D)
            lowerTerm = -2*(innerTerm) + (1+ (innerTerm**2))*math.cos(2*math.pi*self.lambda_odd/self.lambda_D)

            return upperTerm / lowerTerm

        afterTan = math.atan(rightHandSide())
        ret = (afterTan * self.lambdaD)/(4*math.pi * self.mu)

        return ret

if __name__ == "__main__":
    #dist = distance(560.1765, 572.9366, 560.720, muMica=1.5971, mu = 1.34,n=43)
    
    lambda_odd = float(input("Enter Lambda odd (nm): "))
    lambda_even = float(input("Enter Lambda even (nm): "))
    lambda_D = float(input("Enter Lambda D (nm): "))

    # Prompt for optional inputs with defaults
    mu_input = input("Enter mu (default 1.34): ").strip()
    n_input = input("Enter n (default 3): ").strip()

    # Use defaults if no input is given
    mu = float(mu_input) if mu_input else 1.34
    n = float(n_input) if n_input else 3

    # Create distance object using entered values
    dist = distance(lambda_odd, lambda_even, lambda_D, mu=mu, n=n)

    # Print results
    print(f"The distance is {dist.realDCalc()} nm")

