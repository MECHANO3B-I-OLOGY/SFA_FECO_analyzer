import math
import numpy as np

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

            return upperTerm, lowerTerm + (-1)**(self.n+1)*pm

        num, denom = rightHandSide()
        afterTan = math.atan2(num, denom)
        ret = (afterTan * self.lambdaD)/(2*math.pi * self.mu)

        return abs(ret)

    def asymMicaAu(self):
        def rightHandSide():
            muMica = self.calcMuMica(self.lambdaD)

            innerTerm = (muMica - self.mu)/(muMica + self.mu)
            upperTerm = (1- innerTerm**2)*math.sin(2*math.pi*self.lambdaOdd/self.lambdaD)
            lowerTerm = -2*(innerTerm) + (1+ (innerTerm**2))*math.cos(2*math.pi*self.lambdaOdd/self.lambdaD)

            return upperTerm, lowerTerm

        num, denom = rightHandSide()

        afterTan = math.atan2(num, denom)
        ret = (afterTan * self.lambdaD)/(4*math.pi * self.mu)

        return ret

    def arrayDistance(self, lambdaDArray, func, run_direction="out"):
        """
        Calculate distance for an array of lambdaD values, with phase unwrapping
        to handle fringe order crossings.

        Phase unwrapping detects jumps > π in the raw atan2 output across consecutive
        frames and adds/subtracts 2π to keep the angle sequence continuous. This
        prevents the distance from jumping or folding back when a fringe crosses a
        fringe order boundary.

        run_direction : "in" or "out" (default "out")
            "out" — unwrap left-to-right (forward), anchoring at the start of the run.
                    Use for the out (separation) run where the fringe moves away.
            "in"  — unwrap right-to-left (backward), anchoring at the end of the run
                    (the closest-approach point). This ensures the approach is treated
                    as continuously decreasing even when it crosses a fringe order.

        The unwrapped angle sequence is then converted to distances using the same
        formula as the per-point functions, but with the cumulative phase offset applied.
        """
        if not hasattr(self, func):
            raise ValueError(f"Unknown function '{func}'")

        is_realD = (func == "realDCalc")
        is_asymMicaAu = (func == "asymMicaAu")

        # --- Collect raw atan2 angles and per-point scale factors ---
        raw_angles = []
        scale_factors = []   #holds thing to be multiplied by angle

        for val in lambdaDArray:
            self.lambdaD = val

            if is_realD:
                muBar = self.calcMuBar(self.calcMuMica(val))
                upperInner = 1 - (self.lambdaOdd / val)
                lowerInner = 1 - (self.lambdaOdd / self.lambdaEven)
                innerTerm = (upperInner / lowerInner) * math.pi
                num = 2 * muBar * math.sin(innerTerm)
                denom = (1 + muBar**2) * math.cos(innerTerm) + (-1)**(self.n + 1) * (muBar**2 - 1)
                scale = val / (2 * math.pi * self.mu)

            elif is_asymMicaAu: 
                muMica = self.calcMuMica(val)
                r = (muMica - self.mu) / (muMica + self.mu)
                num = (1 - r**2) * math.sin(2 * math.pi * self.lambdaOdd / val)
                denom = -2 * r + (1 + r**2) * math.cos(2 * math.pi * self.lambdaOdd / val)
                scale = val / (4 * math.pi * self.mu)

            raw_angles.append(math.atan2(num, denom))
            scale_factors.append(scale)

        # --- Unwrap the angle sequence ---
        if run_direction == "in":
            unwrapped = np.unwrap(raw_angles[::-1])[::-1] #in run reversed
        else:
            unwrapped = np.unwrap(raw_angles)

        # --- Convert unwrapped angles back to distances ---
        results = []
        for angle, scale in zip(unwrapped, scale_factors):
            d = angle * scale
            results.append(d)

        return results

if __name__ == "__main__":
    lambda_odd = float(input("Enter Lambda odd (nm): "))
    lambda_even = float(input("Enter Lambda even (nm): "))

    mu_input = input("Enter mu (default 1.34): ").strip()
    n_input = input("Enter n (default 3): ").strip()

    mu = float(mu_input) if mu_input else 1.34
    n = float(n_input) if n_input else 3

    dist = distance(lambda_odd, lambda_even, mu=mu, n=n)

    result = dist.arrayDistance([
    567.021, 566.542, 566.087, 565.665, 565.259, 564.845, 564.438, 564.038, 563.64,
    563.257, 562.882, 562.522, 562.162, 561.828, 561.495, 561.172, 560.893, 560.643,
    560.414, 560.226, 560.057, 559.911, 559.785, 559.675, 559.588, 559.509, 559.431,
    559.374, 559.32, 559.271, 559.224, 559.182, 559.148, 559.118, 559.089, 559.059,
    559.026, 559.011, 558.979, 558.962, 558.942, 558.919, 558.894, 558.879, 558.861,
    558.841, 558.833, 558.817, 558.802, 558.784, 558.768, 558.749, 558.744, 558.735,
    558.718, 558.707, 558.701, 558.696, 558.676, 558.672, 558.666, 558.659, 558.643,
    558.634, 558.625, 558.628, 558.615, 558.61, 558.607, 558.59, 558.591, 558.586,
    558.571, 558.579, 558.557, 558.562, 558.551, 558.545, 558.532, 558.536, 558.529,
    558.522, 558.514, 558.517, 558.511, 558.508, 558.497, 558.496, 558.487, 558.487,
    558.476, 558.483, 558.472, 558.475, 558.471, 558.466, 558.462, 558.459, 558.45,
    558.458, 558.449, 558.442, 558.446, 558.439, 558.436, 558.424, 558.427, 558.425,
    558.425, 558.42, 558.414, 558.407, 558.407, 558.406, 558.408, 558.404, 558.401,
    558.393, 558.395, 558.391, 558.383, 558.391, 558.382, 558.374, 558.372, 558.374,
    558.374, 558.373, 558.366, 558.36, 558.36, 558.365, 558.36, 558.349, 558.352,
    558.354, 558.348, 558.341, 558.346, 558.344, 558.337, 558.34, 558.339, 558.333,
    558.327, 558.329, 558.327, 558.329, 558.325, 558.327, 558.321, 558.322, 558.309,
    558.317, 558.323, 558.313, 558.307, 558.304, 558.304, 558.313, 558.306, 558.306,
    558.3, 558.295, 558.299, 558.303, 558.299, 558.298, 558.289, 558.288, 558.291,
    558.291, 558.288, 558.283, 558.284, 558.289, 558.278, 558.278, 558.274, 558.28,
    558.273, 558.267, 558.282, 558.276, 558.262, 558.266, 558.264, 558.267, 558.264,
    558.266, 558.261, 558.262, 558.254, 558.264, 558.25, 558.258, 558.25, 558.261,
    558.253, 558.251, 558.247, 558.243, 558.244, 558.245, 558.239, 558.246, 558.242,
    558.236, 558.239, 558.236, 558.237, 558.239, 558.229, 558.234, 558.226, 558.237,
    558.225, 558.226, 558.232, 558.229, 558.221, 558.226
], "asymMicaAu")

    for i in result:
        print(f"The distance is {i} nm")