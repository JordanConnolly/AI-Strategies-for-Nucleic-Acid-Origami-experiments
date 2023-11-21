from pymoo.factory import get_performance_indicator
import numpy as np

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))

population = np.array([[0, 0, 0, 0]])

calculated = hv.do(population)
print(calculated)
