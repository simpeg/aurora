"""
Placeholder for example
===========================

This example is a placeholder that uses the sphinx-gallery syntax
for creating examples
"""

import aurora
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Step 1
# ------
# Some description of what we are doing in step one

x = np.linspace(0, 4*np.pi, 100)

# take the sin(x)
y = np.sin(x)


###############################################################################
# Step 2
# ------
# Plot it

fig, ax = plt.subplots(1, 1)
ax.plot(x, y)


