import numpy as np
import matplotlib.pyplot as plt

# # Create Arrays for the plot
materials = ['HMN', 'LGC', 'HD','BHD']

x_pos = np.arange(len(materials))

###Advertisement###############
# CTEs = [12.78,14.03,14.41,5.94]
# error = [0.39,0.17,0.13,0.18]

# ###Boston Housing###############
# CTEs = [22.94,23.64,23.50,11.55]
# error = [0.38,0.20,0.23,0.33]

# ###Parkinson###############
# CTEs = [10.68,28.30,29.21,10.61]
# error = [0.054,0.087,0.069,0.031]

# ####White Wine###############
# CTEs = [2.64,9.62,9.96,1.52]
# error = [0.093,0.0163,0.043,0.09]

# ####Red Wine###############
# CTEs = [2.50,9.55,9.86,1.59]
# error = [0.017,0.017,0.011,0.013]

####Airfoil###############
CTEs = [66.84,114.29,118.22,11.33]
error = [2.64,0.12,0.091,0.16]




# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.9, ecolor='black', capsize=10)
ax.set_ylabel('RMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)

ax.set_title('Air Foil Self Noise')
ax.yaxis.grid(True)
plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
#plt.figure(figsize=(20,8))
plt.tight_layout()
plt.savefig('images/10.eps')
plt.show()
