from matplotlib import pyplot as plt

_, axes = plt.subplots(3, 7,figsize=(3.5,4.5))
axes = axes.flatten()
# axes[0].imshow()
axes[0].axes.get_xaxis().set_visible(False)
plt.show()