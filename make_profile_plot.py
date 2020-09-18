u"""
make_profile_plot.py
Yara Mohajerani
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#-- Directory setup
base_dir = os.path.expanduser('~/GL_learning_data/')


#-- Set up figure
fig = plt.figure(1,figsize=(7,5))
ax1 = fig.add_subplot(111)
#-- make right hand axis
ax2 = ax1.twinx() 
#-- make seafloor
xf1 = np.arange(0,11,1)
xf2 = np.arange(10,30)
ax1.plot(xf1,-0.01*xf1-1.2,color='green')
ax1.plot(xf2,-0.005*xf2-1.25,color='green')
ax1.axvline(x=0,color='lightgray',linestyle='--')

#-- make x axis
x = np.arange(-20,30)

#-- make ice surface
#-- sigmoid for tidal motion
ax2.plot(x,1/(1+np.exp(-0.5*(-x+5)))-1,color='deepskyblue')
ax2.set_ylabel('Ice Tidal Motion',color='deepskyblue',fontweight='bold',fontsize=15)
ax1.set_ylabel('Ice Elevation',fontweight='bold',fontsize=15)
ax1.set_xlabel('Distance from Grounding Line (km)',fontweight='bold',fontsize=15)
ax1.set_ylim([-1.5,0.5])
ax2.set_ylim([-1.5,0.5])
ax1.get_xaxis().set_ticks([0,10])
ax1.get_yaxis().set_ticks([0])
ax2.get_yaxis().set_ticks([0])
plt.tight_layout()
plt.savefig(os.path.join(base_dir,'profile_plot.pdf'),format='PDF')
plt.close(fig)