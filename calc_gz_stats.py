#!/usr/bin/env python
u"""
calc_gz_stats.py
by Yara Mohajerani

Comparison stats of expected and delineated
GZ widths

Last Update 01/2021
"""
import os
import pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

infile = os.path.join(pathlib.Path.home(),'GL_learning_data',\
	'6d_results','GZ_widths-hybrid_Getz_comparison.csv')
#-- read data
df = pd.read_csv(infile)
ml = np.array(df['ML_WIDTH(m)'])/1e3
he = np.array(df['HE_WIDTH(m)'])/1e3
ra = np.array(df['ML/HE ratio'])
print(len(ra))
print(np.std(ra))
print("mean and std of ratios before filtering {0:.2f} +/- {1:.2f}".format(np.mean(ra),np.std(ra)/np.sqrt(len(ra))))

#-- remove non-valid elements
# ii = np.where((he<0.1) | (he>8))
ii = np.where((he<0.1) | (he>20))
ml[ii] = np.nan
he[ii] = np.nan
ra[ii] = np.nan

print("Mean Ratio (first 14) {0:.2f}".format(np.nanmean(ra[:14])))
print("Mean Ratio (ALL) {0:.2f}".format(np.nanmean(ra)))
jj = np.where(ml < 10)
print("Mean without Berry Glacier {0:.2f}".format(np.nanmean(ra[jj])))

#-- get the uncertainty in the mean ratio
N = np.count_nonzero(~np.isnan(ra))
# ra_err = np.nanstd(ra)/np.sqrt(N)
ra_err = (np.nanmax(ra)-np.nanmin(ra))/(2*np.sqrt(N))

#-- make histogram
fig,ax = plt.subplots(1,1,figsize=(6,5))
ax.hist(np.array([ml,he]).transpose(),bins=100,color=['green','darkorange'],label=['ML','HE'])
ax.legend(prop={'size': 18})
ax.set_xlim([0,8])
ax.set_xlabel("GZ Width Bins (km)",fontsize=18)
ax.set_ylabel('Number of Transects',fontsize=18)
ax.text(0.25,0.5,r"$R_{mean}^{ML:HE} = $ %.1f $\pm$ %.1f"%(np.nanmean(ra),ra_err),transform=fig.transFigure,fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.savefig(infile.replace('.csv','_distribution.pdf'),format='PDF')
plt.close(fig)

#-- make scatter plot
fig = plt.figure(1,figsize=(8,5))
sc = plt.scatter(ml,he,c=ra,s=20, vmin=0,vmax=100, cmap='viridis',alpha=0.7)
cbar = plt.colorbar(sc)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel('ML:HE width ratio', rotation=270,fontweight='bold',fontsize=18)
cbar.ax.tick_params(labelsize=14)
#-- regression 
x = sm.add_constant(ml)
model = sm.OLS(he,x,missing='drop')
fit = model.fit()
#-- get trend and uncertainty
tr = fit.params[1]
er = fit.bse[1]
#-- plot 45-degree line
l = np.arange(np.nanmax(ml))
# plt.plot(l,l,color='darkred',linestyle='--',linewidth=0.5,label=r"$x=y$ line")
#-- plot trend line
pred = fit.params[0] + l*tr
plt.plot(l,pred,color='red',linestyle='-',linewidth=1.2,\
	label=r"Fitted $HE:ML$ Ratio"+"\n"+r"%.1e $\pm$ %.1e"%(tr,er))
plt.legend(fontsize=18)
plt.xlabel("ML widths (km)",fontsize=18)
plt.ylabel('HE widths (km)',fontsize=18)
plt.tick_params(labelsize=14)
plt.xticks([0,5,10,15,20])
plt.yticks([0,5,10,15,20])
plt.savefig(infile.replace('.csv','_regression.pdf'),format='PDF')
plt.close(fig)