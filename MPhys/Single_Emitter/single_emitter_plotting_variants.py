#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load in data from different mitsuba variant tests
variants = ["llvm_mono", "llvm_ad_rgb", "cuda_mono", "cuda_ad_rgb"]

timing_data = []
for variant in variants:
    data = np.loadtxt("cluster_new_"+variant+"_tests/"+variant+"_timing_for_n_photons.csv", delimiter=",")
    timing_data.append([variant, data])

print(timing_data)


# In[3]:


# Plot each variant for each column 
columns = ["load", "load_and_render", "full_time"]

for ncol, column in enumerate(columns):
    fig = plt.figure()
    plt.plot([tn[0] for tn in timing_data[0][1]], [tn[ncol+1] for tn in timing_data[0][1]], label=column+' '+timing_data[0][0])
    plt.plot([tn[0] for tn in timing_data[1][1]], [tn[ncol+1] for tn in timing_data[1][1]], label=column+' '+timing_data[1][0])
    plt.plot([tn[0] for tn in timing_data[2][1]], [tn[ncol+1] for tn in timing_data[2][1]], label=column+' '+timing_data[2][0])
    plt.plot([tn[0] for tn in timing_data[3][1]], [tn[ncol+1] for tn in timing_data[3][1]], label=column+' '+timing_data[3][0])
    plt.legend()
    plt.xscale('log')
    plt.ylabel('time (s)')
    plt.xlabel('Number of photons')
    plt.title('timing vs n_photons, mitsuba3 single photon emitter variants ('+column+')')
    plt.savefig('png/'+column+'_timing_for_n_photons')

# Also plot just rendering time (column 2 minus column 1)
fig = plt.figure()
plt.plot([tn[0] for tn in timing_data[0][1]], [tn[2]-tn[1] for tn in timing_data[0][1]], label='render '+timing_data[0][0])
plt.plot([tn[0] for tn in timing_data[1][1]], [tn[2]-tn[1] for tn in timing_data[1][1]], label='render '+timing_data[1][0])
plt.plot([tn[0] for tn in timing_data[2][1]], [tn[2]-tn[1] for tn in timing_data[2][1]], label='render '+timing_data[2][0])
plt.plot([tn[0] for tn in timing_data[3][1]], [tn[2]-tn[1] for tn in timing_data[3][1]], label='render '+timing_data[3][0])
plt.legend()
plt.xscale('log')
plt.ylabel('time (s)')
plt.xlabel('Number of photons')
plt.title('timing vs n_photons, mitsuba3 single photon emitter variants (render)')
plt.savefig('png/render_timing_for_n_photons')


# In[ ]:




