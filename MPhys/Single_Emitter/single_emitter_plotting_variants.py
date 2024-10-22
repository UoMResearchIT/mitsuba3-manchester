#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Load in data from different mitsuba variant tests
variants = ["llvm_mono", "llvm_ad_rgb"]  # , "cuda_mono", "cuda_ad_rgb"]

max_time = 0
timing_data = []
for variant in variants:
    data = np.loadtxt("csv/"+variant+"_stats_timing_for_n_photons.csv", delimiter=",")
    max_full = np.max([d[7] for d in data])
    if max_full > max_time:
        max_time = max_full
    timing_data.append([variant, data])

print(timing_data)

# Plot each variant for each column 
columns = ["load", "render", "load_and_render", "full_time"]

for ncol, column in enumerate(columns):
    fig = plt.figure()
    for n_var, variant in enumerate(variants):
        plt.plot([tn[0] for tn in timing_data[n_var][1]], [tn[2*ncol+1] for tn in timing_data[n_var][1]], label=column+' '+timing_data[n_var][0])
    plt.legend()
    plt.ylim(1e-3, max_time+(0.1*max_time))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('time (s)')
    plt.xlabel('Number of photons')
    plt.title('timing vs n_photons, mitsuba3 variants ('+column+')')
    plt.savefig('png/'+column+'_timing_for_n_photons')
