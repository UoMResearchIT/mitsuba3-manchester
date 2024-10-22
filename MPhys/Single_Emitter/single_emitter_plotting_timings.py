#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    sys.exit("Usage: python single_emitter_plotting_timings.py variant \n")

variant = sys.argv[1]
timing_data = np.loadtxt("csv/"+variant+"_full_timing_for_n_photons.csv", delimiter=",")

print("----- full timing_data -----")
print(timing_data)

# Loop to run experiments and compare to old images
timing_vs_nphotons_dict = {}

for t_data in timing_data:
    [n_photons, load_time, render_time, load_and_render_time, elapsed_time] = t_data
    timing_vs_nphotons_dict[n_photons] = []

for t_data in timing_data:
    [n_photons, load_time, render_time, load_and_render_time, elapsed_time] = t_data
    timing_vs_nphotons_dict[n_photons].append([load_time, render_time, load_and_render_time, elapsed_time])

stats_timing_vs_nphotons_data = []
for key in timing_vs_nphotons_dict.keys():
    timing_vals = timing_vs_nphotons_dict[key]

    check_val = 0
    for timing in timing_vals:
        check_val += timing[0]
    load_avg = np.average([timing[0] for timing in timing_vals])
    print(key, load_avg, check_val)
    load_sd = np.std([timing[0] for timing in timing_vals])
    render_avg = np.average([timing[1] for timing in timing_vals])
    render_sd = np.std([timing[1] for timing in timing_vals])
    load_and_render_avg = np.average([timing[2] for timing in timing_vals])
    load_and_render_sd = np.std([timing[2] for timing in timing_vals])
    full_elapsed_avg = np.average([timing[3] for timing in timing_vals])
    full_elapsed_sd = np.std([timing[3] for timing in timing_vals])

    stats_timing_vs_nphotons_data.append([key, load_avg, load_sd, render_avg, render_sd,
                                          load_and_render_avg, load_and_render_sd, full_elapsed_avg, full_elapsed_sd])

print("-------- timings --------")
max_time = 0
for n in range(len(stats_timing_vs_nphotons_data)):
    print(stats_timing_vs_nphotons_data[n])
    max_full = stats_timing_vs_nphotons_data[n][7]
    if max_full > max_time:
        max_time = max_full

# Save timing data to CSV files
np.savetxt('csv/' + variant + '_stats_timing_for_n_photons.csv', np.array(stats_timing_vs_nphotons_data), delimiter=',')

fig = plt.figure()
plt.plot([tn[0] for tn in stats_timing_vs_nphotons_data], [tn[1] for tn in stats_timing_vs_nphotons_data], label='load')
plt.plot([tn[0] for tn in stats_timing_vs_nphotons_data], [tn[3] for tn in stats_timing_vs_nphotons_data], label='render')
plt.plot([tn[0] for tn in stats_timing_vs_nphotons_data], [tn[5] for tn in stats_timing_vs_nphotons_data], label='load_and_render')
plt.plot([tn[0] for tn in stats_timing_vs_nphotons_data], [tn[7] for tn in stats_timing_vs_nphotons_data], label='full_elapsed')
plt.legend()
plt.ylim(1e-3, max_time+(0.1*max_time))
plt.xscale('log')
plt.yscale('log')
plt.ylabel('time (s)')
plt.xlabel('Number of photons')
plt.title('timing vs n_photons, mitsuba3 single photon emitter (' + variant + ')')
plt.savefig('png/' + variant +'_timing_for_n_photons')
plt.close(fig)
