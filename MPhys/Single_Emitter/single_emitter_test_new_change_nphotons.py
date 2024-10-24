#!/usr/bin/env python
# coding: utf-8

import sys
import mitsuba as mi
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use agg backend for matplotlib since we're just writing to files
# Note: in WSL2 with python3-tk installed, allowing matplotlib to select automatically
#       (I assume it picks 'TkAgg' in this instance) causes a segmentation fault!
matplotlib.use('agg')

# Tell user to supply mitsuba variant if they haven't
if len(sys.argv) != 3:
    sys.exit("Usage: python single_emitter_test_new_change_nphotons.py variant n_repeats \n"
             "(where n_repeats is the number of times the csv data file is repeated)")

mi.set_variant(sys.argv[1])
# The number of repeats my local machine can cope with (using 16GB RAM) is somewhere around 20
# In order to get to 10^8 photons then n_repeats needs to be approximately 108 
# (but you can check when running as to how many photons this gives you in the long run)
n_repeats = int(sys.argv[2])
# mi.set_variant('cuda_mono')
# mi.set_variant('llvm_mono')
# mi.set_variant('llvm_ad_rgb')
print(f"{mi.variant()} with {n_repeats} repeats")
mi.variants()

# First section is only necessary to run if the photon_detected CSV file does not yet exist ?

# nm_per_ev_constant = (float(6.6260715e-34)*float(3.00e8)*float(1e9))/(float(1.6021e-19)*float(1e6))

# #Values for linear interpolation.
# wavelength_steps = [100, 200, 230, 270, 300, 330, 370, 400, 430, 470, 500, 530, 570, 600, 630, 670, 700, 1000]
# qe_steps = [0, 0, 0.02, 0.20, 0.31, 0.35, 0.35, 0.33, 0.31, 0.24, 0.18, 0.08, 0.05, 0.02, 0.01, 0.002, 0, 0]
# rows_to_drop =[]

# #load data from original G4 output (csv format)
# column_names = ["time (ps)", "x", "y", "z", "px", "py", "pz", "E (MeV)"]
# photon_data_full = pd.read_csv('./csv/Photons_1000000_filtered.csv', names = column_names)
# initial_number_of_photons = photon_data_full.count()[0]

# def ev_to_nm (energy):
#     return nm_per_ev_constant/energy

# #gives linear relation between two wavelegths thanks to two efficiency values
# def linear_interp (j,w_input):
#     a = (qe_steps[j] - qe_steps[j-1])/(wavelength_steps[j] - wavelength_steps[j-1])
#     b = qe_steps[j-1] - a*wavelength_steps[j-1]
#     return a*w_input+b

# #Convert energy column to wavelength in nm.
# photon_data_full['E (MeV)'] = photon_data_full['E (MeV)'].apply(ev_to_nm)
# photon_data_full.rename(columns={"E (MeV)": "Wavelength (nm)"}, inplace = True)

# #iterate over each photon.
# for i in range(initial_number_of_photons):
#     for j in range(len(wavelength_steps)):
#         #find index in wavelength_steps corresponding to the photon.
#         if photon_data_full.loc[i, 'Wavelength (nm)']<wavelength_steps[j]:
#             break
#     #Use linear interpolation to calculate QE at this wavelength.
#     qe_estimated = linear_interp(j, photon_data_full.loc[i, 'Wavelength (nm)']) 
#     x = random.uniform(0, 1)
#     if x > qe_estimated:
#         rows_to_drop.append(i)

# #generate new frame only with photons that will be detected.
# photon_detected = photon_data_full.drop(rows_to_drop)
# #I don't know why a first column is created and keeps track of the old index of photons.
# photon_detected.reset_index(drop=True, inplace= True)

# #Can be removed later. Outputs characteristics of saved photons.
# photon_detected.to_csv('./csv/test_new_photons_detected_spectral.csv' )
# final_number_of_photons = photon_detected.count()[0]
# fraction_detected = final_number_of_photons/initial_number_of_photons
# print (photon_detected)
# print ("{0:.4} of emitted photon ({1} of {2}) in G4 are actually detected.".format(
#     fraction_detected, final_number_of_photons, initial_number_of_photons))

# Load in the CSV file
# column_names = ["time (ps)", "x", "y", "z", "px", "py", "pz", "E (MeV)"]
# The first column in the CSV file is an index column and we don't want to load that in
photon_detected_load = pd.read_csv('csv/test_new_photons_detected_spectral.csv', index_col=0) #, names = column_names)
# photon_detected_load.reset_index(drop=True, inplace=True)
# The number of photons actually in this file is somewhere just under 10^6
photon_detected = pd.concat([photon_detected_load for _ in range(n_repeats)], ignore_index=True)

print(photon_detected_load)
print(photon_detected)

# photon_detected = pd.read_csv('csv/test_new_photons_detected_spectral.csv')

def generate_emitter_data(photon_data):
    """
    Generates the data for the photon_emitter plugin of Mitsuba using the photon data
    """
    x_position, y_position, z_position = photon_data.values[:, 1:4].T
    x_momentum, y_momentum, z_momentum = photon_data.values[:, 4:7].T
    # calculate the target coordinates of the photons
    x_target = x_position + x_momentum
    y_target = y_position + y_momentum
    z_target = z_position + z_momentum
    # combine them into a single array
    # emitter_data = np.column_stack((x_position, y_position, z_position, x_target, y_target, z_target)).flatten()
    # The y- and z- positions in other examples were inverted for some reason... ?
    emitter_data = np.column_stack((x_position, z_position, y_position, x_target, z_target, y_target)).flatten()
    emitter_data = np.insert(emitter_data, 0, len(x_position))
    # create a 3D array of the emitter data
    result = np.zeros((1, 1, len(emitter_data)), dtype=np.float32)
    result[0, 0, :] = emitter_data
    return result

# gen_photon_datas = []
photon_lists = []

# Loop to generate photon data from the photon_detected data frame

for n_photons in [10**n_size for n_size in range(9)]:
    # If the size is bigger than the whole dataset then pandas is sensible and just gives you the whole dataset
    gen_photon_data = generate_emitter_data(photon_detected.head(n_photons))
    photon_list = mi.VolumeGrid(gen_photon_data)
    print("--------------------------------")
    photon_lists.append(photon_list)
    print (photon_list)
    print (np.array(photon_list))
    print (gen_photon_data)
    print ("n_photons into mitsuba ", (len(np.array(photon_list)[0][0]) - 1) // 6)

# Do it for the whole dataset too
gen_photon_data = generate_emitter_data(photon_detected)
photon_list = mi.VolumeGrid(gen_photon_data)
photon_lists.append(photon_list)

print("--------------------------------")
print (photon_list)
print (np.array(photon_list))
print (gen_photon_data)

print ("n_photons into mitsuba ", (len(np.array(photon_list)[0][0]) - 1) // 6)

# sleep before experiments, just to see if first experiment is being affected by start?
print("sleeping...")
time.sleep(10)


def run_experiment(photon_list):
    
    start_time = time.time()
    # scene = mi.load_file("./xml/real_geometry_int10000.xml")
    
    # Set up the scene description
    scene_description = {
        'type': 'scene',
    
        'integrator': {
            'type': 'ptracer_c',  # ptracer?
            'max_depth': 50,
            'hide_emitters': False,
        },
    
        'sensor': {
            'type': 'perspective',
            'fov': 40,
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, 1100, 950],
                                                     target=[0, 1500, 1500],
                                                     up=[0, 0, 1]),
            'sampler': {
                'type': 'independent',
                'sample_count': 1,
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 1024,
                'file_format': 'openexr',
                'pixel_format': 'luminance',
                'component_format': 'uint32',
                'filter': {
                    'type': 'tent',
                },
            },      
        },
    
        'MirrorBSDF': {
            'type': 'twosided',
            'bsdf_id': {
                'type': 'conductor',
                'material': 'none',
            },
        },
    
        # 'RoughMirrorBSDF': {
        #     'type': 'conductor',
        #     'material': 'none',
        #     'alpha': 0.01,
        # },
    
        'test': {
            'type': 'twosided',
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [1, 1, 1],
                },
            },
        },
    
        'test2': {
            'type': 'twosided',
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.9, 0.5, 0.2],
                },
            },
        },
    
        'spherical_mirror': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 2000, 355], target=[0, 1000, 600], up=[0, 0, 1]).scale([1500, 650, 33]),
            'bsdf_id': {
                'type': 'ref',
                'id': 'MirrorBSDF',
            },
        },
    
        'flat_mirror': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1000, 710], target=[0, 2000, 900], up=[0, 0, 1]).scale([740, 440, 0.1]),
            'bsdf_id': {
                'type': 'ref',
                'id': 'MirrorBSDF',
            },
        },
    
        'detector': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1500, 1120], target=[0, 1000, 710], up=[0, 0, 1]).scale([1000, 500, 0.5]),
            'bsdf_id': {
                'type': 'ref',
                'id': 'test',
            },        
        },
    
        'backwall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[-1000, 1500, 500], target=[0, 1500, 500], up=[0, 1, 0]).scale([3000, 3000, 1]),
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.4, 1, 0.2],
                },
            },        
        },
            
        'ceiling': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1500, 3000], target=[0, 1500, 0], up=[0, 1, 0]).scale([3000, 3000, 1]),
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.8, 0.3, 0.45],
                },
            },        
        },
            
        'leftwall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 500], target=[0, 1500, 500], up=[0, 0, 1]).scale([3000, 3000, 3000]),
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.32, 0.46, 0.23],
                },
            },        
        },
            
        'rightwall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 3000, 500], target=[0, 1500, 500], up=[0, 0, 1]).scale([3000, 3000, 3000]),
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.92, 0.58, 1],
                },
            },        
        },
    
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1500, -1000], target=[0, 1500, 3000], up=[0, 1, 0]).scale([3000, 3000, 3000]),
            'bsdf_id': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.56, 0.23, 0.54],
                },
            },        
        },
    
        'photons': {
            'type': 'photon_emitter',
            'photon_list': photon_list,
            'intensity': 1000.0,
        },        
    }
    
    # Timing for dictionary creation
    create_dict_mark = time.time()
    create_dict_time = create_dict_mark - start_time
    print(f"Create dict time: {create_dict_time:.3f} seconds")

    scene = mi.load_dict(scene_description)

    # Timing for loading
    load_time_mark = time.time()
    load_time = load_time_mark - create_dict_mark
    print(f"Load time: {load_time:.3f} seconds")
    
    original_image = mi.render(scene)

    # Timing for render
    render_time_mark = time.time()
    render_time = render_time_mark - load_time_mark

    print(f"Render time: {render_time:.2f} seconds")
    print(f"Load / render time: {load_time+render_time:.3f} seconds")
    # print(original_image)
    fig = plt.figure(figsize = (20,20))
    plt.axis('off')
    plt.imshow(original_image ** (1.0 / 2.2)); 

    # Full timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Full elapsed time: {elapsed_time:.3f} seconds")
    
    # There are 6 columns in each of the lists so divide by this to get the number of photons
    n_photons = (len(np.array(photon_list)[0][0]) - 1) // 6
    print(mi.variant(), "intensity = 1000, n photons = ", n_photons)
    plt.savefig('png/' + mi.variant() + ' new intensity = 1000, n photons = ' + str(n_photons))
    plt.close(fig)

    return (n_photons, create_dict_time, load_time, render_time, elapsed_time)

def compare_images(old_fname, new_fname):
    old_image = plt.imread(old_fname)
    new_image = plt.imread(new_fname)
    diff_image = old_image - new_image

    print(type(diff_image))
    print(np.max(diff_image), np.min(diff_image))
    print(diff_image.shape)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_figheight(30)
    fig.set_figwidth(10)
    ax1.set_title('original image (from loaded XML) ' + old_fname)
    ax1.imshow(old_image)
    ax2.set_title('new image (from loaded vector via dict) ' + new_fname)
    ax2.imshow(new_image)
    ax3.set_title('difference image')
    ax3.imshow(diff_image - np.min(diff_image))
    
    print("diff image")
    plt.savefig('png/diff image ' + new_fname[4:-4])
    plt.close(fig)

# Loop to run experiments and compare to old images
full_timing_vs_nphotons = []

for photon_list in photon_lists:
    # Run the experiments multiple times (take averages in plotting script)
    n_exps = 11
    for n in range(n_exps):
        (n_photons, create_dict_time, load_time, render_time, elapsed_time) = run_experiment(photon_list)
        # The first run is often longer than any other, so don't count it
        if n!=0:
            full_timing_vs_nphotons.append([n_photons, create_dict_time, load_time, render_time, load_time + render_time, elapsed_time])

    # Compare this image to the previous one
    old_fname = 'png/intensity = 1000.png'
    new_fname = 'png/' + mi.variant() + ' new intensity = 1000, n photons = '+str((len(np.array(photon_list)[0][0]) - 1) // 6)+'.png'

    compare_images(old_fname, new_fname)

print("-------- timings --------")
for n in range(len(full_timing_vs_nphotons)):
    print(full_timing_vs_nphotons[n])

# Save timing data to CSV files
np.savetxt('csv/' + mi.variant()+'_full_timing_for_n_photons.csv', np.array(full_timing_vs_nphotons), delimiter=',')

# fig = plt.figure()
# plt.plot([tn[0] for tn in average_timing_vs_nphotons], [tn[1] for tn in average_timing_vs_nphotons], label='load')
# plt.plot([tn[0] for tn in average_timing_vs_nphotons], [tn[2] for tn in average_timing_vs_nphotons], label='load and render')
# plt.plot([tn[0] for tn in average_timing_vs_nphotons], [tn[3] for tn in average_timing_vs_nphotons], label='full elapsed')
# plt.plot([tn[0] for tn in average_timing_vs_nphotons], [tn[2]-tn[1] for tn in average_timing_vs_nphotons], label='render')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('time (s)')
# plt.xlabel('Number of photons')
# plt.title('timing vs n_photons, mitsuba3 single photon emitter (' + mi.variant() + ')')
# plt.savefig('png/' + mi.variant()+'_timing_for_n_photons')
# plt.close(fig)
