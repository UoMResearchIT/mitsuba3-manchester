#!/usr/bin/env python
# coding: utf-8

import random
import pandas as pd

# Takes filtered photons output (e.g. created via Geant4 / additional methods)
# and converts to photon data that can be read into Mitsuba3

nm_per_ev_constant = (float(6.6260715e-34)*float(3.00e8)*float(1e9))/(float(1.6021e-19)*float(1e6))

#Values for linear interpolation.
wavelength_steps = [100, 200, 230, 270, 300, 330, 370, 400, 430, 470, 500, 530, 570, 600, 630, 670, 700, 1000]
qe_steps = [0, 0, 0.02, 0.20, 0.31, 0.35, 0.35, 0.33, 0.31, 0.24, 0.18, 0.08, 0.05, 0.02, 0.01, 0.002, 0, 0]
rows_to_drop =[]

#load data from original G4 output (csv format)
column_names = ["time (ps)", "x", "y", "z", "px", "py", "pz", "E (MeV)"]
photon_data_full = pd.read_csv('./csv/Photons_1000000_filtered.csv', names = column_names)
initial_number_of_photons = photon_data_full.count()[0]

def ev_to_nm (energy):
    return nm_per_ev_constant/energy

#gives linear relation between two wavelegths thanks to two efficiency values
def linear_interp (j,w_input):
    a = (qe_steps[j] - qe_steps[j-1])/(wavelength_steps[j] - wavelength_steps[j-1])
    b = qe_steps[j-1] - a*wavelength_steps[j-1]
    return a*w_input+b

#Convert energy column to wavelength in nm.
photon_data_full['E (MeV)'] = photon_data_full['E (MeV)'].apply(ev_to_nm)
photon_data_full.rename(columns={"E (MeV)": "Wavelength (nm)"}, inplace = True)

#iterate over each photon.
for i in range(initial_number_of_photons):
    for j in range(len(wavelength_steps)):
        #find index in wavelength_steps corresponding to the photon.
        if photon_data_full.loc[i, 'Wavelength (nm)']<wavelength_steps[j]:
            break
    #Use linear interpolation to calculate QE at this wavelength.
    qe_estimated = linear_interp(j, photon_data_full.loc[i, 'Wavelength (nm)']) 
    x = random.uniform(0, 1)
    if x > qe_estimated:
        rows_to_drop.append(i)

#generate new frame only with photons that will be detected.
photon_detected = photon_data_full.drop(rows_to_drop)
#I don't know why a first column is created and keeps track of the old index of photons.
photon_detected.reset_index(drop=True, inplace= True)

#Can be removed later. Outputs characteristics of saved photons.
photon_detected.to_csv('./csv/test_new_photons_detected_spectral.csv' )
final_number_of_photons = photon_detected.count()[0]
fraction_detected = final_number_of_photons/initial_number_of_photons
print (photon_detected)
print ("{0:.4} of emitted photon ({1} of {2}) in G4 are actually detected.".format(
    fraction_detected, final_number_of_photons, initial_number_of_photons))