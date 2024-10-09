import pandas as pd

photon_detected = pd.read_csv('test_new_photons_detected_spectral.csv')

# There were just under 10^6 elements in this dataset, so to get to 10^8, concatenate 110 of these together...

photon_detected_large = pd.concat([photon_detected for _ in range(110)])

photon_detected_large.to_csv('test_new_photons_detected_spectral_100000000.csv')
