import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Absorbance Spectra Accessed Via HITRAN (cite both the data source and the HITRAN database)

def normalise_column(array):
    max_value = np.max(array)
    print(max_value)
    return array / max_value

def convert_wavenumber_to_wavelength(wavenumber):
    return 1/wavenumber * 10000000

convert_wn_to_wl_array = np.vectorize(convert_wavenumber_to_wavelength)

solar_irr_data = pd.read_excel("C:/Users/ggp24ash/Documents/Spectra/ChanceandKurucz2010SolarIrradiance.xlsx")
so2_data = pd.read_excel("C:/Users/ggp24ash/Documents/Spectra/Hermansetal.2009 SO2 AbsoprtionViaHITRAN298K.xlsx")
ozone_data = pd.read_excel("C:/Users/ggp24ash/Documents/Spectra/BirkandWagner2018 OzoneAbsorptionviaHITRAN273K.xlsx")


solar_wavelengths = np.array(solar_irr_data["Wavelength (nm)"])
irradiances = np.array(solar_irr_data["Watts m-2 nm-1"])
rel_indexes = np.where(np.logical_and(solar_wavelengths >= 240, solar_wavelengths<=340))
solar_wavelengths = solar_wavelengths[rel_indexes]
irradiances = irradiances[rel_indexes]
chosen_max = np.max(irradiances)
irradiances = irradiances/chosen_max

so2_absorbance = so2_data["Cross Section cm2molecule-1"].to_numpy()
so2_absorbance = normalise_column(so2_absorbance)
so2_wavenumbers = np.linspace(23995, 43985, num=so2_absorbance.shape[0])
so2_wavelengths = convert_wn_to_wl_array(so2_wavenumbers)

ozone_absorbance = ozone_data["Cross Section (cm2molecule-1)"].to_numpy()
ozone_absorbance = normalise_column(ozone_absorbance)
ozone_wavenumbers = np.linspace(28901, 40999, num=ozone_absorbance.shape[0])
ozone_wavelengths = convert_wn_to_wl_array(ozone_wavenumbers)

cm = 1 / 2.54
fig, axs = plt.subplots(figsize=(18*cm, 12*cm))
#Boundary of x-ray
axs.axvline(x=0.01, color="slategray")
#Boundaries of UV
axs.axvline(x=10, color="slategray")
#Boundaries of visible
axs.axvline(x=400, color="slategray")
axs.axvline(x=700, color="slategray")
#Boundary of IR

axs.plot(solar_wavelengths, irradiances, color ="gold", label="Solar Irradiance")
axs.plot(so2_wavelengths, so2_absorbance, color="orange", label=r"$\text{SO}_2$ Absorption")
axs.plot(ozone_wavelengths, ozone_absorbance, color="steelblue", label=r"$\text{O}_3$ Absorption")
leg = axs.legend(loc='upper right')
axs.set_xlim(240, 340)
axs.set_ylim(0, 1.3)
axs.set_ylabel("Normalised cross section", fontsize=14)
axs.set_xlabel("Wavelength (nm)", fontsize=14)
#plt.show()
plt.savefig("C:/Users/ggp24ash/Documents/ThesisFigures/SolarAndAbsorbanceCrossSections.jpg", dpi=300)