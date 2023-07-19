import os
import re
import glob

def mass_spectra_to_mgf(filename, mass_values_list, intensity_values_list, meta_list = None, filter_by_instrument_type = None, names=None, casnos=None, hmdb_ids=None):
    r"""Write mass spectra in Mascot format file.

    Parameters
    ----------
    filename :
        New file filename.
    mass_values_list :
        Array of spectra mass values
    intensity_values_list :
        Array of spectra intensities
    meta_list : optional
        Array of metadata for each spectrum
    filter_by_instrument_type : optional
        Skip spectrum if spectrum is different from instrument filter_by_instrument_type
    names : optional
        List of names for each spectrum metadata
    casnos : optional
        List of casnos for each spectrum metadata
    hmdb_ids : optional
        List of hmdb_ids for each spectrum metadata
    """
    with open(filename, "w", encoding="utf-8") as mgf_file:
        for i in range(len(mass_values_list)):
            if (meta_list):
                databaseid, intrumenttype = meta_list[i]
                if (filter_by_instrument_type and filter_by_instrument_type != intrumenttype):
                    continue
                mgf_file.write("BEGIN IONS\n")
                mgf_file.write("DATABASEID=" + databaseid + "\n")
                mgf_file.write("INSTRUMENT=" + intrumenttype + "\n")
            else:
                mgf_file.write("BEGIN IONS\n")
            if (names != None):
                mgf_file.write("NAME=" + names[i] + "\n")
            if (casnos != None):
                mgf_file.write("CASNO=" + casnos[i] + "\n")
            if (hmdb_ids != None):
                mgf_file.write("DATABASEID=" + hmdb_ids[i] + "\n")

            for j in range(len(mass_values_list[i])):
                mass, intensity = mass_values_list[i][j], intensity_values_list[i][j]
                mgf_file.write(str(mass) + " " + str(intensity) + "\n")
            mgf_file.write("END IONS\n")

def nist_msl_to_mgf(filename, new_filename=None):
    r"""Convert NIST MSL file to Mascot file format.

    Parameters
    ----------
    filename :
        NIST MSL file filename.
    filename_mgf : optional
        New Mascot file format filename.
    """
    with open(filename, "r") as f:
        data = f.read()
    compounds = data.split("NAME:")[1:]
    names = []
    casnos = []
    masses_array = []
    intensities_array = []
    for compound in compounds[:]:
        fields = compound.split("\n")
        #Skip the space at the start
        name = fields[0][1:]
        casno = None
        num_peaks = 0
        masses = []
        intensities = []
        for field in fields[1:]:
            if ('CONTRIB' in field or 'FORM' in field or 'NIST' in field or 'COMMENTS' in field):
                continue
            elif ('CASNO' in field):
                casno = field[len('CASNO') + 1:]
            elif ('NUM PEAKS' in field):
                num_peaks = int(field[len('NUM PEAKS') + 1:])
            else:
                #mass - values
                mass_int_values = re.findall(r'[0-9]+', field)
                for i in range(len(mass_int_values)):
                    if (i % 2 == 0):
                        masses.append(mass_int_values[i])
                    else:
                        intensities.append(mass_int_values[i])
        if (len(masses) != num_peaks):
            print("Error: [" + name + "]" + num_peaks + "_ != num_masses " + str(len(masses)))
            continue
        if (len(masses) != num_peaks):
            print("Error: [" + name + "]" + num_peaks + " != num_intensities " + str(len(intensities)))
            continue
        names.append(name)
        casnos.append(casno)
        masses_array.append(masses)
        intensities_array.append(intensities)
    if (new_filename == None):
        new_filename = filename[:-3 ] + "mgf"
        print(new_filename)
    mass_spectra_to_mgf(new_filename, masses_array, intensities_array, names=names, casnos=casnos)

def MSP_files_to_MGF_file(path, filename_mgf=None):
    r"""Create Mascot format file from multiple MSP files.

    Parameters
    ----------
    path :
        Path to MSP files.
    filename_mgf : optional
        New Mascot file format filename
    """
    msp_filenames = glob.glob(path + "/*")
    names = []
    casnos = []
    masses_array = []
    intensities_array = []
    for msp_filename in msp_filenames:
        with open(msp_filename.replace("\\", "/"), "r") as input:
            lines = input.readlines()
        name = None
        casno = None
        nb_peaks = 0
        masses = []
        intensities = []
        for i in range(len(lines)):
            line = lines[i]
            if ('Name: ' in line):
                name = line[len('Name: '):-1]
                continue
            elif ('CAS#: ' in line):
                split = line.split(";")
                casno = split[0][len('CAS#: '):]
                continue
            elif ('Num Peaks: ' in line):
                nb_peaks = int(line[len('Num Peaks: '):-1])
            elif (line == "\n"):
                continue
            elif (len(re.findall(r'[a-zA-Z]+', line)) == 0):
                mass_int_values = re.findall(r'[0-9]+', line)
                for j in range(len(mass_int_values)):
                    if (j % 2 == 0):
                        masses.append(mass_int_values[j])
                    else:
                        intensities.append(mass_int_values[j])
        if (len(masses) != nb_peaks):
            print("Error: [" + name + "]" + nb_peaks + "_ != num_masses " + str(len(masses)))
            continue
        if (len(masses) != nb_peaks):
            print("Error: [" + name + "]" + nb_peaks + " != num_intensities " + str(len(intensities)))
            continue
        names.append(name)
        casnos.append(casno)
        masses_array.append(masses)
        intensities_array.append(intensities)
    if (filename_mgf == None):
        filename_mgf = path + "/nist_msp_to_mgf.mgf"
    mass_spectra_to_mgf(filename_mgf, masses_array, intensities_array, names=names, casnos=casnos)

#MSP_files_to_MGF_file('./data/NIST_MSP/', './data/NIST_from_mssearch.mgf')
#nist_msl_to_gmf('./data/NISTFF.MSL')