import xml.etree.ElementTree as ET
import glob
from write_masspec import mass_spectra_to_mgf
def parse_xml_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    databaseid = root.find("./database-id").text
    intrumenttype = root.find("./instrument-type").text
    peaks = root.find("./c-ms-peaks")
    mass_values = []
    intensity_values = []
    for i in range(len(peaks)):
        mv = peaks[i][2]
        iv = peaks[i][3]
        mass_values.append(float(mv.text))
        intensity_values.append(float(iv.text))
    return mass_values, intensity_values, (databaseid, intrumenttype)

def xml_lib_to_gmf(path, lib_filename, filter_by_instrument_type = None):
    files = glob.glob(path + "/HMDB*")
    mass_values_list = []
    intensity_values_list = []
    meta_list = []
    for file in files:
        mass_values, intensity_values, meta = parse_xml_file(file.replace("\\", "/"))
        mass_values_list.append(mass_values)
        intensity_values_list.append(intensity_values)
        meta_list.append(meta)
    mass_spectra_to_mgf(lib_filename, mass_values_list, intensity_values_list, meta_list, filter_by_instrument_type)

xml_lib_to_gmf("./hmdb/", "./lib_EIB.mgf", "EI-B")

#mass_values, intensity_values = parse_xml_file('./hmdb/HMDB0000001_c_ms_spectrum_1469_experimental.xml')
