import sys
import logging

from matchms.importing import load_from_mgf, scores_from_json
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from utils import generate_lib_scores_from_lib

DEFAULT_LIB_FILENAME="./lib_EIB.mgf"

if __name__ == "__main__":
    if (len(sys.argv[1]) == 2):
        generate_lib_scores_from_lib(sys.argv[1])
    else:
        print("lib filename: ", DEFAULT_LIB_FILENAME)
        generate_lib_scores_from_lib(DEFAULT_LIB_FILENAME)
