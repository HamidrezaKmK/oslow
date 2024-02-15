# codes are kindly borrowed from
# https://github.com/AlexImmer/loci
import os
from source.methods.lsnm.lsnm import LSNM

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = "/".join(PACKAGE_DIR.split("/")[:-1])
DATA_DIR = ROOT + "/data"
