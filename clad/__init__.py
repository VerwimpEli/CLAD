from clad.classification.cladc import *
from clad.classification.cladc_utils import get_matching_classification_set, get_cladc_domain_sets
from clad.detection.cladd import *
from clad.utils.test_cladc import test_cladc, AMCAtester
from clad.utils.meta import SODA_DOMAINS, SODA_CATEGORIES

try:
    from clad.detection.cladd_detectron import register_cladd_detectron
except ModuleNotFoundError:
    print("[INFO] No Detectron installation found, continuing without.")
