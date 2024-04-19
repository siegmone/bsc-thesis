from utils import get_impedance_data
from iminuit import Minuit
from iminuit.cost import LeastSquares
from models import R_RC_RC


model_func = R_RC_RC().func_flat




