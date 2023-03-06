import numpy as np
import analysis_functions_vicsek_nd as fun
import matplotlib.pyplot as plt
import gzip
import shutil


with open('coupling.txt', 'rb') as f_in, gzip.open('coupling.txt.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)