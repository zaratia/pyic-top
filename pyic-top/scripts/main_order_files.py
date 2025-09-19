import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import sort_df_by_idba_seq

infold = (
    "D:/25_ARFFS/59_progetti/02_arffs/"
    "f06_ARFFS_TEXT_gfortran/topmelt_ichymod_ics/"
    "PY-TOP_V1.0/INPUT/parameters"
)
fname = "parameters_TOPMELT.csv"

basins = pd.read_csv(
    "D:/25_ARFFS/59_progetti/02_arffs/"
    "06_ARFFS_TEXT_gfortran/topmelt_ichymod_ics/"
    "PY-TOP_V1.0/INPUT/topological_elements/basins.txt"
)
basin_id = np.array(basins.idba)

df = pd.read_csv(os.path.join(infold, fname), skipinitialspace=True)

# df = sort_df_by_idba_time_seq(df, basin_id)
df = sort_df_by_idba_seq(df, basin_id)

df.to_csv(fname, index=False)
