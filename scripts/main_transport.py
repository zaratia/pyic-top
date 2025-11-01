import json
import os

import numpy as np
import pandas as pd
from numba import njit

from pyic_top.ictop_utils import (
    init_basin_vars,
    init_junction_vars,
    init_qnodelink,
    init_reach_vars,
    init_reservoir_vars,
)
from pyic_top.module_mc import MC, MC_params


@njit
def transport_loop(
    basin_count,
    reservoir_count,
    reach_count,
    junction_count,
    nodelink_count,
    Qnodelink,
    n_sub_reaches,
    mod_seq,
    c1_mc,
    c2_mc,
    c3_mc,
    Qsubreach_in,
    Qsubreach_out,
    t,
):
    # transport loop over transport sequence
    for m in range(len(mod_seq)):
        if mod_seq[m] == 5:
            # basin, do nothing
            basin_count = basin_count + 1
            nodelink_count = nodelink_count + 1
        elif mod_seq[m] == 4:  # Reservoir
            # neutral model only, for now
            # turbined flow is 0
            Qnodelink[nodelink_count, t] = 0
            nodelink_count = nodelink_count + 1
            # All spilled flow, the upstream nodelink is the second last
            Qnodelink[nodelink_count, t] = Qnodelink[nodelink_count - 2, t]
            nodelink_count = nodelink_count + 1
            reservoir_count = reservoir_count + 1  # the reservoir is already computed!
        elif mod_seq[m] == 2:  # MC
            # Muskingum-Cunge
            Qsubreach_in, Qsubreach_out = MC(
                reach_count,
                Qupstream=Qnodelink[nodelink_count - 1, t],
                nstep=n_sub_reaches[reach_count],
                Qsubreach_in=Qsubreach_in,
                Qsubreach_out=Qsubreach_out,
                c1_mc=c1_mc[reach_count],
                c2_mc=c2_mc[reach_count],
                c3_mc=c3_mc[reach_count],
            )
            Qnodelink[nodelink_count, t] = Qsubreach_out[-1, reach_count]
            # next reach
            nodelink_count = nodelink_count + 1
            reach_count = reach_count + 1
        elif mod_seq[m] == 3:  # Junction
            # sum upstream flows
            Qnodelink[nodelink_count, t] = (
                Qnodelink[nodelink_count - 1, t] + Qnodelink[nodelink_count - 2, t]
            )
            nodelink_count = nodelink_count + 1
            # next junction
            junction_count = junction_count + 1
    return


def _read_init(cfg_path: str = "config.json") -> dict:
    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return {}


config_dir = _read_init("config.json")
init_info = _read_init("init.json")

# global variables
INPUT_FOLDER = os.path.join(
    config_dir["main_dir"],
    config_dir["input_dir"],
)
OUTPUT_FOLDER = os.path.join(
    config_dir["main_dir"],
    config_dir["output_dir"],
)
INITCOND_FOLDER = config_dir["initcond_dir"]
TOPOLOGICAL_ELEMENT_FOLDER = config_dir["topo_ele_dir"]
PARAMETER_FOLDER = config_dir["param_dir"]
EEB_FOLDER = config_dir["eeb_dir"]
TOPOLOGY_FOLDER = config_dir["topology_dir"]
METEO_FOLDER = config_dir["meteo_dir"]
TO_PDM_FOLDER = config_dir["to_pdm_dir"]
TO_TRNSPRT_FOLDER = config_dir["to_trnsprt_dir"]
START_TIME = init_info["start_time"]
END_TIME = init_info["end_time"]
AVG_LAT = init_info["average_lat"]
AVG_LON = init_info["average_lon"]
WE_THRESHOLD = init_info["sca_we_threshold"]
QBASE_TYPE = init_info["qbase_type"]
FLOAT_FORMAT_SM = "%.4f"

# first hour is initial condition

if __name__ == "__main__":
    start_time = pd.to_datetime(START_TIME, format="%Y-%m-%d %H:%M:%S") + pd.Timedelta(
        hours=1
    )
    end_time = pd.to_datetime(END_TIME, format="%Y-%m-%d %H:%M:%S")

    # count numer of simulated hours (first is IC)
    n_hours = int((end_time - start_time).total_seconds() / 3600 + 1)
    # build time array
    time_array = pd.date_range(start=start_time, end=end_time, freq="h")

    # read reaches list, reach id must be the model sequence
    df_reaches, reaches_id, n_reach, reach_in, reach_out, n_sub_reaches, n_step_max = (
        init_reach_vars(
            os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reaches.txt")
        )
    )

    # read nodelinks list
    # NOTE: nodelinks are not in sequence order!!!
    df_nodelinks = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "nodelinks.txt"),
        skipinitialspace=True,
    )

    # read transport sequence
    df_trans_seq = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "topseq.txt"),
        skipinitialspace=True,
    )
    # read element list
    df_elements = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "elements.txt"),
        skipinitialspace=True,
    ).set_index("idel")

    # read reaches params
    df_reach_params = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "parameters_MC.csv"),
        skipinitialspace=True,
    ).set_index("idre")
    # calc MC params
    df_reach_params["c1_mc"], df_reach_params["c2_mc"], df_reach_params["c3_mc"] = (
        MC_params(
            cel=df_reach_params["CEL"].to_numpy(),
            disp=df_reach_params["DISP"].to_numpy(),
            dx=df_reach_params["DX"].to_numpy(),
            dt=1.0 * 3600.0,  # time step in seconds
            n_reach=n_reach,
        )
    )
    c1_mc = np.asarray(df_reach_params["c1_mc"])
    c2_mc = np.asarray(df_reach_params["c2_mc"])
    c3_mc = np.asarray(df_reach_params["c3_mc"])

    # read basin list, basin_id must be the model sequence
    df_basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse, basin_node = (
        init_basin_vars(
            os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "basins.txt")
        )
    )

    # read reservoir list, reservoir_id must be the model sequence
    df_reservoirs, reservoir_id, n_reservoir = init_reservoir_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reservoirs.txt")
    )

    # read junction list, junction_id must be the model sequence
    df_junctions, junction_id, n_junction = init_junction_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "junctions.txt")
    )

    # All dataframe must respect the same sequence order of the reaches!!!
    # init (n_nodelinks, n_hours) real vars
    Qnodelink = np.full((len(df_nodelinks), n_hours + 1), -999.0)

    # ini last flows everywhere at time 0
    # NOTE: not in sequence order!!!
    Qnodelink[:, 0] = np.asarray(
        pd.read_csv(
            os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_last_flow.txt"),
            skipinitialspace=True,
        )["value"]
    )

    # init flows at nodelinks, for every time step but only at catchments
    Qnodelink = init_qnodelink(
        df_nodelinks,
        Qnodelink,
        os.path.join(INPUT_FOLDER, TO_TRNSPRT_FOLDER, "discharge.txt"),
        START_TIME,
        END_TIME,
    )

    # init (n_reach) real vars
    df_Qsubreach_in = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_MC_Qin.txt"),
        skipinitialspace=True,
    )
    df_Qsubreach_out = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_MC_Qout.txt"),
        skipinitialspace=True,
    )
    # create subreach vars. we use fillna because some reaches may have less subreaches
    # than the max number of subreaches
    Qsubreach_in = np.asarray(
        df_Qsubreach_in.pivot(index="n_sm", columns="idre", values="value").fillna(
            -999.0
        )
    )
    Qsubreach_out = np.asarray(
        df_Qsubreach_out.pivot(index="n_sm", columns="idre", values="value").fillna(
            -999.0
        )
    )

    # model sequence array
    mod_seq = np.asarray(df_trans_seq["idmo"])

    # loop over time steps excluded time 0
    for t in range(1, len(time_array) + 1, 1):
        if time_array.hour[t - 1] == 0:
            print(
                "Transport",
                time_array.year[t - 1],
                time_array.month[t - 1],
                time_array.day[t - 1],
            )
        # init
        basin_count = 0
        reservoir_count = 0
        reach_count = 0
        junction_count = 0
        nodelink_count = 0

        transport_loop(
            basin_count,
            reservoir_count,
            reach_count,
            junction_count,
            nodelink_count,
            Qnodelink,
            n_sub_reaches,
            mod_seq,
            c1_mc,
            c2_mc,
            c3_mc,
            Qsubreach_in,
            Qsubreach_out,
            t,
        )

    # just a little check
    print(Qnodelink[:, :].max())
