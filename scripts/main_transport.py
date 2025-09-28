import json
import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_reach_vars, init_qnodelink, init_basin_vars, init_reservoir_vars,init_junction_vars
from pyic_top.module_mc import MC, MC_params


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
    config_dir['main_dir'],
    config_dir['input_dir'],
)
OUTPUT_FOLDER = os.path.join(
    config_dir['main_dir'],
    config_dir['output_dir'],
)
INITCOND_FOLDER = config_dir['initcond_dir']
TOPOLOGICAL_ELEMENT_FOLDER = config_dir['topo_ele_dir']
PARAMETER_FOLDER = config_dir['param_dir']
EEB_FOLDER = config_dir['eeb_dir']
TOPOLOGY_FOLDER = config_dir['topology_dir']
METEO_FOLDER = config_dir['meteo_dir']
TO_PDM_FOLDER = config_dir['to_pdm_dir']
TO_TRNSPRT_FOLDER = config_dir['to_trnsprt_dir']
START_TIME = init_info['start_time']
END_TIME = init_info['end_time']
AVG_LAT = init_info['average_lat']
AVG_LON = init_info['average_lon']
WE_THRESHOLD = init_info['sca_we_threshold']
QBASE_TYPE = init_info['qbase_type']
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
    df_reaches, reaches_id, n_reach, reach_in, reach_out, n_sub_reaches, n_step_max = init_reach_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reaches.txt")
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
    df_reach_params["c1_mc"], df_reach_params["c2_mc"], df_reach_params["c3_mc"] = MC_params(
        cel=df_reach_params["CEL"].to_numpy(),
        disp=df_reach_params["DISP"].to_numpy(),
        dx=df_reach_params["DX"].to_numpy(),
        dt=1.0 * 3600.0,  # time step in seconds
        n_reach=n_reach,
    )

    # read basin list, basin_id must be the model sequence
    df_basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse, basin_node = init_basin_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "basins.txt")
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
    Qnodelink[:, 0] = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_last_flow.txt"),
        skipinitialspace=True,
    )["value"])   

    # init flows at nodelinks, for every time step but only at catchments
    Qnodelink = init_qnodelink(
        df_nodelinks,
        Qnodelink,
        os.path.join(INPUT_FOLDER, TO_TRNSPRT_FOLDER, "discharge.txt"),
        START_TIME,
        END_TIME
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
    Qsubreach_in = np.asarray(df_Qsubreach_in.pivot(
        index="n_sm",
        columns="idre",
        values="value"
    ).fillna(-999.0))
    Qsubreach_out = np.asarray(df_Qsubreach_out.pivot(
        index="n_sm",
        columns="idre",
        values="value"
    ).fillna(-999.0))   

    # loop over time steps excluded time 0
    for t in range(1, len(time_array) + 1, 1):
        if time_array.hour[t - 1] == 0:
            print(
                "Transport",
                time_array.year[t - 1],
                time_array.month[t - 1],
                time_array.day[t - 1]
            )
        # init
        basin_count = 0
        reservoir_count = 0
        reach_count = 0
        junction_count = 0
        # transport loop over transport sequence
        for i, row in df_trans_seq.iterrows():
            if row["idmo"] == 5:  # basin, just transfer the flow
                # find basin id
                basin_id = df_elements.loc[row["idel"]]["idxx"]
                #print(f"Basin {basin_id}")

                downstream_nodelink = (
                    df_nodelinks.set_index("nodelink_node_u").loc[
                        int(df_basins.loc[basin_count]["nodeout"])
                        ]
                )
                # the order of the nodelinks is not in sequence order!!!
                # TODO: order the nodelinks according to the sequence!!!
                downstream_flow = Qnodelink[downstream_nodelink["nodelink_id"] - 1, t]
                basin_count += 1  # the basin is already computed! 
            elif row["idmo"] == 4:  # Reservoir
                # find reservoir id
                reservoir_id = df_elements.loc[row["idel"]]["idxx"]
                #print(f"Reservoir {reservoir_id}")

                # find upstream and downstream nodelinks
                upstream_nodelink = (
                    df_nodelinks.set_index("nodelink_node_d").loc[
                        int(df_reservoirs.loc[reservoir_count]["nodein"])
                        ]
                )
                downstream_nodelink_turb = (
                    df_nodelinks.set_index("nodelink_node_u").loc[
                        int(df_reservoirs.loc[reservoir_count]["nodeturb"])
                        ]
                )
                downstream_nodelink_spill = (
                    df_nodelinks.set_index("nodelink_node_u").loc[
                        int(df_reservoirs.loc[reservoir_count]["nodespill"])
                        ]
                )
                # 0 turbinated flow
                Qnodelink[
                    downstream_nodelink_turb["nodelink_id"] - 1, t
                ] = 0
                # All spilled flow
                Qnodelink[
                    downstream_nodelink_spill["nodelink_id"] - 1, t
                ] = Qnodelink[
                    upstream_nodelink["nodelink_id"] - 1, t
                ]
            elif row["idmo"] == 2:  # MC
                # find reach
                reach_id = df_elements.loc[row["idel"]]["idxx"]
                #print(f"Reach {reach_id}")

                # find upstream nodelink
                upstream_node = df_reaches.set_index("idre").loc[reach_id]["idin"]
                upstream_nodelink = df_nodelinks.set_index("nodelink_node_d").loc[upstream_node]
                # find downstream nodelink
                downstream_node = df_reaches.set_index("idre").loc[reach_id]["idout"]
                downstream_nodelink = df_nodelinks.set_index("nodelink_node_u").loc[downstream_node]
                # Muskingum-Cunge
                Qsubreach_in, Qsubreach_out = MC(
                    reach_count,
                    Qupstream=Qnodelink[upstream_nodelink["nodelink_id"] - 1, t],
                    nstep=df_reaches.set_index("idre").loc[reach_id]["nreaches"],
                    Qsubreach_in=Qsubreach_in,
                    Qsubreach_out=Qsubreach_out,
                    c1_mc=df_reach_params.loc[reach_id]["c1_mc"],
                    c2_mc=df_reach_params.loc[reach_id]["c2_mc"],
                    c3_mc=df_reach_params.loc[reach_id]["c3_mc"],
                )
                Qnodelink[downstream_nodelink["nodelink_id"] - 1, t] = Qsubreach_out[-1, reach_count]
                # next reach
                reach_count += 1
            elif row["idmo"] == 3:  # Junction
                # find junction id
                junction_id = df_elements.loc[row["idel"]]["idxx"]
                #print(f"Junction {junction_id}")

                # find upstream nodelinks
                upstream_node1 = df_junctions.set_index("idju").loc[junction_id]["idin1"]
                upstream_nodelink1 = df_nodelinks.set_index("nodelink_node_d").loc[upstream_node1]
                upstream_node2 = df_junctions.set_index("idju").loc[junction_id]["idin2"]
                upstream_nodelink2 = df_nodelinks.set_index("nodelink_node_d").loc[upstream_node2]
                # find downstream nodelinks
                downstream_node = df_junctions.set_index("idju").loc[junction_id]["idout"]
                downstream_nodelink = df_nodelinks.set_index("nodelink_node_u").loc[downstream_node]
                # sum upstream flows
                Qnodelink[downstream_nodelink["nodelink_id"] - 1, t] = (
                    Qnodelink[upstream_nodelink1["nodelink_id"] - 1, t]
                    + Qnodelink[upstream_nodelink2["nodelink_id"] - 1, t]
                )
                # next junction
                junction_count += 1 
