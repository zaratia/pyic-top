import json
import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_reach_vars, init_qnodelink, init_basin_vars, init_reservoir_vars, init_junction_vars, sort_df_by_seq
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

    # read reaches list, reach id must be the model sequence
    df_reaches, reaches_id, n_reach, reach_in, reach_out, n_sub_reaches, n_step_max = init_reach_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reaches.txt")
    )
    df_reaches = df_reaches.set_index("idre")

    # read nodelinks list
    # NOTE: nodelinks are not in sequence order!!!
    df_nodelinks = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "nodelinks.txt"),
        skipinitialspace=True,
    ).set_index("nodelink_id")

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
    df_basins = df_basins.set_index("idba")

    # read reservoir list, reservoir_id must be the model sequence
    df_reservoirs, reservoir_id, n_reservoir = init_reservoir_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reservoirs.txt")
    )
    df_reservoirs = df_reservoirs.set_index("idrs")

    # read junction list, junction_id must be the model sequence
    df_junctions, junction_id, n_junction = init_junction_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "junctions.txt")
    )
    df_junctions = df_junctions.set_index("idju")

    # init (n_reach) real vars
    df_Qsubreach_in = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_MC_Qin.txt"),
        skipinitialspace=True,
    )
    df_Qsubreach_out = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_MC_Qout.txt"),
        skipinitialspace=True,
    )

    # reordered nodelink dataframe
    id_nodelinks_seq = np.full(len(df_nodelinks), -999)
    id_reaches_seq = np.full(len(df_reaches), -999)
    id_junctions_seq = np.full(len(df_junctions), -999)
    id_reservoirs_seq = np.full(len(df_reservoirs), -999)
    id_basins_seq = np.full(len(df_basins), -999)

    basin_count = 0
    reservoir_count = 0
    reach_count = 0
    junction_count = 0
    nodelink_count = 0
    # transport loop over transport sequence
    for i, row in df_trans_seq.iterrows():
        if row["idmo"] == 5:  # basin, just transfer the flow
            # find basin id
            basin_id = df_elements.loc[row["idel"]]["idxx"]
            #print(f"Basin {basin_id}")

            downstream_nodelink = (
                df_nodelinks.reset_index().set_index("nodelink_node_u").loc[
                    int(df_basins.loc[basin_id]["nodeout"])
                    ]
            )

            id_nodelinks_seq[nodelink_count] = (
                downstream_nodelink["nodelink_id"]
            )
            nodelink_count += 1

            id_basins_seq[basin_count] = basin_id
            basin_count += 1 

        elif row["idmo"] == 4:  # Reservoir
            # find reservoir id
            reservoir_id = df_elements.loc[row["idel"]]["idxx"]
            #print(f"Reservoir {reservoir_id}")

            # find upstream and downstream nodelinks
            upstream_nodelink = (
                df_nodelinks.reset_index().set_index("nodelink_node_d").loc[
                    int(df_reservoirs.loc[reservoir_id]["nodein"])
                    ]
            )
            downstream_nodelink_turb = (
                df_nodelinks.reset_index().set_index("nodelink_node_u").loc[
                    int(df_reservoirs.loc[reservoir_id]["nodeturb"])
                    ]
            )
            downstream_nodelink_spill = (
                df_nodelinks.reset_index().set_index("nodelink_node_u").loc[
                    int(df_reservoirs.loc[reservoir_id]["nodespill"])
                    ]
            )

            # usptream nodelink is already in sequence!
            # id_nodelinks_seq[nodelink_count] = (
            #     upstream_nodelink["nodelink_id"]
            # )
            # nodelink_count += 1
            
            id_nodelinks_seq[nodelink_count] = (
                downstream_nodelink_turb["nodelink_id"]
            )
            nodelink_count += 1

            id_nodelinks_seq[nodelink_count] = (
                downstream_nodelink_spill["nodelink_id"]
            )
            nodelink_count += 1

            id_reservoirs_seq[reservoir_count] = reservoir_id
            reservoir_count += 1

        elif row["idmo"] == 2:  # MC
            # find reach
            reach_id = df_elements.loc[row["idel"]]["idxx"]
            #print(f"Reach {reach_id}")

            # find upstream nodelink
            upstream_node = df_reaches.reset_index().set_index("idre").loc[reach_id]["idin"]
            upstream_nodelink = df_nodelinks.reset_index().set_index("nodelink_node_d").loc[upstream_node]
            # find downstream nodelink
            downstream_node = df_reaches.reset_index().set_index("idre").loc[reach_id]["idout"]
            downstream_nodelink = df_nodelinks.reset_index().set_index("nodelink_node_u").loc[downstream_node]
           
            #upstream nodelink is already in sequence!
            # id_nodelinks_seq[nodelink_count] = (
            #     upstream_nodelink["nodelink_id"]
            # )
            # nodelink_count += 1

            id_nodelinks_seq[nodelink_count] = (
                downstream_nodelink["nodelink_id"]
            )
            nodelink_count += 1

            id_reaches_seq[reach_count] = reach_id
            reach_count += 1
        
        elif row["idmo"] == -1:  # MC final reach
            # find reach
            reach_id = df_elements.loc[row["idel"]]["idxx"]
            #print(f"Reach {reach_id}")

            # find upstream nodelink
            upstream_node = df_reaches.reset_index().set_index("idre").loc[reach_id]["idin"]
            upstream_nodelink = df_nodelinks.reset_index().set_index("nodelink_node_d").loc[upstream_node]
            
            # upstream nodelink is already in sequence!
            # id_nodelinks_seq[nodelink_count] = (
            #     upstream_nodelink["nodelink_id"]
            # )
            # nodelink_count += 1

            id_reaches_seq[reach_count] = reach_id
            reach_count += 1

        elif row["idmo"] == 3:  # Junction
            # find junction id
            junction_id = df_elements.loc[row["idel"]]["idxx"]
            #print(f"Junction {junction_id}")

            # find upstream nodelinks
            upstream_node1 = df_junctions.reset_index().set_index("idju").loc[junction_id]["idin1"]
            upstream_nodelink1 = df_nodelinks.reset_index().set_index("nodelink_node_d").loc[upstream_node1]
            upstream_node2 = df_junctions.reset_index().set_index("idju").loc[junction_id]["idin2"]
            upstream_nodelink2 = df_nodelinks.reset_index().set_index("nodelink_node_d").loc[upstream_node2]
            # find downstream nodelinks
            downstream_node = df_junctions.reset_index().set_index("idju").loc[junction_id]["idout"]
            downstream_nodelink = df_nodelinks.reset_index().set_index("nodelink_node_u").loc[downstream_node]
            
            # upstream nodelinks are already in sequence!
            # id_nodelinks_seq[nodelink_count] = (
            #     upstream_nodelink1["nodelink_id"]
            # )           
            # nodelink_count += 1
            # id_nodelinks_seq[nodelink_count] = (
            #     upstream_nodelink2["nodelink_id"]
            # )
            # nodelink_count += 1

            id_nodelinks_seq[nodelink_count] = (
                downstream_nodelink["nodelink_id"]
            )
            nodelink_count += 1

            id_junctions_seq[junction_count] = junction_id
            junction_count += 1

    df_nodelinks = sort_df_by_seq(df_nodelinks.reset_index(), id_nodelinks_seq, "nodelink_id")
    df_nodelinks.to_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "nodelinks_ordered.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
    df_reaches = sort_df_by_seq(df_reaches.reset_index(), id_reaches_seq, "idre")
    df_reaches.to_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reaches_ordered.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
    df_junctions = sort_df_by_seq(df_junctions.reset_index(), id_junctions_seq, "idju")
    df_junctions.to_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "junctions_ordered.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    ) 
    df_reservoirs = sort_df_by_seq(df_reservoirs.reset_index(), id_reservoirs_seq, "idrs")
    df_reservoirs.to_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reservoirs_ordered.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
    df_basins = sort_df_by_seq(df_basins.reset_index(), id_basins_seq, "idba")
    df_basins.to_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "basins_ordered.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
