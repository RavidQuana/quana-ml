from time import strftime, localtime

path_date_format = "%d_%m_%y_%H%M"
path_result_dir = "./Results/"
path_graphs_dir = "/Graphs"
path_sim_graphs_dir = "/similarity_graphs/"
path_radar_graphs_dir = "/radar_graphs/"
path_scatered_graphs_dir = "/scatered_graphs/"
path_random_forest_graphs_dir = "/random_forest_graphs/"
path_features_dir = "/features/"
humidity_col_name = "humidity"
temp_col_name = "temp"
time_col_name = "time"

channel_out_col_title = "Channel"
sample_ID_out_col_title = "Sample ID"
now = None

def get_date_str():
    global now
    if not now:
        now = localtime()
    return strftime(path_date_format, now)