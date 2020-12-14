import sys
import numpy as np
from scipy import stats
import constants

import sample

class protocol_attr:
    iteration_count = 3
    pre_expose_time = 30
    expose_time = 50
    fade_time = 20
    clear_time = 70

class features:
    expose_avg_slope = 1
    expose_pick = 0
    expose_max_slope = 0
    slope_to_next_expose_pick = 0
    clear_avg_slope = 1
    clear_pick = 0
    clear_max_slope = 0
    slope_to_next_clear_pick = 0
    
    

def get_max_slope(chan_data, start, end):
    return max(chan_data.derviate_1[start:end])

def get_picks_indexes(chan_data, start, end):
    return np.where(np.diff(np.sign(chan_data.derviate_1)) != 0)[0]

def get_avg_slope(chan_data, start, end):
    x = chan_data.values['time'][start:end]
    y = chan_data.values[chan_data.values.columns[1]][start:end]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope

def extract_fetures(chan_data, prot_attr: protocol_attr):
    count = 0
    cycle_time = prot_attr.pre_expose_time + prot_attr.expose_time + prot_attr.fade_time + prot_attr.clear_time
    sampling_timing = 0
    chan_column = chan_data.values.columns[1]
    if chan_column == constants.humidity_col_name or chan_column == constants.temp_col_name:
        return
    features_list = []
    while count < prot_attr.iteration_count:
        data_features = features()
        sampling_timing += prot_attr.pre_expose_time
        seg_start = sampling_timing
        seg_end = sampling_timing+prot_attr.expose_time
        #pick_list_index = np.where(chan_data.picks_list > seg_start)[0][0]
        #data_index = chan_data.picks_list[pick_list_index]
        #data_features.expose_pick = chan_data.values[chan_column][data_index]
        data_features.expose_avg_slope = get_avg_slope(chan_data, seg_start, seg_end)
        if data_features.expose_avg_slope > 0:
            data_features.expose_pick = max(chan_data.values[chan_column][seg_start:seg_end])
        elif data_features.expose_avg_slope <= 0:
            data_features.expose_pick = min(chan_data.values[chan_column][seg_start:seg_end])
        data_features.expose_max_slope = get_max_slope(chan_data, seg_start, seg_end)
        sampling_timing += (prot_attr.fade_time + prot_attr.expose_time)
        seg_start = sampling_timing
        seg_end = sampling_timing+prot_attr.clear_time
        #pick_list_index = np.where(chan_data.picks_list > seg_start)[0][0]
        #data_index = chan_data.picks_list[pick_list_index]
        #data_features.clear_pick = chan_data.values[chan_column][data_index]
        data_features.clear_avg_slope = get_avg_slope(chan_data, seg_start, seg_end)
        if data_features.clear_avg_slope > 0:
            data_features.clear_pick = max(chan_data.values[chan_column][seg_start:seg_end])
        elif data_features.clear_avg_slope <= 0:
            data_features.clear_pick = min(chan_data.values[chan_column][seg_start:seg_end])        
        data_features.clear_max_slope = get_max_slope(chan_data, seg_start, seg_end)        
        if count > 0:
            slope_between_Picks = (data_features.expose_pick - features_list[count -1].expose_pick)/cycle_time
            features_list[count -1].slope_to_next_expose_pick = slope_between_Picks
            slope_between_Picks = (data_features.clear_pick - features_list[count -1].clear_pick)/cycle_time
            features_list[count -1].slope_to_next_clear_pick = slope_between_Picks
        features_list.append(data_features)    
        count += 1

def calculate_prot_timing(prot: protocol_attr):
    xpositions = []
    timing = 0
    for i in range(prot.iteration_count):
        timing += prot.pre_expose_time
        xpositions.append(timing)
        timing += prot.expose_time
        xpositions.append(timing)
        timing += prot.fade_time
        xpositions.append(timing)
        timing += prot.clear_time
        xpositions.append(timing)
    return xpositions