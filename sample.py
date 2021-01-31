import signal_process
import feature_extractor
import matplotlib.pyplot as plt
import os
import constants
sorted_samples = {}

class DataSample:
    ID = 0
    product = ''
    note = ''
    tags = ''
    brand = ''
    sampler_type = ''
    card = ''
    values = None
    
class channel_data:
    sample_id = 0
    values = None
    derviate_1 = None
    derivative_2 = None
    picks_list = None
    features = None
    protocol = None
    
    def __init__(self):
        return

def sort_samples(samples_array, sorter):
    prot = feature_extractor.protocol_attr()
    for sample in samples_array:
        if sample.sampler_type not in sorted_samples:
            sorted_samples[sample.sampler_type] = {}
        brand_prod = sample.brand + "_" + sample.product
        if brand_prod not in sorted_samples[sample.sampler_type]:
            sorted_samples[sample.sampler_type][brand_prod] = {}
        for channel in sample.values.columns[1:]:
            if (sample.values[channel] == 0).all():
                continue
            card_channel = sample.card + "_" + channel
            if card_channel not in sorted_samples[sample.sampler_type][brand_prod]:
                sorted_samples[sample.sampler_type][brand_prod][card_channel] = []
            ch_data = channel_data()
            ch_data.sample_id = sample.ID
            ch_data.note = sample.note
            ch_data.tags = sample.tags
            ch_data.values = sample.values[["time", channel]]
            ch_data.values[channel] -= ch_data.values[channel][30]
            ch_data.values[channel] = signal_process.smooth(ch_data.values[channel])
            ch_data.derviate_1 = signal_process.get_derivative_1(ch_data.values[channel])
            ch_data.derviate_2 = signal_process.get_derivative_2(ch_data.values[channel])
            ch_data.picks_list = feature_extractor.get_picks_indexes(ch_data, 0, ch_data.values.size)
            ch_data.protocol = prot
            feature_extractor.extract_features(ch_data, prot)
            sorted_samples[sample.sampler_type][brand_prod][card_channel].append(ch_data)
    datestr = constants.get_date_str()
    features_results_dir = constants.path_result_dir + datestr + constants.path_features_dir    
    features_file_name = features_results_dir + "features_" + "_" + datestr + ".csv"    
    if not os.path.exists(features_results_dir):
        os.makedirs(features_results_dir)
    feature_extractor.flush_features_data_frame(features_file_name, sorter)

    
            
                                                    