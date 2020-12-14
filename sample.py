import signal_process
import feture_extractor
import matplotlib.pyplot as plt
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
    derivate_2 = None
    picks_list = None
    features = None
    protocol = None
    
    def __init__(self):
        return

def sort_samples(samples_array):
    prot = feture_extractor.protocol_attr()
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
            ch_data.derviate_1 = signal_process.get_derivate_1(ch_data.values[channel])
            ch_data.derviate_2 = signal_process.get_derivate_2(ch_data.values[channel])
            ch_data.picks_list = feture_extractor.get_picks_indexes(ch_data, 0, ch_data.values.size)
            ch_data.features = feture_extractor.extract_fetures(ch_data, prot)
            ch_data.protocol = prot
            sorted_samples[sample.sampler_type][brand_prod][card_channel].append(ch_data)


    
            
                                                    