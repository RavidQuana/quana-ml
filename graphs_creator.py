import os
import constants
from time import strftime, localtime
import sample 
import feture_extractor
import similarity
import sample_file_parser
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

markers_list = ["o", "v", "8", "s", "p", "P", "X", "D", "d"]
markers_list_index = 0;
time_array = []
names = list(mcolors.CSS4_COLORS)

split_by_prod = "prod"
split_by_tag = "tag"

def create_graph_directory():
    datestr = strftime(constants.path_date_format, localtime())
    grap_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir
    grap_types_dirs = { \
        constants.path_radar_graphs_dir, \
        constants.path_sim_graphs_dir, \
        constants.path_scatered_graphs_dir, \
        constants.path_random_forest_graphs_dir \
        }
    
    if not os.path.exists(grap_results_dir):
        os.makedirs(grap_results_dir)
    for dir_name in grap_types_dirs:
        path = grap_results_dir + dir_name
        if not os.path.exists(path):
            os.makedirs(path)        
    
def get_time_array():
    if not sample.sorted_samples:
        print ("No sorted sampler type found")
        return None
    samp_type = list(sample.sorted_samples.keys())[0]
    if not sample.sorted_samples[samp_type]:
        print ("No sorted products found")
        return  None
    first_prod = list(sample.sorted_samples[samp_type].keys())[0]
    if not sample.sorted_samples[samp_type][first_prod]:
        print ("No sorted channels found")
        return  None       
    first_chan = list(sample.sorted_samples[samp_type][first_prod].keys())[0]
    if len(sample.sorted_samples[samp_type][first_prod][first_chan]) == 0:
        print ("No sorted channel data found")
        return  None       
    ch_data = sample.sorted_samples[samp_type][first_prod][first_chan][0]
    time_array = ch_data.values[constants.time_col_name]
    return time_array

def create_graphs_by_prod():
    for samp_type in sample.sorted_samples:
        for prod in sample.sorted_samples[samp_type]:
            for chan in sample.sorted_samples[samp_type][prod]:
                similarity.group_by_similarity(sample.sorted_samples[samp_type][prod][chan])
                create_sim_plots(similarity.similarity_dict)

def create_graphs_by_tags():
    chan_dict = {}
    for samp_type in sample.sorted_samples:
        for prod in sample.sorted_samples[samp_type]:
            for chan in sample.sorted_samples[samp_type][prod]:
                if chan not in chan_dict.keys():
                    chan_dict[chan] = []
                for ch_data in sample.sorted_samples[samp_type][prod][chan]:
                    chan_dict[chan].append(ch_data)
    for chan in chan_dict:
        similarity.group_by_similarity(chan_dict[chan])
        create_sim_plots(similarity.similarity_dict)
                
def create_sim_graphs(split_by):
    create_graph_directory()
    global time_array
    time_array.clear()
    time_array = get_time_array()
    if time_array.empty:
        return
    if split_by == split_by_prod:
        create_graphs_by_prod()
    elif split_by == split_by_tag:
        create_graphs_by_tags()
    
                    
def create_sim_plots(groups_dict):
    global time_array, markers_list, markers_list_index
    figW=18
    figH=10    
    fig, ax = plt.subplots(3, 1, figsize=(figW, figH))
    color_name_index = 35
    prod_tag_markers = {}
    markers_list_index = 0
    card = ""
    channel = ""
    group_statistics = "\n"
    for group in groups_dict:
        line_color = names[color_name_index]
        color_name_index += 2
        Marker = ''
        group_statistics += group + " members->" + str(groups_dict[group].members_count) + "\n" 
        for ch_data in groups_dict[group].chan_data_array:
            xpositions = feture_extractor.calculate_prot_timing(ch_data.protocol)
            if channel == "":
                channel = ch_data.values.columns[1]
            prod = sample_file_parser.get_sample_prod(ch_data.sample_id)
            tags = sample_file_parser.get_sample_tag(ch_data.sample_id)
            if card == "":
                card = sample_file_parser.get_sample_card(ch_data.sample_id)
                
            prod_tag = prod+"_" +tags
            if prod_tag not in prod_tag_markers.keys():
                prod_tag_markers[prod_tag] = markers_list[markers_list_index]
                markers_list_index += 1
            Marker = prod_tag_markers[prod_tag]
            
            Label = prod + ", " + tags
            ax[0].plot(time_array[0:509], ch_data.values[channel][0:509]*100, color = line_color, label=Label, marker=Marker, markersize=0.5)
            ax[0].set_title("Sample relative data")
            ax[1].plot(time_array[0:509], ch_data.derviate_1[0:509]*100, color = line_color, label=Label, marker=Marker, markersize=0.5)
            ax[1].set_title("1st derivate")
            ax[2].plot(time_array[0:509], ch_data.derviate_2[0:509]*100, color = line_color, label=Label, marker=Marker, markersize=0.5)
            ax[2].set_title("2nd derivate")
    for i in range(3):
        for xc in ch_data.picks_list:
            ax[i].axvline(x=xc/10, color='k', linestyle='--')
        for xp in xpositions:
            ax[i].axvline(x=xp/10, color='r', linestyle='--')
        ax[i].axhline(y=0, color='green', linestyle='dotted')
        ax[i].legend(loc='center left', bbox_to_anchor=(0, 1))
    plt.subplots_adjust(hspace=0.3, left=0.18, bottom=1/figH, top=1-1/figH)
    fig.suptitle("card: " + card + ", channel: " + channel + group_statistics, y=1)
    plt.get_current_fig_manager().full_screen_toggle()    
    plt.show()                    
    plt.close(fig)    
        