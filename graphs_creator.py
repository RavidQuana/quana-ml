import os
import plotly.express as px
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import constants
import sample 
import feature_extractor
import similarity
import sample_file_parser

markers_list = ["o", "v", "8", "s", "p", "P", "X", "D", "d"]
markers_list_index = 0;
time_array = []
names = list(mcolors.CSS4_COLORS)

csv_group_data_colums = ["type of data", "group name", "members count", "max similarity value", "min similarity value"]
csv_ch_data_columns =["type of data", "group name", "sample id", "product", "tags" ]

simi_group_df = None
out_file = None
split_by_prod = "prod"
split_by_tag = "tag"
card = ""
channel = ""
prod_tag_markers = {}
color_name_index = 10
graphs_results_dir = ""

light_colors = ['cornsilk']
def create_graph_directory():
    datestr = constants.get_date_str()
    graphs_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir
    grap_types_dirs = { \
        constants.path_radar_graphs_dir, \
        constants.path_sim_graphs_dir, \
        constants.path_scatered_graphs_dir, \
        constants.path_random_forest_graphs_dir \
        }
    
    if not os.path.exists(graphs_results_dir):
        os.makedirs(graphs_results_dir)
    for dir_name in grap_types_dirs:
        path = graphs_results_dir + dir_name
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
                create_sim_plots(similarity.similarity_dict, chan)

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
        create_sim_plots(similarity.similarity_dict, chan)
                
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


def open_sim_out_file(chan):
    global out_file
    datestr = constants.get_date_str()
    graphs_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir
    out_dir_path = graphs_results_dir + constants.path_sim_graphs_dir
    out_file_name = out_dir_path + "similarity_report_"+ chan+"_" + datestr + ".csv"
    out_file = open(out_file_name, "w")
    
def close_sim_out_file():
    global out_file
    out_file.close()

def add_row(type_of_data, group_name, ch_data):
    global out_file
    prod = sample_file_parser.get_sample_prod(ch_data.sample_id)
    tags = sample_file_parser.get_sample_tag(ch_data.sample_id)
    row = type_of_data +"," + group_name + "," + str(ch_data.sample_id) + "," + prod + "," + tags + "\n"
    out_file.write(row)
    
def add_group_data(group, key):
    global out_file
    row = ""
    for title in csv_group_data_colums:
        row += title +","
    row += "\n"
    out_file.write(row)
    group_name = similarity.get_group_name(group, key)
    row = key + "," + group_name + "," + str(group.members_count) + ","  + str(group.max_similarity) + ","+ str(group.min_similarity) + "\n"
    out_file.write(row)
    row=""
    for title in csv_ch_data_columns:
        row += title +","
    row += "\n"
    out_file.write(row)
    
def create_sim_plots(groups_dict, chan):
    global time_array, markers_list, markers_list_index, prod_tag_markers
    global card, channel
    global line_color, color_name_index
    figW=18
    figH=10    
    fig, ax = plt.subplots(3, 1, figsize=(figW, figH))
    color_name_index = 10
    prod_tag_markers.clear()
    markers_list_index = 0
    card = ""
    channel  = ""
    group_statistics = "\n"
    keys_list = [similarity.raw_data_key, similarity.first_derivate_key, similarity.second_derivate_key]
    sub_plot_titles_pre_list = ["Sample relative data", "1st derivate", "2nd derivate"]
    xpositions = []
    subplot_index = 0
    open_sim_out_file(chan)
    for key in keys_list:
        group_statistics = "\n"
        for group in groups_dict[key]:
            add_group_data(groups_dict[key][group], key)
            color_name_index += 2
            if names[color_name_index] in light_colors:
                color_name_index += 2
            Marker = ''
            group_statistics += group + " members->" + str(groups_dict[key][group].members_count) + "\n" 
            for ch_data in groups_dict[key][group].chan_data_array:
                if len(xpositions) == 0:
                    xpositions = feature_extractor.calculate_prot_timing(ch_data.protocol)
                add_graph_line(ax, ch_data, key)
                add_row(key, group, ch_data)
        ax[subplot_index].set_title(sub_plot_titles_pre_list[subplot_index] + group_statistics)
        subplot_index += 1
    close_sim_out_file()
    for i in range(3):
        #for xc in ch_data.picks_list:
            #ax[i].axvline(x=xc/10, color='k', linestyle='--')
        for xp in xpositions:
            ax[i].axvline(x=xp/10, color='r', linestyle='--')
        ax[i].axhline(y=0, color='green', linestyle='dotted')
        ax[i].legend(loc='center left', bbox_to_anchor=(0, 1))
    plt.subplots_adjust(hspace=0.3, left=0.18, bottom=1/figH, top=1-1/figH)
    fig.suptitle("card: " + card + ", channel: " + channel + group_statistics, y=1)
    plt.get_current_fig_manager().full_screen_toggle()  
    datestr = constants.get_date_str()
    graphs_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir    
    graph_file_name = "simgraph_" + card + "_" + channel + "_" + datestr + ".png"
    out_dir_path = graphs_results_dir + constants.path_sim_graphs_dir
    plt.savefig(out_dir_path + graph_file_name)
    #plt.show()
    plt.close(fig)    
        
def add_graph_line(ax, ch_data, key):
    global time_array, markers_list, markers_list_index, channel, card
    global color_name_index, prod_tag_markers
    Marker = ""
    line_color = names[color_name_index]
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
    if key == similarity.raw_data_key:
        ax[0].plot(time_array[0:509], ch_data.values[channel][0:509]*100, color = line_color, label=Label, marker=Marker, markersize=0.5)
    elif key == similarity.first_derivate_key:
        ax[1].plot(time_array[0:509], ch_data.derviate_1[0:509]*100, color = line_color, label=Label, marker=Marker, markersize=0.5)
    elif key == similarity.second_derivate_key:
        ax[2].plot(time_array[0:509], ch_data.derviate_2[0:509]*100, color = line_color, label=Label, marker=Marker, markersize=0.5)
        

def create_scatter_chart(dataframe, feature):
    datestr = constants.get_date_str()
    graphs_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir    
    out_dir_path = graphs_results_dir + constants.path_scatered_graphs_dir
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)    
    figscatter = px.scatter(dataframe, x=constants.channel_out_col_title, y=feature, color=sample_file_parser.tags_col_name, height=1000, width=1600)
    figscatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    scatter_chart_file_name = out_dir_path + '\\' + feature + '_scatter_chart.png'
    figscatter.write_image(file=scatter_chart_file_name, format='png')    

def create_radar_charts(dataframe, feature):
    datestr = constants.get_date_str()
    graphs_results_dir = constants.path_result_dir + datestr + constants.path_graphs_dir    
    out_dir_path = graphs_results_dir + constants.path_radar_graphs_dir
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)        
    #fig = go.Figure()
    fig = px.scatter_polar(dataframe,  r=feature, theta=constants.channel_out_col_title, color=sample_file_parser.tags_col_name, height=1600, width=1800)
    #fig = px.line_polar(dataframe,  
                        #r=feature, 
                        #theta="QCM", 
                        #color="Product", 
                        #height=1600, width=1600, 
                        #)
     
    #for product in dataframe['Product'].tolist():
        #rows_array = dataframe.loc[dataframe['Product'] == product]
        #for q in channels_list:
            #q_rows = rows_array.loc[rows_array['QCM'] == q]
            #for index, row in q_rows.iterrows():
                #print(row)
        
        ##fig.show() 
    fig.update_traces( mode="lines+markers", 
        marker = dict(
            symbol="diamond-open",
            size=6
        )  
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
            ), 
        showlegend=True, 
        title=feature,
        font=dict(
            family="Courier New, monospace",
            size=32,
            color="Black"
            ) ,
        margin=dict(l=270, r=80, t=80, b=20)
    )
    radar_chart_file_name = out_dir_path + '\\' + feature + '_radar_chart.png'
    fig.write_image(file=radar_chart_file_name, format='png')
        
def create_scatter_radar_graph(dataframe):
    dataframe = dataframe.sort_values(by=[constants.channel_out_col_title, sample_file_parser.product_col_name])
    channels_list = dataframe[constants.channel_out_col_title].unique()
    tagsCoulmnIndex = dataframe.columns.get_loc(sample_file_parser.tags_col_name)
    featuresList = dataframe.columns[(tagsCoulmnIndex+1):]
    for feature in featuresList:
        create_scatter_chart(dataframe, feature)
        create_radar_charts(dataframe, feature)
