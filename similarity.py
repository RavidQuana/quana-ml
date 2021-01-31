import scipy.stats
import scipy.signal as sig
import pandas as pd
import sample_file_parser

class group_stat:
    members_count = 1
    max_similarity = 0
    min_similarity = 1
    chan_data_array = None
    def __init__(self, ch_data):
        self.members_count = 1
        self.chan_data_array = []
        self.chan_data_array.append(ch_data)
min_good_similarity = 0.85
similarity_dict = {}
group_prefix = "group_"
group_raw_counter = 1
group_first_der_counter = 1
group_second_der_counter = 1
raw_data_key = "raw"
first_derivative_key = "first_derivative"
second_derivative_key = "second_derivative"

class bad_sample:
    sample_id = 0
    tags = ''
    group =''
    refcount = 0
    key = ''
    chan_card = ''
bad_sample_dict = {}

def getPearsonSimi(array_src, array_compare):
    return scipy.stats.pearsonr(array_src, array_compare)[0]

def getSpearmanSimi(array_src, array_compare):
    return scipy.stats.spearmanr(array_src, array_compare)[0]

def getKendalltauSimi(array_src, array_compare):
    return scipy.stats.kendalltau(array_src, array_compare)[0]

def getSimilarity(array_src, array_compare):
    pearson = getPearsonSimi(array_src, array_compare)
    spearman = getSpearmanSimi(array_src, array_compare)
    kandeltau = getKendalltauSimi(array_src, array_compare)
    #print("Pearson = " + str(pearson) + ", Spearman = " +str(spearman) + ", kandeltau = " + str(kandeltau))
    return (pearson + spearman + kandeltau) / 3

def get_group_name(group: group_stat, key):
    val_index = list(similarity_dict[key].values()).index(group)
    group_name = list(similarity_dict[key].keys())[val_index]
    return group_name
    
def compare_simi_to_group(chan_data, group, key):
    belong_to_group = True
    min_sim = group.min_similarity
    max_sim = group.max_similarity
    data_array = chan_data.values
    if key == raw_data_key:
        src_array = data_array[data_array.columns[1]]
    elif key == first_derivative_key:
        src_array = chan_data.derviate_1
    elif key == second_derivative_key:
        src_array = chan_data.derviate_2
        
    for data in group.chan_data_array:
        if key == raw_data_key:
            comapare_array = data.values[data.values.columns[1]]
        elif key == first_derivative_key:
            comapare_array = data.derviate_1
        elif key == second_derivative_key:
            comapare_array = data.derviate_2
            
        sim_val = getSimilarity(src_array, comapare_array)
        if sim_val < min_good_similarity:
            belong_to_group = False
            break
        else:
            if (sim_val < min_sim):
                min_sim = sim_val
            if (sim_val > max_sim):
                max_sim = sim_val
    if belong_to_group:
        if min_sim < group.min_similarity:
            group.min_similarity = min_sim
        if max_sim > group.max_similarity:
            group.max_similarity = max_sim
        group.chan_data_array.append(chan_data)
        group.members_count +=1
    return belong_to_group

def check_bad_samples(key):
    for data_type in similarity_dict:
        for grp in similarity_dict[data_type]:
            grp_stat = similarity_dict[data_type][grp]
            tags = sample_file_parser.get_sample_tag(grp_stat.chan_data_array[0].sample_id)
            bad_sample_list = []
            is_bad_sample = False
            for ch_data in grp_stat.chan_data_array:
                tag_next = sample_file_parser.get_sample_tag(ch_data.sample_id)
                curr_bad_sample = bad_sample()
                curr_bad_sample.sample_id = ch_data.sample_id
                curr_bad_sample.tags = tag_next
                curr_bad_sample.group = grp
                curr_bad_sample.key = key
                card = sample_file_parser.get_sample_card(ch_data.sample_id)
                chan_name = ch_data.values.columns[1]
                curr_bad_sample.chan_card = card + "_" + chan_name
                bad_sample_list.append(curr_bad_sample)
                if (tag_next != tags):
                    is_bad_sample = True
            if is_bad_sample:
                for bad_samp in bad_sample_list:
                    bad_samp_key = bad_samp.key + "_" + bad_samp.chan_card+ "_" + str(bad_samp.sample_id)
                    if bad_samp_key in bad_sample_dict:
                        bad_sample_dict[bad_samp_key].refcount += 1
                    else:
                        bad_sample_dict[bad_samp_key] = bad_samp
                        bad_sample_dict[bad_samp_key].refcount = 1    

def group_by_similarity(chan_data_array):
    group_key = ""
    similarity_dict.clear()
    similarity_dict[raw_data_key] = {}
    similarity_dict[first_derivative_key] = {}
    similarity_dict[second_derivative_key] = {}
    group_by_similarity_key(chan_data_array, raw_data_key)
    check_bad_samples(raw_data_key)
    group_by_similarity_key(chan_data_array, first_derivative_key)
    check_bad_samples(first_derivative_key)
    group_by_similarity_key(chan_data_array, second_derivative_key)
    check_bad_samples(second_derivative_key)

                      
def group_by_similarity_key(chan_data_array, key):
    global group_raw_counter, group_first_der_counter, group_second_der_counter
    
    if key == raw_data_key:
        group_raw_counter = 1
    elif key == first_derivative_key:
        group_first_der_counter = 1
    elif key == second_derivative_key:
        group_second_der_counter = 1
    for chan_data in chan_data_array:
        create_new_group = False
        if not similarity_dict[key]:
            create_new_group = True
        else:
            create_new_group = True
            for grp in similarity_dict[key]:
                if compare_simi_to_group(chan_data, similarity_dict[key][grp], key):
                    create_new_group = False
                    break
        group_counter = 1
        if create_new_group:
            if key == raw_data_key:
                group_counter = group_raw_counter
                group_raw_counter += 1
            elif key == first_derivative_key:
                group_counter = group_first_der_counter
                group_first_der_counter += 1
            elif key == second_derivative_key:
                group_counter = group_second_der_counter
                group_second_der_counter += 1            
            group_key = group_prefix + str(group_counter)
            similarity_dict[key][group_key] = group_stat(chan_data)    

