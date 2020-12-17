import scipy.stats
import scipy.signal as sig
import pandas as pd

class group_stat:
    members_count = 1
    max_similarity = 0
    min_similarity = 1
    chan_data_array = None
    def __init__(self, ch_data):
        self.members_count = 1
        self.chan_data_array = []
        self.chan_data_array.append(ch_data)
min_good_similarity = 0.65
similarity_dict = {}
group_prefix = "group_"
group_raw_counter = 1
group_first_der_counter = 1
group_second_der_counter = 1
raw_data_key = "raw"
first_derivate_key = "first_derivate"
second_derivate_key = "second_derivate"



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
    elif key == first_derivate_key:
        src_array = chan_data.derviate_1
    elif key == second_derivate_key:
        src_array = chan_data.derviate_2
        
    for data in group.chan_data_array:
        if key == raw_data_key:
            comapare_array = data.values[data.values.columns[1]]
        elif key == first_derivate_key:
            comapare_array = data.derviate_1
        elif key == second_derivate_key:
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

def group_by_similarity(chan_data_array):
    group_key = ""
    similarity_dict.clear()
    similarity_dict[raw_data_key] = {}
    similarity_dict[first_derivate_key] = {}
    similarity_dict[second_derivate_key] = {}
    group_by_similarity_key(chan_data_array, raw_data_key)
    group_by_similarity_key(chan_data_array, first_derivate_key)
    group_by_similarity_key(chan_data_array, second_derivate_key)
    

                      
def group_by_similarity_key(chan_data_array, key):
    global group_raw_counter, group_first_der_counter, group_second_der_counter
    
    if key == raw_data_key:
        group_raw_counter = 1
    elif key == first_derivate_key:
        group_first_der_counter = 1
    elif key == second_derivate_key:
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
            elif key == first_derivate_key:
                group_counter = group_first_der_counter
                group_first_der_counter += 1
            elif key == second_derivate_key:
                group_counter = group_second_der_counter
                group_second_der_counter += 1            
            group_key = group_prefix + str(group_counter)
            similarity_dict[key][group_key] = group_stat(chan_data)    

