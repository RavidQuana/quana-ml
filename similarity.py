import scipy.stats
import scipy.signal as sig

class group_stat:
    members_count = 0
    max_similarity = 0
    min_similarity = 1
    chan_data_array = None

min_good_similarity = 0.65
similarity_dict = {}
group_prefix = "group_"
group_counter = 1

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

def compare_simi_to_group(chan_data, group):
    belong_to_group = True
    min_sim = group.min_similarity
    max_sim = group.max_similarity
    data_array = chan_data.values
    for data in group.chan_data_array:
        comapare_array = data.values[data.values.columns[1]]
        sim_val = getSimilarity(data_array[data_array.columns[1]], comapare_array)
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
    global group_counter
    group_counter = 1
    for chan_data in chan_data_array:
        create_new_group = False
        if not similarity_dict:
            create_new_group = True
        else:
            create_new_group = True
            for grp in similarity_dict:
                if compare_simi_to_group(chan_data, similarity_dict[grp]):
                    create_new_group = False
                    break
        if create_new_group:
            group_key = group_prefix + str(group_counter)
            group_counter += 1
            similarity_dict[group_key] = group_stat()
            similarity_dict[group_key].members_count = 1
            similarity_dict[group_key].chan_data_array = []
            similarity_dict[group_key].chan_data_array.append(chan_data)             
                                           
            
            
    

                            