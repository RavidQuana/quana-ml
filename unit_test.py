import sample_file_parser
import signal_process
import sample
import similarity
import feature_extractor
import graphs_creator
import sample_file_parser
import os
import constants

#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import matplotlib.colors as colors
#import math


#fig = plt.figure()
#ax = fig.add_subplot(111)

#ratio = 1.0 / 3.0
#count = math.ceil(math.sqrt(len(colors.CSS4_COLORS)))
#x_count = count * ratio
#y_count = count / ratio
#x = 0
#y = 0
#w = 1 / x_count
#h = 1 / y_count

#for c in colors.CSS4_COLORS:
    #pos = (x / x_count, y / y_count)
    #ax.add_patch(patches.Rectangle(pos, w, h, color=c))
    #ax.annotate(c, xy=pos)
    #if y >= y_count-1:
        #x += 1
        #y = 0
    #else:
        #y += 1

#plt.show()
zip_file_path = "C:/Users/ravid/Downloads/samples.zip"


Samples = sample_file_parser.extract_zip_file(zip_file_path)

#for samp in Samples:
    #for col in samp.values.columns:
        #if col != 'time':
            #samp.values[col] = signal_process.smooth(samp.values[col])

sample.sort_samples(Samples, sample_file_parser.product_col_name)

graphs_creator.create_sim_graphs(graphs_creator.split_by_prod)
#graphs_creator.create_sim_graphs(graphs_creator.split_by_tag)
#print(sample.sorted_samples['prototype_1_aromabit']['Kanaf_Tilapia']['qcm_3'][0].values)
#similarity.group_by_similarity(sample.sorted_samples['prototype_1_aromabit']['Kanaf_Tilapia']['qcm_3'])
#for prod in sample.sorted_samples['prototype_1_aromabit']:
    #print("==============Product = " + prod + "===============")
    #for chan in sample.sorted_samples['prototype_1_aromabit'][prod]:
        #print( "----------------------Chan = " + chan + "---------------------")
        #similarity.group_by_similarity(sample.sorted_samples['prototype_1_aromabit'][prod][chan])
        
datestr = constants.get_date_str()

out_dir_path = constants.path_result_dir + datestr
out_file_name = out_dir_path + "/badsamples_report_" + datestr + ".csv"
out_file = open(out_file_name, "w")
head_line = "sample id, data type,chan card,tags,refcount\n"
out_file.write(head_line)
for sample_key in sorted(similarity.bad_sample_dict):
    bad_samp = similarity.bad_sample_dict[sample_key]
    line = str(bad_samp.sample_id) + "," + bad_samp.key + "," + bad_samp.chan_card + "," + bad_samp.tags + "," + str(bad_samp.refcount) + "\n"
    out_file.write(line)
    print(line)
out_file.close()