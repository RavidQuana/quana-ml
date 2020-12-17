import sample_file_parser
import signal_process
import sample
import similarity
import feature_extractor
import graphs_creator
import sample_file_parser
zip_file_path = "C:/Users/ravid/Downloads/samples.zip"

Samples = sample_file_parser.extract_zip_file(zip_file_path)

#for samp in Samples:
    #for col in samp.values.columns:
        #if col != 'time':
            #samp.values[col] = signal_process.smooth(samp.values[col])

sample.sort_samples(Samples)

graphs_creator.create_sim_graphs(graphs_creator.split_by_tag)
#print(sample.sorted_samples['prototype_1_aromabit']['Kanaf_Tilapia']['qcm_3'][0].values)
#similarity.group_by_similarity(sample.sorted_samples['prototype_1_aromabit']['Kanaf_Tilapia']['qcm_3'])
for prod in sample.sorted_samples['prototype_1_aromabit']:
    print("==============Product = " + prod + "===============")
    for chan in sample.sorted_samples['prototype_1_aromabit'][prod]:
        print( "----------------------Chan = " + chan + "---------------------")
        similarity.group_by_similarity(sample.sorted_samples['prototype_1_aromabit'][prod][chan])
        for data_type in similarity.similarity_dict:
            for grp in similarity.similarity_dict[data_type]:
                grp_stat = similarity.similarity_dict[data_type][grp]
                print (grp + ": mebers_count = " + str(grp_stat.members_count))
                print ("max sim = " + str(grp_stat.max_similarity))
                print ("min sim = " + str(grp_stat.min_similarity))