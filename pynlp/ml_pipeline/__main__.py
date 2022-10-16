from ml_pipeline import experiment
import argparse

# Directories of test datasets
dir_olid = 'data/olid/'
dir_hasoc = 'data/hasoc/'
dirs =  [dir_olid, dir_hasoc]

# Counter used to identify the current experiment. Used in utils.eval
domain = 0

for d in dirs:

    
    if domain ==0:
        print('\n-----------IN-DOMAIN EXPERIMENT------------\n')
        
    else:
        print('\n----------CROSS-DOMAIN EXPERIMENT----------\n')


    parser = argparse.ArgumentParser(description='run classifier on data')
    parser.add_argument('--task', dest='task', default="vua_format")
    parser.add_argument('--data_dir', dest='data_dir', default= d)
    parser.add_argument('--pipeline', dest='pipeline', default= 'svm_libsvc_counts_12')
    parser.add_argument('--print_predictions', dest='print_predictions', default=False)
    args = parser.parse_args()

    experiment.run(args.task, args.data_dir, args.pipeline, args.print_predictions)

    domain =+1
