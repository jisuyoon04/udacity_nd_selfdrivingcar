import argparse
import glob
import os

import shutil
import numpy as np

from utils import get_module_logger

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # create folders for train, val, test
    train = os.path.join(data_dir, 'train')
    if os.path.exists(train) == False:
        os.makedirs(train)
        
    val = os.path.join(data_dir, 'val')
    if os.path.exists(val) == False:
        os.makedirs(val)

    test = os.path.join(data_dir, 'test')
    if os.path.exists(test) == False:
        os.makedirs(test)
        
    # split data
    files = [filename for filename in glob.glob(f'{data_dir}/*.tfrecord')]
    np.random.shuffle(files)

    train_data, val_data, test_data = np.split(files, [int(.7*len(files)), int(.9*len(files))])
    
    # move data to created directories
    for data in train_data:
        shutil.move(data, train)
    
    for data in val_data:
        shutil.move(data, val)
    
    for data in test_data:
        shutil.move(data, test)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)