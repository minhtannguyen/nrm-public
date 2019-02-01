import matplotlib as mpl

mpl.use('Agg')
from pylab import *

import os

import argparse
import cPickle


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/mnt/ebs2/data/all_cifar/real/cifar-100-python')
parser.add_argument('--out_dir', type=str, default='/mnt/ebs2/data/cifar4/real')

opt = parser.parse_args()
print(opt)

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

train_dat = unpickle(os.path.join(opt.data_dir, 'train'))
test_dat = unpickle(os.path.join(opt.data_dir, 'test'))
meta_dat = unpickle(os.path.join(opt.data_dir, 'meta'))

classes = ['bicycle', 'streetcar', 'motorcycle', 'train']

def extract_images_from_class(c, dat):
    c_id = meta_dat['fine_label_names'].index(c)
    x_id = [i for i, y in enumerate(dat['fine_labels']) if y == c_id]
    x = dat['data'][x_id,:]
    fine_y = np.asarray(dat['fine_labels'])[x_id]
    coarse_y = np.asarray(dat['coarse_labels'])[x_id]
    x_name = [dat['filenames'][i] for i in x_id]
    fine_y_name = [c for i in x_id]
    coarse_y_name = [meta_dat['coarse_label_names'][i] for i in coarse_y.tolist()]
    return x, fine_y, coarse_y, x_name, fine_y_name, coarse_y_name

def dump_pickle(x, fine_y, coarse_y, x_name, fine_y_name, coarse_y_name, save_name):
    out_dat = {'data': x, 'fine_labels': fine_y, 'coarse_labels': coarse_y,
                 'filenames': x_name, 'fine_label_names': fine_y_name,
                 'coarse_label_names': coarse_y_name}
    with open(save_name, 'wb') as fp:
        cPickle.dump(out_dat, fp)

def save_dat(dat, save_name):
    x_dat = []; fine_y_dat = []; coarse_y_dat = []; x_name_dat = []; fine_y_name_dat = []; coarse_y_name_dat = []

    for c in classes:
        x, fine_y, coarse_y, x_name, fine_y_name, coarse_y_name = extract_images_from_class(c, dat)
        x_dat.append(x)
        fine_y_dat.append(fine_y)
        coarse_y_dat.append(coarse_y)
        x_name_dat = x_name_dat + x_name
        fine_y_name_dat = fine_y_name_dat + fine_y_name
        coarse_y_name_dat = coarse_y_name_dat + coarse_y_name
        dump_pickle(x, fine_y, coarse_y, x_name, fine_y_name, coarse_y_name, save_name+'_%s'%c)

    x_dat = np.concatenate(x_dat)
    fine_y_dat = np.concatenate(fine_y_dat)
    coarse_y_dat = np.concatenate(coarse_y_dat)
    dump_pickle(x_dat, fine_y_dat, coarse_y_dat, x_name_dat, fine_y_name_dat, coarse_y_name_dat, save_name)

if __name__ == '__main__':
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    save_train_name = os.path.join(opt.out_dir, 'train')
    save_test_name = os.path.join(opt.out_dir, 'test')

    save_dat(dat=train_dat, save_name=save_train_name)
    save_dat(dat=test_dat, save_name=save_test_name)