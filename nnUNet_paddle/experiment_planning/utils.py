import os
import json
import shutil
import pickle
from collections import OrderedDict

from nnUNet_paddle.utils import maybe_mkdir_p
from nnUNet_paddle.preprocessing.cropping import ImageCropper

"""
    Code Reference: 
    https://github.com/MIC-DKFZ/nnUNet
"""

def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = os.path.join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(os.path.join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(os.path.join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def crop(task_string, raw_data_dir, cropped_data_dir, override=False, num_threads=8):
    cropped_out_dir = os.path.join(cropped_data_dir, task_string)
    maybe_mkdir_p(cropped_data_dir)

    if override and os.path.isdir(cropped_data_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)
    
    splitted_4d_output_dir_task = os.path.join(raw_data_dir, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(os.path.join(raw_data_dir, task_string, 'dataset.json'), cropped_out_dir)


def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.
    """
    npz_file, pkl_file, all_classes = args
    seg_map = np.load(npz_file)['data'][-1]
    with open(pkl_file, 'rb') as f:
        props = pickle.load(f)
    #if props.get('classes_in_slice_per_axis') is not None:
    print(pkl_file)
    # this will be a dict of dict where the first dict encodes the axis along which a slice is extracted in its keys.
    # The second dict (value of first dict) will have all classes as key and as values a list of all slice ids that
    # contain this class
    classes_in_slice = OrderedDict()
    for axis in range(3):
        other_axes = tuple([i for i in range(3) if i != axis])
        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0)[0]
            classes_in_slice[axis][c] = valid_slices

    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        number_of_voxels_per_class[c] = np.sum(seg_map == c)

    props['classes_in_slice_per_axis'] = classes_in_slice
    props['number_of_voxels_per_class'] = number_of_voxels_per_class

    with open(pkl_file, 'wb') as f:
        pickle.dump(props, f)

