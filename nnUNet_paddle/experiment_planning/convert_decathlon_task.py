import os
import argparse
import shutil
from multiprocessing import Pool

import sys
sys.path.append('/home/aistudio')
# print(os.path.dirname(__file__))

from nnUNet_paddle.utils import subdirs, subfiles, remove_trailing_slash, maybe_mkdir_p
from nnUNet_paddle.experiment_planning.common_utils import split_4d_nifti


def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    subf = subdirs(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [os.remove(i) for i in subfiles(os.path.join(folder, 'imagesTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(os.path.join(folder, 'labelsTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(os.path.join(folder, 'imagesTs'), prefix=".")]


def split_4d(input_folder, num_processes=8, overwrite_task_output_id=None, output_dir='./'):
    assert os.path.isdir(os.path.join(input_folder, "imagesTr")) and os.path.isdir(os.path.join(input_folder, "labelsTr")) and \
           os.path.isfile(os.path.join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "imagesTr and labelsTr subdirs and the dataset.json file"

    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    full_task_name = input_folder.split("/")[-1]

    assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]

    output_folder = os.path.join(output_dir, "Task%03.0d_" % overwrite_task_output_id + task_name)

    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = os.path.join(output_folder, subdir)
        if not os.path.isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = os.path.join(input_folder, subdir)
        nii_files = [os.path.join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(os.path.join(input_folder, "labelsTr"), os.path.join(output_folder, "labelsTr"))

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(os.path.join(input_folder, "dataset.json"), output_folder)


def main():
    parser = argparse.ArgumentParser(description="The MSD provides data as 4D Niftis with the modality being the first"
                                                 " dimension. We think this may be cumbersome for some users and "
                                                 "therefore expect 3D niftixs instead, with one file per modality. "
                                                 "This utility will convert 4D MSD data into the format nnU-Net "
                                                 "expects")
    parser.add_argument("-i", help="Input folder. Must point to a TaskXX_TASKNAME folder as downloaded from the MSD "
                                   "website", required=True)
    parser.add_argument("-o", help="Output folder. output folder for dataset", default="./", required=False)
    parser.add_argument("-p", required=False, default=8, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % 8)
    parser.add_argument("-output_task_id", required=False, default=None, type=int,
                        help="If specified, this will overwrite the task id in the output folder. If unspecified, the "
                             "task id of the input folder will be used.")
    args = parser.parse_args()

    crawl_and_remove_hidden_from_decathlon(args.i)

    split_4d(args.i, args.p, args.output_task_id, output_dir=args.o)

if __name__ == '__main__':
    main()

