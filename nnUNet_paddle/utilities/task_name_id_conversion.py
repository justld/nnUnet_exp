import os
import numpy as np

from nnUNet_paddle.utils import subdirs, subfiles


def convert_id_to_task_name(task_id: int, preprocessed_data_dir, raw_data_dir, cropped_data_dir, training_output_dir):
    startswith = "Task%03.0d" % task_id
    if preprocessed_data_dir is not None:
        candidates_preprocessed = subdirs(preprocessed_data_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if raw_data_dir is not None:
        candidates_raw = subdirs(raw_data_dir, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if cropped_data_dir is not None:
        candidates_cropped = subdirs(cropped_data_dir, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if training_output_dir is not None:
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            if os.path.isdir(os.path.join(training_output_dir, m)):
                candidates_trained_models += subdirs(os.path.join(training_output_dir, m), prefix=startswith, join=False)

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (task_id, raw_data_dir, preprocessed_data_dir,
                                                               cropped_data_dir))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                           "nnU-Net knows where raw and preprocessed data are located.")
    return unique_candidates[0]


def convert_task_name_to_id(task_name: str):
    assert task_name.startswith("Task")
    task_id = int(task_name[4:7])
    return task_id
