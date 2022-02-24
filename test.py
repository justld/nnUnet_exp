import os
import shutil

from nnUNet_paddle.experiment_planning.utils import crop
from nnUNet_paddle.utils import maybe_mkdir_p, load_json
from nnUNet_paddle.preprocessing.sanity_checks import verify_dataset_integrity
from nnUNet_paddle.utilities.task_name_id_conversion import convert_id_to_task_name
from nnUNet_paddle.experiment_planning.dataset_analyzer import DatasetAnalyzer
from nnUNet_paddle.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnUNet_paddle.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21

task_id = 4

raw_data_dir = '/home/aistudio/converted_data'
task_string = "Task004_Hippocampus"
cropped_data_dir = "/home/aistudio/cropped_data"
preprocessed_data_dir = "/home/aistudio/preprocessed_data"
training_output_dir = "/home/aistudio/result_folder"

# 创建文件夹
raw_data = os.path.join(raw_data_dir, 'raw_data')
cropped_data = os.path.join(raw_data_dir, 'cropped_data')
maybe_mkdir_p(raw_data)
maybe_mkdir_p(cropped_data)
maybe_mkdir_p(preprocessed_data_dir)
maybe_mkdir_p(os.path.join(training_output_dir, "nnUNet"))


task_name = convert_id_to_task_name(task_id, preprocessed_data_dir=preprocessed_data_dir, raw_data_dir=raw_data_dir, cropped_data_dir=cropped_data_dir, training_output_dir=training_output_dir)
print(task_name)
verify_dataset_integrity(os.path.join(raw_data_dir, task_name))              # 检查数据完整性
crop(task_string, raw_data_dir, cropped_data_dir)                            # 裁剪数据

cropped_out_dir = os.path.join(cropped_data_dir, task_name)
preprocessing_output_dir_this_task = os.path.join(preprocessed_data_dir, task_name)

dataset_json = load_json(os.path.join(cropped_out_dir, 'dataset.json'))
modalities = list(dataset_json["modality"].values())
collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=8)  # this class creates the fingerprint
_ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

maybe_mkdir_p(preprocessing_output_dir_this_task)
shutil.copy(os.path.join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
shutil.copy(os.path.join(raw_data_dir, task_name, "dataset.json"), preprocessing_output_dir_this_task)

threads = (8, 8)
print('number of threads: ', threads)

# plan3d
exp_planner_3d = ExperimentPlanner3D_v21(cropped_out_dir, preprocessing_output_dir_this_task)
exp_planner_3d.plan_experiment()

exp_planner_3d.run_preprocessing(threads)

# plan2d
exp_planner_2d = ExperimentPlanner2D_v21(cropped_out_dir, preprocessing_output_dir_this_task)
exp_planner_2d.plan_experiment()
exp_planner_2d.run_preprocessing(threads)