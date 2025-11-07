from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .training_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml
from typing import List

@dataclass
class TrainingPipelineParams:
    project_name: str
    task_name: str
    tags: List[str]
    configuration_path: str
    s3_data_path: str
    output_data_folder: str
    input_data_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    training_params: TrainingParams
    metric_file_path: str
    local_model_save_path: str
    model_s3_path: str


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        params = yaml.safe_load(input_stream)
        schema = schema.load(params)
        return schema
