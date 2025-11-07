import os
import pandas as pd
import click
from clearml import Task, OutputModel
from dotenv import load_dotenv
from src.data import read_data, split_train_validate_data, load_data_from_s3
from src.entities import (
    TrainingPipelineParams,
    read_training_pipeline_params,
    report_text,
    report_table,
)
from src.features import make_features, build_transformer, extract_target
from src.models import train_model, model_predict, evaluate_model, save_model

load_dotenv()


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    task = Task.init(
        project_name=training_pipeline_params.project_name,
        task_name=training_pipeline_params.task_name,
        tags=training_pipeline_params.tags,
    )
    config_file_yaml = task.connect_configuration(
        name="config file",
        configuration=training_pipeline_params.configuration_path,
    )

    report_text(f"Load data from minio...")
    load_data_from_s3(
        training_pipeline_params.s3_data_path,
        training_pipeline_params.output_data_folder,
    )
    report_text(f"Read data file...")
    df = read_data(training_pipeline_params.input_data_path)
    report_text(f"Data loaded successfully, total number of objects: {df.shape[0]}\n")
    report_text(f"Split data to train and validate...")
    df_train, df_validate = split_train_validate_data(
        df, training_pipeline_params.splitting_params
    )
    report_text(
        f"Train data size: {df_train.shape[0]}, validate data size: {df_validate.shape[0]}\n"
    )
    report_text(f"Start build transformer for features...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(df_train)
    report_text(f"Pipeline for features was built and fitted!!!\n")
    report_text(f"Start make features...")
    train_features = make_features(df_train, transformer)
    validate_features = make_features(df_validate, transformer)
    report_text(
        f"Prepared train and validate features, train features shape: {train_features.shape}, validate features shape: {validate_features.shape}\n"
    )
    train_target = extract_target(df_train, training_pipeline_params.feature_params)
    validate_target = extract_target(
        df_validate, training_pipeline_params.feature_params
    )
    report_text(f"Start training model...")
    model = train_model(
        train_features, train_target, training_pipeline_params.training_params
    )
    report_text(f"Model was trained successfully!!!\n")
    report_text(f"Start getting metrics...!!!\n")
    predict = model_predict(validate_features, model)

    metrics: pd.DataFrame = evaluate_model(predict, validate_target)
    report_table(metrics)
    model = save_model(
        model,
        training_pipeline_params.local_model_save_path,
        training_pipeline_params.model_s3_path,
    )
    # Create output model and connect it to the task
    output_model = OutputModel(task=task)
    output_model.update_weights(training_pipeline_params.local_model_save_path)
    report_text(
        f"Model saved locally at {training_pipeline_params.local_model_save_path} and to minio: {training_pipeline_params.model_s3_path}!!!\n"
    )


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    train_pipeline(training_pipeline_params)


if __name__ == "__main__":
    train_pipeline_command()
