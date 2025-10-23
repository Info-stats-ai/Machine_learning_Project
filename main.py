from src.logged.logger import logging
from src.pipeline.stage_01_data_ingestion import run as run_ingestion
from src.pipeline.stage_02_data_transformation import run as run_transform
from src.pipeline.stage_03_model_trainer import run as run_trainer
from src.pipeline.stage_04_model_evaluation import run as run_eval


if __name__ == "__main__":
    logging.info("Pipeline: start")
    train_path, test_path = run_ingestion()
    train_arr, test_arr, pre_path = run_transform(train_path, test_path)
    name, score, model_path = run_trainer(train_arr, test_arr)
    report = run_eval()
    logging.info(
        f"Pipeline: done | best={name} r2={score:.4f} | pre={pre_path} | report={report}"
    )


