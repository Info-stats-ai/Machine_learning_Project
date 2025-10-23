from src.logged.logger import logging
from src.components.model_trainer import ModelTrainer


def run(train_arr, test_arr):
    logging.info("Stage 03: Model Training — started")
    name, score, model_path = ModelTrainer().initiate_model_training(train_arr, test_arr)
    logging.info(
        f"Stage 03: Model Training — done | best={name} r2={score:.4f} model={model_path}"
    )
    return name, score, model_path


if __name__ == "__main__":
    pass
