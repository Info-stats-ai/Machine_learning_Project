from src.logged.logger import logging
from src.components.data_transformation import DataTransformation


def run(train_path: str, test_path: str):
    logging.info("Stage 02: Data Transformation — started")
    train_arr, test_arr, pre_path = DataTransformation().initiate_data_transformation(
        train_path, test_path
    )
    logging.info(
        f"Stage 02: Data Transformation — done | preprocessor={pre_path}"
    )
    return train_arr, test_arr, pre_path


if __name__ == "__main__":
    # Optional manual run
    run("artifacts/train.csv", "artifacts/test.csv")


