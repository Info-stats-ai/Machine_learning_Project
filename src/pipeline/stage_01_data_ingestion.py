from src.logged.logger import logging
from src.components.data_ingestion import DataIngestion


def run():
    logging.info("Stage 01: Data Ingestion — started")
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    logging.info(
        f"Stage 01: Data Ingestion — done | train={train_path} test={test_path}"
    )
    return train_path, test_path


if __name__ == "__main__":
    run()


