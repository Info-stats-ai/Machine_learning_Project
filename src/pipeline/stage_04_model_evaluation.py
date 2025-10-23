from src.logged.logger import logging
from src.components.model_evaluation import ModelEvaluator


def run():
    logging.info("Stage 04: Model Evaluation — started")
    report = ModelEvaluator().evaluate_on_test()
    logging.info(f"Stage 04: Model Evaluation — done | {report}")
    return report


if __name__ == "__main__":
    run()


