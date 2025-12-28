import logging
from zenml import step

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        # Simulate data ingestion logic
        logging.info(f"Ingesting data from {self.data_path}")
        return self.data_path
    

@step
def ingest_data_step(data_path: str) -> str:
    """Step to ingest data from a specified path."""
    try:
        logging.info("Starting data ingestion step.")
        ingestion = DataIngestion(data_path)
        data = ingestion.get_data()
        logging.info("Data ingestion completed successfully.")
        return data
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        raise e