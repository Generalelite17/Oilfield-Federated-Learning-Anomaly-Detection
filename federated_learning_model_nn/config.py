from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "csv_data" / "CIC" / "BenignTraffic.pcap_Flow.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CONTAMINATION = 0.01

NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001

TEST_SIZE = 0.2