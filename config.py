from pathlib import Path

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 10 ** -4
SEQUENCE_LENGTH = 350
DIMENSION_MODEL = 512
LANG_SRC = "en"
LANG_TGT = "it"
MODEL_FOLDER = "weights"
MODEL_BASE_NAME = "tmodel_"
PRELOAD = None
TOKENIZER_FILE = "tokenizer_{0}.json"
EXPERIMENT_NAME = "runs/tmodel"


def get_config():
    return {
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "seq_len": SEQUENCE_LENGTH,
        "d_model": DIMENSION_MODEL,
        "lang_src": LANG_SRC,
        "lang_tgt": LANG_TGT,
        "model_folder": MODEL_FOLDER,
        "model_basename": MODEL_BASE_NAME,
        "preload": PRELOAD,
        "tokenizer_file": TOKENIZER_FILE,
        "experiment_name": EXPERIMENT_NAME,
    }

def get_translate_config():
    return {
        "batch_size": BATCH_SIZE,
        "num_epochs": 10,
        "lr": LEARNING_RATE,
        "seq_len": SEQUENCE_LENGTH,
        "d_model": DIMENSION_MODEL,
        "lang_src": LANG_SRC,
        "lang_tgt": LANG_TGT,
        "model_folder": MODEL_FOLDER,
        "model_basename": MODEL_BASE_NAME,
        "preload": PRELOAD,
        "tokenizer_file": TOKENIZER_FILE,
        "experiment_name": EXPERIMENT_NAME,
    }


def  get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    modle_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / modle_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config, epoch: str):
    # model_folder = f"{config['datasource']}_{config['model_folder']}"
    # model_filename = f"{config['model_basename']}*"
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    modle_filename = f"{model_basename}{epoch}.pt"
    weights_files = list(Path(model_folder).glob(modle_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])