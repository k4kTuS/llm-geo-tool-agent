from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'data'
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'

ASSETS_DIR = SRC_ROOT / 'assets'
RESOURCES_DIR = SRC_ROOT / 'resources'