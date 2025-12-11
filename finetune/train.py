import sys
from pathlib import Path

import sys
import os
# Get the absolute path of the current file (e.g., .../project/inference)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (the parent directory of 'inference')
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Add the project root directory to Python's module search path
# This ensures imports such as 'from finetune.datasets.utils import ...' work,
# regardless of where the script is executed.
sys.path.append(ROOT_DIR)

from finetune.models.utils import get_model_cls
from finetune.schemas import Args

def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()