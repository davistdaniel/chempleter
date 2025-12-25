# chempleter

__version__ = "0.1.0"

from pathlib import Path
import logging

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def start_experiment(experiment_name,working_dir=None):
    """
    Docstring for start_experiment
    
    :param experiment_name: Description
    :param working_dir: Description
    """

    if not working_dir:
        working_dir = Path().cwd() / experiment_name
    else:
        working_dir = Path(working_dir) /experiment_name
    
    # make dir
    working_dir.mkdir(parents=True, exist_ok=True)

    return working_dir


