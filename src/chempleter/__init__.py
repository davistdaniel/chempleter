# chempleter


import logging

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


from pathlib import Path

def start_workflow(experiment_name,working_dir=None):

    if not working_dir:
        working_dir = Path().cwd() / experiment_name
    else:
        working_dir = Path(working_dir) /experiment_name
    
    # make dir
    working_dir.mkdir(parents=True, exist_ok=True)

    return working_dir