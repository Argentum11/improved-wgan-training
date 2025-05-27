from pathlib import Path
from torch_fidelity import calculate_metrics

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch_fidelity.datasets")


def get_inception_score_for_directory(image_directory: str):
    metrics = calculate_metrics(
        input1=image_directory,
        cuda=True,
        isc=True,
        verbose=False
    )

    return metrics['inception_score_mean']


folder_path = Path("temp")

for subdir in folder_path.rglob("*"):
    if subdir.is_dir():
        iteration = int(subdir.name)
        inception_score = get_inception_score_for_directory(str(subdir))
        print(f"iteration: {iteration}\tinception score: {inception_score}")
