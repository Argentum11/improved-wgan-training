from pathlib import Path
from torch_fidelity import calculate_metrics
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch_fidelity.datasets")


def save_cifar10_test_images_from_batch(batch_path="dataset/cifar-10/test_batch", out_dir="dataset/cifar-10/test_images"):
    os.makedirs(out_dir, exist_ok=True)

    # Check if images already exist (e.g., 0.png to 9999.png)
    existing_images = list(Path(out_dir).glob("*.png"))
    if len(existing_images) >= 10000:
        return

    # Load the batch using pickle
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Extract images
    data = batch[b'data']

    for idx, img_flat in enumerate(data):
        # Reshape and convert to RGB (CIFAR-10 images are 32x32 RGB)
        img = img_flat.reshape(3, 32, 32).transpose(
            1, 2, 0)  # from (3, 32, 32) to (32, 32, 3)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(out_dir, f"{idx}.png"))


def get_FID_for_directory(image_directory: str):
    metrics = calculate_metrics(
        input1=image_directory,
        input2='dataset/cifar-10/test_images',
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False
    )

    return metrics['inception_score_mean'], metrics['frechet_inception_distance']


results = []
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
folder_path = Path("temp")
save_cifar10_test_images_from_batch()
for subdir in folder_path.rglob("*"):
    if subdir.is_dir():
        iteration = int(subdir.name)
        inception_score, fid = get_FID_for_directory(str(subdir))
        results.append((iteration, inception_score, fid))

# Sort by iteration
results.sort(key=lambda x: x[0])
for iteration, inception_score, fid in results:
    print(f"iteration: {iteration}\tinception score: {inception_score:.3f}\tFID: {fid:.3f}")

# Split values
iterations = [r[0] for r in results]
inception_scores = [r[1] for r in results]
fid_scores = [r[2] for r in results]

# Inception Score Plot
plt.figure(figsize=(8, 5))
plt.plot(iterations, inception_scores, marker='o', color='blue')
plt.title('Inception Score over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Inception Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'inception_score.png'))

# FID Score Plot
plt.figure(figsize=(8, 5))
plt.plot(iterations, fid_scores, marker='o', color='blue')
plt.title('FID Score over Iterations')
plt.xlabel('Iteration')
plt.ylabel('FID Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'FID.png'))
