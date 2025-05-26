import os
import tempfile
import numpy as np
from torch_fidelity import calculate_metrics
from PIL import Image


def get_inception_score_from_generator(generator_func, num_batches=10, batch_size=100):
    """
    Calculate Inception Score from a generator function.
    
    Args:
        generator_func: Function that generates a batch of images when called with batch_size
        num_batches: Number of batches to generate (default: 10)
        batch_size: Size of each batch (default: 100)
    
    Returns:
        float: Inception Score mean
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and save images
        for i in range(num_batches):
            batch = generator_func(batch_size)
            batch_np = batch.numpy()
            batch_np = batch_np.reshape(-1, 3, 32, 32)
            batch_np = ((batch_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            batch_np = batch_np.transpose(0, 2, 3, 1)  # (N, H, W, C)
            
            # Save each image
            for j, img_array in enumerate(batch_np):
                img = Image.fromarray(img_array)
                img.save(os.path.join(temp_dir, f'img_{i}_{j}.png'))
        
        # Calculate metrics using directory path
        metrics = calculate_metrics(
            input1=temp_dir,
            cuda=True,
            isc=True,
            verbose=False
        )
        
        return metrics['inception_score_mean']
