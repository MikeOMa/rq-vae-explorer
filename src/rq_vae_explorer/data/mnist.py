"""MNIST dataset loading utilities."""

import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load MNIST dataset, normalized to [0, 1].

    Returns:
        Tuple of ((train_images, train_labels), (test_images, test_labels))
        Images have shape (N, 28, 28, 1) and dtype float32.
        Labels have shape (N,) and dtype int.
    """
    ds_train = tfds.load("mnist", split="train", as_supervised=True)
    ds_test = tfds.load("mnist", split="test", as_supervised=True)

    def extract_images(ds):
        images = []
        labels = []
        for image, label in tfds.as_numpy(ds):
            images.append(image)
            labels.append(label)
        images = np.stack(images).astype(np.float32) / 255.0
        labels = np.array(labels)
        return images, labels

    train_images, train_labels = extract_images(ds_train)
    test_images, test_labels = extract_images(ds_test)

    return (train_images, train_labels), (test_images, test_labels)


def create_data_iterator(
    data: tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
):
    """Create an infinite iterator over batches.

    Args:
        data: Tuple of (images, labels)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle each epoch
        rng: Random generator for shuffling

    Yields:
        Tuple of (batch_images, batch_labels) as JAX arrays
    """
    images, labels = data
    n_samples = len(images)
    rng = rng or np.random.default_rng()

    while True:
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            if end > n_samples:
                # Skip incomplete batch at end
                break
            batch_idx = indices[start:end]
            yield jnp.array(images[batch_idx]), jnp.array(labels[batch_idx])
