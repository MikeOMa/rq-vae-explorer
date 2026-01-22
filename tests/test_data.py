import jax.numpy as jnp
from rq_vae_explorer.data import load_mnist, create_data_iterator


def test_load_mnist_returns_normalized_images():
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    # Check we have data
    assert len(train_images) == 60000
    assert len(test_images) == 10000

    # Check shape and normalization
    assert train_images[0].shape == (28, 28, 1)
    assert train_images.min() >= 0.0
    assert train_images.max() <= 1.0

    # Check labels
    assert train_labels.min() >= 0
    assert train_labels.max() <= 9


def test_create_data_iterator_yields_batches():
    (train_images, train_labels), _ = load_mnist()
    iterator = create_data_iterator((train_images, train_labels), batch_size=32)

    batch_images, batch_labels = next(iterator)
    assert batch_images.shape == (32, 28, 28, 1)
    assert batch_images.dtype == jnp.float32
    assert batch_labels.shape == (32,)
