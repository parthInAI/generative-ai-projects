"""
Project 01: Variational Autoencoder (VAE)
=========================================
A VAE learns to compress data into a structured latent space and
reconstruct it. Unlike a plain autoencoder, it forces the latent
space to follow a smooth Gaussian distribution, which means you can
sample from it and generate entirely new, realistic data points.

This is the idea that quietly powers image editing tools, drug
discovery pipelines, and anomaly detection systems. Every time a
model "imagines" a new face, molecule, or soundscape, it is drawing
from a latent space -- and VAEs are where that idea was born clearly
enough to be used in practice.

We train on MNIST here to keep things runnable without a GPU, but
the same architecture scales directly to images, audio, and graphs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# Try TensorFlow/Keras first, then PyTorch
USE_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    USE_TF = True
    print("Backend: TensorFlow", tf.__version__)
except ImportError:
    print("TensorFlow not found -- running NumPy simulation")


# ---------------------------------------------------------------
# Option A: TensorFlow/Keras VAE
# ---------------------------------------------------------------
if USE_TF:

    # -- Sampling layer ------------------------------------------
    class Sampling(layers.Layer):
        """Reparameterisation trick: z = mu + eps * sigma"""
        def call(self, inputs):
            mu, log_var = inputs
            eps = tf.random.normal(tf.shape(mu))
            return mu + tf.exp(0.5 * log_var) * eps

    # -- Build VAE -----------------------------------------------
    LATENT_DIM = 2
    IMG_DIM    = 784   # 28x28 MNIST

    # Encoder
    enc_input = keras.Input(shape=(IMG_DIM,))
    x  = layers.Dense(256, activation="relu")(enc_input)
    x  = layers.Dense(128, activation="relu")(x)
    mu      = layers.Dense(LATENT_DIM, name="mu")(x)
    log_var = layers.Dense(LATENT_DIM, name="log_var")(x)
    z       = Sampling()([mu, log_var])
    encoder = keras.Model(enc_input, [mu, log_var, z], name="encoder")

    # Decoder
    dec_input = keras.Input(shape=(LATENT_DIM,))
    y  = layers.Dense(128, activation="relu")(dec_input)
    y  = layers.Dense(256, activation="relu")(y)
    dec_out = layers.Dense(IMG_DIM, activation="sigmoid")(y)
    decoder = keras.Model(dec_input, dec_out, name="decoder")

    # Full VAE with custom loss
    class VAE(keras.Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def call(self, x):
            mu, log_var, z = self.encoder(x)
            return self.decoder(z), mu, log_var

        def train_step(self, data):
            x = data
            with tf.GradientTape() as tape:
                x_hat, mu, log_var = self(x, training=True)
                recon  = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(x, x_hat),
                        axis=-1
                    )
                )
                kl     = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
                )
                loss   = recon + kl
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {"loss": loss, "reconstruction": recon, "kl": kl}

    # -- Load and prepare MNIST ----------------------------------
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))

    print("\nTraining VAE on MNIST (5 epochs)...")
    history = vae.fit(x_train, epochs=5, batch_size=256, verbose=1)

    # -- Visualise reconstructions -------------------------------
    n = 10
    samples = x_test[:n]
    mu_s, _, z_s = encoder.predict(samples, verbose=0)
    recons        = decoder.predict(z_s, verbose=0)

    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    fig.patch.set_facecolor("#0f1117")
    for i in range(n):
        for row, img in enumerate([samples[i], recons[i]]):
            axes[row, i].imshow(img.reshape(28, 28), cmap="gray")
            axes[row, i].axis("off")
    axes[0, 0].set_title("Original",      color="white", fontsize=9)
    axes[1, 0].set_title("Reconstructed", color="white", fontsize=9)
    plt.suptitle("VAE -- Reconstruction Quality", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/01_vae_reconstructions.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/01_vae_reconstructions.png")

    # -- 2-D latent space ----------------------------------------
    mu_all, _, _ = encoder.predict(x_test, verbose=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    sc = ax.scatter(mu_all[:, 0], mu_all[:, 1],
                    c=y_test, cmap="tab10", s=4, alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.set_label("Digit Class", color="white")
    ax.set_xlabel("Latent Dimension 1", color="white")
    ax.set_ylabel("Latent Dimension 2", color="white")
    ax.tick_params(colors="white")
    plt.suptitle("VAE -- 2D Latent Space (MNIST)", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/01_vae_latent_space.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/01_vae_latent_space.png")

    # -- Sample from latent space --------------------------------
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    fig.patch.set_facecolor("#0f1117")
    z_samples = np.random.normal(0, 1, (50, LATENT_DIM)).astype("float32")
    generated  = decoder.predict(z_samples, verbose=0)
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle("VAE -- Generated Digits Sampled from Latent Space",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/01_vae_generated_samples.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/01_vae_generated_samples.png")

    # -- Loss curve ----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(history.history["loss"], color="#4f9cf9", lw=2, label="Total Loss")
    ax.plot(history.history["reconstruction"], color="#22c55e", lw=2,
            label="Reconstruction Loss")
    ax.plot(history.history["kl"], color="#f59e0b", lw=2, label="KL Divergence")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1d27", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")
    plt.suptitle("VAE -- Training Loss Breakdown", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/01_vae_training_loss.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/01_vae_training_loss.png")

# ---------------------------------------------------------------
# Option B: NumPy simulation (no deep learning framework)
# ---------------------------------------------------------------
else:
    print("\nSimulating VAE concepts with NumPy...")
    np.random.seed(42)

    # Simulate a trained latent space
    n_points  = 3000
    labels    = np.random.randint(0, 10, n_points)
    centers   = np.random.randn(10, 2) * 3
    latent    = centers[labels] + np.random.randn(n_points, 2) * 0.6

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")

    # Latent space
    sc = axes[0].scatter(latent[:, 0], latent[:, 1],
                         c=labels, cmap="tab10", s=10, alpha=0.8)
    axes[0].set_title("VAE Latent Space -- Digit Clusters",
                      color="white", fontsize=11)
    axes[0].set_xlabel("z1", color="white")
    axes[0].set_ylabel("z2", color="white")

    # Reconstruction quality simulation
    epochs = np.arange(1, 21)
    recon  = 180 * np.exp(-0.25 * epochs) + 20 + np.random.randn(20) * 2
    kl     = 3   * (1 - np.exp(-0.3  * epochs)) + np.random.randn(20) * 0.1
    total  = recon + kl

    axes[1].plot(epochs, total, color="#4f9cf9", lw=2, label="Total Loss")
    axes[1].plot(epochs, recon, color="#22c55e", lw=2, label="Reconstruction")
    axes[1].plot(epochs, kl,    color="#f59e0b", lw=2, label="KL Divergence")
    axes[1].set_title("VAE Training Loss Curve (Simulated)",
                      color="white", fontsize=11)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss",  color="white")
    axes[1].legend(facecolor="#1a1d27", labelcolor="white")

    plt.suptitle("Variational Autoencoder -- Concepts", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/01_vae_overview.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/01_vae_overview.png")

print("\nProject 01 -- VAE complete.\n")
