"""
Project 02: Generative Adversarial Networks (GAN)
==================================================
A GAN sets two neural networks against each other in a game.
The Generator tries to produce fake data good enough to fool the
Discriminator. The Discriminator tries to tell real data from fake.
Neither can win permanently -- they push each other to improve until
the generator produces data that is indistinguishable from real.

This adversarial loop is one of the most creative ideas in modern
machine learning. It is behind the synthetic faces that never existed,
the medical scan data that lets hospitals train without real patient
records, and the artwork that blurs the line between human and machine.

We train a DCGAN on MNIST here -- straightforward enough to run on a
laptop, but architecturally identical to the models that generate
photorealistic faces and artwork.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

USE_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    USE_TF = True
    print("Backend: TensorFlow", tf.__version__)
except ImportError:
    print("TensorFlow not found -- running NumPy simulation")


if USE_TF:

    LATENT_DIM = 100
    IMG_SHAPE  = (28, 28, 1)

    # -- Generator -----------------------------------------------
    def build_generator():
        model = keras.Sequential([
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(64,  5, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(1,   5, strides=2, padding="same",
                                   use_bias=False, activation="tanh"),
        ], name="generator")
        return model

    # -- Discriminator -------------------------------------------
    def build_discriminator():
        model = keras.Sequential([
            layers.Conv2D(64, 5, strides=2, padding="same", input_shape=IMG_SHAPE),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, 5, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ], name="discriminator")
        return model

    # -- GAN training loop ---------------------------------------
    class DCGAN(keras.Model):
        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator     = generator
            self.discriminator = discriminator

        def compile(self, g_opt, d_opt, loss_fn):
            super().compile()
            self.g_opt   = g_opt
            self.d_opt   = d_opt
            self.loss_fn = loss_fn

        def train_step(self, real_images):
            batch = tf.shape(real_images)[0]
            noise = tf.random.normal([batch, LATENT_DIM])

            # Train discriminator
            with tf.GradientTape() as tape:
                fake    = self.generator(noise, training=True)
                r_pred  = self.discriminator(real_images, training=True)
                f_pred  = self.discriminator(fake,        training=True)
                r_loss  = self.loss_fn(tf.ones_like(r_pred),  r_pred)
                f_loss  = self.loss_fn(tf.zeros_like(f_pred), f_pred)
                d_loss  = r_loss + f_loss
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )

            # Train generator
            noise = tf.random.normal([batch, LATENT_DIM])
            with tf.GradientTape() as tape:
                fake   = self.generator(noise, training=True)
                f_pred = self.discriminator(fake, training=True)
                g_loss = self.loss_fn(tf.ones_like(f_pred), f_pred)
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
            return {"d_loss": d_loss, "g_loss": g_loss}

    # -- Load MNIST ----------------------------------------------
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_train = (x_train - 127.5) / 127.5   # scale to [-1, 1]

    dataset = tf.data.Dataset.from_tensor_slices(x_train)\
                              .shuffle(60000).batch(256)

    gen   = build_generator()
    disc  = build_discriminator()
    gan   = DCGAN(gen, disc)
    gan.compile(
        g_opt   = keras.optimizers.Adam(2e-4, beta_1=0.5),
        d_opt   = keras.optimizers.Adam(2e-4, beta_1=0.5),
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True),
    )

    print("\nTraining DCGAN on MNIST (10 epochs)...")
    g_losses, d_losses = [], []

    FIXED_NOISE = tf.random.normal([25, LATENT_DIM])

    for epoch in range(10):
        epoch_g, epoch_d = [], []
        for batch in dataset:
            logs = gan.train_step(batch)
            epoch_g.append(float(logs["g_loss"]))
            epoch_d.append(float(logs["d_loss"]))
        g_losses.append(np.mean(epoch_g))
        d_losses.append(np.mean(epoch_d))
        print(f"  Epoch {epoch+1:2d} | G={g_losses[-1]:.4f} | D={d_losses[-1]:.4f}")

    # -- Save generated samples ----------------------------------
    fake_imgs = gen(FIXED_NOISE, training=False).numpy()
    fake_imgs = (fake_imgs * 0.5 + 0.5)           # back to [0, 1]

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.patch.set_facecolor("#0f1117")
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_imgs[i, :, :, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("DCGAN -- Generated MNIST Digits After Training",
                 color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/02_gan_generated.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/02_gan_generated.png")

    # -- Loss curves ---------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(g_losses, color="#4f9cf9", lw=2, marker="o", label="Generator Loss")
    ax.plot(d_losses, color="#e74c3c", lw=2, marker="o", label="Discriminator Loss")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss",  color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1d27", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")
    plt.suptitle("DCGAN -- Generator vs Discriminator Loss",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/02_gan_losses.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/02_gan_losses.png")

else:
    # NumPy simulation
    print("\nSimulating GAN training dynamics with NumPy...")
    np.random.seed(42)
    epochs = np.arange(1, 51)

    # Typical GAN loss behaviour: D starts high, G improves
    d_loss = 1.4 * np.exp(-0.03 * epochs) + 0.6 + np.random.randn(50) * 0.05
    g_loss = 0.5 + 1.5 * np.exp(-0.05 * epochs) + np.random.randn(50) * 0.08

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0f1117")

    # Loss curves
    axes[0].set_facecolor("#1a1d27")
    axes[0].plot(epochs, g_loss, color="#4f9cf9", lw=2, label="Generator Loss")
    axes[0].plot(epochs, d_loss, color="#e74c3c", lw=2, label="Discriminator Loss")
    axes[0].axhline(0.693, color="white", lw=1, linestyle="--", alpha=0.5,
                    label="Nash Equilibrium (ln2)")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("Loss",  color="white")
    axes[0].tick_params(colors="white")
    axes[0].legend(facecolor="#1a1d27", labelcolor="white")
    axes[0].set_title("Generator vs Discriminator Loss", color="white", fontsize=11)
    for sp in axes[0].spines.values(): sp.set_edgecolor("#2d3142")

    # Distribution shift: real vs generated over training
    x = np.linspace(-4, 4, 300)
    from scipy.stats import norm
    axes[1].set_facecolor("#1a1d27")
    axes[1].fill_between(x, norm.pdf(x, 0, 1), alpha=0.5,
                         color="#22c55e", label="Real data")
    axes[1].fill_between(x, norm.pdf(x, -2.5, 1.5), alpha=0.5,
                         color="#4f9cf9", label="Generator (early)")
    axes[1].fill_between(x, norm.pdf(x, 0.1, 1.05), alpha=0.5,
                         color="#f59e0b", label="Generator (converged)")
    axes[1].set_xlabel("Data space", color="white")
    axes[1].set_ylabel("Density",    color="white")
    axes[1].tick_params(colors="white")
    axes[1].legend(facecolor="#1a1d27", labelcolor="white")
    axes[1].set_title("Distribution Matching Over Training", color="white", fontsize=11)
    for sp in axes[1].spines.values(): sp.set_edgecolor("#2d3142")

    plt.suptitle("Generative Adversarial Network -- Training Dynamics",
                 color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/02_gan_overview.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/02_gan_overview.png")

print("\nProject 02 -- GAN complete.\n")
