"""
Project 03: Diffusion Models
=============================
Diffusion models learn to generate data by learning to reverse a
noise process. You start with a real image, gradually add Gaussian
noise over hundreds of steps until it is pure static, then train a
neural network to run that process backwards -- to denoise step by
step until a clean image emerges from noise.

This is the core idea behind Stable Diffusion, DALL-E 2, and Imagen.
It is arguably the most important generative modelling breakthrough
of the past decade. The reason it works so well is that denoising is
a well-defined, learnable task at every step -- unlike GANs, there is
no adversarial instability and no mode collapse.

This implementation covers the full forward and reverse diffusion
process, a simple U-Net denoiser, noise scheduling (linear and cosine),
and sampling with DDPM. Trained on MNIST for accessibility, but the
architecture is identical to production text-to-image systems.
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
    print("TensorFlow not found -- demonstrating diffusion concepts with NumPy")


# ---------------------------------------------------------------
# Noise schedules (framework-independent)
# ---------------------------------------------------------------
def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

def cosine_beta_schedule(T=1000, s=0.008):
    steps = np.arange(T + 1, dtype=np.float64)
    f     = np.cos(((steps / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alpha = f / f[0]
    betas = 1 - (alpha[1:] / alpha[:-1])
    return np.clip(betas, 0, 0.999).astype(np.float32)

def precompute_schedule(betas):
    alphas          = 1.0 - betas
    alpha_bar       = np.cumprod(alphas)
    sqrt_ab         = np.sqrt(alpha_bar)
    sqrt_one_minus  = np.sqrt(1.0 - alpha_bar)
    return alphas, alpha_bar, sqrt_ab, sqrt_one_minus

T      = 200   # reduced for fast demo
betas  = cosine_beta_schedule(T)
alphas, alpha_bar, sqrt_ab, sqrt_one_minus = precompute_schedule(betas)

def q_sample(x0, t, noise=None):
    """Forward process: add noise to x0 at timestep t"""
    if noise is None:
        noise = np.random.randn(*x0.shape).astype(np.float32)
    a  = sqrt_ab[t].reshape((-1,) + (1,) * (x0.ndim - 1))
    b  = sqrt_one_minus[t].reshape((-1,) + (1,) * (x0.ndim - 1))
    return a * x0 + b * noise, noise


# ---------------------------------------------------------------
# Visualise the forward diffusion process
# ---------------------------------------------------------------
try:
    from tensorflow import keras as _k
    (x_train, _), _ = _k.datasets.mnist.load_data()
    sample = (x_train[7].astype(np.float32) / 127.5) - 1.0
    sample = sample[np.newaxis, :, :, np.newaxis]
    HAS_MNIST = True
except Exception:
    np.random.seed(42)
    sample    = np.random.randn(1, 28, 28, 1).astype(np.float32) * 0.3
    HAS_MNIST = False

timesteps_to_show = [0, 10, 25, 50, 75, 100, 150, 199]

fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(22, 3))
fig.patch.set_facecolor("#0f1117")

for ax, t in zip(axes, timesteps_to_show):
    t_arr     = np.array([t])
    noisy, _  = q_sample(sample, t_arr)
    img       = noisy[0, :, :, 0]
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t={t}", color="white", fontsize=9)
    ax.axis("off")

plt.suptitle("Diffusion Forward Process -- Gradual Noise Addition",
             color="white", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/03_diffusion_forward.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("Saved: outputs/03_diffusion_forward.png")


# ---------------------------------------------------------------
# Noise schedule visualisation
# ---------------------------------------------------------------
lin_betas   = linear_beta_schedule(T)
cos_betas   = cosine_beta_schedule(T)
_, lin_ab, _, _ = precompute_schedule(lin_betas)
_, cos_ab, _, _ = precompute_schedule(cos_betas)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.patch.set_facecolor("#0f1117")
ts = np.arange(T)

for ax in axes:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")

axes[0].plot(ts, lin_betas, color="#4f9cf9", lw=2, label="Linear")
axes[0].plot(ts, cos_betas, color="#f59e0b", lw=2, label="Cosine")
axes[0].set_title("Beta Schedule (noise added per step)",
                  color="white", fontsize=10)
axes[0].set_xlabel("Timestep t"); axes[0].set_ylabel("Beta")
axes[0].legend(facecolor="#1a1d27", labelcolor="white")

axes[1].plot(ts, lin_ab, color="#4f9cf9", lw=2, label="Linear")
axes[1].plot(ts, cos_ab, color="#f59e0b", lw=2, label="Cosine")
axes[1].set_title("Alpha-bar (signal retained)",
                  color="white", fontsize=10)
axes[1].set_xlabel("Timestep t"); axes[1].set_ylabel("alpha_bar")
axes[1].legend(facecolor="#1a1d27", labelcolor="white")

axes[2].plot(ts, np.sqrt(lin_ab),       color="#4f9cf9", lw=2, label="Signal (linear)")
axes[2].plot(ts, np.sqrt(1 - lin_ab),   color="#4f9cf9", lw=2, linestyle="--",
             label="Noise (linear)")
axes[2].plot(ts, np.sqrt(cos_ab),       color="#f59e0b", lw=2, label="Signal (cosine)")
axes[2].plot(ts, np.sqrt(1 - cos_ab),   color="#f59e0b", lw=2, linestyle="--",
             label="Noise (cosine)")
axes[2].set_title("Signal vs Noise Ratio", color="white", fontsize=10)
axes[2].set_xlabel("Timestep t"); axes[2].set_ylabel("Weight")
axes[2].legend(facecolor="#1a1d27", labelcolor="white", fontsize=7)

plt.suptitle("Diffusion Noise Schedules -- Linear vs Cosine",
             color="white", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/03_diffusion_schedules.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("Saved: outputs/03_diffusion_schedules.png")


# ---------------------------------------------------------------
# Simple denoiser training (TensorFlow only)
# ---------------------------------------------------------------
if USE_TF:

    def sinusoidal_embedding(t, dim=32):
        half  = dim // 2
        freqs = tf.exp(-np.log(10000) * tf.range(half, dtype=tf.float32) / half)
        args  = tf.cast(t[:, None], tf.float32) * freqs[None]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)

    def build_simple_unet(img_size=28, channels=1, time_dim=32):
        img_in  = keras.Input(shape=(img_size, img_size, channels))
        t_in    = keras.Input(shape=(), dtype=tf.int32)

        t_emb   = layers.Lambda(
            lambda t: sinusoidal_embedding(t, time_dim))(t_in)
        t_emb   = layers.Dense(64, activation="swish")(t_emb)
        t_emb   = layers.Dense(64, activation="swish")(t_emb)

        # Encoder
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(img_in)
        t_proj = layers.Dense(32)(t_emb)
        x = x + t_proj[:, None, None, :]
        x = layers.Conv2D(64, 3, padding="same", activation="relu", strides=2)(x)
        x_skip = x

        # Bottleneck
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64,  3, padding="same", activation="relu")(x)

        # Decoder
        x = x + x_skip
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same",
                                   activation="relu")(x)
        out = layers.Conv2D(channels, 1, padding="same")(x)
        return keras.Model([img_in, t_in], out, name="denoiser")

    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 127.5 - 1.0

    denoiser = build_simple_unet()
    denoiser.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="mse")
    denoiser.summary()

    # Training loop
    EPOCHS     = 5
    BATCH_SIZE = 256
    loss_hist  = []

    print("\nTraining diffusion denoiser (5 epochs)...")
    for epoch in range(EPOCHS):
        idx  = np.random.permutation(len(x_train))
        ep_loss = []
        for i in range(0, len(x_train) - BATCH_SIZE, BATCH_SIZE):
            b     = x_train[idx[i:i+BATCH_SIZE]]
            t     = np.random.randint(0, T, size=len(b))
            noisy, noise = q_sample(b, t)
            loss  = denoiser.train_on_batch(
                [noisy.astype(np.float32), t.astype(np.int32)],
                noise.astype(np.float32)
            )
            ep_loss.append(loss)
        loss_hist.append(np.mean(ep_loss))
        print(f"  Epoch {epoch+1} | Loss = {loss_hist[-1]:.5f}")

    # DDPM reverse sampling
    print("\nSampling 16 images via DDPM reverse process...")
    x = tf.random.normal([16, 28, 28, 1])
    for step in reversed(range(T)):
        t_batch  = tf.fill([16], step)
        eps_pred = denoiser([x, t_batch], training=False)
        alpha_t  = alphas[step]
        ab_t     = alpha_bar[step]
        ab_prev  = alpha_bar[step - 1] if step > 0 else 1.0
        coef     = (1 - alpha_t) / tf.sqrt(1 - ab_t)
        x        = (1 / tf.sqrt(alpha_t)) * (x - coef * eps_pred)
        if step > 0:
            noise  = tf.random.normal(x.shape)
            sigma  = tf.sqrt(betas[step] * (1 - ab_prev) / (1 - ab_t))
            x      = x + sigma * noise

    generated = x.numpy()
    generated = np.clip((generated * 0.5 + 0.5), 0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.patch.set_facecolor("#0f1117")
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i, :, :, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("Diffusion Model -- DDPM Generated Samples",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/03_diffusion_generated.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/03_diffusion_generated.png")

    # Training loss
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(range(1, EPOCHS+1), loss_hist, color="#4f9cf9",
            lw=2, marker="o")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")
    plt.suptitle("Diffusion Denoiser -- Training Loss", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/03_diffusion_training.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/03_diffusion_training.png")

else:
    # Simulate reverse denoising visually
    print("\nSimulating DDPM reverse denoising with NumPy...")
    np.random.seed(0)
    pure_noise = np.random.randn(1, 28, 28, 1).astype(np.float32)

    steps_show = [199, 150, 100, 75, 50, 25, 10, 0]
    fig, axes  = plt.subplots(1, len(steps_show), figsize=(22, 3))
    fig.patch.set_facecolor("#0f1117")

    for ax, step in zip(axes, steps_show):
        ratio = step / 199.0
        img   = pure_noise * ratio + sample * (1 - ratio)
        img   = img[0, :, :, 0]
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t={step}", color="white", fontsize=9)
        ax.axis("off")

    plt.suptitle("Diffusion Reverse Process -- Noise to Image (Simulated)",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/03_diffusion_reverse.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: outputs/03_diffusion_reverse.png")

print("\nProject 03 -- Diffusion Models complete.\n")
