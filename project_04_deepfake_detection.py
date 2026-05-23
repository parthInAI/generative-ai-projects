"""
Project 04: Deepfake Detection System
======================================
Generative models can now produce faces, voices, and videos that are
nearly indistinguishable from real ones. This is remarkable technology --
and a serious problem. A deepfake of a CEO announcing fake news, a
politician saying something they never said, or a person placed in a
scene they were never in -- these are not hypothetical harms. They are
happening now, at scale.

This project is the other side of the generative coin. If we can build
models that create convincing fakes, we must also build models that can
detect them. Deepfake detectors look for the subtle artefacts that
generative models leave behind -- unnatural blending at face boundaries,
frequency domain anomalies, inconsistent noise patterns, and physiological
signals like blinking that GANs often get wrong.

We implement a CNN-based binary classifier trained on real vs synthetic
face features, with frequency-domain analysis (DCT), texture consistency
checks, and a multi-signal ensemble that mirrors production deepfake
detectors used by social media platforms today.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, confusion_matrix)
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  Project 04: Deepfake Detection System")
print("=" * 60)

# ---------------------------------------------------------------
# Simulate realistic face feature dataset
# ---------------------------------------------------------------
# Real faces have: consistent high-frequency texture, natural
# blending, normal DCT energy, natural skin tone variance
# Fake faces have: GAN artefacts in DCT, smooth over-blending,
# checkerboard patterns, abnormal frequency peaks
# ---------------------------------------------------------------
np.random.seed(42)
n_real  = 3000
n_fake  = 3000
IMG_SZ  = 64

def extract_features(imgs, is_fake):
    """Extract 12 detection features per image."""
    feats = []
    for img in imgs:
        # DCT energy distribution
        dct2d     = dct(dct(img, axis=0), axis=1)
        high_freq = np.mean(np.abs(dct2d[16:, 16:]))
        low_freq  = np.mean(np.abs(dct2d[:8,  :8]))
        dct_ratio = high_freq / (low_freq + 1e-8)

        # Texture consistency (Laplacian variance)
        gy      = np.diff(img, axis=0)
        gx      = np.diff(img, axis=1)
        grad_var = np.var(np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2))

        # Colour channel statistics
        r, g, b     = img[:,:,0], img[:,:,1], img[:,:,2]
        rb_corr     = np.corrcoef(r.flat, b.flat)[0, 1]
        colour_var  = np.mean([np.var(r), np.var(g), np.var(b)])

        # Noise analysis
        noise       = img - np.clip(img, 0.1, 0.9)
        noise_mean  = np.mean(np.abs(noise))
        noise_var   = np.var(noise)

        # Blending boundary (face region vs background)
        centre      = img[24:40, 24:40]
        border      = np.concatenate([img[:8,:].flat, img[-8:,:].flat])
        blend_diff  = np.abs(np.mean(centre) - np.mean(border))

        # Frequency peak (GANs leave checkerboard at stride freq)
        fft2d       = np.abs(np.fft.fft2(img[:,:,0]))
        freq_peak   = np.max(fft2d[1:, 1:]) / (np.mean(fft2d) + 1e-8)

        feats.append([
            dct_ratio, grad_var, rb_corr, colour_var,
            noise_mean, noise_var, blend_diff, freq_peak,
            np.mean(img), np.std(img), np.percentile(img, 5),
            np.percentile(img, 95)
        ])
    return np.array(feats, dtype=np.float32)

# Generate synthetic face images
def make_real_faces(n):
    imgs = []
    for _ in range(n):
        # Natural skin tones, organic texture
        base = np.random.uniform(0.3, 0.8, (IMG_SZ, IMG_SZ, 3))
        base[:,:,0] += np.random.uniform(-0.1, 0.1)  # skin tone variation
        noise = np.random.normal(0, 0.02, base.shape)
        imgs.append(np.clip(base + noise, 0, 1).astype(np.float32))
    return imgs

def make_fake_faces(n):
    imgs = []
    for _ in range(n):
        # GAN artefacts: checkerboard noise, over-smooth blending
        base  = np.random.uniform(0.4, 0.7, (IMG_SZ, IMG_SZ, 3))
        # Checkerboard artefact at 8-pixel stride
        check = np.zeros((IMG_SZ, IMG_SZ))
        check[::8, ::8] = np.random.uniform(0.02, 0.08)
        base[:,:,0] += check
        # Over-smoothed boundaries
        base  = np.clip(base, 0, 1)
        # Unnaturally low noise
        imgs.append(base.astype(np.float32))
    return imgs

print("\nGenerating face image dataset...")
real_imgs = make_real_faces(n_real)
fake_imgs = make_fake_faces(n_fake)

print("Extracting detection features...")
real_feats = extract_features(real_imgs, is_fake=False)
fake_feats = extract_features(fake_imgs, is_fake=True)

X = np.vstack([real_feats, fake_feats])
y = np.array([0]*n_real + [1]*n_fake)   # 0=real, 1=fake

FEATURE_NAMES = [
    "DCT Ratio", "Gradient Variance", "RB Correlation", "Colour Variance",
    "Noise Mean", "Noise Variance", "Blend Diff", "Freq Peak",
    "Mean Intensity", "Std Intensity", "5th Percentile", "95th Percentile"
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------------------------------------------------------------
# Train multiple detectors
# ---------------------------------------------------------------
models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0),
    "Random Forest":        RandomForestClassifier(n_estimators=300,
                                                    max_depth=12, random_state=42,
                                                    n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=250,
                                                        learning_rate=0.07,
                                                        max_depth=4, random_state=42),
}

results = {}
print("\nTraining deepfake detectors...")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    results[name] = {"model": model, "pred": pred, "prob": prob, "auc": auc}
    print(f"  {name:<25}  AUC = {auc:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\nBest detector: {best_name}  (AUC = {best['auc']:.4f})")
print(f"\nClassification Report ({best_name}):\n")
print(classification_report(y_test, best["pred"],
                             target_names=["Real", "Fake"]))

# ---------------------------------------------------------------
# DCT frequency analysis visualisation
# ---------------------------------------------------------------
sample_real = real_imgs[0]
sample_fake = fake_imgs[0]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.patch.set_facecolor("#0f1117")
titles = ["Face Image", "DCT Energy Map", "Gradient Map", "FFT Spectrum"]

for row, (img, label) in enumerate([(sample_real, "Real Face"),
                                     (sample_fake, "Fake Face")]):
    # Original image
    axes[row, 0].imshow(np.clip(img, 0, 1))
    axes[row, 0].set_title(f"{label}", color="white", fontsize=10)
    axes[row, 0].axis("off")

    # DCT energy
    dct2d = np.abs(dct(dct(img[:,:,0], axis=0), axis=1))
    axes[row, 1].imshow(np.log1p(dct2d), cmap="inferno")
    axes[row, 1].set_title("DCT Energy (log)", color="white", fontsize=10)
    axes[row, 1].axis("off")

    # Gradient map
    gy = np.diff(img[:,:,0], axis=0)
    gx = np.diff(img[:,:,0], axis=1)
    grad = np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2)
    axes[row, 2].imshow(grad, cmap="hot")
    axes[row, 2].set_title("Gradient Map", color="white", fontsize=10)
    axes[row, 2].axis("off")

    # FFT spectrum
    fft2d = np.abs(np.fft.fftshift(np.fft.fft2(img[:,:,0])))
    axes[row, 3].imshow(np.log1p(fft2d), cmap="viridis")
    axes[row, 3].set_title("FFT Spectrum", color="white", fontsize=10)
    axes[row, 3].axis("off")

for ax in axes.flat:
    ax.set_facecolor("#1a1d27")

plt.suptitle("Deepfake Detection -- Frequency Domain Analysis",
             color="white", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/04_deepfake_frequency.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("Saved: outputs/04_deepfake_frequency.png")

# ---------------------------------------------------------------
# Detection dashboard
# ---------------------------------------------------------------
DARK_BG = "#1a1d27"
C = {"Logistic Regression":"#4f9cf9",
     "Random Forest":"#22c55e",
     "Gradient Boosting":"#f59e0b"}

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor("#0f1117")

# ROC curves
ax = axes[0, 0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    ax.plot(fpr, tpr, color=C[name], lw=2,
            label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
ax.fill_between(*roc_curve(y_test, best["prob"])[:2],
                alpha=0.12, color="#22c55e")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax, "ROC Curves -- Deepfake Detector Comparison")

# Feature importance
ax = axes[0, 1]
rf  = results["Random Forest"]["model"]
fi  = np.array(rf.feature_importances_)
idx = np.argsort(fi)
bar_colors = ["#e74c3c" if v > np.median(fi) else "#3498db" for v in fi[idx]]
ax.barh([FEATURE_NAMES[i] for i in idx], fi[idx], color=bar_colors)
ax.set_xlabel("Importance Score")
style_ax(ax, "Detection Feature Importance (Random Forest)")

# Score distribution
ax = axes[0, 2]
ax.hist(best["prob"][y_test == 0], bins=50, alpha=0.75,
        color="#22c55e", label="Real Face",  density=True)
ax.hist(best["prob"][y_test == 1], bins=50, alpha=0.75,
        color="#e74c3c", label="Fake Face",  density=True)
ax.axvline(0.5, color="white", lw=1.5, linestyle="--", label="Threshold")
ax.set_xlabel("Fake Probability Score")
ax.set_ylabel("Density")
ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax, "Detection Score Distribution")

# Confusion matrix
ax = axes[1, 0]
cm  = confusion_matrix(y_test, best["pred"])
im  = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_xticklabels(["Real","Fake"], color="white", fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(["Real","Fake"], color="white", fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black",
                fontsize=14, fontweight="bold")
style_ax(ax, f"Confusion Matrix -- {best_name}")

# Feature distribution: real vs fake
ax = axes[1, 1]
feat_idx = 0   # DCT Ratio
ax.hist(X_test[y_test==0, feat_idx], bins=40, alpha=0.75,
        color="#22c55e", label="Real", density=True)
ax.hist(X_test[y_test==1, feat_idx], bins=40, alpha=0.75,
        color="#e74c3c", label="Fake", density=True)
ax.set_xlabel(FEATURE_NAMES[feat_idx])
ax.set_ylabel("Density")
ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax, f"Feature Distribution: {FEATURE_NAMES[feat_idx]}")

# AUC bar chart
ax = axes[1, 2]
names  = list(results.keys())
aucs   = [results[n]["auc"] for n in names]
colors = [C[n] for n in names]
bars   = ax.bar(range(len(names)), aucs, color=colors,
                edgecolor="#0f1117", linewidth=1.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(" ", "\n") for n in names],
                   color="white", fontsize=8)
ax.set_ylabel("AUC Score")
ax.set_ylim(0.5, 1.0)
for bar, v in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"{v:.4f}", ha="center", color="white", fontsize=9,
            fontweight="bold")
style_ax(ax, "Model AUC Comparison")

plt.suptitle("Deepfake Detection System -- Analysis Dashboard",
             color="white", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_deepfake_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("Saved: outputs/04_deepfake_dashboard.png")

# ---------------------------------------------------------------
# Live detection demo
# ---------------------------------------------------------------
print("\nLive Detection Demo:")
print("-" * 55)
test_cases = [
    {"label": "Real -- natural skin, organic texture",
     "img":   make_real_faces(1)[0]},
    {"label": "Fake -- GAN artefacts, smooth blending",
     "img":   make_fake_faces(1)[0]},
    {"label": "Real -- high variance, natural noise",
     "img":   make_real_faces(1)[0]},
    {"label": "Fake -- checkerboard noise, low texture",
     "img":   make_fake_faces(1)[0]},
]
rf = results["Random Forest"]["model"]
for case in test_cases:
    feats = extract_features([case["img"]], is_fake=False)
    feats_s = scaler.transform(feats)
    prob  = rf.predict_proba(feats_s)[0][1]
    verdict = "FAKE DETECTED" if prob > 0.5 else "REAL (authentic)"
    conf    = prob if prob > 0.5 else (1 - prob)
    print(f"  {case['label'][:45]:<45}  {prob:.3f}  -> {verdict} ({conf:.1%} conf)")

print("\nProject 04 -- Deepfake Detection complete.\n")
