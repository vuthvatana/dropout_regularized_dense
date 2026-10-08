# ============================================================
# MNIST Experiment — Dropout Regularized Dense Model
# Splits: 70% train / 20% test (with 15% gradient noise) / 10% val
# Subset sizes: 2500 and 5000
# Epochs: 50 and 100
# Learning rate: 0.01
# Fixed seed for reproducibility
# ============================================================

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# GLOBAL SEED — same for every run
# ------------------------------------------------------------
SEED = 42
Noise_Image = 0.40
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# Load MNIST (70,000 total)
# ------------------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_all, y_all), (x_test_full, y_test_full) = mnist.load_data()
x_all = x_all / 255.0
x_test_full = x_test_full / 255.0

# Combine train+test so we can re-split 70/20/10 ourselves
X = np.concatenate([x_all, x_test_full], axis=0)
Y = np.concatenate([y_all, y_test_full], axis=0)
print(f"Total MNIST samples: {len(X)}")

# ------------------------------------------------------------
# Model builder — Dropout Regularized Dense
# ------------------------------------------------------------
def build_model():
    tf.random.set_seed(SEED)
    dropout_regularized_dense = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    dropout_regularized_dense.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),    
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


    return dropout_regularized_dense


# ------------------------------------------------------------
# Add 40% Gaussian gradient noise to test images
# ------------------------------------------------------------
def add_gradient_noise(images, noise_std=Noise_Image):
    noise = np.random.normal(0, noise_std, images.shape)
    noisy = images + noise
    return np.clip(noisy, 0.0, 1.0)

# ------------------------------------------------------------
# Run one experiment
# ------------------------------------------------------------
def run_experiment(subset_size, epochs):
    print(f"\n{'='*60}")
    print(f"  Subset: {subset_size} | Epochs: {epochs} | LR: 0.01")
    print(f"{'='*60}")

    # 1. Sample subset with fixed seed
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), subset_size, replace=False)
    X_sub, Y_sub = X[idx], Y[idx]

    # 2. Split 70% train / 20% test / 10% val
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sub, Y_sub, test_size=0.30, random_state=SEED, stratify=Y_sub)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=SEED, stratify=y_temp)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Val: {len(X_val)}")

    # 3. Add 40% gradient noise to TEST images only
    X_test_noisy = add_gradient_noise(X_test, noise_std=Noise_Image)

    # 4. Build and train
    model = build_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    # 5. Evaluate on clean and noisy test
    clean_loss, clean_acc = model.evaluate(X_test, y_test, verbose=0)
    noisy_loss, noisy_acc = model.evaluate(X_test_noisy, y_test, verbose=0)

    print(f"  ✅ Clean test acc : {clean_acc:.4f}")
    print(f"  🌫️  Noisy test acc : {noisy_acc:.4f}  (40% noise)")

    return {
        'subset': subset_size,
        'epochs': epochs,
        'history': history.history,
        'model': model,
        'X_test': X_test,
        'X_test_noisy': X_test_noisy,
        'y_test': y_test,
        'y_val': y_val,
        'clean_acc': clean_acc,
        'noisy_acc': noisy_acc,
    }

# ------------------------------------------------------------
# Run all four configurations
# ------------------------------------------------------------
configs = [
    (2500, 50), 
    (2500, 100),
    (5000, 50),
    (5000, 100),
]

results = {}
for size, ep in configs:
    results[(size, ep)] = run_experiment(size, ep)

# ============================================================
# PLOT 1: Accuracy curves for all 4 runs
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (size, ep) in zip(axes.flatten(), configs):
    h = results[(size, ep)]['history']
    ax.plot(h['accuracy'], label='Train')
    ax.plot(h['val_accuracy'], label='Val')
    ax.set_title(f"Subset={size}, Epochs={ep}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
plt.suptitle('Training vs Validation Accuracy — Dropout Regularized Dense')
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2: Loss curves for all 4 runs
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (size, ep) in zip(axes.flatten(), configs):
    h = results[(size, ep)]['history']
    ax.plot(h['loss'], label='Train')
    ax.plot(h['val_loss'], label='Val')
    ax.set_title(f"Subset={size}, Epochs={ep}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
plt.suptitle('Training vs Validation Loss — Dropout Regularized Dense')
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 3: Summary bar chart — clean vs noisy test accuracy
# ============================================================
# labels = [f"{s} / {e}ep" for s, e in configs]
# clean_accs = [results[c]['clean_acc'] for c in configs]
# noisy_accs = [results[c]['noisy_acc'] for c in configs]

# x = np.arange(len(labels))
# width = 0.35

# plt.figure(figsize=(10, 5))
# plt.bar(x - width/2, clean_accs, width, label='Clean Test', color='steelblue')
# plt.bar(x + width/2, noisy_accs, width, label='Noisy Test (40%)', color='salmon')
# plt.xticks(x, labels)
# plt.ylabel('Accuracy')
# plt.title('Clean vs Noisy Test Accuracy — All Configurations')
# plt.ylim(0, 1)
# plt.legend()
# plt.grid(axis='y', alpha=0.3)
# for i, (c, n) in enumerate(zip(clean_accs, noisy_accs)):
#     plt.text(i - width/2, c + 0.01, f'{c:.3f}', ha='center', fontsize=9)
#     plt.text(i + width/2, n + 0.01, f'{n:.3f}', ha='center', fontsize=9)
# plt.tight_layout()
# plt.show()

# ============================================================
# PLOT 4: Sample noisy test predictions (best config)
# ============================================================
best_cfg = max(configs, key=lambda c: results[c]['noisy_acc'])
print(f"\n🏆 Best config: subset={best_cfg[0]}, epochs={best_cfg[1]}")

r = results[best_cfg]
preds = np.argmax(r['model'].predict(r['X_test_noisy'][:10], verbose=0), axis=1)

# plt.figure(figsize=(15, 3))
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(r['X_test_noisy'][i], cmap='gray')
#     color = 'green' if preds[i] == r['y_test'][i] else 'red'
#     plt.title(f"P:{preds[i]}\nT:{r['y_test'][i]}", color=color, fontsize=9)
#     plt.axis('off')
# plt.suptitle(f'Noisy Test Predictions — Best Config {best_cfg}')
# plt.tight_layout()
# plt.show()

# ============================================================
# PLOT 5: Confusion matrix on noisy test (best config)
# ============================================================
y_pred_all = np.argmax(r['model'].predict(r['X_test_noisy'], verbose=0), axis=1)
cm = confusion_matrix(r['y_test'], y_pred_all)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix — Noisy Test — {best_cfg}')
plt.show()

# ============================================================
# Final summary table
# ============================================================
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"{'Subset':<10}{'Epochs':<10}{'Clean Acc':<12}{'Noisy Acc':<12}")
for s, e in configs:
    r = results[(s, e)]
    print(f"{s:<10}{e:<10}{r['clean_acc']:<12.4f}{r['noisy_acc']:<12.4f}")

# ============================================================
# Evaluation Figure: Original | Test (noisy) | Result (predicted)
# ============================================================

# Use best config model
r = results[best_cfg]
model = r['model']
X_clean = r['X_test']
X_noisy = r['X_test_noisy']
y_true  = r['y_test']

# Pick 10 samples (one variety of digits)
N = 10
sample_idx = np.random.RandomState(SEED).choice(len(X_clean), N, replace=False)

X_clean_s = X_clean[sample_idx]
X_noisy_s = X_noisy[sample_idx]
y_true_s  = y_true[sample_idx]

# Predict on noisy inputs
preds = np.argmax(model.predict(X_noisy_s, verbose=0), axis=1)

# ------------------------------------------------------------
# Build the figure: 10 rows × 3 columns
# ------------------------------------------------------------
fig, axes = plt.subplots(N, 3, figsize=(5, 1.3 * N))

for i in range(N):
    # Column 1 — Original
    axes[i, 0].imshow(X_clean_s[i].squeeze(), cmap='gray')
    axes[i, 0].axis('off')

    # Column 2 — Test (noisy input)
    axes[i, 1].imshow(X_noisy_s[i].squeeze(), cmap='gray')
    axes[i, 1].axis('off')

    # Column 3 — Result
    # For a classifier: show the noisy image again, color-tinted by correctness
    # (There is no "reconstructed" image — the model outputs a class label.)
    axes[i, 2].imshow(X_noisy_s[i].squeeze(), cmap='gray')
    color = 'green' if preds[i] == y_true_s[i] else 'red'
    axes[i, 2].set_title(f"Pred: {preds[i]}", color=color, fontsize=9)
    axes[i, 2].axis('off')

# Column headers
axes[0, 0].set_title('Original', fontsize=11)
axes[0, 1].set_title('Test (Noisy)', fontsize=11)
axes[0, 2].set_title('Result', fontsize=11)

plt.suptitle('Evaluation — Original vs Noisy Test vs Predicted',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()