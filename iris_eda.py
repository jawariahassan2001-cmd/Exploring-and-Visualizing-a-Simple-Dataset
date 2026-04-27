import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris

# ── STEP 1: Load Dataset ──────────────────────────────────
raw = load_iris()
df = pd.DataFrame(raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = pd.Categorical.from_codes(raw.target, ['setosa', 'versicolor', 'virginica'])

# ── STEP 2: Inspect the Data ──────────────────────────────
print("=" * 50)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
df.info()
print("\nDescribe:")
print(df.describe())

# ── STEP 3: Scatter Plot ──────────────────────────────────
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', palette='Set1', s=80)
plt.title('Scatter Plot: Sepal Length vs Sepal Width', fontsize=13, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.tight_layout()
plt.show()

# ── STEP 4: Histograms ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for i, (feature, color) in enumerate(zip(features, colors)):
    ax = axes[i//2][i%2]
    ax.hist(df[feature], bins=15, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
    ax.set_xlabel(f'{feature} (cm)')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Histograms — Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── STEP 5: Box Plots ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, feature in enumerate(features):
    sns.boxplot(data=df, x='species', y=feature, ax=axes[i//2][i%2], palette='pastel')
    axes[i//2][i%2].set_title(f'{feature.replace("_", " ").title()} by Species')

plt.suptitle('Box Plots — Outlier Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── STEP 6: Pair Plot ─────────────────────────────────────
g = sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde', plot_kws={'alpha': 0.7})
g.figure.suptitle('Pair Plot — All Features', y=1.02, fontsize=14)
plt.show()

print("\n All done!")