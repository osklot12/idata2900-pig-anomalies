import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def raw():
    # Your manual confusion matrix
    confusion = np.array([
        [20, 0, 0, 0, 14],
        [0, 0, 0, 0, 10],
        [1, 0, 471, 0, 805],
        [0, 0, 0, 0, 12],
        [482, 465, 1384, 0, 0]
    ])
    labels = ['Tail Biting', 'Ear Biting', 'Belly Nosing', 'Tail Down', 'Background']

    # Normalize row-wise for color intensity
    conf_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(conf_norm, annot=confusion, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=False, linewidths=0.5)

    ax.set_xlabel('Predicted Label', fontsize=14, labelpad=20)
    ax.set_ylabel('True Label', fontsize=14, labelpad=20)
    ax.set_title('Confusion Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save in high-quality format
    plt.savefig("confusion_matrix.pdf", format='pdf')  # Vector format for thesis
    plt.savefig("confusion_matrix.png", dpi=300)  # Raster format as fallback
    plt.show()

def normalized():
    confusion = np.array([
        [20, 0, 0, 0, 14],
        [0, 0, 0, 0, 10],
        [1, 0, 471, 0, 805],
        [0, 0, 0, 0, 12],
        [482, 465, 1384, 563, 0]
    ])
    labels = ['Tail Biting', 'Ear Biting', 'Belly Nosing', 'Tail Down', 'Background']

    # Normalize per row
    conf_norm = confusion.astype('float') / confusion.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(conf_norm, annot=np.round(conf_norm, 2), fmt='.2f', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=True, linewidths=0.5)

    ax.set_xlabel('Predicted Label', fontsize=14, labelpad=20)
    ax.set_ylabel('True Label', fontsize=14, labelpad=20)
    ax.set_title('Normalized Confusion Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("confusion_matrix_normalized.pdf")
    plt.savefig("confusion_matrix_normalized.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    normalized()