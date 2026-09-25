import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")
sns.set_context("paper", font_scale=2.2)


log_passo64 = """"""
log_passo128 = """"""
log_passo256 = """"""

def genera_grafico_da_log(log_text, result_dir, passo):
    # Regex per estrarre Train Loss e Val Loss dal log
    pattern = r"Train Loss:\s*([\d\.]+)\s*-\s*Val\.\s*Loss:\s*([\d\.]+)"
    matches = re.findall(pattern, log_text)

    if not matches:
        print(f"Errore: Nessun dato trovato per il passo {passo}")
        return

    history = [
        {"train_loss": float(m[0]), "valid_loss": float(m[1])} for m in matches
    ]
    epochs = list(range(1, len(history) + 1))
    train_loss = [row["train_loss"] for row in history]
    valid_loss = [row["valid_loss"] for row in history]

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    color_train = "#1f77b4"  
    color_val = "#d62728"  

    ax.plot(epochs, train_loss, label="Train Loss", color=color_train, linewidth=2.2)
    ax.plot(epochs, valid_loss, label="Validation Loss", color=color_val, linewidth=2.2)

    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Loss", labelpad=8)
    ax.set_title(f"Loss History", pad=12, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine(top=True, right=True)

    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()

    filename_base = f"grafico_passo{passo}"
    plt.savefig(
        result_dir / f"{filename_base}.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(result_dir / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Salvato: {filename_base}.png e .pdf in {result_dir}")

if __name__ == "__main__":
    output_dir = Path(r"C:\Users\giorg\Desktop\Tirocinio\unet\ultimigrafici")
    genera_grafico_da_log(log_passo64, output_dir, 64)
    genera_grafico_da_log(log_passo128, output_dir, 128)
    genera_grafico_da_log(log_passo256, output_dir, 256)