import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione dello stile per la tesi
# 'paper' con font_scale=1.4 garantisce che i testi siano ben leggibili sia a schermo che stampati
sns.set_theme(style="ticks")
sns.set_context("paper", font_scale=2.2)

# Testo dei tre log (incolla i tuoi log qui dentro)
log_passo64 = """[01/150] | Epoch Time: 0m 57s - Train Loss: 0.6487 - Val. Loss: 0.5119
Checkpoint saved at epoch 1
[02/150] | Epoch Time: 0m 56s - Train Loss: 0.4611 - Val. Loss: 0.4501
Checkpoint saved at epoch 2
[03/150] | Epoch Time: 0m 56s - Train Loss: 0.4214 - Val. Loss: 0.4279
Checkpoint saved at epoch 3
[04/150] | Epoch Time: 0m 56s - Train Loss: 0.4038 - Val. Loss: 0.4116
Checkpoint saved at epoch 4
[05/150] | Epoch Time: 0m 56s - Train Loss: 0.3946 - Val. Loss: 0.3927
Checkpoint saved at epoch 5
[06/150] | Epoch Time: 0m 56s - Train Loss: 0.3781 - Val. Loss: 0.3914
Checkpoint saved at epoch 6
[07/150] | Epoch Time: 0m 56s - Train Loss: 0.3683 - Val. Loss: 0.3863
Checkpoint saved at epoch 7
[08/150] | Epoch Time: 0m 56s - Train Loss: 0.3644 - Val. Loss: 0.3865
EarlyStopping counter: 1 of 10
[09/150] | Epoch Time: 0m 56s - Train Loss: 0.3558 - Val. Loss: 0.3771
Checkpoint saved at epoch 9
[10/150] | Epoch Time: 0m 57s - Train Loss: 0.3506 - Val. Loss: 0.3762
Checkpoint saved at epoch 10
[11/150] | Epoch Time: 0m 56s - Train Loss: 0.3458 - Val. Loss: 0.3844
EarlyStopping counter: 1 of 10
[12/150] | Epoch Time: 0m 56s - Train Loss: 0.3380 - Val. Loss: 0.3952
EarlyStopping counter: 2 of 10
[13/150] | Epoch Time: 0m 56s - Train Loss: 0.3409 - Val. Loss: 0.3832
EarlyStopping counter: 3 of 10
[14/150] | Epoch Time: 0m 56s - Train Loss: 0.3309 - Val. Loss: 0.3864
EarlyStopping counter: 4 of 10
[15/150] | Epoch Time: 0m 56s - Train Loss: 0.3201 - Val. Loss: 0.3861
EarlyStopping counter: 5 of 10
[16/150] | Epoch Time: 0m 56s - Train Loss: 0.3140 - Val. Loss: 0.3761
Checkpoint saved at epoch 16
[17/150] | Epoch Time: 0m 56s - Train Loss: 0.3120 - Val. Loss: 0.3788
EarlyStopping counter: 1 of 10
[18/150] | Epoch Time: 0m 56s - Train Loss: 0.3046 - Val. Loss: 0.4106
EarlyStopping counter: 2 of 10
[19/150] | Epoch Time: 0m 56s - Train Loss: 0.3019 - Val. Loss: 0.3761
Checkpoint saved at epoch 19
[20/150] | Epoch Time: 0m 56s - Train Loss: 0.2947 - Val. Loss: 0.3760
Checkpoint saved at epoch 20
[21/150] | Epoch Time: 0m 56s - Train Loss: 0.2897 - Val. Loss: 0.3865
EarlyStopping counter: 1 of 10
[22/150] | Epoch Time: 0m 56s - Train Loss: 0.2866 - Val. Loss: 0.3760
Checkpoint saved at epoch 22
[23/150] | Epoch Time: 0m 56s - Train Loss: 0.2770 - Val. Loss: 0.3738
Checkpoint saved at epoch 23
[24/150] | Epoch Time: 0m 56s - Train Loss: 0.2751 - Val. Loss: 0.3910
EarlyStopping counter: 1 of 10
[25/150] | Epoch Time: 0m 56s - Train Loss: 0.2786 - Val. Loss: 0.3781
EarlyStopping counter: 2 of 10
[26/150] | Epoch Time: 0m 56s - Train Loss: 0.2709 - Val. Loss: 0.3898
EarlyStopping counter: 3 of 10
[27/150] | Epoch Time: 0m 56s - Train Loss: 0.2617 - Val. Loss: 0.3795
EarlyStopping counter: 4 of 10
[28/150] | Epoch Time: 0m 56s - Train Loss: 0.2543 - Val. Loss: 0.3826
EarlyStopping counter: 5 of 10
[29/150] | Epoch Time: 0m 56s - Train Loss: 0.2548 - Val. Loss: 0.3943
EarlyStopping counter: 6 of 10
[30/150] | Epoch Time: 0m 56s - Train Loss: 0.2354 - Val. Loss: 0.3782
EarlyStopping counter: 7 of 10
[31/150] | Epoch Time: 0m 56s - Train Loss: 0.2317 - Val. Loss: 0.3758
EarlyStopping counter: 8 of 10
[32/150] | Epoch Time: 0m 56s - Train Loss: 0.2291 - Val. Loss: 0.3753
EarlyStopping counter: 9 of 10
[33/150] | Epoch Time: 0m 56s - Train Loss: 0.2280 - Val. Loss: 0.3785
EarlyStopping counter: 10 of 10"""
log_passo128 = """[01/150] | Epoch Time: 0m 21s - Train Loss: 0.8656 - Val. Loss: 0.6228
Checkpoint saved at epoch 1
[02/150] | Epoch Time: 0m 19s - Train Loss: 0.5787 - Val. Loss: 0.5191
Checkpoint saved at epoch 2
[03/150] | Epoch Time: 0m 19s - Train Loss: 0.5080 - Val. Loss: 0.4687
Checkpoint saved at epoch 3
[04/150] | Epoch Time: 0m 19s - Train Loss: 0.4750 - Val. Loss: 0.4433
Checkpoint saved at epoch 4
[05/150] | Epoch Time: 0m 19s - Train Loss: 0.4450 - Val. Loss: 0.4277
Checkpoint saved at epoch 5
[06/150] | Epoch Time: 0m 20s - Train Loss: 0.4376 - Val. Loss: 0.4146
Checkpoint saved at epoch 6
[07/150] | Epoch Time: 0m 21s - Train Loss: 0.4252 - Val. Loss: 0.4317
EarlyStopping counter: 1 of 10
[08/150] | Epoch Time: 0m 21s - Train Loss: 0.4244 - Val. Loss: 0.4599
EarlyStopping counter: 2 of 10
[09/150] | Epoch Time: 0m 21s - Train Loss: 0.4057 - Val. Loss: 0.3998
Checkpoint saved at epoch 9
[10/150] | Epoch Time: 0m 21s - Train Loss: 0.4065 - Val. Loss: 0.4518
EarlyStopping counter: 1 of 10
[11/150] | Epoch Time: 0m 21s - Train Loss: 0.3984 - Val. Loss: 0.4124
EarlyStopping counter: 2 of 10
[12/150] | Epoch Time: 0m 20s - Train Loss: 0.3916 - Val. Loss: 0.3922
Checkpoint saved at epoch 12
[13/150] | Epoch Time: 0m 21s - Train Loss: 0.3886 - Val. Loss: 0.3881
Checkpoint saved at epoch 13
[14/150] | Epoch Time: 0m 21s - Train Loss: 0.3905 - Val. Loss: 0.4125
EarlyStopping counter: 1 of 10
[15/150] | Epoch Time: 0m 21s - Train Loss: 0.3781 - Val. Loss: 0.4041
EarlyStopping counter: 2 of 10
[16/150] | Epoch Time: 0m 21s - Train Loss: 0.3793 - Val. Loss: 0.4080
EarlyStopping counter: 3 of 10
[17/150] | Epoch Time: 0m 21s - Train Loss: 0.3751 - Val. Loss: 0.3904
EarlyStopping counter: 4 of 10
[18/150] | Epoch Time: 0m 21s - Train Loss: 0.3714 - Val. Loss: 0.4090
EarlyStopping counter: 5 of 10
[19/150] | Epoch Time: 0m 21s - Train Loss: 0.3691 - Val. Loss: 0.3801
Checkpoint saved at epoch 19
[20/150] | Epoch Time: 0m 21s - Train Loss: 0.3699 - Val. Loss: 0.3903
EarlyStopping counter: 1 of 10
[21/150] | Epoch Time: 0m 21s - Train Loss: 0.3636 - Val. Loss: 0.3768
Checkpoint saved at epoch 21
[22/150] | Epoch Time: 0m 21s - Train Loss: 0.3675 - Val. Loss: 0.3813
EarlyStopping counter: 1 of 10
[23/150] | Epoch Time: 0m 21s - Train Loss: 0.3586 - Val. Loss: 0.3824
EarlyStopping counter: 2 of 10
[24/150] | Epoch Time: 0m 21s - Train Loss: 0.3562 - Val. Loss: 0.3940
EarlyStopping counter: 3 of 10
[25/150] | Epoch Time: 0m 21s - Train Loss: 0.3567 - Val. Loss: 0.3841
EarlyStopping counter: 4 of 10
[26/150] | Epoch Time: 0m 21s - Train Loss: 0.3497 - Val. Loss: 0.3839
EarlyStopping counter: 5 of 10
[27/150] | Epoch Time: 0m 21s - Train Loss: 0.3530 - Val. Loss: 0.3868
EarlyStopping counter: 6 of 10
[28/150] | Epoch Time: 0m 21s - Train Loss: 0.3378 - Val. Loss: 0.3746
Checkpoint saved at epoch 28
[29/150] | Epoch Time: 0m 21s - Train Loss: 0.3307 - Val. Loss: 0.3684
Checkpoint saved at epoch 29
[30/150] | Epoch Time: 0m 21s - Train Loss: 0.3270 - Val. Loss: 0.3721
EarlyStopping counter: 1 of 10
[31/150] | Epoch Time: 0m 21s - Train Loss: 0.3244 - Val. Loss: 0.3713
EarlyStopping counter: 2 of 10
[32/150] | Epoch Time: 0m 21s - Train Loss: 0.3247 - Val. Loss: 0.3764
EarlyStopping counter: 3 of 10
[33/150] | Epoch Time: 0m 21s - Train Loss: 0.3231 - Val. Loss: 0.3733
EarlyStopping counter: 4 of 10
[34/150] | Epoch Time: 0m 21s - Train Loss: 0.3225 - Val. Loss: 0.3748
EarlyStopping counter: 5 of 10
[35/150] | Epoch Time: 0m 21s - Train Loss: 0.3204 - Val. Loss: 0.3747
EarlyStopping counter: 6 of 10
[36/150] | Epoch Time: 0m 21s - Train Loss: 0.3180 - Val. Loss: 0.3721
EarlyStopping counter: 7 of 10
[37/150] | Epoch Time: 0m 21s - Train Loss: 0.3189 - Val. Loss: 0.3729
EarlyStopping counter: 8 of 10
[38/150] | Epoch Time: 0m 21s - Train Loss: 0.3162 - Val. Loss: 0.3738
EarlyStopping counter: 9 of 10
[39/150] | Epoch Time: 0m 21s - Train Loss: 0.3184 - Val. Loss: 0.3709
EarlyStopping counter: 10 of 10"""
log_passo256 = """[01/150] | Epoch Time: 0m 7s - Train Loss: 1.0989 - Val. Loss: 1.5190
Checkpoint saved at epoch 1
[02/150] | Epoch Time: 0m 6s - Train Loss: 0.8211 - Val. Loss: 0.7625
Checkpoint saved at epoch 2
[03/150] | Epoch Time: 0m 5s - Train Loss: 0.6996 - Val. Loss: 0.6748
Checkpoint saved at epoch 3
[04/150] | Epoch Time: 0m 6s - Train Loss: 0.6184 - Val. Loss: 0.5861
Checkpoint saved at epoch 4
[05/150] | Epoch Time: 0m 6s - Train Loss: 0.5778 - Val. Loss: 0.6383
EarlyStopping counter: 1 of 10
[06/150] | Epoch Time: 0m 6s - Train Loss: 0.5483 - Val. Loss: 0.5615
Checkpoint saved at epoch 6
[07/150] | Epoch Time: 0m 6s - Train Loss: 0.5228 - Val. Loss: 0.5468
Checkpoint saved at epoch 7
[08/150] | Epoch Time: 0m 6s - Train Loss: 0.5059 - Val. Loss: 0.4776
Checkpoint saved at epoch 8
[09/150] | Epoch Time: 0m 6s - Train Loss: 0.5043 - Val. Loss: 0.5005
EarlyStopping counter: 1 of 10
[10/150] | Epoch Time: 0m 6s - Train Loss: 0.4866 - Val. Loss: 0.4826
EarlyStopping counter: 2 of 10
[11/150] | Epoch Time: 0m 6s - Train Loss: 0.4843 - Val. Loss: 0.4774
Checkpoint saved at epoch 11
[12/150] | Epoch Time: 0m 6s - Train Loss: 0.4665 - Val. Loss: 0.4474
Checkpoint saved at epoch 12
[13/150] | Epoch Time: 0m 6s - Train Loss: 0.4491 - Val. Loss: 0.4377
Checkpoint saved at epoch 13
[14/150] | Epoch Time: 0m 6s - Train Loss: 0.4454 - Val. Loss: 0.4615
EarlyStopping counter: 1 of 10
[15/150] | Epoch Time: 0m 6s - Train Loss: 0.4542 - Val. Loss: 0.4293
Checkpoint saved at epoch 15
[16/150] | Epoch Time: 0m 6s - Train Loss: 0.4492 - Val. Loss: 0.4223
Checkpoint saved at epoch 16
[17/150] | Epoch Time: 0m 6s - Train Loss: 0.4496 - Val. Loss: 0.4078
Checkpoint saved at epoch 17
[18/150] | Epoch Time: 0m 6s - Train Loss: 0.4336 - Val. Loss: 0.4078
EarlyStopping counter: 1 of 10
[19/150] | Epoch Time: 0m 6s - Train Loss: 0.4203 - Val. Loss: 0.4005
Checkpoint saved at epoch 19
[20/150] | Epoch Time: 0m 6s - Train Loss: 0.4252 - Val. Loss: 0.4342
EarlyStopping counter: 1 of 10
[21/150] | Epoch Time: 0m 6s - Train Loss: 0.4188 - Val. Loss: 0.5102
EarlyStopping counter: 2 of 10
[22/150] | Epoch Time: 0m 6s - Train Loss: 0.4234 - Val. Loss: 0.4120
EarlyStopping counter: 3 of 10
[23/150] | Epoch Time: 0m 6s - Train Loss: 0.4182 - Val. Loss: 0.4031
EarlyStopping counter: 4 of 10
[24/150] | Epoch Time: 0m 6s - Train Loss: 0.4221 - Val. Loss: 0.3992
Checkpoint saved at epoch 24
[25/150] | Epoch Time: 0m 6s - Train Loss: 0.4140 - Val. Loss: 0.4122
EarlyStopping counter: 1 of 10
[26/150] | Epoch Time: 0m 6s - Train Loss: 0.4160 - Val. Loss: 0.3920
Checkpoint saved at epoch 26
[27/150] | Epoch Time: 0m 6s - Train Loss: 0.4021 - Val. Loss: 0.4075
EarlyStopping counter: 1 of 10
[28/150] | Epoch Time: 0m 6s - Train Loss: 0.4027 - Val. Loss: 0.3788
Checkpoint saved at epoch 28
[29/150] | Epoch Time: 0m 6s - Train Loss: 0.3941 - Val. Loss: 0.3879
EarlyStopping counter: 1 of 10
[30/150] | Epoch Time: 0m 6s - Train Loss: 0.3881 - Val. Loss: 0.3909
EarlyStopping counter: 2 of 10
[31/150] | Epoch Time: 0m 6s - Train Loss: 0.3995 - Val. Loss: 0.4099
EarlyStopping counter: 3 of 10
[32/150] | Epoch Time: 0m 6s - Train Loss: 0.3928 - Val. Loss: 0.3976
EarlyStopping counter: 4 of 10
[33/150] | Epoch Time: 0m 6s - Train Loss: 0.3892 - Val. Loss: 0.3980
EarlyStopping counter: 5 of 10
[34/150] | Epoch Time: 0m 6s - Train Loss: 0.4050 - Val. Loss: 0.4171
EarlyStopping counter: 6 of 10
[35/150] | Epoch Time: 0m 6s - Train Loss: 0.3867 - Val. Loss: 0.3848
EarlyStopping counter: 7 of 10
[36/150] | Epoch Time: 0m 6s - Train Loss: 0.3747 - Val. Loss: 0.3848
EarlyStopping counter: 8 of 10
[37/150] | Epoch Time: 0m 6s - Train Loss: 0.3699 - Val. Loss: 0.3823
EarlyStopping counter: 9 of 10
[38/150] | Epoch Time: 0m 6s - Train Loss: 0.3718 - Val. Loss: 0.3897
EarlyStopping counter: 10 of 10"""

def genera_grafico_da_log(log_text, result_dir, passo):
    # Regex per estrarre Train Loss e Val Loss dal log
    pattern = r"Train Loss:\s*([\d\.]+)\s*-\s*Val\.\s*Loss:\s*([\d\.]+)"
    matches = re.findall(pattern, log_text)

    if not matches:
        print(f"Errore: Nessun dato trovato per il passo {passo}")
        return

    # Ricostruzione della struttura 'history'
    history = [
        {"train_loss": float(m[0]), "valid_loss": float(m[1])} for m in matches
    ]
    epochs = list(range(1, len(history) + 1))
    train_loss = [row["train_loss"] for row in history]
    valid_loss = [row["valid_loss"] for row in history]

    # Cartella di destinazione
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(8, 5))

    # Definizione colori eleganti e ad alto contrasto (palette Seaborn/accademica)
    color_train = "#1f77b4"  # Blu professionale
    color_val = "#d62728"  # Rosso professionale

    # Plot delle curve con spessore maggiorato (linewidth=2.2)
    ax.plot(epochs, train_loss, label="Train Loss", color=color_train, linewidth=2.2)
    ax.plot(epochs, valid_loss, label="Validation Loss", color=color_val, linewidth=2.2)

    # Etichette e titoli ben visibili
    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Loss", labelpad=8)
    ax.set_title(f"Loss History", pad=12, fontweight="bold")

    # Griglia discreta e rimozione dei bordi superiore/destro
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine(top=True, right=True)

    # Legenda chiara senza bordo invadente
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()

    # Salvataggio in PNG ad alta risoluzione (300 DPI) e in PDF vettoriale (ideale per LaTeX)
    filename_base = f"grafico_passo{passo}"
    plt.savefig(
        result_dir / f"{filename_base}.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(result_dir / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Salvato: {filename_base}.png e .pdf in {result_dir}")

if __name__ == "__main__":
    # Esegui per i 3 file
    output_dir = Path(r"C:\Users\giorg\Desktop\Tirocinio\unet\ultimigrafici")
    genera_grafico_da_log(log_passo64, output_dir, 64)
    genera_grafico_da_log(log_passo128, output_dir, 128)
    genera_grafico_da_log(log_passo256, output_dir, 256)