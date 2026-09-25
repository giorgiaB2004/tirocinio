import torch

# Carica il file pth (usa weights_only=False se contiene dizionari custom)
checkpoint = torch.load("/data/giorgiabartoli/modificheUNET/files/checkpoint_baseline.pth", map_location="cpu")

# Stampa le chiavi disponibili nel file per vedere cosa c'è dentro
print("Chiavi trovate nel file:", checkpoint.keys())

print(f"Epoca del checkpoint: {checkpoint['epoch']}")

# Estrai e stampa il contenuto della chiave loss
loss_value = checkpoint['loss']
print(f"Contenuto della chiave 'loss': {loss_value}")