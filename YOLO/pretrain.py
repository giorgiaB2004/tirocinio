from ultralytics import YOLO


model = YOLO("yolo26n-sem.pt")


results = model.train(
    data="datasetPreaddestramento.yaml",
    epochs=50,                  
    imgsz=1024,   #da vedere la dimensione delle immagini
    device=0
)