from ultralytics import YOLO


model = YOLO("yolo26n-sem.pt")


results = model.train(
    data="dataset.yaml",
    epochs=100,                  
    imgsz=256,                   
    device=0                     
)


"""

FINE TUNING dopo phenobench
model = YOLO("runs/semantic/train/weights/best.pt")

results = model.train(
    data="dataset.yaml",
    epochs=100,                  
    imgsz=256,                   
    device=0,
    freeze=12,
    lr0=0.001                 
)

"""