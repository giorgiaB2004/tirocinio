from ultralytics import YOLO


model = YOLO("runs/semantic/train/weights/best.pt") #DA VEDERE DOVE SALVA


results = model.predict(
    source="/data/giorgiabartoli/YOLO/maisweed/images/test",   
    imgsz=256,                           
    save=True                           
)

metrics_test = model.val(data="dataset.yaml", split="test")
print("TEST mIoU:", metrics_test.miou, "| Pixel Acc:", metrics_test.pixel_accuracy)

