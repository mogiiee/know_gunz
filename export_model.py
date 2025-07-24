from ultralytics import YOLO
import pathlib

MODEL_PT    = "best.pt"          # your trained weights
IMGSZ       = 640                 # whatever you trained with
OUT_DIR     = pathlib.Path("exports")
OUT_DIR.mkdir(exist_ok=True)

m = YOLO(MODEL_PT)

# 1) Plain ONNX for Pi CPU/GPU path
onnx_path = OUT_DIR / "best_nano.onnx"
m.export(format="onnx", imgsz=IMGSZ, opset=18, simplify=True, dynamic=False)
(OUT_DIR / "best.onnx").rename(onnx_path)  # Ultralytics names it 'best.onnx'

# 2) IMX500 package (runs INSIDE the sensor)
# Needs INT8 quantization. Provide a folder of real-ish images for calibration (optional but better).
imx_zip = OUT_DIR / "packerOut.zip"
m.export(format="imx500", imgsz=IMGSZ, int8=True, calibrate="calib_images")
# Result comes as 'packerOut.zip' in cwd, move it:
pathlib.Path("packerOut.zip").rename(imx_zip)

print("Exported:")
print("  ONNX :", onnx_path)
print("  IMX500 pack zip:", imx_zip)