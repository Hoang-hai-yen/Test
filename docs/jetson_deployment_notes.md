# Jetson / TensorRT — OUT OF SCOPE (reference only)

This project targets CPU / generic GPU and must run correctly there.
The original research doc proposed Jetson Xavier NX (<=50M params,
TensorRT FP16/INT8) deployment. That is NOT implemented here and is kept
only as a future-work note:

- Export each component via torch.onnx.export (opset 13)
- trtexec --onnx=model.onnx --fp16 (or --int8 with calibration set)
- Conv+BN+ReLU fusion, batch sizes multiple of 32
- Expected ~10-15 FPS end-to-end on Xavier NX @ 15W

Do not add TensorRT/Jetson dependencies to requirements.txt.
