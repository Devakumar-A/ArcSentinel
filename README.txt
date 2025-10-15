NSG AI/ML Video Analysis — Single Page Prototype (index.html)

Contents:
- index.html : Single-file prototype dashboard containing UI placeholders for all requested features.
  - Live feeds grid (placeholders for CCTV, Drone, Bodycam, Robot)
  - Upload section for batch videos
  - Analysis & detection placeholders (weapons, faces, behaviors)
  - Real-time alerts panel and timeline
  - Heatmap placeholder
  - Archive search and reports
  - Model Integration area (upload model file placeholder). This is where you will add trained model files and server-side hooks.
  - Settings & Users placeholders, Performance metrics, Saved clips, and System logs.

Notes:
- This is a client-side UI prototype only (HTML/CSS/JS). There is no backend, no actual AI inference, and no persistent storage.
- When you are ready to integrate your trained model, add server-side endpoints to store model files, and implement inference engine (e.g., TensorFlow/ONNX/PyTorch server). Update the 'Model Integration' section to call those endpoints and provide model path and I/O mapping.
- To run locally: open index.html in a browser. For file uploads / server integration, host using a lightweight server (e.g., python -m http.server) and implement required backend routes.
