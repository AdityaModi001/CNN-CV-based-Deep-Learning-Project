# Leaf Health

Streamlit app for the supplied Keras 3 apple-leaf classifier. The app supports
live WebRTC camera inference, camera photos, and image uploads. It displays
class probabilities, Grad-CAM heat-map overlays, and approximate CAM-derived
boxes for disease classes. The model is not changed or retrained.

## Requirements

- Windows with Python 3.12
- The supplied model at `model\leaf_classifier_fp32.keras`
- Internet access to install Python packages

## Run on Windows

From PowerShell:

```powershell
cd "C:\Users\adity\Desktop\DL Project\leaf-app"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this in the same PowerShell window and
activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the local app at <http://localhost:8501>. Stop the server with `Ctrl+C` in
the terminal where Streamlit is running.

## Verify the model

```powershell
cd "C:\Users\adity\Desktop\DL Project\leaf-app"
.\.venv\Scripts\Activate.ps1
python tests\check_model.py
```

The check prints the installed TensorFlow version, model input/output shapes,
layer names, one prediction's four probabilities and their sum. Successful
completion ends with `MODEL OK`.

## Camera access from a phone

Browsers only grant camera access on HTTPS pages or `http://localhost`. A
phone on the same Wi-Fi cannot use the laptop's camera at
`http://<laptop-ip>:8501`. To try the app from a phone, expose the local server
with an HTTPS tunnel. With Cloudflare Tunnel installed:

```powershell
cloudflared tunnel --url http://localhost:8501
```

Open the HTTPS URL printed by the command on your phone. Alternatively, with
ngrok installed:

```powershell
ngrok http 8501
```

The WebRTC connection uses Google's public STUN server at
`stun:stun.l.google.com:19302`. Some networks or hosted deployments also
require a TURN server; configure TURN credentials in `RTC_CONFIGURATION` in
`app.py` if needed. Do not commit TURN credentials.

## Troubleshooting

- **Model loading fails:** the environment pins TensorFlow 2.20.0, matching
  the version confirmed from the training environment. Do not edit the model.
- **Video is laggy:** increase `inference_interval` in
  `LiveVideoProcessor` in `app.py`, or reduce the requested video resolution.
- **Camera does not start:** use `http://localhost` or an HTTPS URL and allow
  camera permissions in the browser.
- **WebRTC does not connect:** try another network or configure a TURN server
  if direct peer connectivity is blocked.

Highlighted region shows where the model focused (Grad-CAM), not an exact
lesion outline.
