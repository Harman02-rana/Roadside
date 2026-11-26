# ROADGUARD AI

**Intelligent Roads. Safer Journeys.**

ROADGUARD AI is a Streamlit road image analysis app using the project's existing trained YOLO models and OpenCV illumination analysis. It detects potholes and waste, annotates the image, and applies the original rule-based risk classification.

## Run locally

Open this project directory in VS Code, then run in the integrated PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The trained weights are included at:

- `runs/detect/train2/weights/best.pt` — pothole model
- `runs/detect/train3/weights/best.pt` — waste model

## App pages

- **Overview** — introduction and workflow.
- **Road scanner** — JPG/JPEG/PNG upload, trained model inference, annotated detections, confidence, illumination and risk factors.
- **Analytics** — session-only analysis records, charts and CSV download.
- **About project** — architecture, methodology and limitations.

## Risk score

The existing project rule assigns 2 points per pothole, 1 point per waste detection, adds 5 for low illumination or 2 for medium illumination, and classifies scores above 10 as high risk and scores above 5 as medium risk. This is a project heuristic, not a validated accident probability.

History is kept for the active Streamlit session. Map reporting, persistent records, PDF reports and weather analysis are future modules and are not operational.
