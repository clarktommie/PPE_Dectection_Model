# PPE Detection Model

Computer vision project for detecting personal protective equipment (PPE) in images and video. It uses Ultralytics YOLO (v8/11), serves a Streamlit UI, and optionally uploads annotated results and metadata to Supabase.

## Features
- Streamlit app for drag‑and‑drop image/video detection.
- YOLO training script with configurable dataset and model size.
- Batch inference/ETL utilities for downstream analytics.
- Supabase integration: uploads annotated media to a storage bucket and logs detections in a table.
- Modal deployment recipe for hosting the Streamlit app.

## Project Structure
- `src/streamlit_app.py` — UI, prediction, optional Supabase upload.
- `src/train_model.py` — trains YOLO; outputs to `runs/train/ppe_yolov8/`.
- `src/predict_image.py` — run inference on a folder or file.
- `src/predict_violations.py` — flags specific PPE violations.
- `src/etl_pipeline.py` — extracts detections as JSON for further processing.
- `src/modal_app.py` — deployment script for Modal web server.
- `yolo11n.pt`, `yolov8n.pt` — lightweight fallback weights.

## Requirements
- Python 3.12
- `uv` (recommended) or `pip`
- GPU optional (set `device=cpu` in scripts if needed)

## Setup
```bash
# from project root
uv venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
uv sync                    # installs deps from pyproject.toml
```

Create a `.env` in the project root:
```
SUPABASE_URL=your-project-url
SUPABASE_SERVICE_KEY=your-service-role-key   # preferred for uploads; stays server-side
# or SUPABASE_KEY=anon-key if you also add RLS policies for anon
MODEL_PATH=/full/path/to/weights.pt          # optional override
```

## Model Weights
By default the app looks for, in order:
1. `MODEL_PATH` env var (absolute or relative to project root)
2. `runs/train/ppe_yolov8/weights/best.pt` (produced by training)
3. `best.pt` in project root
4. `yolo11n.pt` or `yolov8n.pt` fallbacks

### Train
```bash
uv run python src/train_model.py \
  --  # optional: edit defaults in the script (data path, epochs, imgsz)
```
Ensure `data/data.yaml` points to your dataset.

### Validate
```bash
uv run yolo val model=runs/train/ppe_yolov8/weights/best.pt data=data/data.yaml
```

## Run the Streamlit App
```bash
uv run streamlit run src/streamlit_app.py
```
Uploads require Supabase credentials; without them the app will skip storage/table writes but still show results locally.

## Supabase Setup
1. Create a public bucket `ppe_results`.
2. Create table `detections` with columns: `filename text`, `file_url text`, `confidence numeric`, `label text`, optional timestamps/ids.
3. Prefer using the service role key in `.env`. If you must use the anon key, add RLS policies to allow inserts/selects for `anon` on `ppe_results` and `detections`. Example policy for the bucket:
```sql
create policy "anon insert ppe_results" on storage.objects
for insert to anon
using (bucket_id = 'ppe_results') with check (bucket_id = 'ppe_results');
```

## CLI Utilities
- Inference: `uv run python src/predict_image.py --model_path ... --source ...` (edit defaults in file).
- Violations demo: `uv run python src/predict_violations.py` (adjust paths before running).
- ETL JSON dump: `uv run python src/etl_pipeline.py` and provide a source path when prompted.

## Modal Deployment
```bash
uv run modal deploy src/modal_app.py
```
Ensure `.env` and weights are available to the Modal image (the script copies `.env` and the Streamlit file; adjust if your weights live elsewhere).

## Troubleshooting
- **403 Unauthorized on upload**: use `SUPABASE_SERVICE_KEY` or add appropriate RLS policies; confirm bucket is public for reads.
- **Weights not found**: set `MODEL_PATH` or place a checkpoint at `runs/train/ppe_yolov8/weights/best.pt`.
- **Small preview**: the Streamlit image preview uses the column width; widen the page layout in Streamlit settings if needed.
