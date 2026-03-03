# HIRA — Health Intelligence & Report Analyser

<p align="center">
  <img src="templates/assets/images/logo.png" alt="HIRA Logo" width="160"/>
</p>

<p align="center">
  <strong>AI-Powered Medical Report Analysis & Clinical Triage System</strong><br/>
  B.Tech Final Year Main Project · Ilahia College of Engineering and Technology
</p>

---

## 📋 Project Overview

**HIRA (Health Intelligence & Report Analyser)** is an AI-powered web application designed to assist medical professionals with the analysis of diagnostic reports. Patients can upload medical scans and lab reports, which are then analysed in real-time by deep learning models. Doctors receive AI-generated summaries, risk scores, and Grad-CAM visual explanations to support faster and more informed clinical decisions.

HIRA bridges the gap between raw medical imaging data and actionable clinical insight — making advanced diagnostic assistance accessible through a simple, secure web interface.

---

## 👥 Team

| Name | Role |
|------|------|
| **Ajmal Shan P** | Team Member |
| **Ihsan Jafry Sait** | Team Member |
| **Mazin Muneer** | Team Member |
| **Muhammed Sinan CT** | Team Member |

**Project Guide:** Prof. Ashna Shanavas

**Institution:** Ilahia College of Engineering and Technology

---

## ✨ Key Features

- **Multi-Modal AI Analysis** — Supports X-Ray, MRI, CT Scan, Breast Ultrasound, Kidney Scan, Oral Scan, and Blood Test (PDF) reports.
- **Deep Learning Inference** — TensorFlow/Keras `.h5` models trained for disease classification across 6 imaging modalities.
- **Grad-CAM Heatmaps** — Generates visual attention overlays on images so doctors can see exactly where the AI is focusing.
- **Gemini LLM Summaries** — Integrates Google Gemini to produce professional, patient-age-aware clinical summaries for every report.
- **Multimodal Risk Fusion** — Fuses risk scores across multiple uploaded reports per patient for a holistic triage priority.
- **Role-Based Dashboards** — Separate, secure dashboards for Patients and Doctors with distinct workflows.
- **Doctor Review Workflow** — Doctors can approve cases, modify severity/risk scores, request additional tests, and add clinical notes.
- **Blood Test PDF Parsing** — Automatically extracts Haemoglobin, WBC, Platelets, RBC, and Glucose values from uploaded lab PDFs and flags abnormalities.
- **In-App Messaging** — Real-time chat channel between patients and doctors.
- **Priority Triage Engine** — Automatically assigns Critical / High / Moderate / Low priority based on disease severity, model confidence, patient age, and comorbidities.

---

## 🤖 AI Models & Detectable Conditions

| Modality | Model File | Detectable Conditions |
|----------|------------|----------------------|
| X-Ray | `xray.h5` | Pneumonia, Normal |
| MRI (Brain) | `mri.h5` | Glioma, Meningioma, Pituitary Tumour, Normal |
| CT Scan (Lung) | `ct.h5` | Squamous Cell Carcinoma, Adenocarcinoma, Large Cell Carcinoma, Normal |
| Breast | `breast.h5` | Malignant, Benign |
| Kidney | `kidney.h5` | Kidney Tumour, Normal |
| Oral | `oral.h5` | Oral Squamous Cell Carcinoma (SCC), Normal |
| Blood Test (PDF) | Rule-based + Gemini | Anaemia, Infection/Inflammation, Hyperglycaemia, Thrombocytopenia, Normal Blood Profile |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10, Flask |
| **AI / ML** | TensorFlow 2.x, tf-keras, NumPy, Pillow |
| **LLM Integration** | Google Gemini API (`gemini-1.5-flash`) |
| **Database** | SQLite3 (WAL mode) |
| **PDF Parsing** | pypdf / PyPDF2 |
| **Frontend** | HTML5, CSS3 (Jinja2 templates) |
| **Environment Config** | python-dotenv |

---

## 📁 Project Structure

```
HIRA_ULTIMATE_final/
│
├── backend/
│   ├── app.py                  # Main Flask application & all routes
│   ├── migrate.py              # Database migration helper
│   └── ai_analysis/            # Stores latest AI result text files
│       ├── xray_result.txt
│       ├── mri_result.txt
│       └── blood_result.txt
│
├── ai_models/
│   ├── xray/                   # X-Ray model (.h5) + labels
│   ├── mri/                    # MRI brain tumour model
│   ├── ct/                     # CT lung cancer model
│   ├── breast_cancer/          # Breast cancer model
│   ├── kidney_cancer/          # Kidney tumour model
│   ├── oral_cancer/            # Oral cancer model
│   └── blood_tests/            # Blood test rule engine (rules.py)
│
├── templates/                  # Jinja2 HTML templates
│   ├── index.html              # Landing page
│   ├── login.html
│   ├── register.html
│   ├── patient-dashboard.html
│   ├── doctor-dashboard.html
│   ├── patient-case.html
│   ├── doctor-case.html
│   ├── upload.html
│   └── messages.html
│
├── static/
│   ├── style.css
│   └── generated/heatmaps/     # Grad-CAM output images
│
├── storage/
│   ├── uploads/                # Patient-uploaded scan images
│   └── reports/                # Patient-uploaded PDF reports
│
├── users.db                    # SQLite database
├── .env                        # Environment variables (API keys)
├── requirements.txt
├── repair.py                   # DB repair utility
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- pip
- A Google Gemini API Key ([get one here](https://aistudio.google.com/))

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd HIRA_ULTIMATE_final
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create or edit the `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Configure Storage Paths

Open `backend/app.py` and update the following paths to match your local system:

```python
PDF_STORAGE_PATH = 'storage/reports'
IMG_STORAGE_PATH = 'storage/uploads'
AI_RESULTS_PATH  = 'backend/ai_analysis'
HEATMAP_STORAGE_PATH = 'static/generated/heatmaps'
```

> **Note:** The default paths in the repository are set to the original developer's machine. Replace them with relative or absolute paths valid on your system.

Also update the paths inside `AI_MODELS_CONFIG` in `app.py` to point to the correct `.h5` model files.

### 5. Run the Application

```bash
cd backend
python app.py
```

The application will be available at: **http://localhost:5000**

---

## 🗄️ Database

HIRA uses SQLite with the following tables:

- **`users`** — Stores patient and doctor accounts (name, email, role, DOB, past conditions).
- **`reports`** — Stores every uploaded report with AI predictions, risk scores, Grad-CAM paths, doctor notes, and fusion results.
- **`messages`** — Stores the in-app chat messages between patients and doctors.

The database is auto-initialised on first run via `init_db()`.

---

## 🔑 User Roles

### Patient
- Register and log in to a personal dashboard.
- Upload diagnostic scans (JPG/PNG) or blood test PDFs.
- View AI analysis results **only after a doctor has approved the case**.
- Receive doctor notes, updated risk levels, and requests for additional tests.
- Chat with an assigned doctor.

### Doctor
- View all patient cases sorted by AI-computed risk score (highest priority first).
- Inspect Grad-CAM heatmaps and AI summaries for each report.
- Approve cases, override AI risk scores, request additional tests, or add clinical notes.
- Chat with patients.

---

## 🔬 How the AI Pipeline Works

1. **Upload** — Patient uploads a scan image or blood test PDF.
2. **Preprocessing** — Image is resized to 224×224 and normalised for model input.
3. **Inference** — The appropriate `.h5` Keras model classifies the image and returns a predicted class and confidence score.
4. **Grad-CAM** — A gradient-weighted class activation map is generated and overlaid on the original image for visual interpretability.
5. **Gemini Summary** — The predicted class, confidence, patient age, and scan type are sent to Google Gemini, which returns a professional 2–3 sentence clinical summary.
6. **Risk Scoring** — A base risk score (0–10) is computed from disease severity weight, model confidence, patient age, and comorbidities.
7. **Multimodal Fusion** — The base risk score is fused with the patient's historical reports to produce a final fused risk score and priority level.
8. **Doctor Review** — All results are held until a doctor reviews and approves the case, at which point the patient can see the full analysis.

---

## ⚠️ Important Disclaimer

HIRA is an academic research prototype developed as a B.Tech final year project. It is **not a certified medical device** and must **not be used for actual clinical diagnosis**. All AI outputs are intended to assist qualified medical professionals and must be interpreted in conjunction with professional clinical judgement.

---

## 📄 License

This project is developed for academic purposes at **Ilahia College of Engineering and Technology**. All rights reserved by the development team.

---

<p align="center">Made with ❤️ by Team HIRA · Ilahia College of Engineering and Technology</p>
