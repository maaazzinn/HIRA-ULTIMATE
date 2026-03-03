import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import sqlite3
import datetime
import random
import re
import json
import numpy as np
import google.generativeai as genai  
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, abort
from werkzeug.utils import secure_filename
from tf_keras.models import load_model
from PIL import Image, ImageOps
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

def load_local_env(path):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except Exception as e:
        print(f"Warning: failed to load .env file: {e}")

load_local_env(ENV_PATH)

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

app.secret_key = 'hira_ultimate_secret_key_demo'
BASE_DIR = PROJECT_ROOT
DB_NAME = os.path.join(BASE_DIR, "users.db")

# --- CONFIGURATION FOR STORAGE PATHS ---
PDF_STORAGE_PATH = os.path.join(BASE_DIR, "storage", "reports")
IMG_STORAGE_PATH = os.path.join(BASE_DIR, "storage", "uploads")
AI_RESULTS_PATH = os.path.join(BASE_DIR, "backend", "ai_analysis")
HEATMAP_STORAGE_PATH = os.path.join(BASE_DIR, "static", "generated", "heatmaps")

# --- GEMINI API CONFIGURATION ---
GEMINI_MODEL_CANDIDATES = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro"
]
llm_model = None
gemini_init_error = ""

def init_gemini_model():
    global llm_model, gemini_init_error

    if llm_model is not None:
        return llm_model

    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        gemini_init_error = "Missing GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
        return None

    try:
        genai.configure(api_key=gemini_api_key)
        for model_name in GEMINI_MODEL_CANDIDATES:
            try:
                candidate = genai.GenerativeModel(model_name)
                # Lightweight connectivity/auth check.
                candidate.generate_content("Ping")
                llm_model = candidate
                gemini_init_error = ""
                print(f"Gemini initialized with model: {model_name}")
                return llm_model
            except Exception as model_error:
                gemini_init_error = f"{model_name}: {model_error}"
                continue
        return None
    except Exception as e:
        gemini_init_error = str(e)
        return None

# AI Model Configurations
AI_MODELS_CONFIG = {
    'xray': {
        'model': os.path.join(BASE_DIR, 'ai_models', 'xray', 'xray.h5'),
        'labels': os.path.join(BASE_DIR, 'ai_models', 'xray', 'labels.txt'),
        'result_file': 'xray_result.txt'
    },
    'mri': {
        'model': os.path.join(BASE_DIR, 'ai_models', 'mri', 'mri.h5'),
        'labels': os.path.join(BASE_DIR, 'ai_models', 'mri', 'labels.txt'),
        'result_file': 'mri_result.txt'
    },
    'ct': {
        'model': os.path.join(BASE_DIR, 'ai_models', 'ct', 'ct.h5'),
        'labels': os.path.join(BASE_DIR, 'ai_models', 'ct', 'labels.txt'),
        'result_file': 'ct_result.txt'
    },
    'breast': {
        'model': os.path.join(BASE_DIR, 'ai_models', 'breast_cancer', 'breast.h5'),
        'labels': os.path.join(BASE_DIR, 'ai_models', 'breast_cancer', 'labels.txt'),
        'result_file': 'breast_result.txt'
    },
    'kidney': {
        'model': os.path.join(BASE_DIR, 'ai_models', 'kidney_cancer', 'kidney.h5'),
        'labels': os.path.join(BASE_DIR, 'ai_models', 'kidney_cancer', 'labels.txt'),
        'result_file': 'kidney_result.txt'
    },
    'oral': {
        'model': os.path.join(BASE_DIR, 'ai_models', 'oral_cancer', 'oral.h5'),
        'labels': os.path.join(BASE_DIR, 'ai_models', 'oral_cancer', 'labels.txt'),
        'result_file': 'oral_result.txt'
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
os.makedirs(IMG_STORAGE_PATH, exist_ok=True)
os.makedirs(AI_RESULTS_PATH, exist_ok=True)
os.makedirs(HEATMAP_STORAGE_PATH, exist_ok=True)

SEVERITY_WEIGHTS = {
    'cancer': 9.0,
    'carcinoma': 9.0,
    'malignant': 9.0,
    'metastasis': 9.0,
    'tumor': 8.0,
    'stroke': 8.0,
    'aneurysm': 8.0,
    'hemorrhage': 8.0,
    'pneumonia': 6.0,
    'infection': 5.0,
    'fracture': 4.0,
    'cyst': 3.0,
    'benign': 2.0,
    'normal': 1.0,
    'healthy': 1.0
}

DOMAIN_KEYWORDS = {
    'oncology': ['cancer', 'tumor', 'carcinoma', 'malignant', 'metastasis', 'neoplasm'],
    'pulmonary': ['lung', 'pneumonia', 'copd', 'effusion', 'pleural', 'asthma', 'opacity'],
    'neurology': ['brain', 'stroke', 'aneurysm', 'hemorrhage', 'seizure'],
    'renal': ['kidney', 'renal', 'nephro'],
    'cardiac': ['heart', 'cardiac', 'coronary', 'myocardial'],
    'oral': ['oral', 'mouth', 'tongue', 'gingiva']
}

PAST_CONDITION_RISK_KEYWORDS = [
    'diabetes',
    'hypertension',
    'copd',
    'asthma',
    'ckd',
    'chronic kidney',
    'heart disease',
    'coronary',
    'cancer',
    'stroke',
    'smoker'
]

# --- Helper Functions ---

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def calculate_age_from_dob(dob_text):
    if not dob_text:
        return None
    try:
        dob = datetime.datetime.strptime(dob_text, "%Y-%m-%d").date()
        today = datetime.date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except (ValueError, TypeError):
        return None

def parse_past_conditions(past_conditions_text):
    if not past_conditions_text:
        return []
    return [cond.strip().lower() for cond in re.split(r'[;,|]', past_conditions_text) if cond.strip()]

def get_disease_severity_weight(disease_name):
    if not disease_name:
        return 2.0

    label = disease_name.lower()
    best_weight = 2.0
    for keyword, weight in SEVERITY_WEIGHTS.items():
        if keyword in label:
            best_weight = max(best_weight, weight)
    return best_weight

def get_condition_domain(disease_name):
    if not disease_name:
        return 'general'
    label = disease_name.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in label for keyword in keywords):
            return domain
    return 'general'

def has_comorbidity(past_conditions_text):
    conditions = parse_past_conditions(past_conditions_text)
    if not conditions:
        return False

    for condition in conditions:
        if any(keyword in condition for keyword in PAST_CONDITION_RISK_KEYWORDS):
            return True
    return len(conditions) > 0

def calculate_risk_score(predicted_class, confidence, age=None, past_conditions=""):
    severity = get_disease_severity_weight(predicted_class)
    confidence_ratio = confidence if confidence <= 1 else confidence / 100.0
    confidence_ratio = clamp(confidence_ratio, 0.0, 1.0)
    confidence_pct = confidence_ratio * 100

    # Core rule: model confidence multiplied by disease severity weight.
    risk_score = confidence_ratio * severity

    # Modifiers requested by user.
    if age is not None and age > 60:
        risk_score += 1.0
    if has_comorbidity(past_conditions):
        risk_score += 1.0
    if confidence_pct > 90:
        risk_score += 0.5

    risk_score = round(min(risk_score, 10.0), 2)

    if risk_score >= 8:
        priority_level = "High"
    elif risk_score >= 5:
        priority_level = "Medium"
    else:
        priority_level = "Low"

    return risk_score, priority_level

def get_priority_level_from_risk(risk_score):
    if risk_score >= 9.0:
        return "Critical"
    if risk_score >= 7.0:
        return "High"
    if risk_score >= 4.0:
        return "Moderate"
    return "Low"

def map_priority_to_status(priority_level):
    level = (priority_level or '').strip().lower()
    if level in ['critical', 'high']:
        return 'Critical'
    if level in ['moderate', 'medium']:
        return 'Moderate'
    return 'Normal'

def is_abnormal_condition(predicted_class):
    if not predicted_class:
        return False
    label = predicted_class.lower()
    return not any(normal_word in label for normal_word in ['normal', 'healthy', 'no finding'])

def get_multimodal_fusion_summary(fusion_context):
    modalities = fusion_context.get('modalities', [])
    findings = fusion_context.get('findings', [])
    fused_risk_score = fusion_context.get('fused_risk_score', 0.0)
    disagreement = fusion_context.get('disagreement_flag', False)
    related_domains = fusion_context.get('related_abnormal_domains', [])

    prompt = f"""
    You are an AI clinical triage assistant.
    Create a concise consolidated cross-modal summary for a doctor.
    Modalities reviewed: {', '.join(modalities) if modalities else 'single modality'}.
    Findings: {'; '.join(findings) if findings else 'No specific findings'}.
    Fused risk score (0-10): {fused_risk_score}.
    Related abnormal domains detected across modalities: {', '.join(related_domains) if related_domains else 'none'}.
    Cross-modal disagreement flag: {disagreement}.

    Output 2-3 sentences with:
    1) Overall integrated interpretation
    2) Whether modalities are concordant or conflicting
    3) Suggested urgency level
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Multimodal Gemini Summary Error: {e}")
        agreement_text = "cross-modal disagreement present" if disagreement else "modalities are broadly concordant"
        related_text = (
            f" Related abnormalities converge in {', '.join(related_domains)} domains."
            if related_domains else ""
        )
        return (
            f"Integrated triage across {max(len(modalities), 1)} modality inputs yields "
            f"a fused risk score of {fused_risk_score}/10; {agreement_text}.{related_text} "
            f"Prioritize clinical review based on this combined context."
        )

def build_multimodal_fusion_context(user_id, current_report):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT report_type, predicted_class, confidence, confidence_score, risk_score
        FROM reports
        WHERE user_id = ?
        ORDER BY upload_date DESC
        LIMIT 12
    """, (user_id,))
    previous_reports = cursor.fetchall()
    conn.close()

    contexts = []
    for row in previous_reports:
        existing_conf = row['confidence'] if row['confidence'] is not None else row['confidence_score']
        contexts.append({
            'report_type': row['report_type'] or 'unknown',
            'predicted_class': row['predicted_class'] or 'N/A',
            'confidence': float(existing_conf or 0.0),
            'risk_score': float(row['risk_score'] or 0.0)
        })
    contexts.append(current_report)

    risk_values = [clamp(float(c.get('risk_score', 0.0)), 0.0, 10.0) for c in contexts]
    if risk_values:
        fused_risk_score = (max(risk_values) * 0.6) + (float(np.mean(risk_values)) * 0.4)
    else:
        fused_risk_score = clamp(float(current_report.get('risk_score', 0.0)), 0.0, 10.0)

    modality_set = set()
    abnormal_domains = {}
    findings = []
    for c in contexts:
        modality = (c.get('report_type') or 'unknown').lower()
        modality_set.add(modality)
        predicted = c.get('predicted_class') or 'N/A'
        findings.append(f"{modality}: {predicted}")

        if is_abnormal_condition(predicted):
            domain = get_condition_domain(predicted)
            abnormal_domains.setdefault(domain, set()).add(modality)

    related_abnormal_domains = [domain for domain, mods in abnormal_domains.items() if len(mods) >= 2]
    if related_abnormal_domains:
        fused_risk_score += 1.0

    disagreement_flag = len(abnormal_domains.keys()) >= 2 and len(modality_set) >= 2
    fused_risk_score = round(clamp(fused_risk_score, 0.0, 10.0), 2)
    priority_level = get_priority_level_from_risk(fused_risk_score)

    fusion_context = {
        'fused_risk_score': fused_risk_score,
        'priority_level': priority_level,
        'disagreement_flag': disagreement_flag,
        'related_abnormal_domains': related_abnormal_domains,
        'related_boost': 1.0 if related_abnormal_domains else 0.0,
        'modalities': sorted(list(modality_set)),
        'findings': findings
    }
    fusion_context['consolidated_summary'] = get_multimodal_fusion_summary(fusion_context)
    return fusion_context

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
    return None

def generate_gradcam_overlay(model, img_array, pred_index, original_image_path, output_stem):
    try:
        last_conv_layer_name = get_last_conv_layer_name(model)
        if not last_conv_layer_name:
            return "", ""

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return "", ""

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap.numpy(), 0)

        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        original = Image.open(original_image_path).convert("RGB")
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255), mode='L').resize(original.size, Image.Resampling.BILINEAR)

        # Colorized attention map (red-yellow over black) for clinical readability.
        colored_heatmap = ImageOps.colorize(heatmap_img, black="#000000", mid="#ff7f00", white="#ffff00")
        overlay = Image.blend(original, colored_heatmap, alpha=0.45)

        heatmap_filename = f"heatmap_{output_stem}.png"
        overlay_filename = f"overlay_{output_stem}.png"
        heatmap_abs_path = os.path.join(HEATMAP_STORAGE_PATH, heatmap_filename)
        overlay_abs_path = os.path.join(HEATMAP_STORAGE_PATH, overlay_filename)

        colored_heatmap.save(heatmap_abs_path, format='PNG')
        overlay.save(overlay_abs_path, format='PNG')

        heatmap_rel_path = os.path.join('generated', 'heatmaps', heatmap_filename).replace("\\", "/")
        overlay_rel_path = os.path.join('generated', 'heatmaps', overlay_filename).replace("\\", "/")
        return heatmap_rel_path, overlay_rel_path
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return "", ""

def get_gemini_summary(disease_name, confidence_score, patient_age, scan_type):
    """Generates a professional medical summary using Gemini LLM."""
    confidence_ratio = clamp(confidence_score if confidence_score is not None else 0.0, 0.0, 1.0)
    confidence_percent = round(confidence_ratio * 100, 2)
    model = init_gemini_model()
    prompt = f"""
    You are a medical AI assistant. A patient (Age: {patient_age}) has an uploaded {scan_type} report.
    The AI analysis detected: {disease_name} with confidence {confidence_percent}%.
    
    Write a 2-3 sentence summary for the attending doctor explaining:
    1. The general nature of this condition.
    2. Potential clinical implications for a patient of this age.
    Keep the tone professional, concise, and informative.
    """
    try:
        if model is None:
            raise RuntimeError(gemini_init_error or "Gemini model unavailable")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini LLM Error: {e}")
        # Dynamic fallback when Gemini is unavailable.
        return (
            f"The AI model predicts {disease_name} with {confidence_percent}% confidence. "
            f"Clinical correlation is recommended."
        )

def extract_text_from_pdf(pdf_path):
    try:
        text_parts = []
        try:
            from pypdf import PdfReader  # Preferred modern package
        except Exception:
            from PyPDF2 import PdfReader  # Fallback
        reader = PdfReader(pdf_path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                pass
        for page in reader.pages:
            page_text = ""
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Blood PDF extraction error: {e}")
        return ""

def _normalize_pdf_text(text):
    if not text:
        return ""
    normalized = text.replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    return normalized

def _extract_value_with_context(text, label_patterns):
    if not text:
        return None, ""

    candidates = []
    normalized_text = _normalize_pdf_text(text)
    lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]
    number_part = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    unit_part = r"([A-Za-z/%][A-Za-z0-9/%\.\-\s]{0,24})?"

    for line in lines:
        for label_pattern in label_patterns:
            pattern = rf"(?:{label_pattern})[^\n\r:]{{0,40}}[:\-]?\s*{number_part}\s*{unit_part}"
            for match in re.finditer(pattern, line, re.IGNORECASE):
                raw_value = match.group(1)
                unit_text = (match.group(2) or "").strip()
                try:
                    numeric_value = float(raw_value.replace(",", ""))
                    candidates.append((numeric_value, unit_text, line))
                except Exception:
                    continue

    if not candidates:
        for label_pattern in label_patterns:
            pattern = rf"(?:{label_pattern})[^\n\r:]{{0,40}}[:\-]?\s*{number_part}\s*{unit_part}"
            for match in re.finditer(pattern, normalized_text, re.IGNORECASE):
                raw_value = match.group(1)
                unit_text = (match.group(2) or "").strip()
                try:
                    numeric_value = float(raw_value.replace(",", ""))
                    candidates.append((numeric_value, unit_text, ""))
                except Exception:
                    continue

    if not candidates:
        return None, ""

    return candidates[0][0], f"{candidates[0][1]} {candidates[0][2]}".strip()

def _scale_count_value(value, unit_context):
    if value is None:
        return None

    unit_text = (unit_context or "").lower()
    scaled = value
    if "lakh" in unit_text or "lac" in unit_text:
        scaled = value * 100000
    elif "million" in unit_text or "millions" in unit_text:
        scaled = value * 1000000
    return scaled

def run_blood_test_analysis(pdf_path, patient_age="N/A"):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        details = {
            "markers": [],
            "abnormal_findings": [],
            "extraction_status": "failed",
            "extraction_message": "Unable to read blood report PDF text."
        }
        return "Blood Test Analysis Unavailable", 0.0, "Unable to read blood report PDF text.", details

    hb, hb_ctx = _extract_value_with_context(
        text,
        [r"haemoglobin", r"hemoglobin", r"\bhb\b"]
    )
    wbc_raw, wbc_ctx = _extract_value_with_context(
        text,
        [r"\bwbc\b", r"total\s*(?:count|leucocyte\s*count|leukocyte\s*count)", r"\btlc\b"]
    )
    platelets_raw, platelets_ctx = _extract_value_with_context(
        text,
        [r"platelet(?:\s*count)?", r"platelets"]
    )
    rbc_raw, rbc_ctx = _extract_value_with_context(
        text,
        [r"rbc(?:\s*count)?"]
    )
    glucose, glucose_ctx = _extract_value_with_context(
        text,
        [r"glucose", r"blood\s*sugar", r"\bfbs\b", r"\brbs\b"]
    )

    wbc = _scale_count_value(wbc_raw, wbc_ctx)
    platelets = _scale_count_value(platelets_raw, platelets_ctx)
    rbc = rbc_raw
    if rbc is not None:
        rbc_ctx_text = (rbc_ctx or "").lower()
        if "million" in rbc_ctx_text or "millions" in rbc_ctx_text:
            pass
        elif rbc > 1000:
            rbc = rbc / 1000000.0

    abnormalities = []
    severity_points = 0.0

    if hb is not None and hb < 12.0:
        abnormalities.append(f"Low Hemoglobin ({hb})")
        severity_points += min(3.0, (12.0 - hb) * 0.5)
    if wbc is not None and wbc > 11000:
        abnormalities.append(f"High WBC ({wbc})")
        severity_points += min(2.5, (wbc - 11000) / 3000)
    if platelets is not None and platelets < 150000:
        abnormalities.append(f"Low Platelets ({platelets})")
        severity_points += min(3.0, (150000 - platelets) / 50000)
    if rbc is not None and rbc < 4.0:
        abnormalities.append(f"Low RBC ({rbc})")
        severity_points += min(2.0, (4.0 - rbc) * 1.2)
    if glucose is not None and glucose > 140:
        abnormalities.append(f"High Glucose ({glucose})")
        severity_points += min(2.5, (glucose - 140) / 40)

    marker_ranges = {
        "Hemoglobin": (12.0, 17.5, "g/dL", hb),
        "WBC": (4500, 11000, "/uL", wbc),
        "Platelets": (150000, 450000, "/uL", platelets),
        "RBC": (4.0, 6.0, "M/uL", rbc),
        "Glucose": (70, 140, "mg/dL", glucose),
    }
    marker_rows = []
    for marker_name, (low, high, unit, value) in marker_ranges.items():
        if value is None:
            continue
        if value < low:
            status = "LOW"
        elif value > high:
            status = "HIGH"
        else:
            status = "NORMAL"
        marker_rows.append({
            "name": marker_name,
            "value": round(value, 2),
            "unit": unit,
            "status": status,
            "reference_range": f"{low}-{high}",
        })

    if abnormalities:
        if any("Low Hemoglobin" in x for x in abnormalities):
            predicted_class = "Possible Anemia"
        elif any("High WBC" in x for x in abnormalities):
            predicted_class = "Possible Infection/Inflammation Pattern"
        elif any("High Glucose" in x for x in abnormalities):
            predicted_class = "Possible Hyperglycemia Pattern"
        elif any("Low Platelets" in x for x in abnormalities):
            predicted_class = "Possible Thrombocytopenia Pattern"
        else:
            predicted_class = "Abnormal Blood Profile"
    else:
        predicted_class = "Normal Blood Profile"

    if not marker_rows:
        predicted_class = "Blood Test Analysis Limited"
        confidence_score = 0.45
        ai_summary = (
            "Blood report text was detected, but key CBC markers could not be reliably extracted. "
            "Please verify PDF quality or upload a clearer report."
        )
        details = {
            "markers": [],
            "abnormal_findings": [],
            "extraction_status": "partial",
            "extraction_message": "No key markers could be parsed from extracted text."
        }
        return predicted_class, confidence_score, ai_summary, details

    confidence_score = clamp(0.55 + min(0.4, severity_points * 0.12 + len(abnormalities) * 0.05), 0.50, 0.95)
    ai_summary = get_gemini_summary(predicted_class, confidence_score, patient_age, "Blood Test")
    if abnormalities:
        ai_summary = f"{ai_summary} Key abnormalities: {', '.join(abnormalities[:4])}."

    result_file_path = os.path.join(AI_RESULTS_PATH, "blood_result.txt")
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write("Scan Type: Blood Test\n")
            f.write(f"Predicted Class: {predicted_class}\n")
            f.write(f"Confidence Score: {confidence_score:.4f}\n")
            f.write(f"Gemini Clinical Summary: {ai_summary}\n")
            f.write(f"Analysis Date: {datetime.datetime.now()}\n")
    except Exception as e:
        print(f"Blood result save error: {e}")

    details = {
        "markers": marker_rows,
        "abnormal_findings": abnormalities,
        "extraction_status": "success",
        "extraction_message": ""
    }
    return predicted_class, confidence_score, ai_summary, details

def run_ai_prediction(scan_type, image_path, output_stem, patient_age="N/A"):
    """Handles AI inference and triggers Gemini summary generation."""
    scan_key = scan_type.lower()
    if scan_key not in AI_MODELS_CONFIG:
        return "Unknown Scan Type", 0.0, "N/A", "", ""

    config = AI_MODELS_CONFIG[scan_key]
    
    try:
        # Load model and labels
        model = load_model(config['model'], compile=False)
        with open(config['labels'], "r", encoding="utf-8") as f:
            class_names = f.readlines()

        # Preprocess image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        disease_clean = re.sub(r'^\d+\s+', '', class_name).strip()
        confidence_score = float(prediction[0][index])
        heatmap_path, overlay_path = generate_gradcam_overlay(model, data, index, image_path, output_stem)

        # Generate Gemini Summary
        ai_summary = get_gemini_summary(disease_clean, confidence_score, patient_age, scan_type)

        # Save results to .txt file
        result_file_path = os.path.join(AI_RESULTS_PATH, config['result_file'])
        with open(result_file_path, "w") as f:
            f.write(f"Scan Type: {scan_type}\n")
            f.write(f"Predicted Class: {disease_clean}\n")
            f.write(f"Confidence Score: {confidence_score:.4f}\n")
            f.write(f"Gemini Clinical Summary: {ai_summary}\n")
            f.write(f"Analysis Date: {datetime.datetime.now()}\n")

        return disease_clean, confidence_score, ai_summary, heatmap_path, overlay_path
    except Exception as e:
        print(f"AI Prediction Error: {e}")
        return "Error in Analysis", 0.0, "N/A", "", ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            dob TEXT,
            past_conditions TEXT DEFAULT ''
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            report_type TEXT,
            upload_date TEXT,
            status TEXT,
            file_size TEXT,
            notes TEXT,
            summary TEXT,
            predicted_class TEXT,
            confidence REAL DEFAULT 0,
            confidence_score REAL DEFAULT 0,
            base_risk_score REAL DEFAULT 0,
            risk_score REAL DEFAULT 0,
            priority_level TEXT DEFAULT 'Low',
            analysis_details TEXT,
            heatmap_path TEXT,
            overlay_path TEXT,
            disagreement_flag INTEGER DEFAULT 0,
            related_boost REAL DEFAULT 0,
            modality_count INTEGER DEFAULT 1,
            doctor_reviewed INTEGER DEFAULT 0,
            doctor_notes TEXT DEFAULT '',
            risk_tier TEXT DEFAULT 'Low',
            filename TEXT,
            file_path TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            FOREIGN KEY(sender_id) REFERENCES users(id),
            FOREIGN KEY(receiver_id) REFERENCES users(id)
        )
    ''')
    # Backward compatibility for existing databases.
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN summary TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN past_conditions TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN predicted_class TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN confidence REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN confidence_score REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN base_risk_score REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN risk_score REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN priority_level TEXT DEFAULT 'Low'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN analysis_details TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN heatmap_path TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN overlay_path TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN disagreement_flag INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN related_boost REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN modality_count INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN doctor_reviewed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN doctor_notes TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN risk_tier TEXT DEFAULT 'Low'")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

init_db()

# --- Routes ---

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/register.html', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Defensive: recreate required tables if DB file is empty/corrupted state.
        init_db()

        fullname = request.form.get('fullname')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        role = request.form.get('role')
        dob = request.form.get('dob')
        past_conditions = request.form.get('past_conditions', '')

        if not all([fullname, email, phone, password, role, dob]):
            flash("Please fill in all required registration fields.")
            return render_template('register.html')

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (fullname, email, phone, password, role, dob, past_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fullname, email, phone, password, role, dob, past_conditions))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            try:
                conn.close()
            except Exception:
                pass
            flash("Email already exists.")
            return render_template('register.html')
        except sqlite3.OperationalError as e:
            try:
                conn.close()
            except Exception:
                pass
            # Auto-heal common DB initialization issue and retry once.
            if "no such table: users" in str(e).lower():
                try:
                    init_db()
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO users (fullname, email, phone, password, role, dob, past_conditions)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (fullname, email, phone, password, role, dob, past_conditions))
                    conn.commit()
                    conn.close()
                    return redirect(url_for('login'))
                except Exception as retry_error:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    print(f"Registration Retry Error: {retry_error}")
                    flash("Registration failed due to DB setup issue. Please try again.")
                    return render_template('register.html')
            print(f"Registration DB Error: {e}")
            flash("Registration failed due to database error. Please try again.")
            return render_template('register.html')
        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            print(f"Registration Error: {e}")
            flash("Registration failed. Please try again.")
            return render_template('register.html')
            
    return render_template('register.html')

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['name'] = user['fullname']
            session['dob'] = user['dob']
            session['past_conditions'] = user['past_conditions'] if 'past_conditions' in user.keys() else ''

            if user['role'] == 'patient':
                return redirect(url_for('patient_dashboard'))
            elif user['role'] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
        else:
            flash("Invalid Credentials.")

    return render_template('login.html')

@app.route('/upload.html', methods=['GET', 'POST'])
def upload_report():
    if 'user_id' not in session or session['role'] != 'patient':
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        report_type = request.form.get('reportType')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            user_name = session.get('name', 'User')
            raw_filename = f"{user_name}_{report_type}"
            clean_filename = re.sub(r'[^\w\s-]', '', raw_filename).strip().replace(' ', '_')
            final_filename = f"{clean_filename}.{file_ext}"
            
            save_folder = PDF_STORAGE_PATH if file_ext == 'pdf' else IMG_STORAGE_PATH
            full_path = os.path.join(save_folder, final_filename)
            
            if os.path.exists(full_path):
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                final_filename = f"{clean_filename}_{timestamp}.{file_ext}"
                full_path = os.path.join(save_folder, final_filename)

            try:
                file.save(full_path)
                
                # --- INTEGRATED AI & GEMINI LOGIC ---
                predicted_class = "N/A"
                confidence = 0.0
                ai_summary = ""
                analysis_details = None
                heatmap_path = ""
                overlay_path = ""
                if file_ext in ['jpg', 'jpeg', 'png']:
                    output_stem = os.path.splitext(final_filename)[0]
                    predicted_class, confidence, ai_summary, heatmap_path, overlay_path = run_ai_prediction(
                        report_type,
                        full_path,
                        output_stem,
                        session.get('dob', 'N/A')
                    )
                elif file_ext == 'pdf' and (report_type or '').strip().lower() in ['blood test', 'bloodtest', 'cbc', 'lipid']:
                    predicted_class, confidence, ai_summary, analysis_details = run_blood_test_analysis(
                        full_path,
                        session.get('dob', 'N/A')
                    )
                
            except Exception as e:
                flash(f"Error saving file: {e}")
                return redirect(request.url)

            file_size_mb = f"{os.path.getsize(full_path) / (1024 * 1024):.2f} MB"
            upload_date = datetime.date.today().strftime("%Y-%m-%d")

            patient_age = calculate_age_from_dob(session.get('dob'))
            past_conditions = session.get('past_conditions', '')
            base_risk_score, base_priority_level = calculate_risk_score(
                predicted_class,
                confidence,
                patient_age,
                past_conditions
            )

            fusion_result = build_multimodal_fusion_context(session['user_id'], {
                'report_type': report_type or 'unknown',
                'predicted_class': predicted_class,
                'confidence': confidence,
                'risk_score': base_risk_score
            })
            risk_score = fusion_result['fused_risk_score']
            priority_level = fusion_result['priority_level']
            consolidated_summary = fusion_result['consolidated_summary']
            disagreement_flag = 1 if fusion_result['disagreement_flag'] else 0
            related_boost = fusion_result['related_boost']
            modality_count = len(fusion_result['modalities'])

            if priority_level in ['Critical', 'High']:
                simulated_status = 'Critical'
            elif priority_level == 'Moderate':
                simulated_status = 'Moderate'
            else:
                simulated_status = 'Normal'

            if predicted_class != "N/A":
                simulated_notes = (
                    f"Inference Result: {predicted_class}. "
                    f"Confidence: {confidence:.2f}. "
                    f"Base Risk: {base_risk_score}/10 ({base_priority_level}). "
                    f"Fused Risk: {risk_score}/10 ({priority_level}). "
                    f"Disagreement: {'Yes' if disagreement_flag else 'No'}."
                )
            else:
                simulated_notes = (
                    f"Standard file upload. Base Risk: {base_risk_score}/10 ({base_priority_level}). "
                    f"Fused Risk: {risk_score}/10 ({priority_level}). "
                    f"Disagreement: {'Yes' if disagreement_flag else 'No'}."
                )

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reports (
                    user_id, report_type, upload_date, status, file_size, notes, summary,
                    predicted_class, confidence, confidence_score, base_risk_score, risk_score, priority_level,
                    analysis_details, heatmap_path, overlay_path, disagreement_flag, related_boost, modality_count,
                    doctor_reviewed, doctor_notes, risk_tier, filename, file_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                report_type,
                upload_date,
                simulated_status,
                file_size_mb,
                simulated_notes,
                consolidated_summary if consolidated_summary else ai_summary,
                predicted_class,
                confidence,
                confidence,
                base_risk_score,
                risk_score,
                priority_level,
                json.dumps(analysis_details) if analysis_details else None,
                heatmap_path,
                overlay_path,
                disagreement_flag,
                related_boost,
                modality_count,
                0,
                '',
                priority_level,
                final_filename,
                full_path
            ))
            conn.commit()
            conn.close()

            return redirect(url_for('patient_dashboard'))

    return render_template('upload.html')

@app.route('/patient-dashboard.html')
def patient_dashboard():
    if 'user_id' in session and session['role'] == 'patient':
        user_id = session['user_id']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports WHERE user_id = ? ORDER BY upload_date DESC", (user_id,))
        report_rows = cursor.fetchall()
        conn.close()

        reports = []
        for row in report_rows:
            report = dict(row)
            reviewed = bool(report.get('doctor_reviewed'))
            ai_summary = report.get('summary') or ''
            doctor_notes = (report.get('doctor_notes') or '').strip()
            report_status = (report.get('status') or '').strip()

            if reviewed:
                if doctor_notes:
                    report['summary'] = f"{ai_summary} Doctor Conclusion: {doctor_notes}".strip()
                else:
                    report['summary'] = ai_summary
                report['notes'] = report.get('notes') or "Doctor finalized this case."
            else:
                report['summary'] = ''
                if report_status == 'Pending Additional Tests':
                    report['notes'] = (
                        report.get('notes')
                        or doctor_notes
                        or "Doctor requested additional tests. Please upload the requested reports."
                    )
                elif doctor_notes:
                    report['notes'] = f"Doctor update: {doctor_notes}"
                else:
                    report['notes'] = "Under specialist review. Final conclusion will be available after doctor approval."

            reports.append(report)

        total_reports = len(reports)
        last_analysis = reports[0]['upload_date'] if reports else "N/A"
        health_status = "Stable"
        status_color = "green"

        if reports:
            for r in reports:
                if r['status'] == 'Critical':
                    health_status = "Action Required"
                    status_color = "red"
                    break
                elif r['status'] == 'Moderate' and health_status != "Action Required":
                    health_status = "Review Needed"
                    status_color = "orange"

        return render_template('patient-dashboard.html', 
                               name=session['name'],
                               user_id=user_id,
                               reports=reports,
                               total_reports=total_reports,
                               last_analysis=last_analysis,
                               health_status=health_status,
                               status_color=status_color)
                               
    return redirect(url_for('login'))

@app.route('/doctor-dashboard.html')
def doctor_dashboard():
    if 'user_id' in session and session['role'] == 'doctor':
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT r.*, u.fullname, u.id as patient_id
            FROM reports r 
            JOIN users u ON r.user_id = u.id 
            ORDER BY COALESCE(r.risk_score, 0) DESC, r.upload_date DESC
        """)
        report_rows = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM reports")
        total_cases = cursor.fetchone()[0]
        conn.close()

        prioritized_cases = []
        for row in report_rows:
            case = dict(row)
            risk_score = float(case.get('risk_score') or 0.0)
            raw_confidence = case.get('confidence')
            if raw_confidence is None:
                raw_confidence = case.get('confidence_score') or 0.0
            raw_confidence = float(raw_confidence)

            confidence_percent = raw_confidence * 100 if raw_confidence <= 1 else raw_confidence
            case['confidence_percent'] = round(clamp(confidence_percent, 0.0, 100.0), 2)
            case['predicted_condition'] = case.get('predicted_class') or 'Unknown'
            case['ai_summary'] = case.get('summary') or 'No AI summary available.'
            case['disagreement_flag'] = bool(case.get('disagreement_flag'))

            if risk_score >= 9.0:
                case['priority_badge'] = 'Critical'
            elif risk_score >= 7.0:
                case['priority_badge'] = 'High'
            elif risk_score >= 4.0:
                case['priority_badge'] = 'Moderate'
            else:
                case['priority_badge'] = 'Low'

            prioritized_cases.append(case)

        urgent_count = sum(1 for c in prioritized_cases if c['priority_badge'] in ['Critical', 'High'])
        completed_count = total_cases - urgent_count

        return render_template('doctor-dashboard.html', 
                               name=session['name'],
                               urgent_cases=prioritized_cases,
                               prioritized_cases=prioritized_cases,
                               total_cases=total_cases,
                               urgent_count=urgent_count,
                               completed_count=completed_count)
                               
    return redirect(url_for('login'))

@app.route('/patient-case.html')
def patient_case():
    if 'user_id' in session and session['role'] == 'patient':
        user_id = session['user_id']
        report_id = request.args.get('report_id', type=int)

        conn = get_db_connection()
        cursor = conn.cursor()
        if report_id:
            cursor.execute("""
                SELECT *
                FROM reports
                WHERE id = ? AND user_id = ?
            """, (report_id, user_id))
        else:
            cursor.execute("""
                SELECT *
                FROM reports
                WHERE user_id = ?
                ORDER BY upload_date DESC, id DESC
                LIMIT 1
            """, (user_id,))
        report = cursor.fetchone()
        conn.close()

        if not report:
            flash("No report found for this case.")
            return redirect(url_for('patient_dashboard'))

        case = dict(report)
        analysis_details = {}
        try:
            if case.get('analysis_details'):
                analysis_details = json.loads(case.get('analysis_details'))
        except Exception:
            analysis_details = {}
        raw_confidence = case.get('confidence')
        if raw_confidence is None:
            raw_confidence = case.get('confidence_score') or 0.0
        raw_confidence = float(raw_confidence or 0.0)
        confidence_percent = raw_confidence * 100 if raw_confidence <= 1 else raw_confidence
        confidence_percent = round(clamp(confidence_percent, 0.0, 100.0), 2)

        risk_score = float(case.get('risk_score') or 0.0)
        priority_badge = get_priority_level_from_risk(risk_score)
        doctor_reviewed = bool(case.get('doctor_reviewed'))
        doctor_notes = (case.get('doctor_notes') or '').strip()

        if doctor_reviewed:
            final_summary = case.get('summary') or "No AI summary available."
            if doctor_notes:
                final_summary = f"{final_summary} Doctor Conclusion: {doctor_notes}"
            display_confidence_percent = confidence_percent
            display_confidence_width = f"{confidence_percent:.0f}%"
            display_risk_score = round(risk_score, 2)
            display_predicted_condition = case.get('predicted_class') or 'Unknown'
            case_status_label = "Approved by Doctor"
        else:
            final_summary = "This case is under specialist review. Final conclusion will appear after doctor approval."
            display_confidence_percent = None
            display_confidence_width = "0%"
            display_risk_score = None
            display_predicted_condition = "Hidden until doctor approval"
            case_status_label = "Under Specialist Review"

        case_file_ext = os.path.splitext(case.get('file_path') or '')[1].lower()
        case_is_pdf = case_file_ext == '.pdf'
        case_is_blood = 'blood' in (case.get('report_type') or '').lower()

        return render_template(
            'patient-case.html',
            case_data=case,
            case_image_url=url_for('report_image', report_id=case['id']),
            confidence_percent=display_confidence_percent,
            confidence_width=display_confidence_width,
            risk_score=display_risk_score,
            priority_badge=priority_badge,
            predicted_condition=display_predicted_condition,
            doctor_reviewed=doctor_reviewed,
            case_status_label=case_status_label,
            final_summary=final_summary,
            patient_note=case.get('notes') or '',
            case_is_pdf=case_is_pdf,
            case_is_blood=case_is_blood,
            blood_markers=analysis_details.get('markers', []),
            blood_abnormal_findings=analysis_details.get('abnormal_findings', [])
        )
    return redirect(url_for('login'))

@app.route('/doctor-case.html', methods=['GET', 'POST'])
def doctor_case():
    if 'user_id' in session and session['role'] == 'doctor':
        report_id = request.values.get('report_id', type=int)

        if request.method == 'POST':
            if not report_id:
                flash('Missing report ID for doctor action.')
                return redirect(url_for('doctor_dashboard'))

            action = (request.form.get('action') or '').strip().lower()
            doctor_notes = (request.form.get('doctor_notes') or '').strip()
            selected_priority = (request.form.get('priority_level') or '').strip()
            modified_risk = request.form.get('modified_risk_score')
            additional_tests = (request.form.get('additional_tests') or '').strip()

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, priority_level, risk_score, doctor_notes FROM reports WHERE id = ?", (report_id,))
            existing = cursor.fetchone()

            if not existing:
                conn.close()
                flash('Report not found.')
                return redirect(url_for('doctor_dashboard'))

            current_priority = existing['priority_level'] or get_priority_level_from_risk(float(existing['risk_score'] or 0.0))
            new_priority = selected_priority if selected_priority else current_priority
            new_risk_score = float(existing['risk_score'] or 0.0)

            if modified_risk not in [None, '']:
                try:
                    new_risk_score = round(clamp(float(modified_risk), 0.0, 10.0), 2)
                    new_priority = get_priority_level_from_risk(new_risk_score)
                except ValueError:
                    pass

            merged_notes = doctor_notes
            if action == 'request_tests' and additional_tests:
                merged_notes = f"{doctor_notes} Additional tests requested: {additional_tests}".strip()

            if action == 'approve':
                patient_update_note = (
                    "Doctor finalized this case. "
                    + (merged_notes if merged_notes else "Final clinical review completed.")
                )
                cursor.execute("""
                    UPDATE reports
                    SET doctor_reviewed = 1,
                        doctor_notes = ?,
                        priority_level = ?,
                        risk_score = ?,
                        status = 'Closed',
                        notes = ?
                    WHERE id = ?
                """, (merged_notes, new_priority, new_risk_score, patient_update_note, report_id))
                flash('Case approved and finalized.')
            elif action == 'modify_severity':
                patient_update_note = (
                    f"Doctor updated severity to {new_priority} (risk {new_risk_score}/10). "
                    + (merged_notes if merged_notes else "No additional note provided.")
                )
                cursor.execute("""
                    UPDATE reports
                    SET priority_level = ?,
                        risk_score = ?,
                        status = ?,
                        doctor_notes = ?,
                        notes = ?
                    WHERE id = ?
                """, (new_priority, new_risk_score, map_priority_to_status(new_priority), merged_notes, patient_update_note, report_id))
                flash('Severity updated.')
            elif action == 'request_tests':
                tests_message = additional_tests if additional_tests else "Additional diagnostic workup"
                patient_update_note = (
                    f"Doctor requested more tests: {tests_message}. "
                    + (doctor_notes if doctor_notes else "Please upload the requested reports.")
                )
                cursor.execute("""
                    UPDATE reports
                    SET doctor_reviewed = 0,
                        doctor_notes = ?,
                        status = 'Pending Additional Tests',
                        notes = ?
                    WHERE id = ?
                """, (merged_notes or 'Additional tests requested.', patient_update_note, report_id))
                flash('Additional tests requested.')
            else:
                conn.close()
                flash('Unsupported doctor action.')
                return redirect(url_for('doctor_case', report_id=report_id))

            conn.commit()
            conn.close()
            return redirect(url_for('doctor_case', report_id=report_id))

        conn = get_db_connection()
        cursor = conn.cursor()

        if report_id:
            cursor.execute("""
                SELECT r.*, u.fullname, u.id as patient_id, u.dob
                FROM reports r
                JOIN users u ON r.user_id = u.id
                WHERE r.id = ?
            """, (report_id,))
        else:
            cursor.execute("""
                SELECT r.*, u.fullname, u.id as patient_id, u.dob
                FROM reports r
                JOIN users u ON r.user_id = u.id
                ORDER BY COALESCE(r.risk_score, 0) DESC, r.upload_date DESC
                LIMIT 1
            """)

        report = cursor.fetchone()
        conn.close()

        if not report:
            return redirect(url_for('doctor_dashboard'))

        case = dict(report)
        analysis_details = {}
        try:
            if case.get('analysis_details'):
                analysis_details = json.loads(case.get('analysis_details'))
        except Exception:
            analysis_details = {}
        raw_confidence = case.get('confidence')
        if raw_confidence is None:
            raw_confidence = case.get('confidence_score') or 0.0
        raw_confidence = float(raw_confidence)
        confidence_percent = raw_confidence * 100 if raw_confidence <= 1 else raw_confidence
        confidence_percent = round(clamp(confidence_percent, 0.0, 100.0), 2)

        risk_score = float(case.get('risk_score') or 0.0)
        priority_badge = get_priority_level_from_risk(risk_score)

        case_image_url = url_for('report_image', report_id=case['id'])
        case_file_ext = os.path.splitext(case.get('file_path') or '')[1].lower()
        case_is_pdf = case_file_ext == '.pdf'
        case_is_blood = 'blood' in (case.get('report_type') or '').lower()
        heatmap_url = url_for('static', filename=case['heatmap_path']) if case.get('heatmap_path') else ''
        overlay_url = url_for('static', filename=case['overlay_path']) if case.get('overlay_path') else ''

        heatmap_disclaimer = (
            "Grad-CAM heatmaps visualize where the model focused attention. "
            "They are supportive interpretability cues and not definitive evidence of pathology."
        )

        return render_template(
            'doctor-case.html',
            case_data=case,
            case_image_url=case_image_url,
            heatmap_url=heatmap_url,
            overlay_url=overlay_url,
            confidence_percent=confidence_percent,
            confidence_width=f"{confidence_percent:.0f}%",
            predicted_condition=case.get('predicted_class') or 'Unknown',
            ai_summary=case.get('summary') or 'No AI summary available.',
            risk_score=round(risk_score, 2),
            priority_badge=priority_badge,
            heatmap_disclaimer=heatmap_disclaimer,
            case_is_pdf=case_is_pdf,
            case_is_blood=case_is_blood,
            blood_markers=analysis_details.get('markers', []),
            blood_abnormal_findings=analysis_details.get('abnormal_findings', [])
        )
    return redirect(url_for('login'))

@app.route('/report-image/<int:report_id>')
def report_image(report_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, file_path FROM reports WHERE id = ?", (report_id,))
    report = cursor.fetchone()
    conn.close()

    if not report:
        abort(404)

    # Doctors can view all reports, patients only their own report files.
    if session.get('role') != 'doctor' and report['user_id'] != session.get('user_id'):
        abort(403)

    file_path = report['file_path']
    if not file_path or not os.path.exists(file_path):
        abort(404)

    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.pdf': 'application/pdf'
    }
    return send_file(file_path, mimetype=mime_map.get(ext, 'application/octet-stream'))

@app.route('/messages.html')
@app.route('/messages.html', methods=['GET', 'POST'])
def messages():
    if 'user_id' in session:
        user_id = session['user_id']
        role = session.get('role')

        conn = get_db_connection()
        cursor = conn.cursor()

        partner_id = request.values.get('partner_id', type=int)
        if not partner_id and role == 'doctor':
            partner_id = request.values.get('patient_id', type=int)
        if not partner_id and role == 'patient':
            partner_id = request.values.get('doctor_id', type=int)

        if not partner_id:
            if role == 'patient':
                cursor.execute("SELECT id FROM users WHERE role = 'doctor' ORDER BY id ASC LIMIT 1")
                row = cursor.fetchone()
                partner_id = row['id'] if row else None
            else:
                cursor.execute("""
                    SELECT DISTINCT u.id
                    FROM reports r
                    JOIN users u ON u.id = r.user_id
                    WHERE u.role = 'patient'
                    ORDER BY r.upload_date DESC, r.id DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                partner_id = row['id'] if row else None

        if request.method == 'POST':
            text = (request.form.get('message_text') or '').strip()
            if not partner_id:
                flash("No recipient available for this conversation.")
            elif not text:
                flash("Message cannot be empty.")
            else:
                cursor.execute("""
                    INSERT INTO messages (sender_id, receiver_id, content, sent_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, partner_id, text, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                return redirect(url_for('messages', partner_id=partner_id))

        partner_name = "No Recipient"
        partner_role = "N/A"
        if partner_id:
            cursor.execute("SELECT id, fullname, role FROM users WHERE id = ?", (partner_id,))
            partner = cursor.fetchone()
            if partner:
                partner_name = partner['fullname']
                partner_role = partner['role']
            else:
                partner_id = None

        chat_messages = []
        if partner_id:
            cursor.execute("""
                SELECT m.*, s.fullname AS sender_name
                FROM messages m
                JOIN users s ON s.id = m.sender_id
                WHERE (m.sender_id = ? AND m.receiver_id = ?)
                   OR (m.sender_id = ? AND m.receiver_id = ?)
                ORDER BY m.sent_at ASC, m.id ASC
            """, (user_id, partner_id, partner_id, user_id))
            chat_messages = cursor.fetchall()

        conn.close()
        back_endpoint = 'doctor_dashboard' if session.get('role') == 'doctor' else 'patient_dashboard'
        return render_template(
            'messages.html',
            back_endpoint=back_endpoint,
            partner_id=partner_id,
            partner_name=partner_name,
            partner_role=partner_role,
            chat_messages=chat_messages,
            current_user_id=user_id
        )
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
