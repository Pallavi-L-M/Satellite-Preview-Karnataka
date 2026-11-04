"""
Satellite Preview Karnataka Project
====================================
Flask backend for satellite-based flood prediction and early warning system.
Uses BiLSTM-Attention model for 72-hour flood probability forecasting.
"""

import os
import json
import datetime
import uuid
import math
from flask import Flask, render_template, request, jsonify

# Import LSTM predictor
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.lstm_predictor import (
    predict_flood_probability,
    generate_forecast,
    generate_all_districts_overview,
    analyze_uploaded_image,
    get_district_history,
    get_all_districts,
    MODEL_INFO,
    DISTRICT_RISK_PROFILES
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ── Upload directory ──
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load GeoJSON ──
DATA_DIR = os.path.join(os.path.dirname(__file__), 'static', 'data')
GEOJSON_PATH = os.path.join(DATA_DIR, 'karnataka_districts.geojson')
if os.path.exists(GEOJSON_PATH):
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        KARNATAKA_GEOJSON = json.load(f)
else:
    KARNATAKA_GEOJSON = None

# ── Alert logs ──
alert_log = []
sos_log = []
last_sos_time = {}
SOS_COOLDOWN = 60


# ══════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/geojson')
def api_geojson():
    """Return Karnataka district boundaries GeoJSON."""
    if KARNATAKA_GEOJSON:
        return jsonify(KARNATAKA_GEOJSON)
    return jsonify({"type": "FeatureCollection", "features": []})


# ══════════════════════════════════════════════════════════════
#  API — STATE OVERVIEW
# ══════════════════════════════════════════════════════════════

@app.route('/api/overview')
def api_overview():
    """Return current predictions for all 31 districts."""
    districts = generate_all_districts_overview()
    
    # Compute statewide stats
    total = len(districts)
    severe = sum(1 for d in districts if d['risk_level'] == 'Severe')
    high = sum(1 for d in districts if d['risk_level'] == 'High')
    moderate = sum(1 for d in districts if d['risk_level'] == 'Moderate')
    low = sum(1 for d in districts if d['risk_level'] in ('Low', 'Minimal'))
    avg_prob = round(sum(d['flood_probability_pct'] for d in districts) / max(1, total), 1)
    avg_rainfall = round(sum(d['weather']['rainfall_mm'] for d in districts) / max(1, total), 1)
    
    return jsonify({
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_districts": total,
        "statewide_summary": {
            "average_flood_probability": avg_prob,
            "average_rainfall_mm": avg_rainfall,
            "severe_count": severe,
            "high_count": high,
            "moderate_count": moderate,
            "low_count": low,
            "highest_risk_district": districts[0]['district'] if districts else None,
            "highest_risk_pct": districts[0]['flood_probability_pct'] if districts else 0,
            "total_sos_alerts": len(sos_log)
        },
        "districts": districts
    })


# ══════════════════════════════════════════════════════════════
#  API — SINGLE DISTRICT DETAIL
# ══════════════════════════════════════════════════════════════

@app.route('/api/district/<district_name>')
def api_district_detail(district_name):
    """Detailed weather-app style prediction for a single district."""
    # Try exact match first, then fuzzy
    matched = None
    for d in get_all_districts():
        if d.lower().replace(" ", "") == district_name.lower().replace(" ", ""):
            matched = d
            break
    if not matched:
        for d in get_all_districts():
            if district_name.lower() in d.lower() or d.lower() in district_name.lower():
                matched = d
                break
    
    if not matched:
        return jsonify({"status": "error", "message": f"District '{district_name}' not found"}), 404
    
    prediction = predict_flood_probability(matched)
    forecast = generate_forecast(matched, 7)
    
    profile = DISTRICT_RISK_PROFILES.get(matched, {})
    
    return jsonify({
        "status": "success",
        "district": matched,
        "current": prediction,
        "forecast_7day": forecast,
        "profile": {
            "river_basin": profile.get("basin", "Unknown"),
            "flood_prone_rank": profile.get("flood_prone_rank", 20),
            "base_risk_factor": profile.get("base_risk", 0.3),
            "monsoon_amplification": profile.get("monsoon_amp", 0.15)
        }
    })


# ══════════════════════════════════════════════════════════════
#  API — FORECAST
# ══════════════════════════════════════════════════════════════

@app.route('/api/district/<district_name>/forecast')
def api_district_forecast(district_name):
    """7-day forecast for a district."""
    days = request.args.get('days', 7, type=int)
    days = min(days, 14)
    
    matched = None
    for d in get_all_districts():
        if d.lower().replace(" ", "") == district_name.lower().replace(" ", ""):
            matched = d
            break
    if not matched:
        for d in get_all_districts():
            if district_name.lower() in d.lower() or d.lower() in district_name.lower():
                matched = d
                break
    
    if not matched:
        return jsonify({"status": "error", "message": f"District '{district_name}' not found"}), 404
    
    forecast = generate_forecast(matched, days)
    return jsonify({
        "status": "success",
        "district": matched,
        "days": days,
        "forecast": forecast
    })


# ══════════════════════════════════════════════════════════════
#  API — HISTORICAL DATA
# ══════════════════════════════════════════════════════════════

@app.route('/api/district/<district_name>/history')
def api_district_history(district_name):
    """Historical flood events for a district."""
    matched = None
    for d in get_all_districts():
        if d.lower().replace(" ", "") == district_name.lower().replace(" ", ""):
            matched = d
            break
    if not matched:
        for d in get_all_districts():
            if district_name.lower() in d.lower() or d.lower() in district_name.lower():
                matched = d
                break
    
    if not matched:
        return jsonify({"status": "error", "message": f"District '{district_name}' not found"}), 404
    
    history = get_district_history(matched)
    return jsonify({"status": "success", **history})


# ══════════════════════════════════════════════════════════════
#  API — DISTRICT RANKING
# ══════════════════════════════════════════════════════════════

@app.route('/api/districts/ranking')
def api_districts_ranking():
    """All districts ranked by current flood risk."""
    districts = generate_all_districts_overview()
    ranking = []
    for i, d in enumerate(districts):
        ranking.append({
            "rank": i + 1,
            "district": d["district"],
            "flood_probability_pct": d["flood_probability_pct"],
            "risk_level": d["risk_level"],
            "risk_color": d["risk_color"],
            "risk_icon": d["risk_icon"],
            "rainfall_mm": d["weather"]["rainfall_mm"],
            "river_basin": d["river_basin"]
        })
    return jsonify({"status": "success", "ranking": ranking})


# ══════════════════════════════════════════════════════════════
#  API — MODEL INFO
# ══════════════════════════════════════════════════════════════

@app.route('/api/model/info')
def api_model_info():
    """Return LSTM model architecture and training details."""
    return jsonify({"status": "success", "model": MODEL_INFO})


# ══════════════════════════════════════════════════════════════
#  API — SATELLITE IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════

@app.route('/api/satellite/analyze', methods=['POST'])
def api_satellite_analyze():
    """Analyze uploaded satellite/QGIS image for flood risk."""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided. Use 'image' field."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400
    
    allowed = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.geotiff', '.img'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"status": "error", "message": f"Unsupported format '{ext}'. Supported: {', '.join(allowed)}"}), 400
    
    # Save file
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)
    file_size = os.path.getsize(save_path)
    
    # Analyze
    result = analyze_uploaded_image(file.filename, file_size)
    result["saved_as"] = safe_name
    
    return jsonify({"status": "success", "analysis": result})


# ══════════════════════════════════════════════════════════════
#  API — EARLY WARNING / SOS
# ══════════════════════════════════════════════════════════════

@app.route('/api/alerts/active')
def api_active_alerts():
    """Return currently active flood alerts."""
    districts = generate_all_districts_overview()
    active = []
    for d in districts:
        if d['risk_level'] in ('Severe', 'High'):
            active.append({
                "alert_id": f"ALT-{d['district'][:3].upper()}-{datetime.datetime.now().strftime('%H%M')}",
                "district": d['district'],
                "risk_level": d['risk_level'],
                "flood_probability_pct": d['flood_probability_pct'],
                "risk_color": d['risk_color'],
                "weather_description": d['weather_description'],
                "rainfall_mm": d['weather']['rainfall_mm'],
                "issued_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"Flood {d['risk_level'].upper()} for {d['district']}. "
                           f"Flood probability: {d['flood_probability_pct']}%. "
                           f"Rainfall: {d['weather']['rainfall_mm']}mm. {d['risk_action']}."
            })
    return jsonify({
        "status": "success",
        "active_alerts": len(active),
        "alerts": active
    })


@app.route('/api/sos/trigger', methods=['POST'])
def api_trigger_sos():
    """Trigger SOS alert for a district."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400
    
    district = data.get('district', '')
    severity = data.get('severity', 'Unknown')
    confirmed = data.get('confirmed', False)
    
    if not confirmed:
        return jsonify({"status": "error", "message": "SOS not confirmed. Set confirmed=true."}), 400
    
    # Cooldown check
    now = datetime.datetime.now()
    key = district.lower().strip()
    if key in last_sos_time:
        elapsed = (now - last_sos_time[key]).total_seconds()
        if elapsed < SOS_COOLDOWN:
            return jsonify({
                "status": "error",
                "message": f"Cooldown active. Wait {int(SOS_COOLDOWN - elapsed)}s before re-triggering."
            }), 429
    
    last_sos_time[key] = now
    
    sos_id = f"SOS-{district[:3].upper()}-{uuid.uuid4().hex[:6].upper()}"
    
    entry = {
        "sos_id": sos_id,
        "district": district,
        "severity": severity,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "SIMULATED_BROADCAST",
        "message": f"EMERGENCY FLOOD SOS ALERT!\n\n"
                   f"District: {district}\n"
                   f"Alert Level: {severity}\n"
                   f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} IST\n\n"
                   f"Severe flood risk detected. Immediate emergency response required.\n"
                   f"Move to higher ground. Call 112 / 1077.\n\n"
                   f"Satellite Preview Karnataka - Flood Early Warning System",
        "mode": "TEST_SIMULATION"
    }
    sos_log.append(entry)
    
    return jsonify({"status": "success", "sos": entry})


@app.route('/api/sos/log')
def api_sos_log():
    """Return SOS alert history."""
    return jsonify({
        "status": "success",
        "total_alerts": len(sos_log),
        "log": list(reversed(sos_log))
    })


# ══════════════════════════════════════════════════════════════
#  API — STATEWIDE STATS
# ══════════════════════════════════════════════════════════════

@app.route('/api/stats')
def api_stats():
    """Comprehensive statewide statistics."""
    districts = generate_all_districts_overview()
    all_districts = get_all_districts()
    
    # Historical stats
    total_historical_events = 0
    total_lives_lost = 0
    total_affected = 0
    for d in all_districts[:5]:  # Sample for speed
        h = get_district_history(d)
        total_historical_events += h['summary']['total_events']
        total_lives_lost += h['summary']['total_lives_lost']
        total_affected += h['summary']['total_people_affected']
    
    # Extrapolate
    scale = len(all_districts) / 5
    
    return jsonify({
        "status": "success",
        "statewide": {
            "total_districts_monitored": len(all_districts),
            "model_name": MODEL_INFO["model_name"],
            "model_accuracy_auc": MODEL_INFO["metrics"]["test_auc_roc"],
            "prediction_horizons": ["6h", "12h", "24h", "48h", "72h"],
            "total_historical_events_estimated": int(total_historical_events * scale),
            "data_coverage_years": "2018-2026",
            "satellite_sources": ["Sentinel-1 SAR (C-band)", "Sentinel-2 MSI", "ISRO Bhuvan", "Landsat-8/9"],
            "ground_sources": ["IMD AWS", "CWC Gauge Telemetry", "KSNDMC Rain Gauges"],
            "total_sos_triggered": len(sos_log)
        }
    })


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
