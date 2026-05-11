"""Tests for the MRM framework."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.models.mrm import (
    ModelCard, OutOfTimeValidator, SensitivityAnalyser,
    CalibrationChecker, DriftMonitor, OODDetector,
    ExpertOverrideManager, PredictionRiskScorer,
    ModelAuditTrail, RegimeDetector, TailRiskQuantifier,
    ValidationReportGenerator,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def simple_model():
    """A minimal trained model for MRM tests."""
    from src.utils.synthetic_data import generate_synthetic_data
    from src.features.engineering import FeatureEngineer, get_feature_cols
    from src.models.ensemble import HousingEnsemble
    from sklearn.model_selection import train_test_split

    df = generate_synthetic_data("Shanghai", n=1000, seed=0)
    fe = FeatureEngineer()
    y  = df["unit_price"]
    df_e = fe.fit_transform(df, y)
    fc = get_feature_cols(df_e)
    X  = df_e[fc].fillna(-999)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    m = HousingEnsemble(seed=0, n_folds=3)
    m.fit(X_tr, y_tr, feature_cols=fc)
    return m, X_tr, X_te, y_tr, y_te, fc, df, df_e


# ── MODEL CARD ────────────────────────────────────────────────────────────────
def test_model_card_fields():
    mc = ModelCard()
    assert mc.model_id
    assert len(mc.assumptions) >= 5
    assert len(mc.limitations) >= 5
    assert len(mc.prohibited_uses) >= 3
    assert "max_mape_pct" in mc.performance_thresholds

def test_model_card_to_json():
    import json
    mc   = ModelCard(model_id="TEST-001")
    j    = mc.to_json()
    data = json.loads(j)
    assert data["model_id"] == "TEST-001"
    assert "intended_use" in data
    assert "prohibited_uses" in data


# ── OUT-OF-TIME VALIDATOR ─────────────────────────────────────────────────────
def test_oot_split(simple_model):
    _, _, _, _, _, _, df, _ = simple_model
    oot = OutOfTimeValidator(train_end_year=2020, test_start_year=2021)
    tr, te = oot.split(df)
    assert tr["year"].max() <= 2020
    assert te["year"].min() >= 2021
    assert len(tr) + len(te) == len(df)


# ── SENSITIVITY ───────────────────────────────────────────────────────────────
def test_sensitivity_shape(simple_model):
    m, _, X_te, _, _, fc, _, _ = simple_model
    sens  = SensitivityAnalyser(m, fc)
    df_s  = sens.analyse(X_te, top_n=10)
    assert len(df_s) <= 10
    assert "feature"  in df_s.columns
    assert "abs_range" in df_s.columns

def test_sensitivity_top_feature_meaningful(simple_model):
    m, _, X_te, _, _, fc, _, _ = simple_model
    sens = SensitivityAnalyser(m, fc)
    df_s = sens.analyse(X_te, top_n=20)
    top  = df_s.iloc[0]["feature"]
    # Most sensitive feature should be economically meaningful
    meaningful = {
        "comp_weighted_median", "dist_mean_price", "dist_median_price",
        "sub_mean_price", "proj_mean_price", "log_area", "sqrt_area",
        "district_te", "dist_relative_price", "accessibility_score",
        "comparable_median_price", "district_mean_price", "log_area_sq"
    }
    assert top in meaningful, f"Top feature '{top}' not in expected set"


# ── CALIBRATION ───────────────────────────────────────────────────────────────
def test_calibration_columns(simple_model):
    m, _, X_te, _, y_te, _, _, _ = simple_model
    cal = CalibrationChecker()
    unc = m.predict_with_uncertainty(X_te)
    df_cal = cal.check_coverage(y_te.values, unc)
    assert "Interval" in df_cal.columns
    assert "Kupiec p-value" in df_cal.columns
    assert len(df_cal) >= 1

def test_kupiec_extreme_cases():
    cal  = CalibrationChecker()
    # Perfect coverage → high p-value (don't reject H0)
    p1   = cal._kupiec_test(0.20, 0.20, 1000)
    assert p1 > 0.05
    # Zero coverage when 20% expected → very low p-value
    p2   = cal._kupiec_test(0.001, 0.20, 1000)
    assert p2 < 0.001


# ── DRIFT MONITOR ─────────────────────────────────────────────────────────────
def test_psi_stable(simple_model):
    _, _, _, _, _, fc, df, df_e = simple_model
    drift = DriftMonitor()
    drift.fit_reference(df_e, fc)
    # Monitor on same data → PSI should be near 0
    df_psi = drift.monitor(df_e)
    assert df_psi["PSI"].max() < 0.05

def test_psi_detects_shift(simple_model):
    _, _, _, _, _, fc, df, df_e = simple_model
    drift = DriftMonitor()
    drift.fit_reference(df_e, fc)
    # Create shifted data: shift log_area by +3 sigma (large distributional shift)
    df_shifted = df_e.copy()
    if "log_area" in df_shifted.columns:
        mu  = df_e["log_area"].mean()
        std = df_e["log_area"].std()
        df_shifted["log_area"] = mu + 3 * std  # all values at +3 sigma
    if "school_quality" in df_shifted.columns:
        df_shifted["school_quality"] = df_e["school_quality"].max() + 5.0
    df_psi = drift.monitor(df_shifted)
    # Should detect significant drift on shifted features
    assert df_psi["PSI"].max() > 0.10

def test_concept_drift_insufficient_data():
    drift  = DriftMonitor()
    result = drift.concept_drift_test([{"MAPE": 5.0}, {"MAPE": 6.0}])
    assert result["trend"] == "insufficient data"

def test_concept_drift_increasing_trend():
    drift  = DriftMonitor()
    history = [{"MAPE": 4+i*1.5} for i in range(8)]
    result = drift.concept_drift_test(history)
    assert result["trend"] == "increasing"


# ── OOD DETECTOR ─────────────────────────────────────────────────────────────
def test_ood_fit_predict(simple_model):
    _, X_tr, X_te, _, _, _, _, _ = simple_model
    ood = OODDetector(threshold_pct=97.5)
    ood.fit(X_tr.values)
    scores = ood.score(X_te.values)
    flags  = ood.flag(X_te.values)
    assert len(scores) == len(X_te)
    assert flags.dtype == bool
    # ~2.5% should be flagged
    flag_rate = flags.mean()
    assert 0.0 < flag_rate < 0.15

def test_ood_extreme_point(simple_model):
    _, X_tr, X_te, _, _, _, _, _ = simple_model
    ood = OODDetector(threshold_pct=97.5)
    ood.fit(X_tr.values)
    # Create a wildly out-of-distribution point
    extreme = np.ones((1, X_tr.shape[1])) * 9999
    assert ood.flag(extreme)[0] == True


# ── EXPERT OVERRIDE ───────────────────────────────────────────────────────────
def test_override_records():
    mgr = ExpertOverrideManager()
    ov  = mgr.apply_override("P001", 100000, 108000,
                              "renovation", "New kitchen", "ANA-001")
    assert ov.override_pct == pytest.approx(8.0, abs=0.1)
    assert not ov.requires_approval

def test_override_requires_approval():
    mgr = ExpertOverrideManager()
    ov  = mgr.apply_override("P002", 100000, 65000,
                              "distressed_sale", "Forced sale", "ANA-002")
    assert ov.requires_approval  # -35% > 20% threshold

def test_override_invalid_reason():
    mgr = ExpertOverrideManager()
    with pytest.raises(ValueError):
        mgr.apply_override("P003", 100000, 90000,
                           "INVALID_REASON", "test", "ANA-003")

def test_override_analytics():
    mgr = ExpertOverrideManager()
    mgr.apply_override("P001", 100000, 95000, "local_knowledge", "d1", "A1")
    mgr.apply_override("P002", 100000, 92000, "policy_change",   "d2", "A2")
    stats = mgr.override_analytics()
    assert stats["n_overrides"] == 2
    assert stats["mean_override_pct"] < 0


# ── PREDICTION RISK SCORER ────────────────────────────────────────────────────
def test_risk_scorer_low(simple_model):
    _, X_tr, X_te, _, _, _, _, _ = simple_model
    ood = OODDetector(); ood.fit(X_tr.values)
    scorer = PredictionRiskScorer(model_mape=4.0)
    result = scorer.score(
        ood_score=0.5, ood_threshold=ood._threshold,
        q10=90000, q90=110000, point_pred=100000,
        data_age_days=10,
    )
    assert result["risk_score"] < 50
    assert "LOW" in result["risk_rating"] or "MEDIUM" in result["risk_rating"]

def test_risk_scorer_high():
    scorer = PredictionRiskScorer(model_mape=12.0, max_acceptable_mape=8.0)
    result = scorer.score(
        ood_score=50.0, ood_threshold=5.0,
        q10=30000, q90=170000, point_pred=100000,
        data_age_days=120,
    )
    assert result["risk_score"] > 60


# ── AUDIT TRAIL ───────────────────────────────────────────────────────────────
def test_audit_trail_logs():
    trail = ModelAuditTrail()
    pid   = trail.log_prediction(
        property_inputs  = {"area": 90, "district": "test"},
        point_prediction = 100000.0,
        quantile_preds   = {"q10": 85000.0, "q90": 115000.0},
        risk_score       = {"risk_score": 25, "risk_rating": "🟢 LOW RISK"},
    )
    assert len(pid) > 0
    df = trail.to_dataframe()
    assert len(df) == 1
    assert df["point_prediction"].iloc[0] == 100000.0

def test_audit_trail_integrity():
    trail = ModelAuditTrail()
    for i in range(5):
        trail.log_prediction({"i": i}, float(i*1000),
                              {"q10": i*900.0, "q90": i*1100.0}, {})
    assert trail.verify_integrity()


# ── REGIME DETECTOR ───────────────────────────────────────────────────────────
def test_regime_bubble():
    rd = RegimeDetector()
    r  = rd.detect(yoy_price_growth=25, yoy_volume_change=30,
                   price_to_income=38, sentiment_score=0.85)
    assert "BUBBLE" in r["regime"]

def test_regime_crash():
    rd = RegimeDetector()
    r  = rd.detect(yoy_price_growth=-20, yoy_volume_change=-50,
                   price_to_income=25, sentiment_score=-0.8)
    assert "CRASH" in r["regime"]

def test_regime_normal():
    rd = RegimeDetector()
    r  = rd.detect(yoy_price_growth=4.0, yoy_volume_change=5.0,
                   price_to_income=20, sentiment_score=0.1)
    assert "NORMAL" in r["regime"]

def test_regime_stagnation():
    rd = RegimeDetector()
    r  = rd.detect(yoy_price_growth=0.3, yoy_volume_change=-5.0,
                   price_to_income=18, sentiment_score=-0.1)
    assert "STAGNATION" in r["regime"]


# ── TAIL RISK ─────────────────────────────────────────────────────────────────
def test_tail_risk_shape():
    tq = TailRiskQuantifier()
    fc = pd.DataFrame({
        "year"  : range(2024, 2035),
        "p5"    : np.linspace(90000, 70000, 11),
        "p10"   : np.linspace(95000, 75000, 11),
        "p25"   : np.linspace(100000, 90000, 11),
        "median": np.linspace(110000, 120000, 11),
        "p75"   : np.linspace(120000, 150000, 11),
        "p90"   : np.linspace(130000, 170000, 11),
        "p95"   : np.linspace(140000, 190000, 11),
    })
    df_var = tq.compute(fc, current_value=110000,
                        horizons_yr=[1, 3, 5, 10])
    assert len(df_var) == 4
    assert "VaR Price" in df_var.columns

def test_tail_risk_var_less_than_median():
    tq = TailRiskQuantifier()
    fc = pd.DataFrame({
        "year"  : range(2024, 2030),
        "p5"    : [80000]*6,
        "p10"   : [85000]*6,
        "p25"   : [92000]*6,
        "median": [100000]*6,
        "p75"   : [108000]*6,
        "p90"   : [115000]*6,
        "p95"   : [120000]*6,
    })
    df_var = tq.compute(fc, current_value=100000, horizons_yr=[1])
    assert df_var.iloc[0]["VaR Price"] < df_var.iloc[0]["Median Price"]


# ── VALIDATION REPORT ─────────────────────────────────────────────────────────
def test_validation_report_generates():
    mc  = ModelCard(model_id="TEST-RPT")
    rpt = ValidationReportGenerator(mc)
    rpt.add_section("Test Section", {"metric": "value"}, "✅")
    report = rpt.generate()
    assert "TEST-RPT" in report
    assert "INTENDED USE" in report
    assert "LIMITATIONS" in report
    assert "TEST SECTION" in report
    assert "SIGN-OFF" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
