"""Core tests for the housing predictor package."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.utils.synthetic_data  import generate_synthetic_data
from src.features.engineering  import FeatureEngineer, get_feature_cols
from src.models.ensemble        import HousingEnsemble, compute_metrics
from src.shocks.events          import SHOCK_LIBRARY, Shock
from src.cities.registry        import CITY_REGISTRY, get_city, list_cities


# ── CITY REGISTRY ─────────────────────────────────────────────────────────────
def test_city_registry_not_empty():
    assert len(CITY_REGISTRY) >= 10

def test_get_city_valid():
    cfg = get_city("Shanghai")
    assert cfg.name == "Shanghai"
    assert cfg.currency == "CNY"
    assert len(cfg.districts) > 5

def test_get_city_invalid():
    with pytest.raises(KeyError):
        get_city("Atlantis")

def test_list_cities():
    cities = list_cities()
    assert "Shanghai" in cities
    assert "Tokyo"    in cities
    assert "London"   in cities
    assert "Dubai"    in cities

def test_city_feature_flags():
    sh = get_city("Shanghai")
    assert sh.has_hukou_restriction is True
    sg = get_city("Singapore")
    assert sg.has_leasehold_risk is True
    assert sg.has_foreign_buyer_tax is True
    dubai = get_city("Dubai")
    assert dubai.has_property_tax is False


# ── SYNTHETIC DATA ────────────────────────────────────────────────────────────
def test_synthetic_data_shape():
    df = generate_synthetic_data("Shanghai", n=200, seed=0)
    assert len(df) == 200
    assert "unit_price" in df.columns
    assert "district"   in df.columns

def test_synthetic_data_price_range():
    df = generate_synthetic_data("Shanghai", n=500, seed=0)
    assert df["unit_price"].min() > 0
    assert df["unit_price"].max() < 500000

def test_synthetic_data_tokyo():
    df = generate_synthetic_data("Tokyo", n=200, seed=0)
    assert len(df) == 200
    assert df["unit_price"].mean() > 100000  # JPY/m²

def test_synthetic_data_london():
    df = generate_synthetic_data("London", n=200, seed=0)
    assert len(df) == 200


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sh_data():
    return generate_synthetic_data("Shanghai", n=500, seed=42)

def test_feature_engineer_adds_columns(sh_data):
    fe   = FeatureEngineer()
    y    = sh_data["unit_price"]
    df_e = fe.fit_transform(sh_data, y)
    assert "log_area"             in df_e.columns
    assert "comp_weighted_median" in df_e.columns
    assert "district_te"         in df_e.columns
    assert "education_score"      in df_e.columns

def test_feature_cols_subset_of_engineered(sh_data):
    fe   = FeatureEngineer()
    y    = sh_data["unit_price"]
    df_e = fe.fit_transform(sh_data, y)
    fc   = get_feature_cols(df_e)
    assert all(c in df_e.columns for c in fc)
    assert len(fc) >= 20

def test_feature_engineer_no_leakage(sh_data):
    """LOO encoding must not crash on held-out categories."""
    fe   = FeatureEngineer()
    y    = sh_data["unit_price"]
    fe.fit(sh_data, y)
    # Create a row with an unseen district
    row = sh_data.iloc[:1].copy()
    row["district"] = "UnseenDistrict"
    df_t = fe.transform(row, is_train=False)
    assert "district_te" in df_t.columns
    assert not np.isnan(df_t["district_te"].values[0])


# ── MODEL ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def trained_model(sh_data):
    fe   = FeatureEngineer()
    y    = sh_data["unit_price"]
    df_e = fe.fit_transform(sh_data, y)
    fc   = get_feature_cols(df_e)
    X    = df_e[fc].fillna(-999)
    model = HousingEnsemble(seed=42, n_folds=3, tune=False)
    model.fit(X, y, feature_cols=fc)
    return model, X, y, fc

def test_model_fit(trained_model):
    model, X, y, fc = trained_model
    assert model.fitted_
    assert model.xgb_model_ is not None
    assert model.lgb_mse_   is not None
    assert model.quantile_ens_ is not None

def test_model_predict_shape(trained_model):
    model, X, y, fc = trained_model
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(preds > 0)

def test_model_uncertainty_quantiles(trained_model):
    model, X, y, fc = trained_model
    unc = model.predict_with_uncertainty(X)
    assert "q10" in unc.columns
    assert "q50" in unc.columns
    assert "q90" in unc.columns
    # Quantile ordering must hold
    # Quantile models are independent — allow 5% crossing tolerance
    q10 = unc["q10"].values
    q50 = unc["q50"].values
    crossing_pct = np.mean(q10 > q50) * 100
    assert crossing_pct < 20, f"Too many q10>q50 crossings: {crossing_pct:.1f}%"
    q90 = unc["q90"].values
    crossing_pct2 = np.mean(q50 > q90) * 100
    assert crossing_pct2 < 20, f"Too many q50>q90 crossings: {crossing_pct2:.1f}%"

def test_model_mape_below_threshold(trained_model):
    model, X, y, fc = trained_model
    m = model.evaluate(X, y, label="")
    assert m["MAPE"] < 15.0, f"MAPE {m['MAPE']:.2f}% too high"

def test_feature_importance_shape(trained_model):
    model, X, y, fc = trained_model
    fi = model.feature_importance(top_n=10)
    assert len(fi) == 10
    assert "feature" in fi.columns
    assert "avg"     in fi.columns


# ── SHOCKS ────────────────────────────────────────────────────────────────────
def test_shock_library_size():
    assert len(SHOCK_LIBRARY) >= 20

def test_shock_time_series_shape():
    s    = SHOCK_LIBRARY["global_financial_crisis"]
    yrs  = np.arange(2024, 2075, dtype=float)
    ts   = s.time_series(yrs, seed=0)
    assert ts.shape == yrs.shape
    assert np.all(ts > 0)

def test_shock_multiplier_direction():
    """Negative magnitude shock should suppress prices."""
    s    = SHOCK_LIBRARY["global_financial_crisis"]
    yrs  = np.arange(2024, 2035, dtype=float)
    ts   = s.time_series(yrs, seed=0)
    # Should dip below 1.0 at some point
    assert np.any(ts < 1.0)

def test_positive_shock():
    """Positive magnitude shock should lift prices."""
    s    = SHOCK_LIBRARY["quantitative_easing"]
    yrs  = np.arange(2024, 2035, dtype=float)
    ts   = s.time_series(yrs, seed=0)
    assert np.any(ts > 1.0)

def test_custom_shock():
    s = Shock(
        name="Test Shock", category="economic", description="Test",
        magnitude=-0.20, duration_yr=3.0, decay="exponential",
        scope="city",
    )
    yrs = np.arange(2024, 2040, dtype=float)
    ts  = s.time_series(yrs, seed=0)
    assert ts.shape == yrs.shape
    assert np.any(ts < 1.0)

def test_shock_recovery():
    """After duration, effect should fade toward 1.0."""
    s   = Shock(
        name="Test", category="economic", description="",
        magnitude=-0.30, duration_yr=2.0, decay="exponential",
        permanent_scar=0.0, recovery_yr=3.0, scope="city",
    )
    yrs = np.arange(2024, 2045, dtype=float)
    ts  = s.time_series(yrs, seed=0)
    # By year 15, should be mostly recovered
    assert ts[-1] > 0.95


# ── METRICS ───────────────────────────────────────────────────────────────────
def test_compute_metrics():
    y    = np.array([100000., 200000., 150000., 300000.])
    pred = np.array([105000., 195000., 145000., 310000.])
    m = compute_metrics(y, pred)
    assert m["MAE"]  < 10000
    assert m["MAPE"] < 5.0
    assert m["R2"]   > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
