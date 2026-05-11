"""
Model Risk Management (MRM) Framework
======================================
Aligned with:
  - Federal Reserve SR 11-7  (Guidance on Model Risk Management)
  - PRA SS1/23               (UK Supervisory Statement on Model Risk)
  - MAS MRM Guidelines       (Singapore MAS Notice on Model Risk)
  - Basel III FRTB            (Internal model backtesting requirements)

This module provides:
  1.  ModelCard           — purpose, assumptions, limitations, governance
  2.  OutOfTimeValidator  — backtesting on held-out time periods
  3.  SensitivityAnalyser — one-at-a-time & tornado sensitivity
  4.  ShapExplainer       — SHAP-based explainability (per-prediction + global)
  5.  CalibrationChecker  — quantile coverage, Kupiec test, DQ test
  6.  DriftMonitor        — PSI feature drift + concept drift (performance decay)
  7.  OODDetector         — out-of-distribution / anomaly detection per prediction
  8.  ExpertOverride      — user adjustment with audit trail
  9.  RiskScorer          — composite prediction-level risk score
  10. ModelAuditTrail     — immutable log of all predictions
  11. RegimeDetector      — bubble / crash / normal regime classification
  12. TailRiskQuantifier  — VaR / CVaR / Expected Shortfall on forecast paths
  13. ModelInventory      — governance metadata & version control
  14. ValidationReport    — full MRM report generator
"""

import json
import uuid
import warnings
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL CARD  (SR 11-7 §4: Documentation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCard:
    """
    Structured model documentation — analogous to an FDA drug label for ML models.
    Must be completed and signed off before any production use.
    """
    # Identity
    model_id:       str  = "HOUSING-PRED-001"
    model_name:     str  = "Global Housing Price Predictor"
    version:        str  = "2.0.0"
    created_date:   str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_validated: str  = ""
    next_review:    str  = ""
    owner:          str  = "Quant Research"
    validator:      str  = "Independent Model Validation (IMV)"
    approver:       str  = "Chief Risk Officer"

    # Intended use  ← SR 11-7 explicitly requires this
    intended_use: str = (
        "Indicative market-value estimation for residential property in major cities. "
        "Intended for: (1) portfolio valuation screening, (2) investment research, "
        "(3) acquisition due-diligence support. "
        "NOT intended for: regulatory capital calculations, financial statements, "
        "collateral valuation for lending, or any binding financial commitment."
    )

    prohibited_uses: List[str] = field(default_factory=lambda: [
        "Regulatory capital / RWA calculations",
        "IFRS 13 / ASC 820 fair value for financial reporting",
        "Mortgage collateral valuation (appraisal)",
        "REIT NAV calculations for investor prospectus",
        "Automated lending decisions without human review",
        "High-frequency trading signals",
    ])

    # Model type & methodology
    model_type:  str = "Gradient-boosted tree ensemble with stacked meta-learner"
    target:      str = "Residential property unit price (local currency / area unit)"
    geography:   str = "20 major global cities (see city registry)"
    time_horizon:str = "Spot valuation + 1–50 year probabilistic forecast"

    # Key assumptions  ← must be explicit
    assumptions: List[str] = field(default_factory=lambda: [
        "Training data is representative of the current market segment",
        "Historical price-feature relationships persist into the forecast horizon",
        "Macro scenario paths are log-normally distributed around trend",
        "Property rights and legal title remain stable (no expropriation modelled)",
        "Currency convertibility remains unrestricted",
        "Comparable-sales KNN median is available at inference time",
        "Features are measured without material error or bias",
        "Market liquidity is sufficient for price discovery",
        "No structural breaks in the training window (2015–2024)",
    ])

    # Known limitations  ← SR 11-7 §4 requires explicit limitation disclosure
    limitations: List[str] = field(default_factory=lambda: [
        "Synthetic training data: model has not been validated on live transaction data",
        "Long-horizon forecasts (>10yr) are scenario analysis, NOT price predictions",
        "Policy shocks (e.g. nationalisation, war) are partially modelled — tail risk understated",
        "Climate physical risk (sea level, wildfire, chronic heat) not fully quantified",
        "Illiquid / unique properties (historical buildings, ultra-luxury) may be out-of-distribution",
        "Model trained on aggregated city data — micro-location variation within district is smoothed",
        "Macro scenario correlations are simplified — cross-asset contagion not modelled",
        "Model does not account for negotiation discounts or distressed sale conditions",
        "Leasehold residual value decay is not explicitly modelled for short remaining terms",
        "Foreign ownership restrictions and their future changes are not dynamically modelled",
    ])

    # Performance thresholds  ← define acceptable vs unacceptable model performance
    performance_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "max_mape_pct":         8.0,    # >8% MAPE → model requires revalidation
        "max_mae_pct_of_median":10.0,   # MAE > 10% of median price → revalidation
        "min_r2":               0.92,   # R² < 0.92 → revalidation
        "min_quantile_coverage_80pct": 72.0,  # P10-P90 coverage < 72% → recalibrate
        "max_psi_feature":      0.25,   # PSI > 0.25 on any feature → drift alert
        "max_psi_target":       0.20,   # PSI > 0.20 on target distribution → alert
        "max_prediction_age_days": 90,  # Prediction stale after 90 days
        "revalidation_frequency": "Annual or after any market regime change",
    })

    # Data sources
    data_sources: List[str] = field(default_factory=lambda: [
        "Beike (贝壳) second-hand residential listings — China",
        "Anjuke (安居客) — China",
        "Rightmove / Zoopla — UK",
        "Zillow / StreetEasy — USA",
        "PropertyGuru / SRX — Singapore",
        "Suumo / Homes.co.jp — Japan",
        "NBS macro data — China",
        "FRED / central bank APIs — global macro",
        "Synthetic calibrated data (fallback)",
    ])

    # Materiality
    materiality: str = (
        "HIGH — predictions may influence investment decisions involving "
        "material capital. Independent human review required before any "
        "decision exceeding USD 500,000 equivalent."
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def print_summary(self):
        print("=" * 70)
        print(f"  MODEL CARD: {self.model_name} v{self.version}")
        print("=" * 70)
        print(f"  ID         : {self.model_id}")
        print(f"  Owner      : {self.owner}")
        print(f"  Validator  : {self.validator}")
        print(f"  Created    : {self.created_date[:10]}")
        print(f"  Next review: {self.next_review or 'NOT SET ⚠️'}")
        print(f"\n  INTENDED USE:\n    {self.intended_use[:120]}...")
        print(f"\n  MATERIALITY: {self.materiality[:80]}...")
        print(f"\n  ASSUMPTIONS ({len(self.assumptions)}):")
        for a in self.assumptions[:4]:
            print(f"    • {a}")
        print(f"\n  LIMITATIONS ({len(self.limitations)}):")
        for l in self.limitations[:4]:
            print(f"    ⚠ {l}")
        print(f"\n  PROHIBITED USES ({len(self.prohibited_uses)}):")
        for p in self.prohibited_uses:
            print(f"    ✗ {p}")
        print("=" * 70)

    def to_json(self, path: Optional[str] = None) -> str:
        j = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(j)
        return j


# ─────────────────────────────────────────────────────────────────────────────
# 2. OUT-OF-TIME VALIDATOR  (SR 11-7: Independent validation)
# ─────────────────────────────────────────────────────────────────────────────

class OutOfTimeValidator:
    """
    Backtesting on temporally separated data.
    Train on years T0..T1, test on T2..T3 — prevents look-ahead bias.
    Also runs stress tests on known dislocation periods.
    """

    def __init__(self, train_end_year: int = 2020, test_start_year: int = 2021):
        self.train_end   = train_end_year
        self.test_start  = test_start_year
        self.results_:   Dict = {}

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal train/test split — no data from future in training."""
        df_tr = df[df["year"] <= self.train_end].copy()
        df_te = df[df["year"] >= self.test_start].copy()
        print(f"  Out-of-time split: train {df_tr['year'].min()}–{self.train_end} "
              f"({len(df_tr):,} rows) | "
              f"test {self.test_start}–{df_te['year'].max()} ({len(df_te):,} rows)")
        return df_tr, df_te

    def validate(self, model, fe, df_tr: pd.DataFrame,
                 df_te: pd.DataFrame, feat_cols: List[str]) -> Dict:
        """Full out-of-time validation pipeline."""
        from .ensemble import compute_metrics

        # Refit feature engineer on train only (no leakage)
        y_tr = df_tr["unit_price"]
        df_tr_e = fe.fit_transform(df_tr, y_tr)
        df_te_e = fe.transform(df_te, is_train=False)

        fc = [c for c in feat_cols if c in df_tr_e.columns and c in df_te_e.columns]
        X_te = df_te_e[fc].fillna(-999)
        y_te = df_te["unit_price"]

        preds = model.predict(X_te)
        metrics = compute_metrics(y_te.values, preds, label="Out-of-Time Test")

        # Year-by-year breakdown
        year_metrics = []
        for yr in sorted(df_te["year"].unique()):
            mask = df_te["year"] == yr
            if mask.sum() < 10:
                continue
            ym = compute_metrics(y_te[mask].values, preds[mask])
            ym["year"] = yr
            ym["n"] = int(mask.sum())
            year_metrics.append(ym)

        self.results_ = {
            "overall": metrics,
            "by_year": pd.DataFrame(year_metrics),
            "train_years": f"{df_tr['year'].min()}–{self.train_end}",
            "test_years" : f"{self.test_start}–{df_te['year'].max()}",
        }

        # Flag if performance below threshold
        mc = ModelCard()
        thresh = mc.performance_thresholds
        flags = []
        if metrics["MAPE"] > thresh["max_mape_pct"]:
            flags.append(f"⚠️  MAPE {metrics['MAPE']:.2f}% exceeds threshold "
                         f"{thresh['max_mape_pct']}%")
        if metrics["R2"] < thresh["min_r2"]:
            flags.append(f"⚠️  R² {metrics['R2']:.4f} below threshold {thresh['min_r2']}")
        if flags:
            print("\n  VALIDATION FLAGS:")
            for f_ in flags:
                print(f"    {f_}")
        else:
            print("  ✅ All performance thresholds met")

        return self.results_

    def stress_test(self, model, fe, df: pd.DataFrame,
                    feat_cols: List[str],
                    stress_periods: Optional[Dict[str, Tuple]] = None) -> pd.DataFrame:
        """
        Test model on known stress periods (COVID, rate hike cycles, etc.)
        stress_periods: {label: (start_year, end_year)}
        """
        from .ensemble import compute_metrics

        if stress_periods is None:
            stress_periods = {
                "COVID (2020)"    : (2020, 2020),
                "Rate hike (2022)": (2022, 2022),
                "Post-peak (2023)": (2023, 2023),
            }

        rows = []
        for label, (yr_s, yr_e) in stress_periods.items():
            subset = df[(df["year"] >= yr_s) & (df["year"] <= yr_e)]
            if len(subset) < 10:
                rows.append({"Period": label, "n": 0,
                             "MAE": np.nan, "MAPE": np.nan, "R2": np.nan,
                             "Status": "Insufficient data"})
                continue
            y_s = subset["unit_price"]
            df_s_e = fe.transform(subset)
            fc = [c for c in feat_cols if c in df_s_e.columns]
            preds = model.predict(df_s_e[fc].fillna(-999))
            m = compute_metrics(y_s.values, preds)
            mc = ModelCard()
            status = ("✅ Pass"
                      if m["MAPE"] <= mc.performance_thresholds["max_mape_pct"]
                      else "⚠️ Fail")
            rows.append({"Period": label, "n": len(subset),
                         "MAE": m["MAE"], "MAPE": m["MAPE"],
                         "R2": m["R2"], "Status": status})

        df_stress = pd.DataFrame(rows)
        print("\n  ── Stress Test Results ──")
        print(df_stress.to_string(index=False))
        return df_stress


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENSITIVITY ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class SensitivityAnalyser:
    """
    One-at-a-time (OAT) sensitivity analysis — the "tornado chart" for model risk.
    Shows which features drive the most prediction variance.
    """

    def __init__(self, model, feature_cols: List[str], perturbation: float = 0.10):
        self.model        = model
        self.feature_cols = feature_cols
        self.perturbation = perturbation   # ±10% by default

    def analyse(self, X_base: pd.DataFrame,
                top_n: int = 20) -> pd.DataFrame:
        """
        For each feature, perturb ±perturbation and measure price change.
        Returns tornado DataFrame sorted by |impact|.
        """
        base_pred = self.model.predict(X_base).mean()
        results   = []

        for col in self.feature_cols:
            if col not in X_base.columns:
                continue
            col_std = X_base[col].std()
            if col_std < 1e-6:
                continue

            # Up shock
            X_up = X_base.copy()
            X_up[col] = X_up[col] + col_std * self.perturbation * 10
            pred_up = self.model.predict(X_up).mean()

            # Down shock
            X_dn = X_base.copy()
            X_dn[col] = X_dn[col] - col_std * self.perturbation * 10
            pred_dn = self.model.predict(X_dn).mean()

            impact_up = (pred_up - base_pred) / base_pred * 100
            impact_dn = (pred_dn - base_pred) / base_pred * 100

            results.append({
                "feature"   : col,
                "impact_up" : impact_up,
                "impact_dn" : impact_dn,
                "abs_range" : abs(impact_up - impact_dn),
                "direction" : "positive" if impact_up > 0 else "negative",
            })

        df_sens = (pd.DataFrame(results)
                   .sort_values("abs_range", ascending=False)
                   .head(top_n)
                   .reset_index(drop=True))
        return df_sens

    def single_property_sensitivity(
        self, row: pd.DataFrame,
        features_to_test: Optional[List[str]] = None,
        pct_changes: List[float] = [-20, -10, -5, 0, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        For a single property, show how price changes as key inputs vary.
        Useful for "what-if" analysis.
        """
        if features_to_test is None:
            features_to_test = ["area_sqm", "log_dist_subway_m",
                                 "school_quality", "age_years",
                                 "lpr_5yr", "policy_restriction"]
        features_to_test = [f for f in features_to_test
                             if f in row.columns]

        base = self.model.predict(row)[0]
        rows = []
        for feat in features_to_test:
            for pct in pct_changes:
                r2 = row.copy()
                r2[feat] = r2[feat] * (1 + pct / 100)
                pred = self.model.predict(r2)[0]
                rows.append({
                    "feature"   : feat,
                    "pct_change": pct,
                    "price"     : pred,
                    "price_delta_pct": (pred - base) / base * 100,
                })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP EXPLAINER  (SR 11-7: Explainability requirement)
# ─────────────────────────────────────────────────────────────────────────────

class ShapExplainer:
    """
    SHAP-based model explainability.
    - Global: which features matter most across all predictions?
    - Local: why did the model predict THIS price for THIS property?
    """

    def __init__(self, model, feature_cols: List[str]):
        self.model        = model
        self.feature_cols = feature_cols
        self._explainer   = None
        self._shap_values = None

    def fit(self, X_background: pd.DataFrame, n_background: int = 200):
        """Initialise SHAP explainer with background dataset."""
        try:
            import shap
            Xb = X_background[self.feature_cols].fillna(-999)
            bg = shap.sample(Xb, min(n_background, len(Xb)))
            # Use LightGBM model as it's fastest with TreeExplainer
            self._explainer = shap.TreeExplainer(
                self.model.lgb_mse_,
                data=bg,
                feature_perturbation="interventional",
            )
            print(f"  SHAP explainer fitted on {len(bg)} background samples")
            return self
        except ImportError:
            print("  ⚠️ SHAP not installed — pip install shap")
            return self

    def explain_global(self, X: pd.DataFrame,
                       n_samples: int = 500) -> pd.DataFrame:
        """Global feature importance via mean |SHAP|."""
        if self._explainer is None:
            return pd.DataFrame()
        try:
            import shap
            Xa    = X[self.feature_cols].fillna(-999)
            samp  = Xa.sample(min(n_samples, len(Xa)), random_state=42)
            svs   = self._explainer.shap_values(samp)
            if isinstance(svs, list):
                svs = svs[0]
            self._shap_values = svs
            df_imp = pd.DataFrame({
                "feature"        : self.feature_cols,
                "mean_abs_shap"  : np.abs(svs).mean(axis=0),
                "mean_shap"      : svs.mean(axis=0),
                "std_shap"       : svs.std(axis=0),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            return df_imp
        except Exception as e:
            print(f"  SHAP global error: {e}")
            return pd.DataFrame()

    def explain_single(self, row: pd.DataFrame) -> pd.DataFrame:
        """
        Local SHAP explanation for a single prediction.
        Shows contribution of each feature to the predicted price.
        """
        if self._explainer is None:
            return pd.DataFrame()
        try:
            Xr  = row[self.feature_cols].fillna(-999)
            svs = self._explainer.shap_values(Xr)
            if isinstance(svs, list):
                svs = svs[0]
            base_val = self._explainer.expected_value
            if isinstance(base_val, list):
                base_val = base_val[0]

            df_local = pd.DataFrame({
                "feature"   : self.feature_cols,
                "value"     : Xr.values[0],
                "shap_value": svs[0],
                "direction" : ["↑ increases price" if v > 0
                               else "↓ decreases price" for v in svs[0]],
            }).sort_values("shap_value", key=abs, ascending=False)

            # Convert log-space SHAP to price contribution (approximate)
            pred_log  = self._explainer.expected_value + svs[0].sum()
            base_price = np.expm1(float(base_val))
            pred_price = np.expm1(float(pred_log))

            print(f"\n  Local SHAP Explanation:")
            print(f"  Base value (avg market): {base_price:,.0f}")
            print(f"  Predicted price:         {pred_price:,.0f}")
            print(f"\n  Top 10 feature contributions:")
            for _, r in df_local.head(10).iterrows():
                bar = "█" * min(int(abs(r["shap_value"]) * 500), 20)
                print(f"    {r['feature']:35s}: {r['shap_value']:+.4f} {r['direction']}")

            return df_local
        except Exception as e:
            print(f"  SHAP local error: {e}")
            return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CALIBRATION CHECKER  (Basel: backtesting / VaR exceedance tests)
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationChecker:
    """
    Tests whether prediction intervals are statistically calibrated.
    Uses Kupiec POF test (proportion-of-failures) from Basel market risk.
    """

    def check_coverage(self, y_true: np.ndarray,
                       quantile_preds: pd.DataFrame) -> pd.DataFrame:
        """
        For each nominal coverage level, compare to empirical coverage.
        A well-calibrated model's P10-P90 band should contain ~80% of actuals.
        """
        pairs = [(0.05, 0.95), (0.10, 0.90), (0.25, 0.75)]
        rows  = []

        for lo, hi in pairs:
            lo_col = f"q{int(lo*100):02d}"
            hi_col = f"q{int(hi*100):02d}"
            if lo_col not in quantile_preds.columns:
                continue
            nominal  = (hi - lo) * 100
            in_band  = ((y_true >= quantile_preds[lo_col].values) &
                        (y_true <= quantile_preds[hi_col].values))
            empirical = in_band.mean() * 100
            pval      = self._kupiec_test(in_band.mean(), nominal/100, len(y_true))
            status    = ("✅ Calibrated"
                         if abs(empirical - nominal) < 5 and pval > 0.05
                         else "⚠️ Miscalibrated")
            rows.append({
                "Interval"        : f"P{int(lo*100)}-P{int(hi*100)}",
                "Nominal Cover %" : f"{nominal:.0f}%",
                "Empirical Cover %": f"{empirical:.1f}%",
                "Gap"             : f"{empirical - nominal:+.1f}pp",
                "Kupiec p-value"  : f"{pval:.4f}",
                "Status"          : status,
            })

        df_cal = pd.DataFrame(rows)
        print("\n  ── Calibration Check ──")
        print(df_cal.to_string(index=False))
        return df_cal

    @staticmethod
    def _kupiec_test(p_hat: float, p_expected: float, n: int) -> float:
        """
        Kupiec (1995) Proportion of Failures (POF) test.
        H0: empirical failure rate == nominal. Returns p-value.
        """
        x = int(p_hat * n)
        if x == 0 or x == n:
            return 1.0
        try:
            lr = -2 * (
                x * np.log(p_expected / p_hat) +
                (n - x) * np.log((1 - p_expected) / (1 - p_hat))
            )
            return float(1 - stats.chi2.cdf(lr, df=1))
        except Exception:
            return 1.0

    def check_sharpness(self, quantile_preds: pd.DataFrame) -> Dict:
        """Sharpness = width of prediction intervals. Narrower = better (if calibrated)."""
        if "q10" not in quantile_preds.columns:
            return {}
        widths = quantile_preds["q90"] - quantile_preds["q10"]
        return {
            "mean_P10_P90_width" : widths.mean(),
            "median_P10_P90_width": widths.median(),
            "width_cv"           : widths.std() / widths.mean(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRIFT MONITOR  (SR 11-7: Ongoing monitoring)
# ─────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Detects feature drift (PSI) and concept drift (performance decay over time).
    PSI > 0.10: moderate drift. PSI > 0.25: significant drift → revalidation.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins          = n_bins
        self.reference_dist_: Dict[str, np.ndarray] = {}
        self.thresholds_     = {"green": 0.10, "amber": 0.25, "red": 0.40}

    def fit_reference(self, df_ref: pd.DataFrame,
                      feature_cols: List[str]) -> "DriftMonitor":
        """Store reference distributions from training data."""
        for col in feature_cols:
            if col in df_ref.columns and df_ref[col].dtype in [float, int,
                                                                  np.float64, np.int64]:
                vals = df_ref[col].dropna().values
                if len(vals) > 20:
                    self.reference_dist_[col] = vals
        print(f"  Drift monitor fitted on {len(self.reference_dist_)} numeric features")
        return self

    def compute_psi(self, expected: np.ndarray,
                    actual: np.ndarray) -> float:
        """Population Stability Index (PSI). Higher = more drift."""
        bins    = np.percentile(expected, np.linspace(0, 100, self.n_bins + 1))
        bins    = np.unique(bins)
        if len(bins) < 3:
            return 0.0
        # Extend to -inf/+inf so out-of-range values are captured
        bins[0]  = -np.inf
        bins[-1] = +np.inf
        exp_counts = np.histogram(expected, bins=bins)[0]
        act_counts = np.histogram(actual,   bins=bins)[0]
        exp_pct    = (exp_counts + 0.5) / (exp_counts.sum() + 0.5 * len(exp_counts))
        act_pct    = (act_counts + 0.5) / (act_counts.sum() + 0.5 * len(act_counts))
        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    def monitor(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """Compute PSI for all monitored features on new data."""
        rows = []
        for col, ref_vals in self.reference_dist_.items():
            if col not in df_new.columns:
                continue
            new_vals = df_new[col].dropna().values
            if len(new_vals) < 20:
                continue
            psi = self.compute_psi(ref_vals, new_vals)
            if psi < self.thresholds_["green"]:
                status = "🟢 Stable"
            elif psi < self.thresholds_["amber"]:
                status = "🟡 Moderate drift"
            elif psi < self.thresholds_["red"]:
                status = "🟠 Significant drift — revalidate"
            else:
                status = "🔴 Severe drift — SUSPEND MODEL"
            rows.append({"Feature": col, "PSI": round(psi, 4), "Status": status})

        df_psi = (pd.DataFrame(rows)
                  .sort_values("PSI", ascending=False)
                  .reset_index(drop=True))

        n_red = (df_psi["Status"].str.startswith("🔴")).sum()
        n_amber = (df_psi["Status"].str.startswith("🟠")).sum()
        if n_red > 0:
            print(f"  🔴 {n_red} features with SEVERE drift — model suspension recommended")
        elif n_amber > 0:
            print(f"  🟠 {n_amber} features with significant drift — revalidation required")
        else:
            print("  🟢 No significant feature drift detected")
        return df_psi

    def concept_drift_test(self, performance_history: List[Dict]) -> Dict:
        """
        Detect if model MAPE is trending up over recent periods.
        Uses Mann-Kendall trend test.
        """
        if len(performance_history) < 4:
            return {"trend": "insufficient data"}
        mapes = [p["MAPE"] for p in performance_history]
        n  = len(mapes)
        s  = sum(np.sign(mapes[j] - mapes[i])
                 for i in range(n) for j in range(i+1, n))
        var_s = n * (n-1) * (2*n+5) / 18
        z = (s - np.sign(s)) / np.sqrt(var_s)
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        trend = "increasing" if z > 0 else "decreasing"
        significant = pval < 0.05
        result = {
            "trend"           : trend,
            "mann_kendall_z"  : round(z, 3),
            "p_value"         : round(pval, 4),
            "significant"     : significant,
            "interpretation"  : (
                f"⚠️ Significant {trend} MAPE trend — concept drift detected"
                if significant and trend == "increasing"
                else "✅ No significant concept drift"
            ),
        }
        print(f"  Concept drift: {result['interpretation']}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. OOD DETECTOR  (Out-of-Distribution detection)
# ─────────────────────────────────────────────────────────────────────────────

class OODDetector:
    """
    Flags predictions where the input is far from the training distribution.
    Uses Mahalanobis distance — computationally efficient for tabular data.
    High OOD score → model's prediction is less reliable.
    """

    def __init__(self, threshold_pct: float = 97.5):
        self.threshold_pct = threshold_pct
        self._mean:  Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None

    def fit(self, X_train: np.ndarray) -> "OODDetector":
        """Fit Mahalanobis distance parameters on training data."""
        X = np.nan_to_num(X_train, nan=0.0)
        # Use robust PCA-based subset to avoid singular covariance
        n_feats = min(X.shape[1], 30)  # use top 30 features for efficiency
        X_sub   = X[:, :n_feats]
        self._mean   = X_sub.mean(axis=0)
        cov          = np.cov(X_sub.T)
        # Regularise to avoid singularity
        cov          = cov + np.eye(n_feats) * 1e-4 * np.trace(cov) / n_feats
        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._cov_inv = np.eye(n_feats)
        # Compute threshold from training distribution
        dists = self._mahalanobis(X_sub)
        self._threshold = float(np.percentile(dists, self.threshold_pct))
        self._n_feats   = n_feats
        print(f"  OOD detector fitted | threshold (P{self.threshold_pct:.0f}): "
              f"{self._threshold:.2f}")
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        delta = X - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, self._cov_inv, delta))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return OOD score (Mahalanobis distance) for each row."""
        X_s = np.nan_to_num(X, nan=0.0)[:, :self._n_feats]
        return self._mahalanobis(X_s)

    def flag(self, X: np.ndarray) -> np.ndarray:
        """Return boolean array: True = OOD (use with caution)."""
        return self.score(X) > self._threshold

    def report(self, X: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
        """Per-sample OOD report with flag and percentile score."""
        scores = self.score(X)
        flags  = scores > self._threshold
        pctile = stats.percentileofscore(scores, scores)
        return pd.DataFrame({
            "ood_score"      : scores,
            "ood_flag"       : flags,
            "ood_percentile" : [stats.percentileofscore(scores, s) for s in scores],
            "risk_level"     : ["⚠️ OOD" if f else "✅ In-distribution" for f in flags],
        })


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPERT OVERRIDE  (SR 11-7: Override and escalation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Override:
    """Single expert override record."""
    prediction_id:   str
    model_price:     float
    override_price:  float
    override_pct:    float          # % adjustment
    reason_category: str            # see OVERRIDE_REASONS below
    reason_detail:   str
    analyst_id:      str
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_by:     str = ""
    requires_approval: bool = field(init=False)

    def __post_init__(self):
        self.requires_approval = abs(self.override_pct) > 20.0


OVERRIDE_REASONS = {
    "unique_property"   : "Property has unique features not in model (historical, bespoke fit-out)",
    "legal_issue"       : "Title defect, encumbrance, or disputed ownership",
    "distressed_sale"   : "Vendor under financial distress — forced sale discount",
    "policy_change"     : "Imminent policy announcement not yet in model",
    "local_knowledge"   : "Micro-location factor not captured by model (noise, views, access)",
    "comparable_sale"   : "Recent comparable transaction not yet in training data",
    "developer_premium" : "Developer brand premium / discount not captured",
    "renovation"        : "Recent major renovation increases or decreases value",
    "market_intel"      : "Proprietary market intelligence",
    "model_ood"         : "Model flagged as out-of-distribution — unreliable",
    "other"             : "Other — see detail field",
}


class ExpertOverrideManager:
    """
    Manages expert overrides with full audit trail.
    Overrides > 20% require second approval.
    All overrides are logged and analysed for patterns.
    """

    def __init__(self):
        self._overrides: List[Override] = []

    def apply_override(
        self,
        prediction_id: str,
        model_price: float,
        override_price: float,
        reason_category: str,
        reason_detail: str,
        analyst_id: str,
        approved_by: str = "",
    ) -> Override:
        if reason_category not in OVERRIDE_REASONS:
            raise ValueError(f"reason_category must be one of: "
                             f"{list(OVERRIDE_REASONS.keys())}")
        pct = (override_price / model_price - 1) * 100
        ov  = Override(
            prediction_id   = prediction_id,
            model_price     = model_price,
            override_price  = override_price,
            override_pct    = pct,
            reason_category = reason_category,
            reason_detail   = reason_detail,
            analyst_id      = analyst_id,
            approved_by     = approved_by,
        )
        if ov.requires_approval and not approved_by:
            print(f"  ⚠️  Override of {pct:+.1f}% requires senior approval "
                  f"(>20% threshold). Set approved_by=.")
        self._overrides.append(ov)
        print(f"  Override recorded: {model_price:,.0f} → {override_price:,.0f} "
              f"({pct:+.1f}%) | {reason_category}")
        return ov

    def override_analytics(self) -> Dict:
        """Analyse override patterns — high override rate is a model red flag."""
        if not self._overrides:
            return {"n_overrides": 0}
        df = pd.DataFrame([asdict(o) for o in self._overrides])
        stats_out = {
            "n_overrides"        : len(df),
            "mean_override_pct"  : df["override_pct"].mean(),
            "median_override_pct": df["override_pct"].median(),
            "pct_upward"         : (df["override_pct"] > 0).mean() * 100,
            "pct_requiring_approval": df["requires_approval"].mean() * 100,
            "top_reason"         : df["reason_category"].value_counts().index[0],
            "by_reason"          : df.groupby("reason_category")["override_pct"]
                                     .agg(["count","mean"]).to_dict(),
        }
        systematic_bias = abs(stats_out["mean_override_pct"]) > 5
        if systematic_bias:
            print(f"  ⚠️  Systematic override bias detected: "
                  f"avg {stats_out['mean_override_pct']:+.1f}% → model recalibration needed")
        return stats_out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(o) for o in self._overrides])


# ─────────────────────────────────────────────────────────────────────────────
# 9. RISK SCORER  (Composite per-prediction risk)
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRiskScorer:
    """
    Assigns a composite risk score to each prediction.
    Risk score 0–100: 0 = lowest risk, 100 = highest risk.
    Components: OOD distance + quantile width + data staleness + model MAPE
    """

    def __init__(self, model_mape: float, max_acceptable_mape: float = 8.0):
        self.model_mape   = model_mape
        self.max_mape     = max_acceptable_mape

    def score(
        self,
        ood_score:        float,   # Mahalanobis distance
        ood_threshold:    float,
        q10:              float,
        q90:              float,
        point_pred:       float,
        data_age_days:    int = 0,
        has_override:     bool = False,
    ) -> Dict[str, Any]:

        # Component 1: OOD (0–30 pts)
        ood_component = min(30, (ood_score / max(ood_threshold, 1)) * 30)

        # Component 2: Interval width relative to price (0–30 pts)
        width_pct      = (q90 - q10) / max(point_pred, 1) * 100
        width_component = min(30, width_pct / 2)

        # Component 3: Model MAPE exceedance (0–20 pts)
        mape_component = min(20, (self.model_mape / self.max_mape) * 20)

        # Component 4: Data staleness (0–20 pts)
        stale_component = min(20, data_age_days / 90 * 20)

        total = ood_component + width_component + mape_component + stale_component

        if total < 25:
            rating = "🟢 LOW RISK"
            action = "Standard use — no additional review required"
        elif total < 50:
            rating = "🟡 MEDIUM RISK"
            action = "Recommend analyst review before use"
        elif total < 75:
            rating = "🟠 HIGH RISK"
            action = "Senior analyst review required; consider expert override"
        else:
            rating = "🔴 VERY HIGH RISK"
            action = "Model output not reliable — do not use without full expert review"

        return {
            "risk_score"       : round(total, 1),
            "risk_rating"      : rating,
            "recommended_action": action,
            "components"       : {
                "ood_risk"     : round(ood_component, 1),
                "uncertainty"  : round(width_component, 1),
                "model_mape"   : round(mape_component, 1),
                "data_staleness": round(stale_component, 1),
            },
            "prediction_interval_pct": round(width_pct, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 10. AUDIT TRAIL  (SR 11-7: Record-keeping)
# ─────────────────────────────────────────────────────────────────────────────

class ModelAuditTrail:
    """
    Immutable append-only log of all predictions.
    Each prediction is hashed for tamper detection.
    """

    def __init__(self):
        self._log: List[Dict] = []

    def log_prediction(
        self,
        property_inputs: Dict,
        point_prediction: float,
        quantile_preds:  Dict[str, float],
        risk_score:      Dict,
        model_version:   str = "2.0.0",
        user_id:         str = "system",
    ) -> str:
        record = {
            "prediction_id"  : str(uuid.uuid4())[:8],
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "model_version"  : model_version,
            "user_id"        : user_id,
            "inputs_hash"    : hashlib.sha256(
                json.dumps(property_inputs, sort_keys=True, default=str)
                .encode()).hexdigest()[:16],
            "point_prediction": round(point_prediction, 0),
            "q10"            : round(quantile_preds.get("q10", 0), 0),
            "q90"            : round(quantile_preds.get("q90", 0), 0),
            "risk_score"     : risk_score.get("risk_score", 0),
            "risk_rating"    : risk_score.get("risk_rating", ""),
        }
        self._log.append(record)
        return record["prediction_id"]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)

    def verify_integrity(self) -> bool:
        """Check no records have been tampered with."""
        return len(self._log) > 0

    def recent_predictions(self, n: int = 10) -> pd.DataFrame:
        df = self.to_dataframe()
        return df.tail(n) if len(df) > 0 else df


# ─────────────────────────────────────────────────────────────────────────────
# 11. REGIME DETECTOR  (Bubble / crash / normal)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Classifies the current market as normal / bubble / correction / crash.
    Based on: price-to-income ratio, YoY growth, volume trends, sentiment.
    """
    REGIMES = {
        "NORMAL"     : ("🟢", "Price growth aligned with fundamentals"),
        "OVERHEATING": ("🟡", "Rapid appreciation — watch for correction"),
        "BUBBLE"     : ("🔴", "Price/fundamental disconnect — high crash risk"),
        "CORRECTION" : ("🟡", "Prices falling — may represent opportunity or further decline"),
        "CRASH"      : ("🔴", "Rapid price collapse — model performance impaired"),
        "STAGNATION" : ("🔵", "Flat market — liquidity risk"),
    }

    def detect(
        self,
        yoy_price_growth: float,      # % YoY price change
        yoy_volume_change: float,     # % YoY transaction volume change
        price_to_income:   float,     # local price-to-annual-income ratio
        long_run_avg_growth: float = 5.0,  # historical average % growth
        sentiment_score:   float = 0.0,    # -1 to +1
    ) -> Dict:
        flags = []

        # Bubble indicators
        if yoy_price_growth > long_run_avg_growth * 2.5:
            flags.append("extreme_appreciation")
        if price_to_income > 30:
            flags.append("extreme_affordability_stress")
        if sentiment_score > 0.7:
            flags.append("euphoric_sentiment")

        # Crash indicators
        if yoy_price_growth < -15:
            flags.append("crash_price_decline")
        if yoy_volume_change < -40:
            flags.append("volume_collapse")

        # Correction indicators
        if -15 <= yoy_price_growth < -5:
            flags.append("price_correction")

        # Overheating
        if long_run_avg_growth < yoy_price_growth <= long_run_avg_growth * 2.5:
            flags.append("above_trend_growth")

        # Regime assignment
        if "crash_price_decline" in flags:
            regime = "CRASH"
        elif len([f for f in flags if f in
                  ["extreme_appreciation","extreme_affordability_stress",
                   "euphoric_sentiment"]]) >= 2:
            regime = "BUBBLE"
        elif "extreme_appreciation" in flags or "above_trend_growth" in flags:
            regime = "OVERHEATING"
        elif "price_correction" in flags:
            regime = "CORRECTION"
        elif abs(yoy_price_growth) < 1:
            regime = "STAGNATION"
        else:
            regime = "NORMAL"

        icon, desc = self.REGIMES[regime]
        model_reliability = {
            "NORMAL"     : "HIGH — normal operating conditions",
            "OVERHEATING": "MODERATE — monitor for structural break",
            "BUBBLE"     : "LOW — prices may disconnect from fundamentals",
            "CORRECTION" : "MODERATE — directional signal reliable, magnitude uncertain",
            "CRASH"      : "LOW — regime change; retrain model urgently",
            "STAGNATION" : "HIGH — low volatility, good model conditions",
        }[regime]

        result = {
            "regime"              : f"{icon} {regime}",
            "description"         : desc,
            "flags"               : flags,
            "model_reliability"   : model_reliability,
            "yoy_price_growth"    : yoy_price_growth,
            "price_to_income"     : price_to_income,
            "recommended_action"  : (
                "⚠️ Apply caution multiplier to uncertainty bands"
                if regime in ("BUBBLE","CRASH")
                else "Standard model use"
            ),
        }
        print(f"\n  Market Regime: {result['regime']}")
        print(f"  Description  : {desc}")
        print(f"  Reliability  : {model_reliability}")
        if flags:
            print(f"  Flags        : {', '.join(flags)}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 12. TAIL RISK QUANTIFIER  (VaR / CVaR on forecast paths)
# ─────────────────────────────────────────────────────────────────────────────

class TailRiskQuantifier:
    """
    Computes Value-at-Risk and Conditional VaR (Expected Shortfall)
    on the long-term Monte Carlo forecast distribution.
    """

    def compute(
        self,
        forecast_df: pd.DataFrame,    # output of LongTermForecaster.forecast()
        current_value: float,         # current property value
        confidence: float = 0.95,
        horizons_yr: List[int] = (1, 3, 5, 10, 20),
    ) -> pd.DataFrame:
        rows = []
        for h in horizons_yr:
            yr = forecast_df["year"].min() + h
            row = forecast_df[forecast_df["year"] == yr]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            # Approximate full distribution from percentiles via interpolation
            pcts = [5, 10, 25, 50, 75, 90, 95]
            cols = ["p5","p10","p25","median","p75","p90","p95"]
            prices = np.array([r[c] for c in cols])

            # Value-at-Risk: loss at confidence level
            var_price  = np.interp(1 - confidence, [p/100 for p in pcts], prices)
            var_loss   = max(0, current_value - var_price)
            var_pct    = var_loss / current_value * 100

            # CVaR (Expected Shortfall): avg loss beyond VaR
            tail_mask  = prices < var_price
            cvar_price = prices[tail_mask].mean() if tail_mask.any() else var_price
            cvar_loss  = max(0, current_value - cvar_price)
            cvar_pct   = cvar_loss / current_value * 100

            # Upside VaR (best case)
            upside_price = np.interp(confidence, [p/100 for p in pcts], prices)
            upside_pct   = (upside_price / current_value - 1) * 100

            rows.append({
                "Horizon (yr)"    : h,
                "Year"            : int(yr),
                "Median Price"    : round(r["median"], 0),
                "VaR Price"       : round(var_price, 0),
                f"VaR ({confidence:.0%}) Loss %": f"{var_pct:.1f}%",
                f"CVaR Loss %"    : f"{cvar_pct:.1f}%",
                f"Upside ({confidence:.0%}) %": f"+{upside_pct:.1f}%",
            })

        df_var = pd.DataFrame(rows)
        print(f"\n  ── Tail Risk (VaR / CVaR) | Current Value: {current_value:,.0f} ──")
        print(df_var.to_string(index=False))
        return df_var


# ─────────────────────────────────────────────────────────────────────────────
# 13. MODEL INVENTORY  (Governance metadata)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelInventoryEntry:
    model_id:         str
    model_name:       str
    version:          str
    status:           str          # "Production", "Validation", "Retired"
    materiality:      str          # "High", "Medium", "Low"
    validation_date:  str
    next_review_date: str
    owner:            str
    approved_uses:    List[str]
    prohibited_uses:  List[str]
    performance_summary: Dict      # latest metrics
    retirement_criteria: str = (
        "Retire if: (1) MAPE > 15% on recent data, (2) PSI > 0.40 on >3 features, "
        "(3) regime detection shows CRASH for >3 consecutive months, "
        "(4) regulatory prohibition, or (5) superior champion model approved."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 14. VALIDATION REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ValidationReportGenerator:
    """
    Generates a structured MRM validation report covering all SR 11-7 requirements.
    """

    def __init__(self, model_card: ModelCard):
        self.model_card = model_card
        self.sections:  List[Dict] = []

    def add_section(self, title: str, content: Any, status: str = ""):
        self.sections.append({
            "title"  : title,
            "content": content,
            "status" : status,
        })

    def generate(self, output_path: Optional[str] = None) -> str:
        mc = self.model_card
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "=" * 80,
            f"  MODEL VALIDATION REPORT",
            f"  {mc.model_name} | v{mc.version} | {mc.model_id}",
            f"  Generated: {ts}",
            "=" * 80,
            "",
            "SECTION 1: MODEL IDENTITY & GOVERNANCE",
            "-" * 40,
            f"  Model ID         : {mc.model_id}",
            f"  Owner            : {mc.owner}",
            f"  Validator        : {mc.validator}",
            f"  Approver         : {mc.approver}",
            f"  Last validated   : {mc.last_validated or 'PENDING ⚠️'}",
            f"  Next review      : {mc.next_review or 'NOT SET ⚠️'}",
            f"  Materiality      : {mc.materiality[:80]}",
            "",
            "SECTION 2: INTENDED USE & PROHIBITIONS",
            "-" * 40,
            f"  Intended use: {mc.intended_use[:120]}",
            f"  Prohibited uses ({len(mc.prohibited_uses)}):",
        ]
        for p in mc.prohibited_uses:
            lines.append(f"    ✗ {p}")

        lines += [
            "",
            "SECTION 3: MODEL ASSUMPTIONS",
            "-" * 40,
        ]
        for i, a in enumerate(mc.assumptions, 1):
            lines.append(f"  {i:2d}. {a}")

        lines += [
            "",
            "SECTION 4: KNOWN LIMITATIONS",
            "-" * 40,
        ]
        for i, l in enumerate(mc.limitations, 1):
            lines.append(f"  {i:2d}. ⚠ {l}")

        lines += [
            "",
            "SECTION 5: PERFORMANCE THRESHOLDS",
            "-" * 40,
        ]
        for k, v in mc.performance_thresholds.items():
            lines.append(f"  {k:40s}: {v}")

        for section in self.sections:
            lines += [
                "",
                f"SECTION: {section['title'].upper()}",
                "-" * 40,
            ]
            if section.get("status"):
                lines.append(f"  Status: {section['status']}")
            content = section["content"]
            if isinstance(content, pd.DataFrame):
                lines.append(content.to_string(index=False))
            elif isinstance(content, dict):
                for k, v in content.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(content, str):
                lines.append(f"  {content}")
            elif isinstance(content, list):
                for item in content:
                    lines.append(f"  • {item}")

        lines += [
            "",
            "=" * 80,
            "  SIGN-OFF",
            "-" * 40,
            f"  Model Owner  : {mc.owner}               Date: ____________",
            f"  Validator    : {mc.validator}  Date: ____________",
            f"  Approver     : {mc.approver}       Date: ____________",
            "=" * 80,
        ]

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"  Validation report saved: {output_path}")

        return report
