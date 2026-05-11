"""
Extreme Conditions & Shock Event System
========================================
Users define shocks in three categories:

  1. POLITICAL   – policy changes, geopolitical events, sanctions
  2. ECONOMIC    – recessions, hyperinflation, credit crises, currency crises
  3. PHYSICAL    – natural disasters, climate events, pandemics

Each shock has:
  - magnitude   : -1.0 (total collapse) → +1.0 (massive boom)
  - duration_yr : how long the shock effect lasts
  - decay        : how the shock fades ('linear', 'exponential', 'step')
  - scope        : 'local', 'national', 'regional', 'global'
  - affected_aspects : which price aspects it impacts

This module generates a time-series multiplier array that is applied on top
of the baseline Monte Carlo forecast.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Dict, Optional, Tuple
import numpy as np


# ── Aspect taxonomy ──────────────────────────────────────────────────────────
# These map to feature groups in the model
ASPECTS = {
    # DEMAND SIDE
    "buyer_demand"         : "Overall buyer demand level",
    "foreign_buyer_demand" : "Demand from overseas investors",
    "investor_demand"      : "Domestic investment demand",
    "end_user_demand"      : "Owner-occupier demand",
    "rental_demand"        : "Rental demand / yield attractiveness",

    # SUPPLY SIDE
    "new_supply"           : "New construction pipeline",
    "land_supply"          : "Available land for development",
    "conversion_supply"    : "Conversion / redevelopment supply",

    # FINANCING
    "mortgage_access"      : "Credit availability / mortgage approval rate",
    "interest_cost"        : "Cost of borrowing (rate environment)",
    "investor_leverage"    : "Leverage available to investors",

    # MACRO
    "gdp_sentiment"        : "Economic growth sentiment",
    "employment"           : "Labour market / income growth",
    "currency_value"       : "Local currency purchasing power",
    "inflation_hedge"      : "Property as inflation protection",

    # POLICY & REGULATION
    "purchase_restriction" : "Government purchase restriction intensity",
    "tax_burden"           : "Transaction / holding tax changes",
    "zoning_regulation"    : "Zoning / development controls",
    "rent_regulation"      : "Rent control / yield compression",

    # RISK
    "physical_risk"        : "Structural / natural hazard risk premium",
    "political_risk"       : "Geopolitical / governance risk premium",
    "liquidity_risk"       : "Market liquidity / transaction volume",
    "legal_title_risk"     : "Property rights / rule of law risk",
}


@dataclass
class Shock:
    """A single extreme-condition shock event."""
    name: str
    category: Literal["political", "economic", "physical", "social"]
    description: str

    # ── Impact parameters ────────────────────────────────────────────────────
    magnitude: float                    # -1.0 to +1.0
    duration_yr: float                  # years the shock lasts
    onset_yr: float = 0.0               # years from now before shock hits
    decay: Literal["linear","exponential","step","log"] = "exponential"

    # ── Scope ────────────────────────────────────────────────────────────────
    scope: Literal["local","city","national","regional","global"] = "city"
    affected_cities: List[str] = field(default_factory=list)  # empty = all

    # ── Aspect weights: which aspects does this shock affect? ─────────────────
    # keys must be in ASPECTS; values are multipliers on magnitude
    aspect_weights: Dict[str, float] = field(default_factory=dict)

    # ── Uncertainty around the shock itself ──────────────────────────────────
    magnitude_uncertainty: float = 0.15  # std dev of magnitude
    timing_uncertainty_yr: float = 0.25  # std dev of onset

    # ── Recovery shape ───────────────────────────────────────────────────────
    recovery_yr: Optional[float] = None  # if None = duration_yr
    permanent_scar: float = 0.0          # fraction of impact that never recovers

    def time_series(self, years: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Returns a multiplier array for each year.
        1.0 = no effect; 0.7 = -30% impact; 1.3 = +30% impact.
        """
        rng = np.random.default_rng(seed)
        mag = np.clip(
            self.magnitude + rng.normal(0, self.magnitude_uncertainty),
            -0.99, 0.99
        )
        onset = max(0, self.onset_yr + rng.normal(0, self.timing_uncertainty_yr))
        recovery = self.recovery_yr if self.recovery_yr else self.duration_yr

        effect = np.zeros(len(years))

        start_yr = float(years[0])
        for i, yr in enumerate(years):
            t = (yr - start_yr) - onset
            if t < 0:
                continue
            elif t <= self.duration_yr:
                phase = t / max(self.duration_yr, 0.01)
                if self.decay == "step":
                    raw = 1.0
                elif self.decay == "linear":
                    raw = 1.0 - phase * 0.3
                elif self.decay == "exponential":
                    raw = np.exp(-phase * 1.5)
                elif self.decay == "log":
                    raw = 1.0 - np.log1p(phase) / np.log1p(1)
                else:
                    raw = 1.0
                effect[i] = mag * raw
            else:
                # Recovery phase
                t_rec = (yr - onset - self.duration_yr) / max(recovery, 0.01)
                scar = mag * self.permanent_scar
                fading = mag * (1 - self.permanent_scar) * np.exp(-t_rec * 2)
                effect[i] = scar + fading

        # Convert effect to price multiplier
        return 1.0 + effect


# ── PRE-BUILT SHOCK LIBRARY ──────────────────────────────────────────────────
SHOCK_LIBRARY: Dict[str, Shock] = {

    # ── POLITICAL SHOCKS ─────────────────────────────────────────────────────
    "china_purchase_ban": Shock(
        name="China Total Purchase Ban",
        category="political",
        description="Government bans all non-resident purchases (e.g. non-hukou holders)",
        magnitude=-0.25, duration_yr=3.0, decay="step",
        scope="city",
        aspect_weights={"buyer_demand":-1.0,"purchase_restriction":1.0,"investor_demand":-1.0},
        permanent_scar=0.05,
    ),
    "taiwan_strait_crisis": Shock(
        name="Taiwan Strait Military Crisis",
        category="political",
        description="Military escalation in the Taiwan Strait triggers capital flight",
        magnitude=-0.35, duration_yr=1.5, decay="exponential",
        scope="regional", affected_cities=["Shanghai","Beijing","Shenzhen","Hong Kong","Taipei"],
        aspect_weights={"political_risk":1.0,"foreign_buyer_demand":-1.0,
                        "capital_flows":-1.0,"currency_value":-0.5},
        magnitude_uncertainty=0.20, permanent_scar=0.10,
    ),
    "hk_political_crisis": Shock(
        name="Hong Kong Political Crisis",
        category="political",
        description="Political unrest and emigration wave (e.g. 2019-style)",
        magnitude=-0.30, duration_yr=4.0, decay="log",
        scope="city", affected_cities=["Hong Kong"],
        aspect_weights={"political_risk":1.0,"foreign_buyer_demand":-0.8,
                        "emigration_pressure":-0.9,"legal_title_risk":0.5},
        permanent_scar=0.15,
    ),
    "russia_sanctions_spillover": Shock(
        name="Russia Sanctions / Capital Flight",
        category="political",
        description="Sanctioned Russian capital floods into safe-haven property markets",
        magnitude=0.15, duration_yr=3.0, decay="exponential",
        scope="global",
        affected_cities=["Dubai","Istanbul","Belgrade","Tbilisi"],
        aspect_weights={"foreign_buyer_demand":1.0,"investor_demand":0.7,"currency_value":0.3},
        permanent_scar=0.05,
    ),
    "us_china_decoupling": Shock(
        name="US-China Full Trade Decoupling",
        category="political",
        description="Complete trade and financial decoupling; supply chains reshored",
        magnitude=-0.20, duration_yr=10.0, decay="linear",
        scope="global",
        aspect_weights={"gdp_sentiment":-0.6,"employment":-0.5,
                        "foreign_buyer_demand":-0.8,"currency_value":-0.4},
        permanent_scar=0.20,
    ),
    "singapore_absd_hike": Shock(
        name="Singapore ABSD Surge (Additional Buyer Stamp Duty)",
        category="political",
        description="Government hikes ABSD for foreigners to 60%+",
        magnitude=-0.15, duration_yr=2.0, decay="step",
        scope="city", affected_cities=["Singapore"],
        aspect_weights={"foreign_buyer_demand":-1.0,"tax_burden":1.0,"investor_demand":-0.5},
        permanent_scar=0.0,
    ),
    "uk_liz_truss_crisis": Shock(
        name="UK Mini-Budget / Fiscal Crisis",
        category="political",
        description="Sudden fiscal shock causes mortgage rates to spike 200bps in weeks",
        magnitude=-0.12, duration_yr=1.5, decay="exponential",
        scope="national",
        aspect_weights={"mortgage_access":-0.8,"interest_cost":1.0,"buyer_demand":-0.7},
        permanent_scar=0.03,
    ),
    "golden_visa_programme": Shock(
        name="Golden Visa / Investor Visa Launch",
        category="political",
        description="Country launches residency-for-investment programme",
        magnitude=0.12, duration_yr=5.0, decay="log",
        scope="national",
        aspect_weights={"foreign_buyer_demand":1.0,"investor_demand":0.6,"rental_demand":0.3},
        permanent_scar=0.10,
    ),
    "beijing_common_prosperity": Shock(
        name="Common Prosperity Policy (China)",
        category="political",
        description="Wealth redistribution drive: property tax pilot, platform crackdowns",
        magnitude=-0.15, duration_yr=5.0, decay="linear",
        scope="national",
        aspect_weights={"investor_demand":-0.8,"tax_burden":0.7,"purchase_restriction":0.5},
        permanent_scar=0.10,
    ),

    # ── ECONOMIC SHOCKS ──────────────────────────────────────────────────────
    "global_financial_crisis": Shock(
        name="Global Financial Crisis (2008-type)",
        category="economic",
        description="Credit markets seize; mortgage availability collapses; recession",
        magnitude=-0.35, duration_yr=3.0, decay="exponential",
        scope="global",
        aspect_weights={"mortgage_access":-1.0,"buyer_demand":-0.9,"gdp_sentiment":-0.9,
                        "employment":-0.7,"investor_leverage":-1.0,"liquidity_risk":1.0},
        magnitude_uncertainty=0.15, permanent_scar=0.05,
    ),
    "china_property_debt_crisis": Shock(
        name="China Property Sector Debt Crisis (Evergrande-type)",
        category="economic",
        description="Developer defaults cascade; new home completions collapse",
        magnitude=-0.30, duration_yr=4.0, decay="log",
        scope="national",
        affected_cities=["Shanghai","Beijing","Shenzhen","Guangzhou"],
        aspect_weights={"buyer_demand":-0.8,"new_supply":-0.7,"investor_demand":-0.9,
                        "mortgage_access":-0.5,"gdp_sentiment":-0.6},
        permanent_scar=0.08,
    ),
    "hyperinflation": Shock(
        name="Hyperinflation (>20% CPI)",
        category="economic",
        description="Currency debasement drives real asset flight to property",
        magnitude=0.30, duration_yr=3.0, decay="exponential",
        scope="national",
        aspect_weights={"inflation_hedge":1.0,"currency_value":-0.8,
                        "interest_cost":1.0,"investor_demand":0.5},
        magnitude_uncertainty=0.25, permanent_scar=0.0,
    ),
    "rate_shock_200bps": Shock(
        name="Rapid Interest Rate Shock +200bps",
        category="economic",
        description="Central bank hikes 200bps in 12 months (2022-Fed style)",
        magnitude=-0.15, duration_yr=2.0, decay="exponential",
        scope="national",
        aspect_weights={"interest_cost":1.0,"mortgage_access":-0.6,
                        "buyer_demand":-0.7,"investor_leverage":-0.5},
        permanent_scar=0.02,
    ),
    "rate_shock_minus_200bps": Shock(
        name="Emergency Rate Cut -200bps",
        category="economic",
        description="Emergency easing cycle; mortgage rates halve",
        magnitude=0.18, duration_yr=3.0, decay="log",
        scope="national",
        aspect_weights={"interest_cost":-1.0,"mortgage_access":0.7,
                        "buyer_demand":0.8,"investor_demand":0.6},
        permanent_scar=0.05,
    ),
    "currency_crisis": Shock(
        name="Currency Crisis / Devaluation 30%+",
        category="economic",
        description="Sharp FX devaluation; foreign buyers gain but locals lose purchasing power",
        magnitude=-0.20, duration_yr=2.0, decay="exponential",
        scope="national",
        aspect_weights={"currency_value":-1.0,"foreign_buyer_demand":0.5,
                        "end_user_demand":-0.7,"inflation_hedge":0.6},
        permanent_scar=0.05,
    ),
    "tech_bubble_burst": Shock(
        name="Tech Sector Bubble Burst",
        category="economic",
        description="Tech valuations collapse; mass layoffs in tech hubs",
        magnitude=-0.25, duration_yr=2.0, decay="exponential",
        scope="city",
        affected_cities=["San Francisco","Seattle","Austin","Shenzhen","Beijing"],
        aspect_weights={"employment":-1.0,"buyer_demand":-0.8,"rental_demand":-0.4},
        permanent_scar=0.05,
    ),
    "quantitative_easing": Shock(
        name="Large-Scale QE / Money Printing",
        category="economic",
        description="Central bank asset purchase programme inflates asset prices",
        magnitude=0.20, duration_yr=4.0, decay="log",
        scope="national",
        aspect_weights={"inflation_hedge":0.8,"investor_demand":0.7,
                        "interest_cost":-0.6,"gdp_sentiment":0.4},
        permanent_scar=0.08,
    ),
    "oil_price_crash": Shock(
        name="Oil Price Crash (>50% decline)",
        category="economic",
        description="Oil exporters face fiscal shock; Gulf property demand collapses",
        magnitude=-0.30, duration_yr=2.0, decay="exponential",
        scope="regional",
        affected_cities=["Dubai","Abu Dhabi","Riyadh","Kuwait City"],
        aspect_weights={"gdp_sentiment":-0.9,"employment":-0.7,
                        "foreign_buyer_demand":-0.5,"buyer_demand":-0.8},
        permanent_scar=0.05,
    ),
    "japan_yield_curve_unwind": Shock(
        name="Japan YCC Unwind / Rate Normalisation",
        category="economic",
        description="BOJ abandons yield curve control; JPY strengthens; rates rise",
        magnitude=-0.15, duration_yr=3.0, decay="exponential",
        scope="national",
        affected_cities=["Tokyo","Osaka"],
        aspect_weights={"interest_cost":0.8,"foreign_buyer_demand":-0.4,
                        "currency_value":0.5,"investor_demand":-0.4},
        permanent_scar=0.05,
    ),

    # ── PHYSICAL / CLIMATE SHOCKS ─────────────────────────────────────────────
    "pandemic_lockdown": Shock(
        name="Pandemic / Lockdown (COVID-type)",
        category="physical",
        description="City lockdown; transactions freeze; remote work reshapes demand",
        magnitude=-0.10, duration_yr=2.0, decay="exponential",
        scope="global",
        aspect_weights={"buyer_demand":-0.6,"liquidity_risk":0.8,"new_supply":-0.5,
                        "rental_demand":-0.3,"end_user_demand":-0.2},
        recovery_yr=1.5, permanent_scar=-0.05,  # negative scar = post-pandemic boom
    ),
    "major_earthquake": Shock(
        name="Major Earthquake (M7.5+)",
        category="physical",
        description="Severe earthquake destroys significant building stock",
        magnitude=-0.30, duration_yr=5.0, decay="log",
        scope="city",
        aspect_weights={"physical_risk":1.0,"buyer_demand":-0.7,"new_supply":0.3,
                        "insurance_cost":0.8,"legal_title_risk":0.3},
        permanent_scar=0.10,
    ),
    "major_flood": Shock(
        name="Catastrophic Flood / Sea Level Event",
        category="physical",
        description="Major flooding event (1-in-100yr) with infrastructure damage",
        magnitude=-0.20, duration_yr=3.0, decay="linear",
        scope="local",
        aspect_weights={"physical_risk":1.0,"buyer_demand":-0.6,"insurance_cost":0.9},
        permanent_scar=0.15,
    ),
    "climate_insurance_crisis": Shock(
        name="Climate Insurance Market Failure",
        category="physical",
        description="Insurers withdraw from high-risk zones; uninsurable properties crash",
        magnitude=-0.25, duration_yr=10.0, decay="linear",
        scope="local",
        aspect_weights={"physical_risk":1.0,"insurance_cost":1.0,"buyer_demand":-0.7,
                        "investor_demand":-0.8,"legal_title_risk":0.3},
        permanent_scar=0.30,
    ),
    "typhoon_super": Shock(
        name="Super Typhoon / Category 5 Hurricane",
        category="physical",
        description="Direct hit from Category 5 typhoon; coastal districts devastated",
        magnitude=-0.15, duration_yr=2.0, decay="exponential",
        scope="city",
        aspect_weights={"physical_risk":1.0,"buyer_demand":-0.5,"new_supply":0.2},
        permanent_scar=0.08,
    ),

    # ── SOCIAL SHOCKS ─────────────────────────────────────────────────────────
    "mass_emigration": Shock(
        name="Mass Emigration Wave",
        category="social",
        description="Large share of population emigrates (e.g. HK 2020-22)",
        magnitude=-0.25, duration_yr=5.0, decay="linear",
        scope="city",
        aspect_weights={"buyer_demand":-0.9,"rental_demand":-0.5,"end_user_demand":-1.0},
        permanent_scar=0.15,
    ),
    "mass_immigration": Shock(
        name="Mass Immigration Surge",
        category="social",
        description="Sudden large influx of residents (refugees, economic migrants)",
        magnitude=0.15, duration_yr=4.0, decay="log",
        scope="city",
        aspect_weights={"rental_demand":1.0,"buyer_demand":0.4,"new_supply":-0.3},
        permanent_scar=0.10,
    ),
    "urban_tech_boom": Shock(
        name="Urban Tech / AI Hub Emergence",
        category="social",
        description="City becomes major tech hub; high-income influx",
        magnitude=0.25, duration_yr=8.0, decay="log",
        scope="city",
        aspect_weights={"employment":1.0,"buyer_demand":0.8,
                        "foreign_buyer_demand":0.4,"rental_demand":0.7},
        permanent_scar=0.20,
    ),
    "work_from_home_permanent": Shock(
        name="Permanent Remote Work Cultural Shift",
        category="social",
        description="Major employer class adopts WFH; suburban/satellite demand rises, CBD falls",
        magnitude=-0.10, duration_yr=10.0, decay="linear",
        scope="global",
        aspect_weights={"end_user_demand":-0.4,"rental_demand":-0.3,
                        "suburban_premium":0.5,"office_district_premium":-0.6},
        permanent_scar=0.20,
    ),
    "demographic_aging": Shock(
        name="Structural Demographic Aging / Birth Rate Collapse",
        category="social",
        description="Secular decline in working-age population reduces structural demand",
        magnitude=-0.15, duration_yr=20.0, decay="linear",
        scope="national",
        aspect_weights={"buyer_demand":-0.6,"end_user_demand":-0.7,
                        "new_supply":-0.4,"gdp_sentiment":-0.3},
        permanent_scar=0.30,
    ),
}


def apply_shocks(
    base_forecast: "np.ndarray",
    years: "np.ndarray",
    shocks: List[Shock],
    n_paths: int = 1,
    seed: int = 42,
) -> "np.ndarray":
    """
    Apply a list of shocks to a base forecast array.

    Parameters
    ----------
    base_forecast : (n_years,) or (n_paths, n_years) array
    years         : (n_years,) array of year values
    shocks        : list of Shock objects to apply
    n_paths       : number of Monte Carlo paths (if base_forecast is 1D)
    seed          : random seed

    Returns
    -------
    shocked_forecast : same shape as base_forecast
    """
    rng = np.random.default_rng(seed)

    if base_forecast.ndim == 1:
        arr = np.tile(base_forecast, (n_paths, 1))
    else:
        arr = base_forecast.copy()

    for shock in shocks:
        for path_i in range(arr.shape[0]):
            ts = shock.time_series(years, seed=rng.integers(0, 99999))
            arr[path_i] *= ts

    return arr


def summarise_shocks(shocks: List[Shock]) -> "pd.DataFrame":
    """Return a summary DataFrame of applied shocks."""
    import pandas as pd
    rows = []
    for s in shocks:
        rows.append({
            "Name"        : s.name,
            "Category"    : s.category,
            "Magnitude"   : f"{s.magnitude:+.0%}",
            "Duration yr" : s.duration_yr,
            "Onset yr"    : s.onset_yr,
            "Decay"       : s.decay,
            "Scope"       : s.scope,
            "Perm. Scar"  : f"{s.permanent_scar:.0%}",
            "Description" : s.description,
        })
    return pd.DataFrame(rows)
