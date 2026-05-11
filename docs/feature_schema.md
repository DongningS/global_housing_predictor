# Feature Schema — Global Housing Price Predictor

All 60+ features used in the model, grouped by domain.

## Property Intrinsic Features

| Feature | Type | Justification |
|---|---|---|
| `area_sqm` | float | Primary price driver — log-transformed |
| `log_area` | float | Linearises price-size relationship |
| `log_area_sq` | float | Captures diminishing returns at large sizes |
| `bedrooms` | int | Room count proxy for family demand |
| `living_rooms` | int | Living space utility |
| `bathrooms` | int | Luxury proxy |
| `floor` | int | Raw floor number |
| `total_floors` | int | Building height |
| `floor_ratio` | float | Relative height (avoids collinearity) |
| `is_high_floor` | bool | Premium above 65th percentile height |
| `is_penthouse` | bool | Top 10% — premium or discount depending on market |
| `age_years` | float | Physical depreciation |
| `log_age` | float | Log-transform of age (diminishing depreciation) |
| `is_new` | bool | New (<= 3yr) — developer premium |
| `is_very_old` | bool | Old (>= 25yr) — structural risk discount |
| `has_elevator` | bool | Critical for mid/high-rise buildings |
| `has_parking` | bool | Urban markets: increasingly scarce |
| `plot_ratio` | float | 容积率 — lower = more spacious = premium |
| `green_ratio` | float | Compound greenery — lifestyle premium |
| `management_fee` | float | Quality signal for compound management |
| `floor_category` | cat | Low/Mid/High/Top label |
| `orientation` | cat | South-facing premium in China/Japan |
| `decoration` | cat | Fit-out level |
| `property_type` | cat | Residential/Apartment/Villa/Co-op etc. |

## Location Features

| Feature | Type | Justification |
|---|---|---|
| `latitude` | float | Direct geospatial signal |
| `longitude` | float | Direct geospatial signal |
| `district` | cat | Administrative area — target-encoded |
| `dist_city_center_km` | float | Access to CBD — exponential decay |
| `log_dist_subway_m` | float | Transit access — log for diminishing returns |
| `subway_lines` | int | Interchange premium |
| `school_quality` | float | School district premium (学区房) |
| `log_dist_school_m` | float | Proximity to school |
| `education_score` | float | Quality × proximity composite |
| `log_dist_hospital_m` | float | Healthcare access |
| `log_dist_mall_m` | float | Retail convenience |
| `log_dist_park_m` | float | Green space access |
| `in_free_trade_zone` | bool | FTZ policy premium (Shanghai, Dubai) |
| `accessibility_score` | float | Composite: subway + city center access |

## Comparable Sales (New in v2)

| Feature | Type | Justification |
|---|---|---|
| `comparable_median_price` | float | **Strongest feature** — KNN median of nearest 20 listings |
| `price_vs_comparable` | float | Relative to comparables — over/underpricing signal |

## District Statistics (New in v2)

| Feature | Type | Justification |
|---|---|---|
| `district_mean_price` | float | District average — market context |
| `district_median_price` | float | Robust central tendency |
| `district_relative_mean` | float | This district vs city average |
| `district_price_std` | float | Price dispersion — heterogeneity signal |
| `district_p25_price` | float | Lower quartile anchor |
| `district_p75_price` | float | Upper quartile anchor |

## Target-Encoded Categoricals (New in v2)

| Feature | Type | Justification |
|---|---|---|
| `district_te` | float | LOO mean target encoding — no data leakage |
| `floor_category_te` | float | Floor tier price signal |
| `orientation_te` | float | Orientation premium encoded in price |
| `decoration_te` | float | Fit-out level price signal |
| `property_type_te` | float | Type-level price signal |

## Macro-Economic Features

| Feature | Type | Justification |
|---|---|---|
| `lpr_5yr` | float | 5-year LPR — mortgage benchmark (China) |
| `gdp_growth_yoy` | float | Economic growth proxy |
| `cpi_yoy` | float | Inflation — real asset demand |
| `m2_growth_yoy` | float | Liquidity driver |
| `policy_restriction` | int | Purchase restriction score 0–5 |
| `macro_momentum` | float | Composite: GDP − CPI + 0.3×M2 |
| `baidu_search_idx` | float | Search interest proxy for demand |
| `social_sentiment` | float | Social media sentiment −1 to +1 |
| `school_quality` | float | Educational quality score |

## Temporal Features

| Feature | Type | Justification |
|---|---|---|
| `year` | int | Time trend |
| `quarter` | int | Seasonal quarter |
| `month_sin` | float | Cyclical month encoding |
| `month_cos` | float | Cyclical month encoding |
| `quarter_sin` | float | Cyclical quarter encoding |
| `quarter_cos` | float | Cyclical quarter encoding |
| `is_golden_week` | bool | October Golden Week — China volume spike |

---

## Shock Aspect Taxonomy

The shock system affects 15 price aspects:

| Aspect | Description |
|---|---|
| `buyer_demand` | Overall buyer demand |
| `foreign_buyer_demand` | Overseas investor demand |
| `investor_demand` | Domestic investment demand |
| `end_user_demand` | Owner-occupier demand |
| `rental_demand` | Rental yield attractiveness |
| `new_supply` | New construction pipeline |
| `land_supply` | Available land |
| `mortgage_access` | Credit availability |
| `interest_cost` | Borrowing cost |
| `investor_leverage` | Leverage available |
| `gdp_sentiment` | Economic growth sentiment |
| `employment` | Labour market / income |
| `currency_value` | Local currency strength |
| `inflation_hedge` | Property as inflation hedge |
| `purchase_restriction` | Govt restriction intensity |
| `tax_burden` | Transaction / holding tax |
| `physical_risk` | Structural / natural hazard |
| `political_risk` | Geopolitical / governance |
| `liquidity_risk` | Transaction volume / liquidity |
| `legal_title_risk` | Property rights security |
