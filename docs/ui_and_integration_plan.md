# UI & Real-Agent Integration — Planning Document

*This is a planning document only. No implementation has been built yet.*
*All design decisions here are proposals for the next development phase.*

---

## 1. Recommended Application Stack

### Why not a `.bat` file UI?
A Windows `.bat` launcher can start a local web server, but the interface itself
should be a proper web application. The recommended stack:

```
Streamlit   (Python-native, fastest to prototype, no JS required)
    ↓ runs via
launcher.bat  (Windows) / launcher.sh  (Mac/Linux)
    ↓ which calls
uvicorn + FastAPI  (REST API backend, separate process)
    ↓ backed by
SQLite  (local prediction cache + audit trail)
```

**Alternative for a richer UI**: Dash (Plotly's framework) gives more control
over layout and supports real-time callbacks without page reloads.
**Alternative for distribution**: package as an Electron app so non-technical
users get a desktop icon without needing Python installed.

---

## 2. Launcher Design (`.bat` / `.sh`)

```batch
:: launcher.bat  — Windows
@echo off
title Global Housing Price Predictor

:: Check Python
python --version >nul 2>&1 || (
    echo Python not found. Please install Python 3.10+ from python.org
    pause & exit /b 1
)

:: Install/update dependencies silently
pip install -r requirements.txt -q --no-warn-script-location

:: Start FastAPI backend in background
start /B python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765 --reload

:: Wait for backend to be ready
timeout /t 3 /nobreak >nul

:: Open Streamlit frontend (opens browser automatically)
streamlit run app/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

pause
```

```bash
#!/bin/bash
# launcher.sh  — Mac / Linux
set -e
echo "Starting Global Housing Price Predictor..."

pip install -r requirements.txt -q
uvicorn src.api.main:app --host 127.0.0.1 --port 8765 --reload &
sleep 2
streamlit run app/streamlit_app.py --server.port 8501
```

---

## 3. Proposed Screen Layout (Streamlit)

```
┌─────────────────────────────────────────────────────────────────┐
│  🏙️ Global Housing Price Predictor              [City: Shanghai▼]│
├──────────────┬──────────────────────────────────────────────────┤
│              │  TABS: [Predict] [Forecast] [Shocks] [MRM] [Map] │
│  PROPERTY    ├──────────────────────────────────────────────────┤
│  INPUTS      │                                                  │
│              │  ┌─────────────────┐  ┌──────────────────────┐  │
│  District ▼  │  │ PREDICTED PRICE │  │  RISK BADGE          │  │
│  社区     ▼  │  │  ¥ 127,500/m²  │  │  🟢 LOW RISK (24/100)│  │
│  楼盘     ▼  │  │  ¥ 1,147万 total│  │  Analyst: no review  │  │
│              │  └─────────────────┘  └──────────────────────┘  │
│  Area: 90m²  │                                                  │
│  Floor:  9/18│  ┌──────────────────────────────────────────┐   │
│  Age:    8yr │  │  P10: ¥102k  ██████████████  P90: ¥155k  │   │
│  Bedrooms: 3 │  │  Confidence interval (CQR-calibrated 80%) │   │
│  School: 8.5 │  └──────────────────────────────────────────┘   │
│  Subway: 300m│                                                  │
│              │  [Why this price? ▼ SHAP waterfall]             │
│  [PREDICT]   │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

### Tab: Forecast
- Forecast horizon slider: 1-50 years
- Scenario selector: Bull / Base / Bear / Stagflation / Deflation
- Active shocks: multi-select from SHOCK_LIBRARY
- Output: Plotly ribbon chart (P10/P25/median/P75/P90)
- Comparison: baseline vs shocked overlay
- Downloadable CSV of forecast table

### Tab: Shocks
- Shock library browser with search/filter by category
- Custom shock builder form:
  - Name, category, magnitude slider (-100% to +100%)
  - Duration, onset delay, decay shape selector
  - Aspect weight sliders (demand, supply, financing, etc.)
- Applied shocks list with remove buttons
- Shock profile chart (time-series of effect)

### Tab: MRM
- Live model metrics (MAPE, R², calibration coverage)
- Drift alerts (PSI for all features)
- Regime gauge (bubble/normal/correction/crash)
- Override entry form
- Prediction audit log table

### Tab: Map
- Folium heatmap embedded in iframe
- Toggle: current prices / forecast 5yr / forecast 10yr
- Click on district → show district stats panel
- Hover on property → mini price card

---

## 4. Real-Agent Website Integration

### Architecture
```
User selects city + search criteria
        ↓
Scraper module (src/scrapers/<city>_scraper.py)
        ↓ runs in background thread
Stores raw listings in SQLite (data/<city>_raw.db)
        ↓
ETL pipeline normalises to standard schema
        ↓
Feature engineer + model inference
        ↓
Results displayed in UI with source attribution
```

### Scraper Registry by City

| City | Primary Source | Secondary | Notes |
|---|---|---|---|
| Shanghai | [Beike 贝壳](https://sh.ke.com/ershoufang/) | [Anjuke 安居客](https://shanghai.anjuke.com/sale/) | Anti-bot: need proxy rotation |
| Beijing | [Beike](https://bj.ke.com/ershoufang/) | Anjuke | Same Beike format |
| Shenzhen | [Beike](https://sz.ke.com/ershoufang/) | Anjuke | — |
| Guangzhou | [Beike](https://gz.ke.com/ershoufang/) | Anjuke | — |
| Hong Kong | [Midland 美聯](https://www.midland.com.hk/) | [Centaline 中原](https://www.centaline.com.hk/) | HTTPS strict, needs headers |
| Singapore | [PropertyGuru](https://www.propertyguru.com.sg/) | [SRX](https://www.srx.com.sg/) | SRX has public API |
| Tokyo | [Suumo](https://suumo.jp/jj/bukken/ichiran/) | [Homes](https://www.homes.co.jp/) | Japanese encoding UTF-8 |
| Seoul | [Naver 부동산](https://land.naver.com/) | [Zigbang](https://www.zigbang.com/) | Naver has semi-public API |
| London | [Rightmove](https://www.rightmove.co.uk/) | [Zoopla](https://www.zoopla.co.uk/) | Rightmove blocks scrapers aggressively |
| New York | [Zillow](https://www.zillow.com/) | [StreetEasy](https://streeteasy.com/) | Zillow has official API ($) |
| Toronto | [Realtor.ca](https://www.realtor.ca/) | [HouseSigma](https://housesigma.com/) | HouseSigma has rich history data |
| Vancouver | [Realtor.ca](https://www.realtor.ca/) | HouseSigma | Same as Toronto |
| Sydney | [realestate.com.au](https://www.realestate.com.au/) | [Domain](https://www.domain.com.au/) | Domain has public API |
| Melbourne | [realestate.com.au](https://www.realestate.com.au/) | Domain | — |
| Dubai | [PropertyFinder](https://www.propertyfinder.ae/) | [Bayut](https://www.bayut.com/) | Both have structured JSON |
| Paris | [SeLoger](https://www.seloger.com/) | [Leboncoin](https://www.leboncoin.fr/) | SeLoger API available |
| Berlin | [ImmobilienScout24](https://www.immobilienscout24.de/) | [Immowelt](https://www.immowelt.de/) | IS24 has developer API |
| Mumbai | [MagicBricks](https://www.magicbricks.com/) | [99acres](https://www.99acres.com/) | — |

### HouseSigma (Canada) — Special Note
HouseSigma is particularly valuable for Toronto/Vancouver because it shows:
- Full transaction history (sold prices, not just asking prices)
- Days on market
- Price reduction history
- Neighbourhood-level sold price statistics
This makes it the best source for the **repeat-sales index** feature.

### Scraper Design Pattern
```python
# src/scrapers/base_scraper.py (already exists)
class BaseScraper:
    def scrape(self, city, pages) -> pd.DataFrame: ...
    def parse_listing(self, html_element) -> dict: ...
    def normalise(self, raw_df) -> pd.DataFrame: ...

# src/scrapers/beike_scraper.py
class BeikeScraper(BaseScraper):
    FIELD_MAP = {
        "totalPrice": "total_price_wan",
        "unitPrice":  "unit_price",
        "area":       "area_sqm",
        # ...
    }
    
# src/scrapers/housesigma_scraper.py
class HouseSigmaScraper(BaseScraper):
    """Extracts sold history — ideal for repeat-sales index."""
    # HouseSigma uses dynamic React rendering — requires Selenium
```

### Anti-Bot Strategy
Real estate sites aggressively protect their data. Recommended approach:

```python
# Tier 1: Simple HTTP with rotating headers (works for ~50% of sites)
headers = {"User-Agent": random.choice(UA_POOL), "Accept-Language": "zh-CN,zh;q=0.9"}

# Tier 2: Playwright headless browser (works for JS-rendered sites)
from playwright.async_api import async_playwright
async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    
# Tier 3: Residential proxy rotation (for aggressive blockers like Rightmove)
proxies = {"http": f"http://{PROXY_USER}:{PROXY_PASS}@gate.dc.smartproxy.com:10000"}

# Tier 4: Official API (preferred where available)
# Zillow API: https://www.zillow.com/howto/api/
# SRX API:    https://www.srx.com.sg/developer
# Domain API: https://developer.domain.com.au/
```

### Data Freshness Strategy
```
Schedule (via APScheduler or cron):
  Every 6 hours : price updates for active search cities
  Every 24 hours: full listing refresh
  Every week    : model retraining trigger check (drift monitor)
  Every month   : model revalidation report
```

---

## 5. Field Mapping: Website → Standard Schema

Different sites use different field names. The normalisation layer maps everything
to the standard schema defined in `docs/feature_schema.md`.

### Beike → Standard
| Beike Field | Standard Field | Notes |
|---|---|---|
| `totalPrice` | `total_price_wan` | In 万元 (10k CNY) |
| `unitPrice` | `unit_price` | CNY/m² |
| `area` | `area_sqm` | m² |
| `houseType` | `bedrooms` + `living_rooms` | e.g. "3室2厅" → 3, 2 |
| `floor` | `floor` + `total_floors` | e.g. "中层(共18层)" |
| `buildYear` | `age_years` | Compute from current year |
| `positionInfo[0]` | `xiaoqu_name` | Compound name |
| `positionInfo[1]` | `subdistrict` | Sub-district name |
| `tags` | `has_elevator`, `has_parking` | Parse from tag list |

### HouseSigma → Standard
| HouseSigma Field | Standard Field | Notes |
|---|---|---|
| `sold_price` | `unit_price` (computed) | Divide by sqft → CNY equiv |
| `address` | `xiaoqu_name` | Building/street |
| `neighbourhood` | `subdistrict` | Community area |
| `sold_date` | `year` + `month` | Transaction date |
| `bedrooms` | `bedrooms` | Direct |
| `baths` | `bathrooms` | Direct |
| `sqft` | `area_sqm` | × 0.0929 |

### Zillow → Standard
| Zillow Field | Standard Field | Notes |
|---|---|---|
| `price` | `unit_price` (computed) | ÷ sqft |
| `bedrooms` | `bedrooms` | Direct |
| `bathrooms` | `bathrooms` | Direct |
| `livingArea` | `area_sqm` | sqft × 0.0929 |
| `yearBuilt` | `age_years` | Compute |
| `homeType` | `property_type` | Normalise to standard types |
| `city` + `zipcode` | `subdistrict` | Zip → neighbourhood lookup |

---

## 6. Proposed File Structure for App Layer

```
global-housing-predictor/
├── app/
│   ├── streamlit_app.py          # Main Streamlit UI
│   ├── pages/
│   │   ├── 01_predict.py         # Point estimate + SHAP
│   │   ├── 02_forecast.py        # Long-term forecast + shocks
│   │   ├── 03_shocks.py          # Shock builder UI
│   │   ├── 04_mrm.py             # MRM dashboard
│   │   └── 05_map.py             # Price heatmap
│   └── components/
│       ├── property_form.py      # Reusable property input form
│       ├── forecast_chart.py     # Plotly ribbon chart component
│       └── shock_builder.py      # Shock configuration UI
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   ├── routes/
│   │   │   ├── predict.py        # POST /predict
│   │   │   ├── forecast.py       # POST /forecast
│   │   │   └── shocks.py         # GET /shocks, POST /shocks/custom
│   │   └── schemas.py            # Pydantic request/response models
│   └── scrapers/
│       ├── base_scraper.py       # Abstract base
│       ├── beike_scraper.py      # China cities
│       ├── housesigma_scraper.py # Canada
│       ├── rightmove_scraper.py  # UK
│       ├── zillow_scraper.py     # USA
│       └── scheduler.py          # APScheduler data refresh
├── launcher.bat                  # Windows one-click start
├── launcher.sh                   # Mac/Linux one-click start
└── docker-compose.yml            # Container deployment option
```

---

## 7. Prioritised Build Order

| Priority | Component | Effort | Value |
|---|---|---|---|
| P0 | `launcher.bat` / `launcher.sh` | 1 day | Immediate usability |
| P0 | FastAPI backend (`/predict` endpoint) | 2 days | Powers all UI |
| P1 | Streamlit predict tab | 3 days | Core user value |
| P1 | Beike scraper (Shanghai live data) | 3 days | Real data for CN cities |
| P2 | Streamlit forecast + shocks tab | 2 days | Key differentiator |
| P2 | HouseSigma scraper (Toronto/Vancouver) | 2 days | Best CA data source |
| P2 | MRM dashboard tab | 2 days | Governance requirement |
| P3 | Zillow scraper (NYC/LA/SF) | 3 days | US expansion |
| P3 | Rightmove scraper (London) | 3 days | Needs Playwright |
| P3 | Docker containerisation | 1 day | Easy deployment |
| P4 | Electron desktop app | 5 days | Non-technical distribution |

---

## 8. Legal & Terms of Service Considerations

| Source | ToS Scraping Clause | Recommended Approach |
|---|---|---|
| Beike / Lianjia | Prohibited without agreement | Contact for data partnership or API access |
| Rightmove | Explicitly prohibited | Use official data product or automated valuation partner |
| Zillow | Prohibited; API available | Use Zillow API (paid, $0.01/call) |
| StreetEasy | Prohibited | StreetEasy Pro API |
| PropertyGuru | API available | Officially request API key |
| SRX | Public API available | Free tier available |
| Domain.com.au | Official API | Free developer tier |
| ImmobilienScout24 | Official API | Developer portal |
| Realtor.ca | No scraping; CREA DDF | Contact Canadian Real Estate Association |
| HouseSigma | No explicit API | Proceed carefully; attribution required |

**Recommendation**: For production use, engage data partnership agreements.
For research/demonstration, use the synthetic data generator.
The scraper code is provided for educational purposes and should only be run
against sites where you have explicit permission or a data agreement.

---

*This document describes planned features only. Implementation estimates are for
a single full-stack developer. No production deployment should occur without
compliance review of applicable terms of service.*
