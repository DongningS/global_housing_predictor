"""
City Registry
=============
Each city entry defines:
  - Geographic center & scrape targets
  - Local feature set (which features matter in THIS jurisdiction)
  - Local macro drivers
  - Currency / price unit
  - Regulatory environment type
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class CityConfig:
    name: str
    name_local: str
    center: Tuple[float, float]          # (lat, lon)
    country: str
    currency: str
    price_unit: str                       # e.g. "CNY/m²", "USD/sqft"
    sqft_market: bool                     # True = sqft, False = m²

    # Scraping
    scrape_urls: List[str]
    scrape_sources: List[str]

    # Feature toggles — which local features apply
    has_hukou_restriction: bool = False   # China-specific
    has_school_district_premium: bool = True
    has_leasehold_risk: bool = False      # Singapore, HK, UK leaseholds
    has_strata_title: bool = False        # SG, AU, MY
    has_homestead_law: bool = False       # USA
    has_mansion_tax: bool = False         # NYC, London
    has_foreign_buyer_tax: bool = False   # SG, CA, AU, NZ
    has_stamp_duty: bool = False          # UK, HK, SG, AU
    has_property_tax: bool = True
    has_hoa_fee: bool = False             # USA
    has_management_fee: bool = True       # CN, HK, SG
    has_ftz_premium: bool = False         # Shanghai FTZ, Dubai freezones
    has_flood_risk: bool = False          # coastal/low-lying
    has_earthquake_risk: bool = False     # Tokyo, Istanbul, SFO
    has_typhoon_risk: bool = False        # HK, Shanghai, Tokyo
    has_rent_control: bool = False        # NYC, Berlin, SF
    has_vacancy_tax: bool = False         # Vancouver, HK

    # Local macro drivers that matter most
    key_macro_drivers: List[str] = field(default_factory=lambda: [
        "gdp_growth", "cpi", "policy_rate", "unemployment"
    ])

    # District/submarket names
    districts: List[str] = field(default_factory=list)

    # Notes for analysts
    analyst_notes: str = ""


CITY_REGISTRY: Dict[str, CityConfig] = {

    # ── CHINA ────────────────────────────────────────────────────────────────
    "Shanghai": CityConfig(
        name="Shanghai", name_local="上海",
        center=(31.2304, 121.4737), country="China",
        currency="CNY", price_unit="CNY/m²", sqft_market=False,
        scrape_urls=["https://sh.ke.com/ershoufang/", "https://shanghai.anjuke.com/sale/"],
        scrape_sources=["beike", "anjuke", "lianjia"],
        has_hukou_restriction=True, has_school_district_premium=True,
        has_ftz_premium=True, has_management_fee=True,
        has_flood_risk=True, has_typhoon_risk=True,
        key_macro_drivers=["gdp_growth", "m2_growth", "lpr_5yr", "cpi",
                           "land_supply", "population_net_inflow", "policy_restriction"],
        districts=["浦东新区","黄浦区","徐汇区","长宁区","静安区","普陀区",
                   "虹口区","杨浦区","闵行区","宝山区","嘉定区","松江区",
                   "青浦区","奉贤区","金山区","崇明区"],
        analyst_notes="购房限制(限购/限贷)是最大政策变量。学区房溢价显著。"
    ),

    "Beijing": CityConfig(
        name="Beijing", name_local="北京",
        center=(39.9042, 116.4074), country="China",
        currency="CNY", price_unit="CNY/m²", sqft_market=False,
        scrape_urls=["https://bj.ke.com/ershoufang/"],
        scrape_sources=["beike", "anjuke"],
        has_hukou_restriction=True, has_school_district_premium=True,
        has_management_fee=True,
        key_macro_drivers=["gdp_growth", "m2_growth", "lpr_5yr", "cpi",
                           "land_supply", "population_net_inflow", "policy_restriction"],
        districts=["朝阳区","海淀区","西城区","东城区","丰台区","石景山区",
                   "通州区","昌平区","大兴区","顺义区","房山区","门头沟区"],
        analyst_notes="政治中心溢价显著。中关村科技园区影响海淀房价。"
    ),

    "Shenzhen": CityConfig(
        name="Shenzhen", name_local="深圳",
        center=(22.5431, 114.0579), country="China",
        currency="CNY", price_unit="CNY/m²", sqft_market=False,
        scrape_urls=["https://sz.ke.com/ershoufang/"],
        scrape_sources=["beike"],
        has_hukou_restriction=True, has_school_district_premium=True,
        has_ftz_premium=True, has_management_fee=True,
        key_macro_drivers=["gdp_growth", "m2_growth", "lpr_5yr", "cpi",
                           "tech_sector_growth", "hk_spillover", "policy_restriction"],
        districts=["南山区","福田区","罗湖区","盐田区","宝安区",
                   "龙岗区","龙华区","坪山区","光明区","大鹏新区"],
        analyst_notes="科技行业景气度是关键驱动。与香港跨境效应明显。"
    ),

    "Guangzhou": CityConfig(
        name="Guangzhou", name_local="广州",
        center=(23.1291, 113.2644), country="China",
        currency="CNY", price_unit="CNY/m²", sqft_market=False,
        scrape_urls=["https://gz.ke.com/ershoufang/"],
        scrape_sources=["beike"],
        has_hukou_restriction=True, has_school_district_premium=True,
        has_management_fee=True, has_flood_risk=True, has_typhoon_risk=True,
        key_macro_drivers=["gdp_growth", "m2_growth", "lpr_5yr", "cpi",
                           "trade_volume", "policy_restriction"],
        districts=["天河区","越秀区","荔湾区","海珠区","白云区",
                   "黄埔区","番禺区","花都区","增城区","从化区"],
        analyst_notes="大湾区一体化政策对广州北部影响大。"
    ),

    # ── HONG KONG ─────────────────────────────────────────────────────────────
    "Hong Kong": CityConfig(
        name="Hong Kong", name_local="香港",
        center=(22.3193, 114.1694), country="Hong Kong SAR",
        currency="HKD", price_unit="HKD/sqft", sqft_market=True,
        scrape_urls=["https://www.midland.com.hk/", "https://www.centaline.com.hk/"],
        scrape_sources=["midland", "centaline", "squarefoot"],
        has_leasehold_risk=True, has_stamp_duty=True,
        has_foreign_buyer_tax=True, has_management_fee=True,
        has_school_district_premium=True, has_vacancy_tax=True,
        has_flood_risk=True, has_typhoon_risk=True,
        key_macro_drivers=["hibor", "usd_hkd_peg", "china_gdp",
                           "capital_flows", "emigration_rate", "policy_restriction"],
        districts=["Central","Wan Chai","Causeway Bay","North Point",
                   "Kowloon City","Mong Kok","Tsim Sha Tsui","Sha Tin",
                   "Tuen Mun","Yuen Long","Lantau Island","Clear Water Bay"],
        analyst_notes="USD peg means rate sensitivity = Fed policy. "
                      "Political risk premium since 2019. Leasehold land."
    ),

    # ── SINGAPORE ─────────────────────────────────────────────────────────────
    "Singapore": CityConfig(
        name="Singapore", name_local="新加坡",
        center=(1.3521, 103.8198), country="Singapore",
        currency="SGD", price_unit="SGD/sqft", sqft_market=True,
        scrape_urls=["https://www.propertyguru.com.sg/", "https://www.srx.com.sg/"],
        scrape_sources=["propertyguru", "srx", "edgeprop"],
        has_leasehold_risk=True, has_strata_title=True,
        has_foreign_buyer_tax=True, has_stamp_duty=True,
        has_school_district_premium=True, has_management_fee=True,
        has_flood_risk=True,
        key_macro_drivers=["sibor", "gdp_growth", "population_growth",
                           "absd_rate", "supply_pipeline", "rental_yield"],
        districts=["District 1-4 (CBD)","District 9-11 (Orchard/Newton)",
                   "District 15 (East Coast)","District 19 (Serangoon)",
                   "District 23 (Bukit Panjang)","District 28 (Seletar)"],
        analyst_notes="ABSD (Additional Buyer Stamp Duty) is key policy lever. "
                      "99yr vs freehold leasehold split is critical. "
                      "HDB resale market separate from private condos."
    ),

    # ── JAPAN ─────────────────────────────────────────────────────────────────
    "Tokyo": CityConfig(
        name="Tokyo", name_local="東京",
        center=(35.6762, 139.6503), country="Japan",
        currency="JPY", price_unit="JPY/m²", sqft_market=False,
        scrape_urls=["https://suumo.jp/jj/bukken/ichiran/",
                     "https://www.homes.co.jp/mansion/"],
        scrape_sources=["suumo", "homes", "athome"],
        has_earthquake_risk=True, has_property_tax=True,
        has_school_district_premium=True,
        key_macro_drivers=["boj_policy_rate", "yen_exchange_rate", "gdp_growth",
                           "population_aging", "inbound_tourism", "supply_pipeline"],
        districts=["千代田区","中央区","港区","新宿区","渋谷区","文京区",
                   "台東区","墨田区","江東区","品川区","目黒区","大田区",
                   "世田谷区","中野区","杉並区","豊島区","北区","足立区"],
        analyst_notes="Depreciation culture: buildings lose value, land appreciates. "
                      "BOJ yield curve control ended 2024 — big rate risk. "
                      "Inbound foreign investment surging with weak yen."
    ),

    "Osaka": CityConfig(
        name="Osaka", name_local="大阪",
        center=(34.6937, 135.5023), country="Japan",
        currency="JPY", price_unit="JPY/m²", sqft_market=False,
        scrape_urls=["https://suumo.jp/jj/bukken/ichiran/"],
        scrape_sources=["suumo", "homes"],
        has_earthquake_risk=True, has_property_tax=True,
        key_macro_drivers=["boj_policy_rate", "yen_exchange_rate",
                           "ir_casino_development", "inbound_tourism"],
        districts=["北区","中央区","西区","浪速区","天王寺区",
                   "住之江区","住吉区","東住吉区","平野区"],
        analyst_notes="IR (Integrated Resort/Casino) development is a unique local driver. "
                      "Expo 2025 legacy effects."
    ),

    # ── SOUTH KOREA ───────────────────────────────────────────────────────────
    "Seoul": CityConfig(
        name="Seoul", name_local="서울",
        center=(37.5665, 126.9780), country="South Korea",
        currency="KRW", price_unit="KRW/m²", sqft_market=False,
        scrape_urls=["https://www.zigbang.com/", "https://www.dabang.com/"],
        scrape_sources=["zigbang", "dabang", "naver_realestate"],
        has_school_district_premium=True, has_property_tax=True,
        key_macro_drivers=["bok_policy_rate", "gdp_growth", "jeonse_ratio",
                           "household_debt", "supply_pipeline"],
        districts=["강남구","서초구","송파구","강동구","용산구",
                   "마포구","영등포구","강서구","노원구","은평구"],
        analyst_notes="Jeonse (전세) system is unique — affects rent/buy dynamics. "
                      "강남 3구 (Gangnam, Seocho, Songpa) have extreme premium. "
                      "Government intervention very active (LTV caps, tax surcharges)."
    ),

    # ── UK ────────────────────────────────────────────────────────────────────
    "London": CityConfig(
        name="London", name_local="London",
        center=(51.5074, -0.1278), country="United Kingdom",
        currency="GBP", price_unit="GBP/m²", sqft_market=False,
        scrape_urls=["https://www.rightmove.co.uk/", "https://www.zoopla.co.uk/"],
        scrape_sources=["rightmove", "zoopla", "onthemarket"],
        has_leasehold_risk=True, has_stamp_duty=True,
        has_foreign_buyer_tax=True, has_mansion_tax=True,
        has_school_district_premium=True, has_rent_control=False,
        key_macro_drivers=["boe_rate", "gdp_growth", "cpi", "sterling_fx",
                           "brexit_impact", "supply_pipeline", "help_to_buy"],
        districts=["City of London","Westminster","Kensington & Chelsea","Camden",
                   "Islington","Hackney","Tower Hamlets","Southwark","Lambeth",
                   "Wandsworth","Greenwich","Lewisham","Hammersmith","Ealing"],
        analyst_notes="Leasehold vs freehold is critical pricing factor. "
                      "Brexit reduced EU buyer demand. Stamp Duty Land Tax (SDLT) "
                      "changes cause sharp transaction volume swings."
    ),

    # ── USA ───────────────────────────────────────────────────────────────────
    "New York": CityConfig(
        name="New York", name_local="New York City",
        center=(40.7128, -74.0060), country="USA",
        currency="USD", price_unit="USD/sqft", sqft_market=True,
        scrape_urls=["https://www.zillow.com/new-york-ny/",
                     "https://streeteasy.com/"],
        scrape_sources=["zillow", "streeteasy", "realtor"],
        has_mansion_tax=True, has_school_district_premium=True,
        has_hoa_fee=True, has_homestead_law=False,
        has_rent_control=True, has_property_tax=True,
        key_macro_drivers=["fed_rate", "gdp_growth", "wall_street_bonus",
                           "unemployment", "nyc_migration_balance", "supply_pipeline"],
        districts=["Manhattan","Brooklyn","Queens","Bronx","Staten Island",
                   "Upper East Side","Upper West Side","Tribeca","SoHo",
                   "Greenwich Village","Williamsburg","Astoria","Long Island City"],
        analyst_notes="Co-op vs condo is unique NYC distinction (co-ops ~75% of market). "
                      "Mansion tax: 1-3.9% on sales over $1M. "
                      "421-a tax abatement expiry affects new development supply."
    ),

    "Los Angeles": CityConfig(
        name="Los Angeles", name_local="Los Angeles",
        center=(34.0522, -118.2437), country="USA",
        currency="USD", price_unit="USD/sqft", sqft_market=True,
        scrape_urls=["https://www.zillow.com/los-angeles-ca/"],
        scrape_sources=["zillow", "redfin", "realtor"],
        has_earthquake_risk=True, has_hoa_fee=True,
        has_homestead_law=True, has_school_district_premium=True,
        has_property_tax=True, has_rent_control=True,
        key_macro_drivers=["fed_rate", "ca_gdp_growth", "tech_sector",
                           "entertainment_industry", "wildfire_insurance"],
        districts=["Beverly Hills","Santa Monica","Venice","West Hollywood",
                   "Silver Lake","Los Feliz","Pasadena","Glendale",
                   "Long Beach","Downtown LA","Culver City","Bel Air"],
        analyst_notes="Prop 13 caps property tax increases — creates lock-in effect. "
                      "Wildfire risk increasing insurance costs dramatically. "
                      "Tech/entertainment sector drives high-end demand."
    ),

    "San Francisco": CityConfig(
        name="San Francisco", name_local="San Francisco",
        center=(37.7749, -122.4194), country="USA",
        currency="USD", price_unit="USD/sqft", sqft_market=True,
        scrape_urls=["https://www.zillow.com/san-francisco-ca/"],
        scrape_sources=["zillow", "redfin"],
        has_earthquake_risk=True, has_hoa_fee=True,
        has_homestead_law=True, has_school_district_premium=True,
        has_rent_control=True, has_property_tax=True,
        key_macro_drivers=["fed_rate", "tech_sector_growth", "vc_funding",
                           "remote_work_rate", "homelessness_index"],
        districts=["Pacific Heights","Nob Hill","Mission","Castro","SoMa",
                   "Marina","Richmond","Sunset","Noe Valley","Haight-Ashbury"],
        analyst_notes="Tech sector is primary driver. Remote work caused 20% price drop 2022-23. "
                      "Prop 13 + rent control creates severe supply constraint."
    ),

    # ── AUSTRALIA ─────────────────────────────────────────────────────────────
    "Sydney": CityConfig(
        name="Sydney", name_local="Sydney",
        center=(-33.8688, 151.2093), country="Australia",
        currency="AUD", price_unit="AUD/m²", sqft_market=False,
        scrape_urls=["https://www.realestate.com.au/buy/in-sydney/",
                     "https://www.domain.com.au/"],
        scrape_sources=["realestate", "domain"],
        has_strata_title=True, has_foreign_buyer_tax=True,
        has_stamp_duty=True, has_school_district_premium=True,
        has_flood_risk=True, has_property_tax=True,
        key_macro_drivers=["rba_rate", "gdp_growth", "immigration_rate",
                           "chinese_buyer_demand", "supply_pipeline", "rental_yield"],
        districts=["CBD","North Shore","Eastern Suburbs","Inner West",
                   "Hills District","Northern Beaches","Western Sydney",
                   "South Sydney","Parramatta","Blacktown"],
        analyst_notes="Chinese buyer demand is key variable (FIRB restrictions). "
                      "Immigration-driven demand surge post-COVID. "
                      "Negative gearing tax policy affects investor demand."
    ),

    "Melbourne": CityConfig(
        name="Melbourne", name_local="Melbourne",
        center=(-37.8136, 144.9631), country="Australia",
        currency="AUD", price_unit="AUD/m²", sqft_market=False,
        scrape_urls=["https://www.realestate.com.au/buy/in-melbourne/"],
        scrape_sources=["realestate", "domain"],
        has_strata_title=True, has_foreign_buyer_tax=True,
        has_stamp_duty=True, has_school_district_premium=True,
        has_property_tax=True, has_vacancy_tax=True,
        key_macro_drivers=["rba_rate", "gdp_growth", "immigration_rate",
                           "supply_pipeline", "rental_yield", "land_tax"],
        districts=["CBD","South Yarra","Toorak","Richmond","Fitzroy",
                   "Brunswick","St Kilda","Brighton","Box Hill","Doncaster"],
        analyst_notes="International student population drives inner-city apartment demand. "
                      "Land tax applies to investment properties — key holding cost."
    ),

    # ── UAE ───────────────────────────────────────────────────────────────────
    "Dubai": CityConfig(
        name="Dubai", name_local="دبي",
        center=(25.2048, 55.2708), country="UAE",
        currency="AED", price_unit="AED/sqft", sqft_market=True,
        scrape_urls=["https://www.propertyfinder.ae/",
                     "https://www.bayut.com/"],
        scrape_sources=["propertyfinder", "bayut"],
        has_ftz_premium=True, has_foreign_buyer_tax=False,
        has_property_tax=False, has_school_district_premium=False,
        has_flood_risk=False,
        key_macro_drivers=["oil_price", "usd_aed_peg", "tourism_growth",
                           "russian_capital_inflow", "visa_policy",
                           "supply_pipeline", "rental_yield"],
        districts=["Downtown Dubai","Dubai Marina","Palm Jumeirah",
                   "Business Bay","Jumeirah","DIFC","Dubai Hills",
                   "Arabian Ranches","JVC","Al Barsha"],
        analyst_notes="No property tax / income tax = pure yield play. "
                      "Oil price → govt spending → infrastructure. "
                      "Russian/Iranian capital inflow since 2022 sanctions. "
                      "Golden Visa program drives demand. Massive supply pipeline."
    ),

    # ── FRANCE ────────────────────────────────────────────────────────────────
    "Paris": CityConfig(
        name="Paris", name_local="Paris",
        center=(48.8566, 2.3522), country="France",
        currency="EUR", price_unit="EUR/m²", sqft_market=False,
        scrape_urls=["https://www.seloger.com/", "https://www.leboncoin.fr/"],
        scrape_sources=["seloger", "pap", "leboncoin"],
        has_rent_control=True, has_stamp_duty=True,
        has_school_district_premium=True, has_property_tax=True,
        key_macro_drivers=["ecb_rate", "french_gdp", "cpi",
                           "wealth_tax", "golden_visa", "tourism"],
        districts=["1er","2e","3e","4e","5e","6e","7e","8e","9e","10e",
                   "11e","12e","13e","14e","15e","16e","17e","18e","19e","20e"],
        analyst_notes="Arrondissement number is key — lower = premium. "
                      "Rent control (encadrement des loyers) limits yields. "
                      "IFI (Impôt sur la Fortune Immobilière) wealth tax on >€1.3M."
    ),

    # ── CANADA ────────────────────────────────────────────────────────────────
    "Toronto": CityConfig(
        name="Toronto", name_local="Toronto",
        center=(43.6532, -79.3832), country="Canada",
        currency="CAD", price_unit="CAD/sqft", sqft_market=True,
        scrape_urls=["https://www.realtor.ca/", "https://www.zolo.ca/"],
        scrape_sources=["realtor", "zolo", "remax"],
        has_foreign_buyer_tax=True, has_stamp_duty=True,
        has_school_district_premium=True, has_property_tax=True,
        has_vacancy_tax=True,
        key_macro_drivers=["boc_rate", "immigration_rate", "gdp_growth",
                           "foreign_buyer_ban", "rental_yield", "condo_supply"],
        districts=["Downtown Core","Midtown","North York","Scarborough",
                   "Etobicoke","East York","York","The Beaches",
                   "Annex","Leslieville","Liberty Village","King West"],
        analyst_notes="Foreign Buyer Ban (2023-) impacts high-end segment. "
                      "Immigration target 500k/yr = structural demand. "
                      "Condo investor market: 30%+ units investor-owned."
    ),

    "Vancouver": CityConfig(
        name="Vancouver", name_local="Vancouver",
        center=(49.2827, -123.1207), country="Canada",
        currency="CAD", price_unit="CAD/sqft", sqft_market=True,
        scrape_urls=["https://www.realtor.ca/"],
        scrape_sources=["realtor", "zolo"],
        has_foreign_buyer_tax=True, has_stamp_duty=True,
        has_school_district_premium=True, has_property_tax=True,
        has_vacancy_tax=True, has_earthquake_risk=True,
        key_macro_drivers=["boc_rate", "chinese_buyer_demand",
                           "foreign_buyer_tax", "immigration_rate", "rental_yield"],
        districts=["Vancouver West","Vancouver East","North Vancouver",
                   "West Vancouver","Richmond","Burnaby","Surrey",
                   "Coquitlam","Langley","White Rock"],
        analyst_notes="Most Chinese-buyer-sensitive market in Canada. "
                      "Empty Homes Tax + Spec & Vacancy Tax layers on. "
                      "Earthquake risk (Cascadia subduction zone) underpriced."
    ),

    # ── GERMANY ───────────────────────────────────────────────────────────────
    "Berlin": CityConfig(
        name="Berlin", name_local="Berlin",
        center=(52.5200, 13.4050), country="Germany",
        currency="EUR", price_unit="EUR/m²", sqft_market=False,
        scrape_urls=["https://www.immobilienscout24.de/",
                     "https://www.immowelt.de/"],
        scrape_sources=["immoscout24", "immowelt"],
        has_rent_control=True, has_stamp_duty=True,
        has_property_tax=True, has_school_district_premium=False,
        key_macro_drivers=["ecb_rate", "german_gdp", "cpi",
                           "rental_yield", "migration_balance", "supply_pipeline"],
        districts=["Mitte","Prenzlauer Berg","Friedrichshain","Kreuzberg",
                   "Charlottenburg","Wilmersdorf","Schöneberg","Tempelhof",
                   "Steglitz","Zehlendorf","Spandau","Reinickendorf"],
        analyst_notes="Mietpreisbremse (rent brake) suppresses yields. "
                      "Historically renter culture — homeownership rate ~17% in Berlin. "
                      "High supply pipeline is bearish for prices."
    ),

    # ── INDIA ─────────────────────────────────────────────────────────────────
    "Mumbai": CityConfig(
        name="Mumbai", name_local="मुंबई",
        center=(19.0760, 72.8777), country="India",
        currency="INR", price_unit="INR/sqft", sqft_market=True,
        scrape_urls=["https://www.magicbricks.com/",
                     "https://www.99acres.com/"],
        scrape_sources=["magicbricks", "99acres", "housing"],
        has_stamp_duty=True, has_school_district_premium=True,
        has_property_tax=True, has_flood_risk=True,
        key_macro_drivers=["rbi_rate", "india_gdp", "finance_sector",
                           "infrastructure_spend", "monsoon_risk", "fsi_regulation"],
        districts=["South Mumbai","Bandra","Andheri","Juhu","Powai",
                   "Navi Mumbai","Thane","Borivali","Kandivali","Malad"],
        analyst_notes="FSI (Floor Space Index) regulations are key supply constraint. "
                      "Stamp duty is 5-6% — big transaction cost. "
                      "Monsoon flooding risk is underpriced in some areas."
    ),

    # ── SOUTHEAST ASIA ────────────────────────────────────────────────────────
    "Bangkok": CityConfig(
        name="Bangkok", name_local="กรุงเทพ",
        center=(13.7563, 100.5018), country="Thailand",
        currency="THB", price_unit="THB/m²", sqft_market=False,
        scrape_urls=["https://www.ddproperty.com/",
                     "https://www.hipflat.co.th/"],
        scrape_sources=["ddproperty", "hipflat"],
        has_property_tax=True, has_flood_risk=True,
        has_school_district_premium=False,
        key_macro_drivers=["bot_rate", "tourism_recovery", "thai_gdp",
                           "chinese_buyer_demand", "supply_pipeline"],
        districts=["Sukhumvit","Silom","Sathorn","Ratchada","Ladprao",
                   "Chatuchak","Phra Khanong","On Nut","Bang Na"],
        analyst_notes="Foreigners cannot own land — only condominiums up to 49% foreign quota. "
                      "Tourism recovery is key driver of condo demand. "
                      "2011 flood risk still affects low-lying areas."
    ),
}


def get_city(name: str) -> CityConfig:
    if name not in CITY_REGISTRY:
        avail = list(CITY_REGISTRY.keys())
        raise KeyError(f"City '{name}' not found. Available: {avail}")
    return CITY_REGISTRY[name]


def list_cities() -> List[str]:
    return sorted(CITY_REGISTRY.keys())


def cities_by_country() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, cfg in CITY_REGISTRY.items():
        out.setdefault(cfg.country, []).append(name)
    return out
