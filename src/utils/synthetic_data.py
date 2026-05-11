"""
Synthetic Data Generator
=========================
Generates statistically realistic property listings for any registered city.
Calibrated to known market data. Used as fallback when live scraping fails.
"""

import numpy as np
import pandas as pd
from typing import Optional


# City-specific calibration parameters
CITY_CALIBRATION = {
    "Shanghai": dict(
        base_prices={
            "黄浦区":120000,"静安区":115000,"徐汇区":108000,"长宁区":100000,
            "虹口区":90000,"杨浦区":85000,"浦东新区":95000,"普陀区":82000,
            "闵行区":72000,"宝山区":60000,"嘉定区":55000,"松江区":50000,
            "青浦区":48000,"奉贤区":42000,"金山区":38000,"崇明区":32000,
        },
        mean_area=85.0, area_sigma=0.40,
        bedrooms_dist=dict(choices=[1,2,3,4,5], probs=[0.08,0.30,0.45,0.13,0.04]),
        yr_mult={2015:0.55,2016:0.65,2017:0.80,2018:0.88,2019:0.90,
                 2020:0.92,2021:1.00,2022:0.97,2023:0.95,2024:1.00},
        lpr_map={2015:4.9,2016:4.75,2017:4.75,2018:4.75,2019:4.8,
                 2020:4.65,2021:4.65,2022:4.3,2023:4.2,2024:3.95},
        dist_center_mean=12, dist_subway_mean=800,
        orientation_keys=["南","南北","东","西","北"],
        orientation_probs=[0.30,0.35,0.15,0.10,0.10],
        deco_keys=["毛坯","简装","精装","豪装"],
        deco_probs=[0.10,0.20,0.55,0.15],
        prop_keys=["住宅","公寓","别墅","商住","老公房"],
        prop_probs=[0.60,0.15,0.05,0.10,0.10],
        fl_keys=["低层","中层","高层","顶层"],
        fl_probs=[0.20,0.45,0.25,0.10],
        fl_mults={"低层":0.95,"中层":1.05,"高层":1.08,"顶层":0.92},
        or_mults={"南":1.06,"南北":1.08,"东":1.01,"西":0.98,"北":0.93},
        de_mults={"毛坯":0.90,"简装":0.96,"精装":1.05,"豪装":1.15},
    ),
    "Tokyo": dict(
        base_prices={
            "千代田区":1500000,"中央区":1200000,"港区":1400000,
            "新宿区":900000,"渋谷区":1100000,"文京区":800000,
            "台東区":700000,"墨田区":650000,"江東区":700000,
            "品川区":800000,"目黒区":900000,"大田区":600000,
            "世田谷区":750000,"中野区":700000,"杉並区":700000,
            "豊島区":750000,"北区":600000,"足立区":500000,
        },
        mean_area=55.0, area_sigma=0.35,
        bedrooms_dist=dict(choices=[1,2,3,4], probs=[0.35,0.35,0.25,0.05]),
        yr_mult={2015:0.75,2016:0.80,2017:0.84,2018:0.88,2019:0.92,
                 2020:0.92,2021:0.96,2022:1.00,2023:1.05,2024:1.10},
        lpr_map={2015:-0.1,2016:-0.1,2017:-0.1,2018:-0.1,2019:-0.1,
                 2020:-0.1,2021:-0.1,2022:-0.1,2023:0.1,2024:0.5},
        dist_center_mean=8, dist_subway_mean=400,
        orientation_keys=["南","南北","東","西","北"],
        orientation_probs=[0.30,0.25,0.20,0.12,0.13],
        deco_keys=["スタンダード","プレミアム","ハイエンド","デベロッパー"],
        deco_probs=[0.40,0.35,0.15,0.10],
        prop_keys=["マンション","一戸建て","タワーマンション","アパート"],
        prop_probs=[0.55,0.20,0.15,0.10],
        fl_keys=["低層","中層","高層","最上階"],
        fl_probs=[0.25,0.40,0.25,0.10],
        fl_mults={"低層":0.94,"中層":1.03,"高層":1.10,"最上階":0.88},
        or_mults={"南":1.05,"南北":1.06,"東":1.01,"西":0.97,"北":0.92},
        de_mults={"スタンダード":0.95,"プレミアム":1.03,"ハイエンド":1.12,"デベロッパー":1.00},
    ),
    "London": dict(
        base_prices={
            "City of London":14000,"Westminster":12000,"Kensington & Chelsea":16000,
            "Camden":9000,"Islington":8500,"Hackney":7000,
            "Tower Hamlets":7500,"Southwark":7000,"Lambeth":6500,
            "Wandsworth":7500,"Greenwich":6000,"Lewisham":5500,
            "Hammersmith":8000,"Ealing":5500,
        },
        mean_area=75.0, area_sigma=0.45,
        bedrooms_dist=dict(choices=[1,2,3,4,5], probs=[0.25,0.35,0.25,0.10,0.05]),
        yr_mult={2015:0.70,2016:0.78,2017:0.80,2018:0.79,2019:0.80,
                 2020:0.82,2021:0.90,2022:1.00,2023:0.95,2024:0.97},
        lpr_map={2015:0.5,2016:0.25,2017:0.5,2018:0.75,2019:0.75,
                 2020:0.1,2021:0.1,2022:2.25,2023:5.25,2024:4.75},
        dist_center_mean=10, dist_subway_mean=600,
        orientation_keys=["South","South-North","East","West","North"],
        orientation_probs=[0.25,0.30,0.18,0.15,0.12],
        deco_keys=["Unfurnished","Part-Furnished","Furnished","Prime"],
        deco_probs=[0.20,0.30,0.35,0.15],
        prop_keys=["Flat","Terraced","Semi-Detached","Detached","Maisonette"],
        prop_probs=[0.55,0.20,0.10,0.05,0.10],
        fl_keys=["Ground","Mid","Upper","Penthouse"],
        fl_probs=[0.20,0.45,0.25,0.10],
        fl_mults={"Ground":0.93,"Mid":1.02,"Upper":1.06,"Penthouse":1.15},
        or_mults={"South":1.04,"South-North":1.06,"East":1.01,"West":0.98,"North":0.94},
        de_mults={"Unfurnished":0.92,"Part-Furnished":0.97,"Furnished":1.03,"Prime":1.18},
    ),
    "Singapore": dict(
        base_prices={
            "District 1-4 (CBD)":2200,"District 9-11 (Orchard/Newton)":2800,
            "District 15 (East Coast)":1600,"District 19 (Serangoon)":1200,
            "District 23 (Bukit Panjang)":1100,"District 28 (Seletar)":1000,
        },
        mean_area=900.0, area_sigma=0.38,  # sqft
        bedrooms_dist=dict(choices=[1,2,3,4,5], probs=[0.15,0.30,0.35,0.15,0.05]),
        yr_mult={2015:0.72,2016:0.72,2017:0.74,2018:0.80,2019:0.84,
                 2020:0.85,2021:0.95,2022:1.05,2023:1.10,2024:1.12},
        lpr_map={2015:0.82,2016:0.78,2017:0.94,2018:1.70,2019:1.80,
                 2020:0.40,2021:0.35,2022:3.20,2023:4.00,2024:3.70},
        dist_center_mean=8, dist_subway_mean=400,
        orientation_keys=["North","South","East","West","North-South"],
        orientation_probs=[0.20,0.25,0.22,0.18,0.15],
        deco_keys=["Standard","Renovated","Premium","Luxury"],
        deco_probs=[0.20,0.30,0.35,0.15],
        prop_keys=["Condo","HDB Resale","Landed","Executive Condo","Apartment"],
        prop_probs=[0.40,0.35,0.08,0.10,0.07],
        fl_keys=["Low","Mid","High","Penthouse"],
        fl_probs=[0.20,0.40,0.30,0.10],
        fl_mults={"Low":0.92,"Mid":1.02,"High":1.10,"Penthouse":1.20},
        or_mults={"North":0.96,"South":1.05,"East":1.02,"West":0.98,"North-South":1.06},
        de_mults={"Standard":0.92,"Renovated":0.98,"Premium":1.06,"Luxury":1.20},
    ),
    "New York": dict(
        base_prices={
            "Manhattan":2200,"Brooklyn":1200,"Queens":900,
            "Bronx":600,"Staten Island":550,
            "Upper East Side":2500,"Upper West Side":2300,"Tribeca":3000,
            "SoHo":2800,"Greenwich Village":2600,"Williamsburg":1400,
            "Astoria":900,"Long Island City":1100,
        },
        mean_area=850.0, area_sigma=0.50,   # sqft
        bedrooms_dist=dict(choices=[0,1,2,3,4], probs=[0.15,0.35,0.30,0.15,0.05]),
        yr_mult={2015:0.72,2016:0.78,2017:0.85,2018:0.88,2019:0.87,
                 2020:0.82,2021:0.90,2022:1.00,2023:0.96,2024:0.98},
        lpr_map={2015:0.25,2016:0.50,2017:1.50,2018:2.50,2019:2.25,
                 2020:0.25,2021:0.25,2022:4.50,2023:5.50,2024:4.75},
        dist_center_mean=5, dist_subway_mean=300,
        orientation_keys=["South","North","East","West","Corner"],
        orientation_probs=[0.25,0.25,0.20,0.20,0.10],
        deco_keys=["Original","Updated","Renovated","Luxury"],
        deco_probs=[0.20,0.30,0.35,0.15],
        prop_keys=["Condo","Co-op","Townhouse","Single Family","Multi Family"],
        prop_probs=[0.40,0.35,0.08,0.08,0.09],
        fl_keys=["Low","Mid","High","Penthouse"],
        fl_probs=[0.25,0.40,0.25,0.10],
        fl_mults={"Low":0.93,"Mid":1.02,"High":1.08,"Penthouse":1.25},
        or_mults={"South":1.05,"North":0.97,"East":1.01,"West":0.99,"Corner":1.08},
        de_mults={"Original":0.88,"Updated":0.97,"Renovated":1.05,"Luxury":1.22},
    ),
    "Dubai": dict(
        base_prices={
            "Downtown Dubai":1800,"Dubai Marina":1500,"Palm Jumeirah":2200,
            "Business Bay":1200,"Jumeirah":1600,"DIFC":2000,
            "Dubai Hills":1300,"Arabian Ranches":900,
            "JVC":700,"Al Barsha":800,
        },
        mean_area=1100.0, area_sigma=0.55,   # sqft
        bedrooms_dist=dict(choices=[1,2,3,4,5], probs=[0.20,0.35,0.28,0.12,0.05]),
        yr_mult={2015:0.78,2016:0.72,2017:0.68,2018:0.65,2019:0.63,
                 2020:0.62,2021:0.70,2022:0.90,2023:1.05,2024:1.15},
        lpr_map={2015:1.5,2016:1.5,2017:1.5,2018:2.5,2019:2.3,
                 2020:0.5,2021:0.5,2022:4.0,2023:5.5,2024:5.25},
        dist_center_mean=10, dist_subway_mean=700,
        orientation_keys=["Sea View","City View","Garden","Pool","Inner"],
        orientation_probs=[0.20,0.30,0.20,0.15,0.15],
        deco_keys=["Standard","Premium","Luxury","Ultra-Luxury"],
        deco_probs=[0.25,0.35,0.28,0.12],
        prop_keys=["Apartment","Villa","Townhouse","Penthouse","Duplex"],
        prop_probs=[0.55,0.20,0.12,0.08,0.05],
        fl_keys=["Low","Mid","High","Penthouse"],
        fl_probs=[0.20,0.40,0.28,0.12],
        fl_mults={"Low":0.90,"Mid":1.00,"High":1.12,"Penthouse":1.35},
        or_mults={"Sea View":1.25,"City View":1.10,"Garden":1.05,"Pool":1.08,"Inner":0.90},
        de_mults={"Standard":0.88,"Premium":1.00,"Luxury":1.15,"Ultra-Luxury":1.40},
    ),
}


def generate_synthetic_data(city: str = "Shanghai", n: int = 5000,
                             seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic property listings for a given city.
    Returns a DataFrame with all standard feature columns.
    """
    if city not in CITY_CALIBRATION:
        # Generic fallback
        city_cal = CITY_CALIBRATION["Shanghai"]
        print(f"  Warning: No calibration for {city}. Using Shanghai as template.")
    else:
        city_cal = CITY_CALIBRATION[city]

    rng = np.random.default_rng(seed)
    districts = list(city_cal["base_prices"].keys())

    dist_arr    = rng.choice(districts, n)
    base_prices = np.array([city_cal["base_prices"][d] for d in dist_arr], dtype=float)

    # Area
    area = rng.lognormal(
        np.log(city_cal["mean_area"]),
        city_cal["area_sigma"], n
    ).clip(city_cal["mean_area"] * 0.2, city_cal["mean_area"] * 8)

    # Bedrooms
    bd = city_cal["bedrooms_dist"]
    bedrooms = rng.choice(bd["choices"], n, p=bd["probs"])
    living_rooms = np.clip(bedrooms - 1, 0, 3)
    bathrooms    = np.clip(bedrooms - 1, 1, 4)

    # Floor
    total_floors = rng.choice(range(3, 40), n)
    floor_num    = (rng.random(n) * total_floors + 1).astype(int)
    fl_idx       = rng.choice(len(city_cal["fl_keys"]), n, p=city_cal["fl_probs"])
    fl_labels    = [city_cal["fl_keys"][i] for i in fl_idx]
    fl_mult      = np.array([city_cal["fl_mults"][k] for k in fl_labels])

    # Orientation
    or_idx    = rng.choice(len(city_cal["orientation_keys"]), n,
                            p=city_cal["orientation_probs"])
    orient    = [city_cal["orientation_keys"][i] for i in or_idx]
    or_mult   = np.array([city_cal["or_mults"][k] for k in orient])

    # Decoration
    de_idx    = rng.choice(len(city_cal["deco_keys"]), n, p=city_cal["deco_probs"])
    deco      = [city_cal["deco_keys"][i] for i in de_idx]
    de_mult   = np.array([city_cal["de_mults"][k] for k in deco])

    # Property type
    pt_idx    = rng.choice(len(city_cal["prop_keys"]), n, p=city_cal["prop_probs"])
    prop_type = [city_cal["prop_keys"][i] for i in pt_idx]

    age_years    = rng.integers(1, 40, n).astype(float)
    has_elevator = (total_floors >= 7).astype(int)
    dist_subway  = rng.exponential(city_cal["dist_subway_mean"], n).clip(50, 5000)
    school_qual  = rng.uniform(0, 10, n)
    dist_center  = rng.exponential(city_cal["dist_center_mean"], n).clip(0.5, 60)

    year  = rng.choice(list(city_cal["yr_mult"].keys()), n)
    month = rng.integers(1, 13, n)

    yr_mult_arr  = np.array([city_cal["yr_mult"][y] for y in year])
    lpr_arr      = np.array([city_cal["lpr_map"][y] for y in year])

    unit_price = (
        base_prices * fl_mult * or_mult * de_mult
        * np.exp(-0.008 * age_years)
        * (1.0 + 0.03 * has_elevator)
        * (1.0 + 0.12 * np.exp(-dist_subway / 500))
        * (1.0 + 0.10 * school_qual / 10)
        * (1.0 + 0.15 * np.exp(-dist_center / 8))
        * yr_mult_arr
        * rng.normal(1.0, 0.07, n)   # tighter noise for lower synthetic MAE
    ).round(0)

    # Geo
    from ..cities.registry import CITY_REGISTRY
    if city in CITY_REGISTRY:
        lat_c, lon_c = CITY_REGISTRY[city].center
    else:
        lat_c, lon_c = 31.2304, 121.4737

    return pd.DataFrame({
        "unit_price"          : unit_price,
        "total_price"         : (unit_price * area).round(0),
        "area_sqm"            : area.round(1),
        "bedrooms"            : bedrooms,
        "living_rooms"        : living_rooms,
        "bathrooms"           : bathrooms,
        "floor"               : floor_num,
        "total_floors"        : total_floors,
        "floor_ratio"         : (floor_num / total_floors).round(3),
        "floor_category"      : fl_labels,
        "orientation"         : orient,
        "decoration"          : deco,
        "property_type"       : prop_type,
        "age_years"           : age_years,
        "has_elevator"        : has_elevator,
        "has_parking"         : rng.choice([0,1], n, p=[0.3,0.7]),
        "plot_ratio"          : rng.uniform(0.8, 4.5, n).round(2),
        "green_ratio"         : rng.uniform(0.20, 0.50, n).round(3),
        "management_fee"      : rng.uniform(1.5, 15, n).round(1),
        "district"            : dist_arr,
        "latitude"            : rng.normal(lat_c, 0.15, n).clip(lat_c-0.5, lat_c+0.5).round(6),
        "longitude"           : rng.normal(lon_c, 0.20, n).clip(lon_c-0.6, lon_c+0.6).round(6),
        "dist_city_center_km" : dist_center.round(2),
        "dist_subway_m"       : dist_subway.round(0).astype(int),
        "subway_lines"        : rng.integers(1, 6, n),
        "school_quality"      : school_qual.round(2),
        "dist_school_m"       : rng.exponential(600, n).clip(50,3000).round(0).astype(int),
        "dist_hospital_m"     : rng.exponential(1200, n).clip(100,5000).round(0).astype(int),
        "dist_mall_m"         : rng.exponential(800, n).clip(100,4000).round(0).astype(int),
        "dist_park_m"         : rng.exponential(500, n).clip(50,3000).round(0).astype(int),
        "in_free_trade_zone"  : ((dist_center < 5) | (dist_center > 45)).astype(int),
        "year"                : year,
        "month"               : month,
        "quarter"             : ((month - 1) // 3 + 1),
        "is_golden_week"      : (month == 10).astype(int),
        "lpr_5yr"             : lpr_arr,
        "gdp_growth_yoy"      : rng.normal(5.0, 1.5, n).round(2),
        "cpi_yoy"             : rng.normal(2.5, 0.8, n).round(2),
        "m2_growth_yoy"       : rng.normal(9.0, 2.5, n).round(2),
        "policy_restriction"  : rng.integers(1, 6, n),
        "baidu_search_idx"    : rng.normal(500, 100, n).clip(100,1000).round(0),
        "social_sentiment"    : rng.uniform(-1, 1, n).round(3),
        "city"                : city,
        "source"              : "synthetic",
    })
