import pandas as pd
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import sys
import os
import time

# ==========================================
# 1. 核心配置区
# ==========================================
EIA_API_KEY = "itwJ48WNrRTPGYaWYLNWiuvtIe0lYaMOcvHZhXJ5"  # <--- 记得填你的 Key
REGION_ID = "PJM"
LAT, LON = 39.9612, -82.9988

# 锁定 2025 完整自然年（绝对稳定，规避气象延迟）
start_date = datetime(2025, 1, 1, 0, 0, 0)
end_date = datetime(2025, 12, 31, 23, 59, 59)


# ==========================================
# 2. 抓取 EIA 真实负荷 (加入自动翻页)
# ==========================================
def get_real_load(api_key, region, start, end):
    print(f"\n⚡ 正在提取 {region} 区域的【真实历史负载】...")
    url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={api_key}"

    all_data = []
    offset = 0
    batch_size = 2000  # 每次安全抓取 2000 行

    while True:
        print(f"   -> 正在下载数据... (已获取 {offset} 行)")
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "D",
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": batch_size,
            "offset": offset  # 核心：每次翻页的偏移量
        }

        resp = requests.get(url, params=params).json()
        if 'response' not in resp or 'data' not in resp['response']:
            print(f"❌ EIA API 报错: {resp}")
            sys.exit()

        batch_data = resp['response']['data']
        if not batch_data:
            break  # 如果这一页是空的，说明抓完了，退出循环

        all_data.extend(batch_data)

        # 如果抓到的数据少于请求的批次大小，说明到底了
        if len(batch_data) < batch_size:
            break

        offset += batch_size
        time.sleep(0.5)  # 稍微停顿，防止把服务器请求崩了

    print(f"✅ 负荷数据提取完毕！共提取 {len(all_data)} 行。")
    df = pd.DataFrame(all_data)
    df = df[['period', 'value']].rename(columns={'period': 'Time', 'value': 'Load_MW'})
    df['Load_MW'] = pd.to_numeric(df['Load_MW'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize(None)

    return df.set_index('Time')


# ==========================================
# 3. 抓取 EIA 真实发电构成 (加入自动翻页)
# ==========================================
def get_generation_mix(api_key, region, start, end):
    print(f"\n⚙️ 正在提取 {region} 区域的【真实燃料发电结构】...")
    url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={api_key}"

    all_data = []
    offset = 0
    batch_size = 2000

    while True:
        print(f"   -> 正在下载数据... (已获取 {offset} 行)")
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": batch_size,
            "offset": offset
        }

        resp = requests.get(url, params=params).json()
        if 'response' not in resp or 'data' not in resp['response']:
            print(f"❌ EIA API 报错: {resp}")
            sys.exit()

        batch_data = resp['response']['data']
        if not batch_data:
            break

        all_data.extend(batch_data)

        if len(batch_data) < batch_size:
            break

        offset += batch_size
        time.sleep(0.5)

    print(f"✅ 发电结构提取完毕！共提取 {len(all_data)} 行。")
    df = pd.DataFrame(all_data)
    df_pivot = df.pivot(index='period', columns='fueltype', values='value')
    df_pivot.index = pd.to_datetime(df_pivot.index).tz_localize(None)

    for col in df_pivot.columns:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

    mapping = {
        'COL': 'Coal_Gen', 'NG': 'Gas_Gen', 'NUC': 'Nuclear_Gen',
        'OIL': 'Oil_Gen', 'SUN': 'Solar_Gen', 'WND': 'Wind_Gen', 'OTH': 'Other_Gen',
        'WAT': 'Hydro_Gen'
    }
    df_pivot = df_pivot.rename(columns=mapping)
    return df_pivot


# ==========================================
# 4. 抓取 Open-Meteo 真实历史天气
# ==========================================
def get_real_weather(lat, lon, start, end):
    print(f"\n🌤️ 正在同步对应周期的【真实历史气象】(温度+湿度)...")
    # Open-Meteo 没有严格的分页限制，可以直接一把抓一年
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "timezone": "UTC"
    }
    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0].Hourly()

    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ).tz_localize(None)

    df_weather = pd.DataFrame(data={
        "Time": times,
        "Temp_C": hourly.Variables(0).ValuesAsNumpy(),
        "Humidity_Pct": hourly.Variables(1).ValuesAsNumpy()
    }).set_index('Time')
    print(f"✅ 气象数据同步完毕！共获取 {len(df_weather)} 行。")
    return df_weather


# ==========================================
# 5. 主流程与特征工程
# ==========================================
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "pjm_full_year_dataset.csv")

        df_load = get_real_load(EIA_API_KEY, REGION_ID, start_date, end_date)
        df_mix = get_generation_mix(EIA_API_KEY, REGION_ID, start_date, end_date)
        df_weather = get_real_weather(LAT, LON, start_date, end_date)

        print("\n🔗 正在进行数据对齐与清洗...")

        df_final = pd.merge(df_load, df_mix, left_index=True, right_index=True, how='inner')
        df_final = pd.merge(df_final, df_weather, left_index=True, right_index=True, how='inner')

        # 数据清洗
        df_final = df_final.interpolate(method='linear')
        df_final = df_final.dropna()

        # 特征工程：计算温度敏感型发电占比
        gen_cols = [c for c in df_final.columns if '_Gen' in c]
        df_final['Total_Gen_MW'] = df_final[gen_cols].sum(axis=1)

        gas = df_final.get('Gas_Gen', 0)
        coal = df_final.get('Coal_Gen', 0)
        df_final['Temp_Sensitive_Share_Pct'] = ((gas + coal) / df_final['Total_Gen_MW']) * 100

        df_final.reset_index().to_csv(output_file, index=False)

        print("-" * 50)
        print(f"🎉 大功告成！全量一年的工业级数据集已生成！")
        print(f"📍 文件路径: {output_file}")
        print(f"📊 最终对齐后数据行数: {len(df_final)} 行 (预期约 8760 行)")

    except Exception as e:
        print(f"\n❌ 程序崩溃！")
        import traceback

        traceback.print_exc()