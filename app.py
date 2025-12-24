import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import altair as alt


# добавим асинхронную функцию
async def get_all_current_weather_async(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    cities = (
        pd.Series(df["city"])
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    url = "https://api.openweathermap.org/data/2.5/weather"
    sem = asyncio.Semaphore(10)

    async def fetch_city(session: aiohttp.ClientSession, city: str) -> dict:
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "ru"}
        async with sem:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 401:
                    raise ValueError('Неверный API ключ')
                data = await resp.json()
                return {"city": city, "temp_now": data["main"]["temp"]}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_city(session, c) for c in cities]
        rows = await asyncio.gather(*tasks)

    return pd.DataFrame(rows)


def run_async(coro):
    """Безопасный запуск async из sync-кода Streamlit (с помощью ChatGPT)"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # На всякий случай (в Streamlit Cloud обычно не нужно),
        # но оставим, чтобы не падало в окружениях с активным loop
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)

def weather_analysis(df):
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # подсчет скользящего среднего по городам за 30 дней
    df["rolling_avg"] = (df.groupby("city", group_keys=False)
                                    .apply(lambda g: pd.Series(g.set_index("timestamp")["temperature"]
                                    .rolling("30D", min_periods=1).mean().values, index=g.index)))

    # подсчет среднего и ско температуры по городам и сезонам
    means = df.groupby(['city', 'season'], as_index=False)['temperature'].mean().rename(columns={"temperature": "mean_seasonal_temperature"})
    stds = df.groupby(['city', 'season'], as_index=False)['temperature'].std().rename(columns={"temperature": "std_seasonal_temperature"})
    df = df.merge(means, on=["city", "season"], how="left").merge(stds, on=["city", "season"], how="left")

    # подсчет выбросов по среднему +- 2ско
    mean_ = df['temperature'].mean()
    std_ = df['temperature'].std()
    lower = mean_ - 2*std_
    higher = mean_ + 2*std_
    df['outlier'] = np.where((df['temperature'] > higher) | (df['temperature'] < lower), 1, 0)

    return df

st.title("Анализ погоды OpenWeatherMap")

uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

st.subheader("Текущая погода")
api_key = st.text_input(
    "Введите API-ключ OpenWeatherMap",
    type="password",
    placeholder="Например: 123abc...",
)

# нет файла - дальше не идём
if not uploaded_file:
    st.info("Загрузите CSV файл, чтобы выбрать город и получить текущую погоду в нем")
    st.stop()

df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

# выбираем город
cities = sorted(df["city"].unique())
if not cities:
    st.warning("В файле нет городов")
    st.stop()

city = st.selectbox("Выберите город", cities)

# если нет ключа, то погоду не показываем
if not api_key:
    st.info("Чтобы показать текущую погоду, введите API-ключ OpenWeatherMap")
    st.stop()

# получаем погоду (асинхронно, но запускаем из sync через helper)
with st.spinner("Запрашиваем текущую погоду..."):
    try:
        weather_df = run_async(get_all_current_weather_async(df, api_key))
    except ValueError:
        st.error('Пожалуйста, введите корректный API-ключ')
        st.stop()

# показываем температуру выбранного города
row = weather_df.loc[weather_df["city"] == city, "temp_now"]
if row.empty:
    st.warning(f"Не удалось получить погоду для города: {city}")
else:
    st.metric(label=f"Температура в {city}", value=f"{row.iloc[0]} °C")

st.subheader(f"Описательная статистика для {city}")
st.table(df[df["city"] == city].describe())

prepared_df = weather_analysis(df.copy())
prepared_df = prepared_df[prepared_df["city"] == city].sort_values("timestamp")

# категория для цвета
prepared_df["cat"] = np.where(
    (prepared_df["outlier"] == 1) & (prepared_df["temperature"] > 0), "high_outlier",
    np.where((prepared_df["outlier"] == 1) & (prepared_df["temperature"] < 0), "low_outlier", "normal")
)

# приглушённые цвета
color_scale = alt.Scale(
    domain=["low_outlier", "normal", "high_outlier"],
    range=["#6f86b7", "#3a3a3a", "#b56b6b"]  # muted blue, anthracite, muted red
)

# границы по времени
max_ts = prepared_df["timestamp"].max()
min_ts = prepared_df["timestamp"].min()

# последние 12 месяцев
start_12m = max_ts - pd.DateOffset(months=12)
df_last12 = prepared_df[prepared_df["timestamp"] >= start_12m].copy()

# "ползунок" (выбор интервала) по оси X
brush = alt.selection_interval(encodings=["x"])

# основной график: ТОЛЬКО последние 12 месяцев + сохраняем окраску outliers
main = (
    alt.Chart(df_last12)
    .mark_bar()
    .encode(
        x=alt.X("timestamp:T", title="Дата"),
        y=alt.Y("temperature:Q", title="Температура (°C)"),
        color=alt.Color("cat:N", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Дата"),
            alt.Tooltip("temperature:Q", title="Температура", format=".2f"),
            alt.Tooltip("outlier:N", title="Outlier"),
        ],
    )
    .properties(height=450)
    .transform_filter(brush)   # фильтр по выбранному диапазону
)

# нижняя панель: вся история (для выбора диапазона)
overview = (
    alt.Chart(prepared_df)
    .mark_area(opacity=0.25)
    .encode(
        x=alt.X("timestamp:T", title=""),
        y=alt.Y("temperature:Q", title=""),
    )
    .properties(height=90)
    .add_params(brush)
)

chart = alt.vconcat(main, overview).resolve_scale(x="shared")
st.altair_chart(chart, use_container_width=True)








