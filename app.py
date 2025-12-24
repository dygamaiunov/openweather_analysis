import streamlit as st
import pandas as pd
import asyncio
import aiohttp


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

df = pd.read_csv(uploaded_file)

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

st.table(df[df["city"] == city].describe())



