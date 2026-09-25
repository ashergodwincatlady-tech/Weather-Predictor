import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = Path(__file__).parent / "Claes.love.csv"
MAX_FORECAST_DAYS = 5
LAGS = 7
N_TREES = 120

FEATURE_COLUMNS = [
    "temperature_max_c",
    "temperature_min_c",
    "humidity_percent",
    "wind_speed_kmh",
    "precipitation_mm",
    "cloud_cover_okta",
]

REGRESSION_TARGETS = [
    "temperature_max_c",
    "temperature_min_c",
    "humidity_percent",
    "wind_speed_kmh",
    "precipitation_mm",
]

# ============================================================
# WEATHER CONDITION
# ============================================================

def calculate_weather_condition(row):
    rain = row["precipitation_mm"]
    clouds = row["cloud_cover_okta"]

    if not pd.isna(rain):
        if rain >= 5:
            return "rainy"
        if rain > 0:
            return "light rain"

    if not pd.isna(clouds):
        if clouds <= 1:
            return "sunny"
        if clouds <= 2:
            return "mostly sunny"
        if clouds <= 4:
            return "partly cloudy"
        if clouds <= 6:
            return "mostly cloudy"
        return "cloudy"

    return "unknown"


# ============================================================
# LOAD CSV
# ============================================================

@st.cache_data
def load_weather():
    # Make sure Streamlit uses the CSV beside app.py
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Could not find Claes.love.csv.\n\nExpected location:\n{CSV_FILE}"
        )

    data = pd.read_csv(CSV_FILE)

    # Separate real historical observations from saved predictions.
    # Only "actual" rows are used to train the machine-learning model.
    if "record_type" not in data.columns:
        data["record_type"] = "actual"

    data["record_type"] = (
        data["record_type"]
        .fillna("actual")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    training_data = data[data["record_type"] == "actual"].copy()

    # Remove accidental spaces/BOM from column names.
    data.columns = data.columns.astype(str).str.replace("\\ufeff", "", regex=False).str.strip()

    # These are the ONLY columns that must exist in the CSV.
    required_columns = {
        "date",
        "temperature_max_c",
        "temperature_min_c",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "cloud_cover_okta",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "The CSV is missing these columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Convert date.
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Convert numeric weather columns.
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # IMPORTANT:
    # weather_condition is NOT required to be in the CSV.
    # If it exists, we clean it. If it does not exist, we create it.
    if "weather_condition" not in data.columns:
        data["weather_condition"] = data.apply(calculate_weather_condition, axis=1)
    else:
        data["weather_condition"] = (
            data["weather_condition"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        missing_condition = data["weather_condition"].isin(["", "nan", "none", "unknown"])
        data.loc[missing_condition, "weather_condition"] = data.loc[
            missing_condition
        ].apply(calculate_weather_condition, axis=1)

    data["weather_condition"] = data["weather_condition"].replace("", "unknown")

    # Only real historical observations are used for training.
    data = (
        training_data.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    if len(data) < 30:
        raise ValueError("The CSV needs at least 30 daily weather rows.")

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):
    result = data.copy()

    # Current value + previous 7 days for every weather variable.
    for column in FEATURE_COLUMNS:
        for lag in range(0, LAGS + 1):
            result[f"{column}_lag_{lag}"] = result[column].shift(lag)

    # Recent averages.
    for column in FEATURE_COLUMNS:
        result[f"{column}_avg_3"] = result[column].rolling(3).mean()
        result[f"{column}_avg_7"] = result[column].rolling(7).mean()

    # Seasonal information.
    day_of_year = result["date"].dt.dayofyear
    result["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return result


def get_model_features():
    features = []

    for column in FEATURE_COLUMNS:
        for lag in range(0, LAGS + 1):
            features.append(f"{column}_lag_{lag}")

    for column in FEATURE_COLUMNS:
        features.append(f"{column}_avg_3")
        features.append(f"{column}_avg_7")

    features.extend(["day_sin", "day_cos"])

    return features


MODEL_FEATURES = get_model_features()


# ============================================================
# RANDOM FOREST MODELS
# ============================================================

def make_regression_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_TREES,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_classification_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_TREES,
                    random_state=42,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


@st.cache_resource
def train_random_forest(data, target_column, days_ahead, model_type):
    model_data = data.copy()

    # Create rain/no-rain target when needed.
    if target_column == "rain_target":
        model_data["rain_target"] = np.where(
            model_data["precipitation_mm"].isna(),
            np.nan,
            (model_data["precipitation_mm"] > 0.1).astype(float),
        )

    features_data = create_features(model_data)
    features_data["target"] = features_data[target_column].shift(-days_ahead)

    # Ignore unknown weather conditions during condition training.
    if target_column == "weather_condition":
        features_data.loc[
            features_data["target"].isin(["unknown", "nan", "none"]),
            "target",
        ] = np.nan

    training = features_data.dropna(subset=["target"])

    if len(training) < 20:
        raise ValueError(
            f"Not enough training rows for {target_column} at {days_ahead} day(s) ahead."
        )

    X = training[MODEL_FEATURES]
    y = training["target"]

    if model_type == "regression":
        model = make_regression_model()
    elif model_type == "classification":
        if y.nunique() < 2:
            raise ValueError(
                f"Not enough different classes to train {target_column}."
            )
        model = make_classification_model()
    else:
        raise ValueError("model_type must be 'regression' or 'classification'.")

    model.fit(X, y)
    return model


def get_latest_features(data):
    features_data = create_features(data)
    return features_data.iloc[[-1]][MODEL_FEATURES]


def predict_numeric(data, target_column, days_ahead):
    model = train_random_forest(
        data,
        target_column,
        days_ahead,
        "regression",
    )
    latest = get_latest_features(data)
    return float(model.predict(latest)[0])


def predict_rain_probability(data, days_ahead):
    model = train_random_forest(
        data,
        "rain_target",
        days_ahead,
        "classification",
    )

    latest = get_latest_features(data)
    probabilities = model.predict_proba(latest)[0]
    classes = list(model.named_steps["model"].classes_)

    if 1.0 in classes:
        probability = probabilities[classes.index(1.0)] * 100
    else:
        probability = 0.0

    return float(np.clip(probability, 0, 100))


def predict_weather_condition(data, days_ahead):
    model = train_random_forest(
        data,
        "weather_condition",
        days_ahead,
        "classification",
    )

    latest = get_latest_features(data)
    return str(model.predict(latest)[0])


# ============================================================
# QUESTION PARSING
# ============================================================

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def get_days_ahead(question):
    q = question.lower().strip()

    if "tomorrow" in q or "next day" in q:
        return 1

    patterns = [
        r"(?:in|after)\s+(\d+|one|two|three|four|five)\s+days?",
        r"(\d+|one|two|three|four|five)\s+days?\s+from\s+now",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            value = match.group(1)
            if value.isdigit():
                return int(value)
            return NUMBER_WORDS[value]

    return 1


def get_topic(question):
    q = question.lower().strip()

    if any(
        phrase in q
        for phrase in [
            "chance of rain",
            "probability of rain",
            "rain probability",
            "will it rain",
            "is it going to rain",
        ]
    ):
        return "rain_probability"

    if any(
        phrase in q
        for phrase in [
            "how much rain",
            "rain amount",
            "amount of rain",
            "how many mm",
            "precipitation amount",
            "precipitation",
            "rain in mm",
        ]
    ):
        return "precipitation"

    if "humidity" in q or "humid" in q:
        return "humidity"

    if "wind" in q or "windy" in q:
        return "wind"

    if any(
        phrase in q
        for phrase in [
            "minimum temperature",
            "minimum temp",
            "lowest temperature",
            "lowest temp",
            "low temperature",
            "low temp",
        ]
    ):
        return "temperature_min"

    if "temperature" in q or "temp" in q or "hot" in q or "cold" in q:
        return "temperature_max"

    if any(
        word in q
        for word in ["weather", "condition", "sunny", "cloudy", "rainy"]
    ):
        return "weather"

    return None


# ============================================================
# SAVE FORECASTS TO CSV
# ============================================================

def save_forecast_to_csv(forecast_rows):
    """
    Save the newest five-day predictions into the same CSV.

    Real historical rows are marked as "actual".
    Model-generated rows are marked as "prediction".
    Old predictions are removed before new predictions are saved.
    """

    full_data = pd.read_csv(CSV_FILE)

    # Make sure the CSV has a record_type column.
    if "record_type" not in full_data.columns:
        full_data["record_type"] = "actual"

    full_data["record_type"] = (
        full_data["record_type"]
        .fillna("actual")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove old predictions so the CSV always contains
    # the newest five-day forecast.
    full_data = full_data[
        full_data["record_type"] != "prediction"
    ].copy()

    predictions = pd.DataFrame(forecast_rows)

    # Convert the forecast table names to CSV column names.
    predictions = predictions.rename(
        columns={
            "Date": "date",
            "Max °C": "temperature_max_c",
            "Min °C": "temperature_min_c",
            "Humidity %": "humidity_percent",
            "Rain mm": "precipitation_mm",
            "Wind km/h": "wind_speed_kmh",
            "Weather": "weather_condition",
        }
    )

    # Rain chance is not one of the original weather columns,
    # but keeping it is useful if the CSV already has that column.
    predictions["record_type"] = "prediction"

    # Make sure every existing CSV column is present.
    csv_columns = list(full_data.columns)

    for column in csv_columns:
        if column not in predictions.columns:
            predictions[column] = np.nan

    predictions = predictions[csv_columns]

    # Add the new predictions after the historical observations.
    combined = pd.concat(
        [full_data, predictions],
        ignore_index=True,
    )

    combined.to_csv(CSV_FILE, index=False)


# ============================================================
# ANSWER QUESTIONS
# ============================================================

def answer_question(question, data):
    if not question.strip():
        return "Please type a weather question."

    days = get_days_ahead(question)

    if days < 1 or days > MAX_FORECAST_DAYS:
        return "I can predict from tomorrow up to 5 days ahead."

    topic = get_topic(question)

    if topic is None:
        return (
            "I can answer questions about temperature, humidity, rain, "
            "precipitation, wind, or weather condition."
        )

    # Forecast dates are based on today's real date, not the last CSV date.
    today = pd.Timestamp(datetime.now().date())
    forecast_date = today + timedelta(days=days)
    date_text = forecast_date.strftime("%B %d, %Y")

    if topic == "temperature_max":
        value = predict_numeric(data, "temperature_max_c", days)
        return (
            f"On {date_text}, the predicted maximum temperature is "
            f"**{value:.1f}°C**."
        )

    if topic == "temperature_min":
        value = predict_numeric(data, "temperature_min_c", days)
        return (
            f"On {date_text}, the predicted minimum temperature is "
            f"**{value:.1f}°C**."
        )

    if topic == "humidity":
        value = np.clip(
            predict_numeric(data, "humidity_percent", days),
            0,
            100,
        )
        return f"On {date_text}, the predicted humidity is **{value:.1f}%**."

    if topic == "wind":
        value = max(0, predict_numeric(data, "wind_speed_kmh", days))
        return f"On {date_text}, the predicted wind speed is **{value:.1f} km/h**."

    if topic == "precipitation":
        value = max(0, predict_numeric(data, "precipitation_mm", days))
        return (
            f"On {date_text}, the predicted precipitation amount is "
            f"**{value:.1f} mm**."
        )

    if topic == "rain_probability":
        probability = predict_rain_probability(data, days)
        return (
            f"On {date_text}, the model predicts a **{probability:.1f}%** "
            "chance of measurable rain."
        )

    if topic == "weather":
        condition = predict_weather_condition(data, days)
        return f"On {date_text}, the predicted weather condition is **{condition}**."

    return "I could not answer that question."


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Senegal Weather Predictor",
    page_icon="🌤️",
    layout="centered",
)

st.title("🌤️ Senegal Weather Predictor")
st.write(
    "Ask about temperature, humidity, rain, precipitation, wind, or "
    "weather condition for tomorrow or up to 5 days ahead."
)
st.caption(
    "The model reads Claes.love.csv automatically. You do not need to enter "
    "today's weather. The forecast dates are based on today's date."
)

# Load data.
try:
    weather = load_weather()
except Exception as error:
    st.error(f"CSV error: {error}")
    st.stop()

today = pd.Timestamp(datetime.now().date())
tomorrow = today + timedelta(days=1)

st.info(
    f"Today is **{today.strftime('%B %d, %Y')}**. "
    f"The 5-day forecast starts tomorrow, "
    f"**{tomorrow.strftime('%B %d, %Y')}**."
)

# Train/cache all models once when the app starts.
try:
    with st.spinner("Training the machine-learning models..."):
        for days in range(1, MAX_FORECAST_DAYS + 1):
            for target in REGRESSION_TARGETS:
                train_random_forest(weather, target, days, "regression")

            train_random_forest(weather, "rain_target", days, "classification")
            train_random_forest(weather, "weather_condition", days, "classification")
except Exception as error:
    st.error(f"The machine-learning models could not be trained: {error}")
    st.stop()

# Ask a natural-language question.
question = st.text_input(
    "Ask your weather question:",
    placeholder="Example: What will the weather be in 5 days?",
)

if st.button("Ask", type="primary"):
    try:
        st.success(answer_question(question, weather))
    except Exception as error:
        st.error(f"The question could not be answered: {error}")

# Five-day forecast table.
st.subheader("📅 Model Forecast: Next 5 Days")

forecast_rows = []

try:
    for days in range(1, MAX_FORECAST_DAYS + 1):
        forecast_rows.append(
            {
                "Date": today + timedelta(days=days),
                "Max °C": predict_numeric(
                    weather, "temperature_max_c", days
                ),
                "Min °C": predict_numeric(
                    weather, "temperature_min_c", days
                ),
                "Humidity %": predict_numeric(
                    weather, "humidity_percent", days
                ),
                "Rain chance %": predict_rain_probability(weather, days),
                "Rain mm": predict_numeric(
                    weather, "precipitation_mm", days
                ),
                "Wind km/h": predict_numeric(
                    weather, "wind_speed_kmh", days
                ),
                "Weather": predict_weather_condition(weather, days),
            }
        )

    forecast_table = pd.DataFrame(forecast_rows)

    # Save the newest five-day predictions into Claes.love.csv.
    save_forecast_to_csv(forecast_rows)

    forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        forecast_table.round(1),
        hide_index=True,
        use_container_width=True,
    )
except Exception as error:
    st.error(f"The 5-day forecast could not be generated: {error}")

# Explanation.
with st.expander("How the machine-learning model works"):
    st.write(
        "The model uses the weather data in the CSV plus the previous 7 days "
        "of weather values."
    )
    st.write(
        "It also creates 3-day and 7-day averages and seasonal date features."
    )
    st.write(
        "Random Forest regression predicts maximum temperature, minimum "
        "temperature, humidity, wind speed, and precipitation."
    )
    st.write(
        "Random Forest classification predicts rain/no-rain probability "
        "and weather condition."
    )
    st.write(
        "There is a separate model for 1, 2, 3, 4, and 5 days ahead."
    )

st.caption(
    "Educational project. Historical data cannot guarantee future weather, "
    "and this app does not use a live weather forecast service."
)
