import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from streamlit_gsheets import GSheetsConnection


# ============================================================
# SETTINGS
# ============================================================

BASE_FOLDER = Path(__file__).parent

CSV_FILE = BASE_FOLDER / "Claes.love.csv"

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

FEEDBACK_COLUMNS = [
    "date",
    "target",
    "predicted",
    "actual",
]


# ============================================================
# GOOGLE SHEETS CONNECTION
# ============================================================

@st.cache_resource
def get_google_connection():

    return st.connection(
        "gsheets",
        type=GSheetsConnection
    )


# ============================================================
# LOAD WEATHER CSV
# ============================================================

@st.cache_data
def load_weather():

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"Could not find Claes.love.csv here:\n{CSV_FILE}"
        )

    data = pd.read_csv(CSV_FILE)

    # Remove accidental spaces from column names
    data.columns = data.columns.str.strip()

    required_columns = {
        "date",
        "temperature_max_c",
        "temperature_min_c",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "cloud_cover_okta",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:

        raise ValueError(
            "The CSV is missing these columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    for column in FEATURE_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # ========================================================
    # CREATE WEATHER CONDITION IF NECESSARY
    # ========================================================

    if "weather_condition" not in data.columns:

        def determine_condition(row):

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

        data["weather_condition"] = data.apply(
            determine_condition,
            axis=1
        )

    data["weather_condition"] = (
        data["weather_condition"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = (
        data
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# GOOGLE SHEETS FEEDBACK
# ============================================================

def load_feedback():

    try:

        conn = get_google_connection()

        feedback = conn.read(
            worksheet="Feedback",
            ttl=0
        )

        if feedback is None:
            return pd.DataFrame(
                columns=FEEDBACK_COLUMNS
            )

        feedback.columns = (
            feedback.columns
            .astype(str)
            .str.strip()
        )

        # If the worksheet is empty
        if len(feedback.columns) == 0:

            return pd.DataFrame(
                columns=FEEDBACK_COLUMNS
            )

        # Make sure all expected columns exist
        for column in FEEDBACK_COLUMNS:

            if column not in feedback.columns:

                feedback[column] = np.nan

        feedback = feedback[
            FEEDBACK_COLUMNS
        ]

        feedback = feedback.dropna(
            how="all"
        )

        return feedback.reset_index(
            drop=True
        )

    except Exception as error:

        st.warning(
            "Google Sheets feedback is not "
            f"available yet: {error}"
        )

        return pd.DataFrame(
            columns=FEEDBACK_COLUMNS
        )


def save_feedback(
    date,
    target,
    predicted,
    actual
):

    conn = get_google_connection()

    # Read current feedback
    try:

        feedback = conn.read(
            worksheet="Feedback",
            ttl=0
        )

        if feedback is None:
            feedback = pd.DataFrame(
                columns=FEEDBACK_COLUMNS
            )

    except Exception:

        feedback = pd.DataFrame(
            columns=FEEDBACK_COLUMNS
        )

    feedback.columns = (
        feedback.columns
        .astype(str)
        .str.strip()
    )

    # Make sure columns exist
    for column in FEEDBACK_COLUMNS:

        if column not in feedback.columns:

            feedback[column] = np.nan

    feedback = feedback[
        FEEDBACK_COLUMNS
    ]

    feedback = feedback.dropna(
        how="all"
    )

    # Add the new correction
    new_feedback = pd.DataFrame(
        [
            {
                "date": str(date),
                "target": str(target),
                "predicted": predicted,
                "actual": actual,
            }
        ]
    )

    updated_feedback = pd.concat(
        [
            feedback,
            new_feedback
        ],
        ignore_index=True
    )

    # Write the complete updated table
    conn.update(
        worksheet="Feedback",
        data=updated_feedback
    )

    # Clear cached Google Sheet data
    st.cache_data.clear()


# ============================================================
# APPLY FEEDBACK TO WEATHER DATA
# ============================================================

def apply_feedback(
    data,
    target_column
):

    feedback = load_feedback()

    if feedback.empty:

        return data

    relevant = feedback[
        feedback["target"] == target_column
    ].copy()

    if relevant.empty:

        return data

    result = data.copy()

    for _, row in relevant.iterrows():

        try:

            feedback_date = pd.to_datetime(
                row["date"]
            )

            actual_value = row["actual"]

            if target_column == "weather_condition":

                actual_value = (
                    str(actual_value)
                    .strip()
                    .lower()
                )

            else:

                actual_value = float(
                    actual_value
                )

            matching_date = (
                result["date"]
                == feedback_date
            )

            if matching_date.any():

                result.loc[
                    matching_date,
                    target_column
                ] = actual_value

            else:

                # Create a new row if the
                # feedback date isn't in CSV
                new_row = {
                    column: np.nan
                    for column in result.columns
                }

                new_row["date"] = feedback_date

                new_row[target_column] = (
                    actual_value
                )

                result = pd.concat(
                    [
                        result,
                        pd.DataFrame([new_row])
                    ],
                    ignore_index=True
                )

        except Exception:

            continue

    return (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(data):

    result = data.copy()

    for column in FEATURE_COLUMNS:

        for lag in range(
            0,
            LAGS + 1
        ):

            result[
                f"{column}_lag_{lag}"
            ] = result[column].shift(lag)

    for column in FEATURE_COLUMNS:

        result[
            f"{column}_avg_3"
        ] = (
            result[column]
            .rolling(3)
            .mean()
        )

        result[
            f"{column}_avg_7"
        ] = (
            result[column]
            .rolling(7)
            .mean()
        )

    day_of_year = (
        result["date"].dt.dayofyear
    )

    result["day_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    result["day_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return result


def get_model_features():

    features = []

    for column in FEATURE_COLUMNS:

        for lag in range(
            0,
            LAGS + 1
        ):

            features.append(
                f"{column}_lag_{lag}"
            )

    for column in FEATURE_COLUMNS:

        features.append(
            f"{column}_avg_3"
        )

        features.append(
            f"{column}_avg_7"
        )

    features.extend(
        [
            "day_sin",
            "day_cos"
        ]
    )

    return features


# ============================================================
# MODELS
# ============================================================

def make_regression_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_TREES,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1
                )
            )
        ]
    )


def make_classification_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_TREES,
                    random_state=42,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

@st.cache_resource
def train_random_forest(
    original_data,
    target_column,
    days_ahead,
    model_type
):

    # Add user corrections
    data = apply_feedback(
        original_data,
        target_column
    )

    features_data = create_features(
        data
    )

    features = get_model_features()

    # --------------------------------------------------------
    # Rain classification
    # --------------------------------------------------------

    if target_column == "rain_target":

        rain_data = data.copy()

        rain_data["rain_target"] = np.where(
            rain_data[
                "precipitation_mm"
            ].isna(),
            np.nan,
            (
                rain_data[
                    "precipitation_mm"
                ] > 0.1
            ).astype(float)
        )

        features_data = create_features(
            rain_data
        )

        features_data["target"] = (
            features_data[
                "rain_target"
            ].shift(-days_ahead)
        )

    else:

        features_data["target"] = (
            features_data[
                target_column
            ].shift(-days_ahead)
        )

    if target_column == "weather_condition":

        features_data.loc[
            features_data["target"]
            == "unknown",
            "target"
        ] = np.nan

    training = features_data.dropna(
        subset=["target"]
    )

    X = training[features]
    y = training["target"]

    if model_type == "regression":

        model = make_regression_model()

    else:

        model = make_classification_model()

    model.fit(X, y)

    return model


# ============================================================
# LATEST FEATURES
# ============================================================

def get_latest_features(data):

    features_data = create_features(
        data
    )

    return features_data.iloc[
        [-1]
    ][get_model_features()]


# ============================================================
# NUMERIC PREDICTION
# ============================================================

def predict_numeric(
    data,
    target_column,
    days_ahead
):

    model = train_random_forest(
        data,
        target_column,
        days_ahead,
        "regression"
    )

    latest = get_latest_features(
        data
    )

    return float(
        model.predict(latest)[0]
    )


# ============================================================
# RAIN PROBABILITY
# ============================================================

def add_rain_target(data):

    result = data.copy()

    result["rain_target"] = np.where(
        result["precipitation_mm"].isna(),
        np.nan,
        (
            result["precipitation_mm"] > 0.1
        ).astype(float)
    )

    return result


def predict_rain_probability(
    data,
    days_ahead
):

    rain_data = add_rain_target(
        data
    )

    model = train_random_forest(
        rain_data,
        "rain_target",
        days_ahead,
        "classification"
    )

    latest = get_latest_features(
        rain_data
    )

    probabilities = (
        model.predict_proba(latest)[0]
    )

    classes = list(
        model.named_steps[
            "model"
        ].classes_
    )

    if 1.0 in classes:

        probability = (
            probabilities[
                classes.index(1.0)
            ] * 100
        )

    else:

        probability = 0.0

    return float(
        np.clip(
            probability,
            0,
            100
        )
    )


# ============================================================
# WEATHER CONDITION
# ============================================================

def predict_weather_condition(
    data,
    days_ahead
):

    model = train_random_forest(
        data,
        "weather_condition",
        days_ahead,
        "classification"
    )

    latest = get_latest_features(
        data
    )

    return str(
        model.predict(latest)[0]
    )


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

    q = question.lower()

    if (
        "tomorrow" in q
        or "next day" in q
    ):

        return 1

    patterns = [

        r"(?:in|after)\s+"
        r"(\d+|one|two|three|four|five)"
        r"\s+days?",

        r"(\d+|one|two|three|four|five)"
        r"\s+days?\s+from\s+now",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            value = match.group(1)

            if value.isdigit():

                return int(value)

            return NUMBER_WORDS[value]

    return 1


def get_topic(question):

    q = question.lower()

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
        ]
    ):

        return "precipitation"

    if (
        "humidity" in q
        or "humid" in q
    ):

        return "humidity"

    if (
        "wind" in q
        or "windy" in q
    ):

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

    if (
        "temperature" in q
        or "temp" in q
        or "hot" in q
        or "cold" in q
    ):

        return "temperature_max"

    if any(
        word in q
        for word in [
            "weather",
            "condition",
            "sunny",
            "cloudy",
            "rainy",
        ]
    ):

        return "weather"

    return None


# ============================================================
# MAKE PREDICTION
# ============================================================

def make_prediction(
    question,
    data
):

    days = get_days_ahead(
        question
    )

    if (
        days < 1
        or days > MAX_FORECAST_DAYS
    ):

        return None

    topic = get_topic(
        question
    )

    if topic is None:

        return None

    last_date = data["date"].iloc[-1]

    forecast_date = (
        last_date
        + timedelta(days=days)
    )

    if topic == "temperature_max":

        value = predict_numeric(
            data,
            "temperature_max_c",
            days
        )

        return {
            "topic": "temperature_max_c",
            "display": f"{value:.1f}°C",
            "value": value,
            "date": forecast_date,
        }

    if topic == "temperature_min":

        value = predict_numeric(
            data,
            "temperature_min_c",
            days
        )

        return {
            "topic": "temperature_min_c",
            "display": f"{value:.1f}°C",
            "value": value,
            "date": forecast_date,
        }

    if topic == "humidity":

        value = np.clip(
            predict_numeric(
                data,
                "humidity_percent",
                days
            ),
            0,
            100
        )

        return {
            "topic": "humidity_percent",
            "display": f"{value:.1f}%",
            "value": value,
            "date": forecast_date,
        }

    if topic == "wind":

        value = max(
            0,
            predict_numeric(
                data,
                "wind_speed_kmh",
                days
            )
        )

        return {
            "topic": "wind_speed_kmh",
            "display": f"{value:.1f} km/h",
            "value": value,
            "date": forecast_date,
        }

    if topic == "precipitation":

        value = max(
            0,
            predict_numeric(
                data,
                "precipitation_mm",
                days
            )
        )

        return {
            "topic": "precipitation_mm",
            "display": f"{value:.1f} mm",
            "value": value,
            "date": forecast_date,
        }

    if topic == "rain_probability":

        value = predict_rain_probability(
            data,
            days
        )

        return {
            "topic": "rain_probability",
            "display": f"{value:.1f}%",
            "value": value,
            "date": forecast_date,
        }

    if topic == "weather":

        value = predict_weather_condition(
            data,
            days
        )

        return {
            "topic": "weather_condition",
            "display": value,
            "value": value,
            "date": forecast_date,
        }

    return None


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Senegal Weather Predictor",
    page_icon="🌤️",
    layout="centered"
)

st.title(
    "🌤️ Senegal Weather Predictor"
)

st.write(
    "Ask about temperature, humidity, rain, "
    "precipitation, wind, or weather condition "
    "for tomorrow or up to 5 days ahead."
)

st.caption(
    "The model reads Claes.love.csv automatically. "
    "You do not need to enter today's weather."
)


# ============================================================
# LOAD WEATHER
# ============================================================

try:

    weather = load_weather()

except Exception as error:

    st.error(
        f"CSV error: {error}"
    )

    st.stop()


last_date = weather["date"].iloc[-1]

st.info(
    "Latest date in the CSV: "
    f"**{last_date.strftime('%B %d, %Y')}**"
)


# ============================================================
# TRAIN MODELS
# ============================================================

try:

    with st.spinner(
        "Training the machine-learning models..."
    ):

        for days in range(
            1,
            MAX_FORECAST_DAYS + 1
        ):

            for target in REGRESSION_TARGETS:

                train_random_forest(
                    weather,
                    target,
                    days,
                    "regression"
                )

            train_random_forest(
                weather,
                "rain_target",
                days,
                "classification"
            )

            train_random_forest(
                weather,
                "weather_condition",
                days,
                "classification"
            )

except Exception as error:

    st.error(
        "The machine-learning models "
        f"could not be trained: {error}"
    )

    st.stop()


# ============================================================
# QUESTION BOX
# ============================================================

question = st.text_input(
    "Ask your weather question:",
    placeholder=(
        "Example: What will the weather be in 5 days?"
    )
)


if st.button(
    "Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please type a weather question."
        )

    else:

        prediction = make_prediction(
            question,
            weather
        )

        if prediction is None:

            st.warning(
                "I can answer questions about "
                "temperature, humidity, rain, "
                "precipitation, wind, or weather "
                "from tomorrow up to 5 days ahead."
            )

        else:

            date_text = prediction[
                "date"
            ].strftime(
                "%B %d, %Y"
            )

            st.success(
                f"Prediction for **{date_text}**: "
                f"**{prediction['display']}**"
            )

            st.session_state[
                "prediction"
            ] = prediction


# ============================================================
# CORRECT / WRONG
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state[
        "prediction"
    ]

    st.subheader(
        "Was this prediction correct?"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "✅ Correct",
            use_container_width=True
        ):

            save_feedback(
                prediction["date"].strftime(
                    "%Y-%m-%d"
                ),
                prediction["topic"],
                prediction["value"],
                prediction["value"]
            )

            train_random_forest.clear()

            st.success(
                "Thanks! I saved that this "
                "prediction was correct."
            )

    with col2:

        if st.button(
            "❌ Wrong",
            use_container_width=True
        ):

            st.session_state[
                "show_correction"
            ] = True


# ============================================================
# CORRECTION FORM
# ============================================================

if st.session_state.get(
    "show_correction",
    False
):

    prediction = st.session_state[
        "prediction"
    ]

    st.subheader(
        "🧠 Help the model learn"
    )

    if prediction["topic"] == (
        "weather_condition"
    ):

        actual = st.selectbox(
            "What was the actual weather condition?",
            [
                "sunny",
                "mostly sunny",
                "partly cloudy",
                "mostly cloudy",
                "cloudy",
                "light rain",
                "rainy",
            ]
        )

    else:

        actual = st.text_input(
            "What was the actual value?",
            placeholder="Example: 32.5"
        )

    if st.button(
        "Save correction",
        type="primary"
    ):

        if not str(actual).strip():

            st.warning(
                "Please enter the actual result."
            )

        else:

            try:

                if prediction["topic"] == (
                    "weather_condition"
                ):

                    actual_value = (
                        str(actual)
                        .strip()
                        .lower()
                    )

                else:

                    actual_value = float(
                        actual
                    )

                save_feedback(
                    prediction["date"].strftime(
                        "%Y-%m-%d"
                    ),
                    prediction["topic"],
                    prediction["value"],
                    actual_value
                )

                train_random_forest.clear()

                st.session_state[
                    "show_correction"
                ] = False

                st.success(
                    "Correction saved to Google Sheets! "
                    "The model can use it during "
                    "future training."
                )

            except ValueError:

                st.error(
                    "Please enter a valid number."
                )


# ============================================================
# FIVE-DAY FORECAST
# ============================================================

st.subheader(
    "📅 Model Forecast: Next 5 Days"
)

forecast_rows = []

try:

    for days in range(
        1,
        MAX_FORECAST_DAYS + 1
    ):

        forecast_rows.append(
            {
                "Date": (
                    last_date
                    + timedelta(days=days)
                ),

                "Max °C": predict_numeric(
                    weather,
                    "temperature_max_c",
                    days
                ),

                "Min °C": predict_numeric(
                    weather,
                    "temperature_min_c",
                    days
                ),

                "Humidity %": predict_numeric(
                    weather,
                    "humidity_percent",
                    days
                ),

                "Rain chance %": (
                    predict_rain_probability(
                        weather,
                        days
                    )
                ),

                "Rain mm": predict_numeric(
                    weather,
                    "precipitation_mm",
                    days
                ),

                "Wind km/h": predict_numeric(
                    weather,
                    "wind_speed_kmh",
                    days
                ),

                "Weather": (
                    predict_weather_condition(
                        weather,
                        days
                    )
                )
            }
        )

    forecast_table = pd.DataFrame(
        forecast_rows
    )

    forecast_table["Date"] = (
        forecast_table["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        forecast_table.round(1),
        hide_index=True,
        use_container_width=True
    )

except Exception as error:

    st.error(
        "The 5-day forecast could not "
        f"be generated: {error}"
    )


# ============================================================
# FEEDBACK HISTORY
# ============================================================

with st.expander(
    "🧠 View what the model has learned"
):

    feedback = load_feedback()

    if feedback.empty:

        st.write(
            "No feedback has been saved yet."
        )

    else:

        st.write(
            f"The model has received "
            f"**{len(feedback)}** feedback records."
        )

        st.dataframe(
            feedback,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# HISTORICAL CSV
# ============================================================

with st.expander(
    "📊 View the historical CSV"
):

    st.dataframe(
        weather,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# EXPLANATION
# ============================================================

with st.expander(
    "ℹ️ How the machine-learning model works"
):

    st.write(
        "The model uses the weather information "
        "in Claes.love.csv and previous 7-day "
        "weather patterns."
    )

    st.write(
        "It creates 3-day and 7-day averages "
        "and seasonal date features."
    )

    st.write(
        "Random Forest regression predicts "
        "temperature, humidity, wind, and "
        "precipitation."
    )

    st.write(
        "Random Forest classification predicts "
        "rain probability and weather condition."
    )

    st.write(
        "User corrections are stored in Google "
        "Sheets and are included when the model "
        "is trained again."
    )

st.caption(
    "Educational project. Historical data "
    "cannot guarantee future weather."
)
