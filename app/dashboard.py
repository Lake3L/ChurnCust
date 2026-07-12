"""Streamlit-дашборд платформы скоринга оттока.

Одна страница, пять секций: статус API, скоринг клиента, экономика порога,
последние предсказания и статус дрифта. Дашборд только читает данные сервиса
(REST /predict, /health) и артефакты пайплайна (reports/, predictions.jsonl либо
таблицу predictions в Postgres) — он ничего не обучает и не изменяет.
"""

import json
import os
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

# --- Константы проекта ------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "reports" / "figures"
DRIFT_SUMMARY_PATH = ROOT / "reports" / "drift" / "drift_summary.json"
PREDICTIONS_JSONL = ROOT / "predictions.jsonl"

# Фирменные цвета проекта.
BLUE = "#2563eb"
ORANGE = "#ea580c"  # noqa: F841 — палитра проекта, зарезервирован для акцентов
GRAY = "#6b7280"
GREEN = "#16a34a"

# Значения категорий берутся строго из app/schemas.py (класс CustomerFeatures).
YES_NO = ["Yes", "No"]
INTERNET_DEP = ["Yes", "No", "No internet service"]

CATEGORY_FIELDS: list[tuple[str, str, list[str]]] = [
    ("gender", "Пол", ["Male", "Female"]),
    ("senior_citizen", "Пенсионер", YES_NO),
    ("partner", "Партнёр", YES_NO),
    ("dependents", "Иждивенцы", YES_NO),
    ("phone_service", "Телефония", YES_NO),
    ("multiple_lines", "Несколько линий", ["Yes", "No", "No phone service"]),
    ("internet_service", "Интернет", ["DSL", "Fiber optic", "No"]),
    ("online_security", "Онлайн-безопасность", INTERNET_DEP),
    ("online_backup", "Онлайн-бэкап", INTERNET_DEP),
    ("device_protection", "Защита устройств", INTERNET_DEP),
    ("tech_support", "Техподдержка", INTERNET_DEP),
    ("streaming_tv", "Стриминг ТВ", INTERNET_DEP),
    ("streaming_movies", "Стриминг фильмов", INTERNET_DEP),
    ("contract", "Контракт", ["Month-to-month", "One year", "Two year"]),
    ("paperless_billing", "Электронные счета", YES_NO),
    (
        "payment_method",
        "Способ оплаты",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    ),
]

# Дефолты — «рискованный» клиент (month-to-month, fiber optic, electronic check).
RISKY_DEFAULTS: dict[str, object] = {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "tenure_months": 2,
    "monthly_charges": 85.0,
}

DISPLAY_COLS = [
    "ts",
    "probability",
    "decision",
    "model_version",
    "contract",
    "tenure_months",
    "monthly_charges",
]


# --- Доступ к API -----------------------------------------------------------


def check_health(api_url: str) -> dict | None:
    """Опросить GET /health. Вернуть JSON-тело либо None, если API недоступен."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def score_customer(api_url: str, payload: dict) -> tuple[dict | None, str | None]:
    """Отправить клиента в POST /predict. Вернуть (ответ, None) или (None, текст ошибки)."""
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    except requests.RequestException as exc:
        return None, f"API недоступен: {exc}"
    if resp.status_code != 200:
        return None, f"Ошибка {resp.status_code}: {resp.text}"
    return resp.json(), None


# --- Чтение логов предсказаний ----------------------------------------------


def _normalize_predictions(records: list[dict]) -> pd.DataFrame:
    """Свести записи лога (из БД или JSONL) к единой таблице для отображения."""
    rows = []
    for rec in records:
        features = rec.get("features") or {}
        if isinstance(features, str):
            with suppress(json.JSONDecodeError):
                features = json.loads(features)
        if not isinstance(features, dict):
            features = {}
        rows.append(
            {
                "ts": rec.get("ts"),
                "probability": rec.get("probability"),
                "decision": rec.get("decision"),
                "model_version": rec.get("model_version"),
                "contract": features.get("contract"),
                "tenure_months": features.get("tenure_months"),
                "monthly_charges": features.get("monthly_charges"),
            }
        )
    return pd.DataFrame(rows, columns=DISPLAY_COLS)


def _load_predictions_from_db(database_url: str) -> pd.DataFrame:
    """Прочитать последние 50 записей из таблицы predictions (Postgres)."""
    engine = create_engine(database_url, pool_pre_ping=True)
    query = text(
        "SELECT ts, probability, decision, threshold, model_version, features "
        "FROM predictions ORDER BY ts DESC LIMIT 50"
    )
    with engine.connect() as conn:
        rows = [dict(row) for row in conn.execute(query).mappings().all()]
    engine.dispose()
    return _normalize_predictions(rows)


def _load_predictions_from_jsonl() -> pd.DataFrame:
    """Прочитать последние 50 записей из predictions.jsonl (новые сверху)."""
    lines = PREDICTIONS_JSONL.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()][-50:]
    records.reverse()
    return _normalize_predictions(records)


@st.cache_data(ttl=30)
def load_predictions() -> tuple[pd.DataFrame, str]:
    """Загрузить лог предсказаний: сначала БД (DATABASE_URL), затем JSONL, иначе пусто."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        with suppress(Exception):
            return _load_predictions_from_db(database_url), "postgres"
    if PREDICTIONS_JSONL.exists():
        with suppress(Exception):
            return _load_predictions_from_jsonl(), "jsonl"
    return pd.DataFrame(columns=DISPLAY_COLS), "none"


@st.cache_data(ttl=30)
def load_drift_summary() -> list[dict] | None:
    """Загрузить reports/drift/drift_summary.json (список сценариев)."""
    if not DRIFT_SUMMARY_PATH.exists():
        return None
    with suppress(json.JSONDecodeError, OSError):
        return json.loads(DRIFT_SUMMARY_PATH.read_text(encoding="utf-8"))
    return None


def _proba_histogram(probs: pd.Series) -> pd.DataFrame:
    """Гистограмма вероятностей по 10 бинам на отрезке [0, 1] для bar-chart."""
    values = pd.to_numeric(probs, errors="coerce").dropna()
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(values, bins=edges)
    labels = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(counts))]
    return pd.DataFrame({"Вероятность": labels, "Количество": counts}).set_index("Вероятность")


# --- Секции UI --------------------------------------------------------------


def render_header(api_url: str) -> None:
    """Заголовок, подзаголовок и статус API (GET /health)."""
    st.title("Churn Scoring Platform")
    st.markdown(
        f"<p style='color:{GRAY};font-size:1.05rem;margin-top:-0.6rem'>"
        "Не «кто уйдёт», а «кому звонить, чтобы заработать»</p>",
        unsafe_allow_html=True,
    )
    health = check_health(api_url)
    if health is None:
        st.error(
            f"API недоступен по адресу {api_url}. "
            "Запусти сервис: `make up` или `uvicorn app.main:app`."
        )
    elif health.get("status") == "ok":
        st.success(
            f"API онлайн · модель {health.get('model_version')} · "
            f"источник {health.get('model_source')}"
        )
    else:
        st.warning("API онлайн, но модель не загружена — запусти `dvc repro`.")


def _render_score_result() -> None:
    """Показать результат последнего скоринга из session_state."""
    error = st.session_state.get("score_error")
    if error:
        st.error(error)
        return
    result = st.session_state.get("score_result")
    if not result:
        return
    col_p, col_ev, col_thr = st.columns(3)
    col_p.metric("Вероятность оттока", f"{result['probability'] * 100:.1f}%")
    col_ev.metric("EV звонка", f"{result['expected_value_of_call']:,.0f} ₽")
    col_thr.metric("Порог", f"{result['threshold']:.3f}")

    is_call = result["decision"] == "call"
    color = GREEN if is_call else GRAY
    label = "Звонить" if is_call else "Не звонить"
    st.markdown(
        f"<div style='display:inline-block;padding:0.4rem 1.3rem;border-radius:0.5rem;"
        f"background:{color};color:#fff;font-weight:600;font-size:1.1rem'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Модель: {result['model_version']}")


def render_scoring(api_url: str) -> None:
    """Форма всех 19 полей клиента и вызов скоринга."""
    st.subheader("Скоринг клиента")
    values: dict[str, object] = {}
    cols = st.columns(3)
    for i, (name, label, options) in enumerate(CATEGORY_FIELDS):
        default = RISKY_DEFAULTS[name]
        values[name] = cols[i % 3].selectbox(label, options, index=options.index(default))

    col_tenure, col_monthly, col_total = st.columns(3)
    tenure = col_tenure.slider(
        "Стаж, мес. (tenure_months)", 0, 72, int(RISKY_DEFAULTS["tenure_months"])
    )
    monthly = col_monthly.slider(
        "Абон. плата, ₽/мес (monthly_charges)",
        18.0,
        120.0,
        float(RISKY_DEFAULTS["monthly_charges"]),
        step=0.5,
    )
    suggested_total = round(monthly * tenure, 2)
    # Ключ зависит от monthly/tenure: авто-подсказка обновляется при их изменении,
    # но остаётся редактируемой между изменениями.
    total = col_total.number_input(
        "Всего оплачено, ₽ (total_charges)",
        min_value=0.0,
        max_value=200000.0,
        value=suggested_total,
        step=10.0,
        key=f"total_charges::{monthly}::{tenure}",
        help="Авто-подсказка = monthly × tenure, можно изменить вручную.",
    )
    values["tenure_months"] = int(tenure)
    values["monthly_charges"] = float(monthly)
    values["total_charges"] = float(total)

    if st.button("Скорить клиента", type="primary", use_container_width=True):
        result, error = score_customer(api_url, values)
        st.session_state["score_result"] = result
        st.session_state["score_error"] = error

    _render_score_result()


def render_economics() -> None:
    """Готовые графики экономики порога и калибровки."""
    st.subheader("Экономика порога")
    profit_png = FIGURES_DIR / "profit_vs_threshold_valid.png"
    calib_png = FIGURES_DIR / "calibration_valid.png"
    if not profit_png.exists() and not calib_png.exists():
        st.info("Графики не найдены — запусти `dvc repro`.")
        return
    col_profit, col_calib = st.columns(2)
    if profit_png.exists():
        col_profit.image(
            str(profit_png),
            caption="Прибыль vs порог: максимум около 0.667 = C/(p·V)",
            use_container_width=True,
        )
    else:
        col_profit.info("profit_vs_threshold_valid.png не найден — запусти `dvc repro`.")
    if calib_png.exists():
        col_calib.image(
            str(calib_png),
            caption="Калибровка: предсказанная вероятность ≈ фактическая частота",
            use_container_width=True,
        )
    else:
        col_calib.info("calibration_valid.png не найден — запусти `dvc repro`.")


def render_recent_predictions() -> None:
    """Последние 50 записей лога и распределение вероятностей."""
    st.subheader("Последние предсказания")
    df, source = load_predictions()
    if df.empty:
        st.info(
            "Пока нет залогированных предсказаний. Сделай скоринг выше или подай нагрузку на API."
        )
        return
    st.caption(f"Источник: {source} · показаны последние {len(df)}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(_proba_histogram(df["probability"]), color=BLUE, use_container_width=True)


def render_drift() -> None:
    """Таблица дрифта по сценариям и итоговый вывод."""
    st.subheader("Дрифт")
    data = load_drift_summary()
    if not data:
        st.info(
            "reports/drift/drift_summary.json не найден — запусти `dvc repro` или `make drift`."
        )
        return
    df = pd.DataFrame(data)
    view = pd.DataFrame(
        {
            "Сценарий": df["scenario"],
            "Dataset drift": df["dataset_drift"],
            "PSI-алерты": df["n_psi_alerts"],
            "PSI(proba)": df["proba_psi"],
            "PR-AUC": df["pr_auc"],
            "Прибыль, ₽": df["profit"],
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)

    has_alerts = bool(
        (df["n_psi_alerts"] > 0).any() or df["proba_psi_alert"].any() or df["dataset_drift"].any()
    )
    if has_alerts:
        st.warning("Есть PSI-алерты хотя бы в одном сценарии — проверь распределения признаков.")
    else:
        st.success("Алертов по дрифту нет.")
    st.caption("Concept drift без разметки не детектируется — виден только по падению метрик.")


def main() -> None:
    """Собрать одностраничный дашборд."""
    st.set_page_config(page_title="Churn Scoring Platform", layout="wide")
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

    render_header(api_url)
    st.divider()
    render_scoring(api_url)
    st.divider()
    render_economics()
    st.divider()
    render_recent_predictions()
    st.divider()
    render_drift()


if __name__ == "__main__":
    main()
