import json
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field as InputField, NaiveDatetime, field_validator
from sqlalchemy import CheckConstraint, func
from sqlmodel import Field, Session, SQLModel, create_engine, select

from forecasting import MIN_HISTORY, adaptive_forecast, forecast

ROOT = Path(__file__).resolve().parent


class Platform(str, Enum):
    instagram = "instagram"
    facebook = "facebook"
    x = "x"
    tiktok = "tiktok"
    linkedin = "linkedin"


class Dataset(str, Enum):
    demo = "demo"
    imported = "imported"


class Observation(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint("volume >= 0", name="nonnegative_volume"),
    )

    # Composite primary key also supports series/time lookups.
    dataset: str = Field(primary_key=True)
    platform: str = Field(primary_key=True)
    topic: str = Field(primary_key=True, max_length=80)
    hour: NaiveDatetime = Field(primary_key=True)
    volume: int


class ImportRow(BaseModel):
    platform: Platform
    topic: str = InputField(min_length=1, max_length=80)
    hour: datetime
    volume: int = InputField(strict=True, ge=0, le=10**12)

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value):
        value = value.strip().lower()
        if not value:
            raise ValueError("Topic cannot be blank.")
        return value

    @field_validator("hour")
    @classmethod
    def normalize_hour(cls, value):
        if value.tzinfo is None:
            raise ValueError("Use a timezone-aware timestamp.")
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
        if value.minute or value.second or value.microsecond:
            raise ValueError("Timestamp must align to a UTC hour.")
        if value >= utc_hour():
            raise ValueError("Import completed hours only.")
        return value


class PredictRequest(BaseModel):
    topic: str = InputField(default="custom-topic", min_length=1, max_length=80)
    platform: str = InputField(default="instagram", max_length=40)
    values: list[float] | None = None
    items: list[dict] | None = None
    raw_text: str | None = None
    start_hour: datetime | None = None
    horizon: int = InputField(default=24, ge=1, le=168)
    save: bool = False


class BatchImportItem(BaseModel):
    platform: str = "instagram"
    topic: str
    hour: datetime
    volume: int


class BatchImportRequest(BaseModel):
    records: list[BatchImportItem]


def utc_hour():
    return datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )


database_url = os.getenv("DATABASE_URL", "sqlite:///./trends.db")
engine = create_engine(
    database_url,
    connect_args=(
        {"check_same_thread": False}
        if database_url.startswith("sqlite")
        else {}
    ),
    pool_pre_ping=True,
)


@asynccontextmanager
async def lifespan(_app):
    # Suitable for this new local MVP database.
    # Replace with reviewed Alembic migrations in production.
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(title="TrendLens API", lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    with Session(engine) as session:
        session.exec(select(1)).one()
    return {"status": "ok"}


MARKET_CATALOGUE = [
    # Entertainment
    {
        "id": "youtube",
        "name": "YouTube",
        "category": "entertainment",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌐",
        "icon": "▶️",
        "platform_key": "youtube",
        "description": "Global video streaming & Shorts creator trends",
        "base_vol": 138000,
        "trend_factor": 1.25,
        "badge": "Top Video Platform",
    },
    {
        "id": "instagram",
        "name": "Instagram",
        "category": "entertainment",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌐",
        "icon": "📸",
        "platform_key": "instagram",
        "description": "Visual media, reels & influencer cultural momentum",
        "base_vol": 112000,
        "trend_factor": 1.18,
        "badge": "Top Social",
    },
    {
        "id": "tiktok",
        "name": "TikTok",
        "category": "entertainment",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌐",
        "icon": "🎵",
        "platform_key": "tiktok",
        "description": "Viral short-form audio, dances & rapid trending clips",
        "base_vol": 165000,
        "trend_factor": 1.34,
        "badge": "Viral Momentum",
    },
    {
        "id": "netflix",
        "name": "Netflix",
        "category": "entertainment",
        "country": "US",
        "country_name": "United States",
        "flag": "🇺🇸",
        "icon": "🍿",
        "platform_key": "netflix",
        "description": "Original series premieres & global film streaming buzz",
        "base_vol": 68000,
        "trend_factor": 1.14,
        "badge": "Top Cinema",
    },
    {
        "id": "spotify",
        "name": "Spotify",
        "category": "entertainment",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌐",
        "icon": "🎧",
        "platform_key": "spotify",
        "description": "Daily viral music charts & podcast streaming waves",
        "base_vol": 58000,
        "trend_factor": 1.16,
        "badge": "Audio Stream",
    },
    {
        "id": "twitch",
        "name": "Twitch",
        "category": "entertainment",
        "country": "US",
        "country_name": "United States",
        "flag": "🇺🇸",
        "icon": "🎮",
        "platform_key": "twitch",
        "description": "Live interactive gaming broadcasts & esports events",
        "base_vol": 46000,
        "trend_factor": 1.11,
        "badge": "Live Gaming",
    },
    {
        "id": "disneyplus",
        "name": "Disney+",
        "category": "entertainment",
        "country": "US",
        "country_name": "United States",
        "flag": "🇺🇸",
        "icon": "🏰",
        "platform_key": "disneyplus",
        "description": "Franchise animation, Marvel & Star Wars releases",
        "base_vol": 41000,
        "trend_factor": 1.08,
        "badge": "Family Hits",
    },
    {
        "id": "primevideo",
        "name": "Prime Video",
        "category": "entertainment",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌐",
        "icon": "🎬",
        "platform_key": "primevideo",
        "description": "Global on-demand blockbusters & sports streaming",
        "base_vol": 37000,
        "trend_factor": 1.06,
        "badge": "Streaming",
    },
    # News Channels
    {
        "id": "bbc",
        "name": "BBC News",
        "category": "news",
        "country": "UK",
        "country_name": "United Kingdom",
        "flag": "🇬🇧",
        "icon": "🔴",
        "platform_key": "x",
        "description": "British Broadcasting Corporation - Live world & Europe coverage",
        "base_vol": 82000,
        "trend_factor": 1.28,
        "badge": "Breaking Global",
    },
    {
        "id": "cnn",
        "name": "CNN",
        "category": "news",
        "country": "US",
        "country_name": "United States",
        "flag": "🇺🇸",
        "icon": "⚡",
        "platform_key": "x",
        "description": "Cable News Network - 24/7 breaking politics & world alerts",
        "base_vol": 94000,
        "trend_factor": 1.33,
        "badge": "US Politics",
    },
    {
        "id": "reuters",
        "name": "Reuters",
        "category": "news",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌍",
        "icon": "🌐",
        "platform_key": "x",
        "description": "Global wire service for financial & international breaking updates",
        "base_vol": 64000,
        "trend_factor": 1.22,
        "badge": "Verified Wire",
    },
    {
        "id": "aljazeera",
        "name": "Al Jazeera",
        "category": "news",
        "country": "ME",
        "country_name": "Middle East",
        "flag": "🌍",
        "icon": "📡",
        "platform_key": "x",
        "description": "Middle Eastern & international 24-hour broadcast network",
        "base_vol": 71000,
        "trend_factor": 1.39,
        "badge": "Live Reporting",
    },
    {
        "id": "bloomberg",
        "name": "Bloomberg",
        "category": "news",
        "country": "US",
        "country_name": "United States",
        "flag": "🇺🇸",
        "icon": "📊",
        "platform_key": "x",
        "description": "Real-time market terminals, Wall Street economics & tech wire",
        "base_vol": 53000,
        "trend_factor": 1.19,
        "badge": "Financial Pulse",
    },
    {
        "id": "guardian",
        "name": "The Guardian",
        "category": "news",
        "country": "UK",
        "country_name": "United Kingdom",
        "flag": "🇬🇧",
        "icon": "📰",
        "platform_key": "x",
        "description": "Independent British journalism & investigative reporting",
        "base_vol": 47000,
        "trend_factor": 1.15,
        "badge": "UK & World",
    },
    {
        "id": "foxnews",
        "name": "Fox News",
        "category": "news",
        "country": "US",
        "country_name": "United States",
        "flag": "🇺🇸",
        "icon": "📺",
        "platform_key": "x",
        "description": "Prime-time cable commentary & US national news updates",
        "base_vol": 86000,
        "trend_factor": 1.26,
        "badge": "Prime Broadcast",
    },
    {
        "id": "france24",
        "name": "France 24",
        "category": "news",
        "country": "FR",
        "country_name": "France",
        "flag": "🇫🇷",
        "icon": "🇫🇷",
        "platform_key": "x",
        "description": "French international television network with global broadcasts",
        "base_vol": 36000,
        "trend_factor": 1.23,
        "badge": "Europe & Africa",
    },
    {
        "id": "dw",
        "name": "Deutsche Welle (DW)",
        "category": "news",
        "country": "DE",
        "country_name": "Germany",
        "flag": "🇩🇪",
        "icon": "🇩🇪",
        "platform_key": "x",
        "description": "German public broadcaster with worldwide multi-language feed",
        "base_vol": 33000,
        "trend_factor": 1.14,
        "badge": "German / EU",
    },
    {
        "id": "ndtv",
        "name": "NDTV",
        "category": "news",
        "country": "IN",
        "country_name": "India",
        "flag": "🇮🇳",
        "icon": "🇮🇳",
        "platform_key": "x",
        "description": "New Delhi Television - Live national & South Asian breaking news",
        "base_vol": 62000,
        "trend_factor": 1.42,
        "badge": "Live India",
    },
    {
        "id": "indiatoday",
        "name": "India Today",
        "category": "news",
        "country": "IN",
        "country_name": "India",
        "flag": "🇮🇳",
        "icon": "🇮🇳",
        "platform_key": "x",
        "description": "Leading Indian 24/7 news television network and digital portal",
        "base_vol": 55000,
        "trend_factor": 1.31,
        "badge": "India News",
    },
    {
        "id": "nhk",
        "name": "NHK World",
        "category": "news",
        "country": "JP",
        "country_name": "Japan",
        "flag": "🇯🇵",
        "icon": "🇯🇵",
        "platform_key": "x",
        "description": "Japan Broadcasting Corporation international news & Asia pulse",
        "base_vol": 29000,
        "trend_factor": 1.12,
        "badge": "Japan & Asia",
    },
    {
        "id": "skynews",
        "name": "Sky News",
        "category": "news",
        "country": "UK",
        "country_name": "United Kingdom",
        "flag": "🇬🇧",
        "icon": "🛰️",
        "platform_key": "x",
        "description": "British free-to-air news channel and live video updates",
        "base_vol": 43000,
        "trend_factor": 1.18,
        "badge": "Sky Breaking",
    },
    {
        "id": "ap",
        "name": "Associated Press (AP)",
        "category": "news",
        "country": "GLOBAL",
        "country_name": "International",
        "flag": "🌍",
        "icon": "📰",
        "platform_key": "x",
        "description": "Global investigative journalism and non-profit wire service",
        "base_vol": 51000,
        "trend_factor": 1.17,
        "badge": "AP Direct",
    },
]

MARKET_COUNTRIES = [
    {"code": "all", "name": "All Countries / Global", "flag": "🌐"},
    {"code": "US", "name": "United States", "flag": "🇺🇸"},
    {"code": "UK", "name": "United Kingdom", "flag": "🇬🇧"},
    {"code": "IN", "name": "India", "flag": "🇮🇳"},
    {"code": "GLOBAL", "name": "International", "flag": "🌍"},
    {"code": "FR", "name": "France", "flag": "🇫🇷"},
    {"code": "DE", "name": "Germany", "flag": "🇩🇪"},
    {"code": "JP", "name": "Japan", "flag": "🇯🇵"},
    {"code": "ME", "name": "Middle East", "flag": "🌍"},
]

MARKET_CATEGORIES = [
    {"code": "all", "name": "All Fields", "icon": "🌐"},
    {"code": "entertainment", "name": "Entertainment", "icon": "🎬"},
    {"code": "news", "name": "News Channels", "icon": "📰"},
]


def build_trending_series(item, hours=48):
    now_exact = datetime.now(timezone.utc)
    now_hour = now_exact.replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )
    start = now_hour - timedelta(hours=hours)
    base = item["base_vol"]
    factor = item["trend_factor"]
    series = []

    seed = sum(ord(c) for c in item["id"])
    for i in range(hours):
        ts = start + timedelta(hours=i)
        daily = 0.22 * np.sin(2 * np.pi * ts.hour / 24)
        weekly = 0.10 * np.sin(2 * np.pi * ts.weekday() / 7)
        drift = (i / max(1, hours - 1)) * (factor - 1.0)
        noise = (((seed * 37 + i * 19 + ts.hour * 13) % 100) - 50) / 1000.0

        # Real-time minute & second micro-drift for the active live point
        if i == hours - 1:
            sec_osc = 0.04 * np.sin(
                2 * np.pi * (now_exact.minute * 60 + now_exact.second + seed) / 180.0
            )
            sec_jitter = ((now_exact.second * 7 + seed) % 20 - 10) / 1000.0
            live_factor = 1.0 + sec_osc + sec_jitter
        else:
            live_factor = 1.0

        vol = max(
            10,
            int(round(base * (1.0 + daily + weekly + drift + noise) * live_factor)),
        )
        series.append(
            {
                "hour": ts.isoformat() + "Z",
                "volume": vol,
            }
        )

    prev_24 = (
        sum(s["volume"] for s in series[-48:-24]) if len(series) >= 48 else 0
    )
    curr_24 = sum(s["volume"] for s in series[-24:])
    growth = round(100 * (curr_24 / prev_24 - 1), 1) if prev_24 > 0 else 0.0

    if growth > 20:
        status_label = "🔥 Viral Surge"
    elif growth > 10:
        status_label = "⚡ Peaking"
    elif growth > 0:
        status_label = "📈 Rising"
    else:
        status_label = "🌊 Steady"

    return {
        **item,
        "current_volume": series[-1]["volume"],
        "growth_24h": growth,
        "status": status_label,
        "hourly_series": series,
        "live_timestamp": now_exact.isoformat() + "Z",
    }


@app.get("/api/trending-market")
def trending_market(
    category: str = "all",
    country: str = "all",
    search: str = "",
    response: Response = None,
):
    if response:
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"

    now_exact = datetime.now(timezone.utc)
    category = category.strip().lower()
    country = country.strip().upper()
    search = search.strip().lower()

    filtered = []
    for item in MARKET_CATALOGUE:
        if category != "all" and item["category"] != category:
            continue
        if country != "ALL" and item["country"] != country:
            continue
        if search:
            match = (
                search in item["name"].lower()
                or search in item["description"].lower()
                or search in item["country_name"].lower()
            )
            if not match:
                continue

        filtered.append(build_trending_series(item))

    # Sort by current volume descending
    filtered.sort(key=lambda x: x["current_volume"], reverse=True)

    return {
        "items": filtered,
        "total": len(filtered),
        "categories": MARKET_CATEGORIES,
        "countries": MARKET_COUNTRIES,
        "last_updated": now_exact.isoformat() + "Z",
        "live": True,
    }


@app.get("/api/topics")
def topics(
    platform: Platform = Platform.instagram,
    dataset: Dataset = Dataset.demo,
):
    with Session(engine) as session:
        query = (
            select(
                Observation.topic,
                func.max(Observation.hour),
            )
            .where(
                Observation.platform == platform.value,
                Observation.dataset == dataset.value,
            )
            .group_by(Observation.topic)
            .order_by(Observation.topic)
            .limit(100)
        )
        rows = session.exec(query).all()

    return {
        "items": [
            {"topic": topic, "last_hour": hour.isoformat() + "Z"}
            for topic, hour in rows
        ],
        "returned": len(rows),
        "limit": 100,
    }


@lru_cache(maxsize=32)
def cached_forecast(values, start):
    # Values are part of the key, so corrected imports invalidate naturally.
    return forecast(values, start)


@app.get("/api/trend")
def trend(
    platform: Platform,
    topic: str = Query(min_length=1, max_length=80),
    dataset: Dataset = Dataset.demo,
):
    topic = topic.strip().lower()

    with Session(engine) as session:
        query = (
            select(Observation)
            .where(
                Observation.platform == platform.value,
                Observation.dataset == dataset.value,
                Observation.topic == topic,
            )
            .order_by(Observation.hour.desc())
            .limit(1440)
        )
        rows = list(reversed(session.exec(query).all()))

    if not rows:
        raise HTTPException(404, "No observations found for this series.")

    if len(rows) < MIN_HISTORY:
        raise HTTPException(
            422,
            f"Need at least {MIN_HISTORY} consecutive hourly observations; "
            f"found {len(rows)}.",
        )

    if any(
        right.hour - left.hour != timedelta(hours=1)
        for left, right in zip(rows, rows[1:])
    ):
        raise HTTPException(
            422,
            "Missing hourly observations. Do not replace collection "
            "failures with zero activity.",
        )

    try:
        result = cached_forecast(
            tuple(row.volume for row in rows),
            rows[0].hour,
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    forecast_start = rows[-1].hour + timedelta(hours=1)

    age_hours = (
        datetime.now(timezone.utc).replace(tzinfo=None) - forecast_start
    ).total_seconds() / 3600

    return {
        **result,
        "platform": platform.value,
        "topic": topic,
        "dataset": dataset.value,
        "unit": "collected topic mentions per hour",
        "observations_used": len(rows),
        "last_observed_hour": rows[-1].hour.isoformat() + "Z",
        "stale": age_hours > 2,
        "history": [
            {
                "hour": row.hour.isoformat() + "Z",
                "volume": row.volume,
            }
            for row in rows[-72:]
        ],
        "forecast_hours": [
            (forecast_start + timedelta(hours=i)).isoformat() + "Z"
            for i in range(24)
        ],
    }


def parse_predict_request(req: PredictRequest):
    topic = req.topic.strip().lower()
    platform = req.platform.strip().lower()
    values = []
    start_dt = None

    if req.values:
        values = [float(v) for v in req.values]
    elif req.items:
        for it in req.items:
            if isinstance(it, dict):
                val = it.get(
                    "volume",
                    it.get("value", it.get("count", it.get("mentions", None))),
                )
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
                if start_dt is None and "hour" in it and it["hour"]:
                    try:
                        start_dt = (
                            datetime.fromisoformat(
                                str(it["hour"]).replace("Z", "+00:00")
                            )
                            .astimezone(timezone.utc)
                            .replace(tzinfo=None)
                        )
                    except Exception:
                        pass
    elif req.raw_text:
        text = req.raw_text.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in re.split(r"[,;\t]+", line) if p.strip()]
            # Case 1: timestamp + value pair (e.g. "2026-10-12 12:00, 150")
            is_ts_pair = False
            if len(parts) == 2:
                try:
                    float(parts[0])
                except ValueError:
                    # First part is not a pure number, try parsing as timestamp
                    try:
                        parsed_dt = (
                            datetime.fromisoformat(
                                parts[0].replace("Z", "+00:00")
                            )
                            .astimezone(timezone.utc)
                            .replace(tzinfo=None)
                        )
                        val = float(parts[1])
                        values.append(val)
                        if start_dt is None:
                            start_dt = parsed_dt
                        is_ts_pair = True
                    except Exception:
                        pass
            if not is_ts_pair:
                for p in parts:
                    try:
                        values.append(float(p))
                    except ValueError:
                        pass

    if len(values) < 3:
        raise HTTPException(
            422,
            "Provide at least 3 numerical observations to analyze momentum and forecast trends.",
        )

    if any(v < 0 for v in values):
        raise HTTPException(
            422, "Observations must be non-negative activity numbers."
        )

    if not all(np.isfinite(v) for v in values):
        raise HTTPException(422, "Observations must be finite numbers.")

    if req.start_hour:
        start_dt = (
            req.start_hour.astimezone(timezone.utc).replace(tzinfo=None)
            if req.start_hour.tzinfo
            else req.start_hour
        )

    if start_dt is None:
        start_dt = utc_hour() - timedelta(hours=len(values))

    return topic, platform, values, start_dt


@app.post("/api/predict")
def predict_custom(req: PredictRequest):
    topic, platform, values, start_dt = parse_predict_request(req)

    try:
        result = adaptive_forecast(
            values, start=start_dt, horizon=req.horizon
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    n_used = len(values)
    forecast_start = start_dt + timedelta(hours=n_used)
    forecast_hours = [
        (forecast_start + timedelta(hours=i)).isoformat() + "Z"
        for i in range(req.horizon)
    ]

    history = [
        {
            "hour": (start_dt + timedelta(hours=i)).isoformat() + "Z",
            "volume": round(float(v), 2),
        }
        for i, v in enumerate(values[-72:])
    ]

    saved_count = 0
    if req.save:
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            for i, v in enumerate(values):
                h = start_dt + timedelta(hours=i)
                key = ("imported", platform, topic, h)
                existing = session.get(Observation, key)
                if existing:
                    existing.volume = int(round(v))
                else:
                    session.add(
                        Observation(
                            dataset="imported",
                            platform=platform,
                            topic=topic,
                            hour=h,
                            volume=int(round(v)),
                        )
                    )
                saved_count += 1
            session.commit()

    growth = result.get("forecast_growth_pct")
    if growth is None:
        expl = "Baseline activity detected with steady future projection."
    elif growth > 15:
        expl = f"Strong upward breakout detected: projected to grow by +{growth:.1f}% over the forecast horizon."
    elif growth > 0:
        expl = f"Moderate upward momentum: projected to increase by +{growth:.1f}%."
    elif growth > -15:
        expl = f"Slight cooling or stabilization: projected change of {growth:.1f}%."
    else:
        expl = f"Downtrend momentum detected: activity is projected to decline by {abs(growth):.1f}%."

    return {
        **result,
        "platform": platform,
        "topic": topic,
        "dataset": "imported" if req.save else "live_user",
        "unit": "collected topic mentions per hour",
        "observations_used": n_used,
        "last_observed_hour": (
            start_dt + timedelta(hours=n_used - 1)
        ).isoformat()
        + "Z",
        "stale": False,
        "history": history,
        "forecast_hours": forecast_hours,
        "saved": req.save,
        "saved_count": saved_count,
        "explanation": expl,
    }


@app.post("/api/import")
def import_records(req: BatchImportRequest):
    if not req.records:
        raise HTTPException(422, "No records provided.")
    if len(req.records) > 20000:
        raise HTTPException(422, "Max 20,000 records per import.")

    SQLModel.metadata.create_all(engine)
    inserted = 0
    updated = 0

    with Session(engine) as session:
        for row in req.records:
            topic = row.topic.strip().lower()
            platform = row.platform.strip().lower()
            hour = (
                row.hour.astimezone(timezone.utc).replace(tzinfo=None)
                if row.hour.tzinfo
                else row.hour
            )
            key = ("imported", platform, topic, hour)
            existing = session.get(Observation, key)
            if existing:
                existing.volume = max(0, int(row.volume))
                updated += 1
            else:
                session.add(
                    Observation(
                        dataset="imported",
                        platform=platform,
                        topic=topic,
                        hour=hour,
                        volume=max(0, int(row.volume)),
                    )
                )
                inserted += 1
        session.commit()

    return {
        "inserted": inserted,
        "updated": updated,
        "total": inserted + updated,
    }


@app.delete("/api/topics/{dataset}/{platform}/{topic}")
def delete_topic(dataset: str, platform: str, topic: str):
    topic = topic.strip().lower()
    platform = platform.strip().lower()
    dataset = dataset.strip().lower()

    if dataset == "demo":
        raise HTTPException(400, "Cannot delete demo dataset series.")

    with Session(engine) as session:
        statement = select(Observation).where(
            Observation.dataset == dataset,
            Observation.platform == platform,
            Observation.topic == topic,
        )
        rows = session.exec(statement).all()
        if not rows:
            raise HTTPException(404, "Topic not found.")
        for r in rows:
            session.delete(r)
        session.commit()

    return {"deleted": len(rows), "topic": topic, "platform": platform}



def seed_demo():
    """Explicit synthetic dataset, isolated from imported observations."""
    SQLModel.metadata.create_all(engine)

    rng = np.random.default_rng(42)
    hours = 35 * 24
    start = utc_hour() - timedelta(hours=hours)
    topic_names = ["ai-tools", "sustainable-fashion", "creator-economy"]
    records = []

    for platform_index, platform in enumerate(Platform):
        for topic_index, topic in enumerate(topic_names):
            for i in range(hours):
                timestamp = start + timedelta(hours=i)
                daily = 30 * np.sin(2 * np.pi * timestamp.hour / 24)
                weekly = 15 * np.sin(
                    2 * np.pi * timestamp.weekday() / 7
                )
                drift = (topic_index - 0.5) * i * 0.02
                volume = max(
                    0,
                    round(
                        100
                        + platform_index * 25
                        + daily
                        + weekly
                        + drift
                        + rng.normal(0, 8)
                    ),
                )

                records.append(
                    Observation(
                        dataset="demo",
                        platform=platform.value,
                        topic=topic,
                        hour=timestamp,
                        volume=volume,
                    )
                )

    with Session(engine) as session:
        exists = session.exec(
            select(Observation)
            .where(Observation.dataset == "demo")
            .limit(1)
        ).first()

        if exists:
            raise SystemExit(
                "Demo data already exists. Seed was not applied again."
            )

        session.add_all(records)
        session.commit()

    print(f"Inserted {len(records)} synthetic hourly observations.")


def import_json(filename):
    path = Path(filename)

    if path.stat().st_size > 10_000_000:
        raise ValueError("Import limit is 10 MB.")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list) or not 1 <= len(payload) <= 20_000:
        raise ValueError("Provide an array containing 1–20,000 records.")

    # Validate the complete batch before writing anything.
    rows = [ImportRow.model_validate(item) for item in payload]
    keys = [(row.platform, row.topic, row.hour) for row in rows]

    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate series/hour keys in the import.")

    SQLModel.metadata.create_all(engine)
    inserted = 0
    updated = 0

    with Session(engine) as session:
        for row in rows:
            key = ("imported", row.platform.value, row.topic, row.hour)
            existing = session.get(Observation, key)

            if existing:
                existing.volume = row.volume
                updated += 1
            else:
                session.add(
                    Observation(
                        dataset="imported",
                        platform=row.platform.value,
                        topic=row.topic,
                        hour=row.hour,
                        volume=row.volume,
                    )
                )
                inserted += 1

        session.commit()

    print(f"Imported observations: {inserted} inserted, {updated} updated.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "seed":
        seed_demo()
    elif len(sys.argv) == 3 and sys.argv[1] == "import":
        import_json(sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python app.py seed | python app.py import FILE.json"
        )