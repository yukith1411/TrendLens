from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import app as application
from forecasting import (
    HORIZON,
    forecast,
    make_windows,
    split_masks,
)


def sample_values(hours=672):
    rng = np.random.default_rng(17)
    t = np.arange(hours)
    return np.maximum(
        0,
        np.round(
            100
            + 25 * np.sin(2 * np.pi * t / 24)
            + rng.normal(0, 4, hours)
        ),
    )


def test_forecast_has_24_nonnegative_values():
    result = forecast(sample_values(), datetime(2025, 1, 1))

    assert len(result["forecast"]) == 24
    assert all(value >= 0 for value in result["forecast"])
    assert result["evaluation"]["test_windows"] >= 3
    assert 0 <= result["evaluation"]["direction_accuracy_pct"] <= 100


def test_training_targets_do_not_overlap_holdout():
    values = sample_values()
    starts, _, _ = make_windows(values, datetime(2025, 1, 1))
    cutoff = int(len(values) * 0.75)
    training, testing = split_masks(starts, cutoff)

    assert np.max(starts[training] + HORIZON) <= np.min(starts[testing])
    assert np.all(np.diff(starts[testing]) == HORIZON)


def test_insufficient_history_is_rejected():
    with pytest.raises(ValueError, match="consecutive"):
        forecast([10] * 100, datetime(2025, 1, 1))


def test_zero_activity_does_not_create_fake_accuracy():
    result = forecast([0] * 672, datetime(2025, 1, 1))

    assert result["evaluation"]["model_wape_pct"] is None
    assert result["forecast_growth_pct"] is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(application, "engine", test_engine)
    application.cached_forecast.cache_clear()

    with TestClient(application.app) as test_client:
        yield test_client, test_engine

    test_engine.dispose()


def insert_series(engine, gap=False):
    start = datetime(2025, 1, 1)
    values = sample_values()

    with Session(engine) as session:
        for i, value in enumerate(values):
            if gap and i == 100:
                continue

            session.add(
                application.Observation(
                    dataset="demo",
                    platform="instagram",
                    topic="ai-tools",
                    hour=start + timedelta(hours=i),
                    volume=int(value),
                )
            )

        session.commit()


def test_api_health(client):
    browser, _ = client
    assert browser.get("/api/health").json() == {"status": "ok"}


def test_api_forecast_and_dataset_isolation(client):
    browser, engine = client
    insert_series(engine)

    response = browser.get(
        "/api/trend",
        params={
            "platform": "instagram",
            "topic": "ai-tools",
            "dataset": "demo",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observations_used"] == 672
    assert len(body["forecast_hours"]) == 24
    assert body["dataset"] == "demo"

    missing = browser.get(
        "/api/trend",
        params={
            "platform": "instagram",
            "topic": "ai-tools",
            "dataset": "imported",
        },
    )
    assert missing.status_code == 404


def test_missing_hours_are_rejected(client):
    browser, engine = client
    insert_series(engine, gap=True)

    response = browser.get(
        "/api/trend",
        params={
            "platform": "instagram",
            "topic": "ai-tools",
            "dataset": "demo",
        },
    )

    assert response.status_code == 422
    assert "Missing hourly" in response.json()["detail"]


def test_invalid_platform_is_rejected(client):
    browser, _ = client

    response = browser.get(
        "/api/trend",
        params={"platform": "unsupported", "topic": "ai-tools"},
    )

    assert response.status_code == 422


def test_import_requires_timezone():
    with pytest.raises(ValueError):
        application.ImportRow.model_validate({
            "platform": "instagram",
            "topic": "ai-tools",
            "hour": "2025-01-01T00:00:00",
            "volume": 100,
        })


def test_predict_custom_values(client):
    browser, _ = client
    payload = {
        "topic": "crypto-pulse",
        "platform": "x",
        "values": [50, 55, 62, 70, 75, 82, 90, 105, 110, 125, 130, 140],
        "horizon": 24,
        "save": False,
    }
    response = browser.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "crypto-pulse"
    assert data["platform"] == "x"
    assert len(data["forecast"]) == 24
    assert len(data["forecast_hours"]) == 24
    assert data["observations_used"] == 12
    assert data["forecast_growth_pct"] is not None
    assert data["saved"] is False


def test_predict_custom_raw_text(client):
    browser, _ = client
    payload = {
        "topic": "custom-brand",
        "platform": "tiktok",
        "raw_text": "100, 110, 120, 130, 140, 150, 160",
        "horizon": 12,
    }
    response = browser.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecast"]) == 12
    assert data["observations_used"] == 7


def test_predict_insufficient_points_rejected(client):
    browser, _ = client
    payload = {
        "topic": "short",
        "platform": "instagram",
        "values": [10, 20],
    }
    response = browser.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "at least 3" in response.json()["detail"].lower()


def test_predict_and_save_to_database(client):
    browser, _ = client
    payload = {
        "topic": "live-stream-gadgets",
        "platform": "instagram",
        "values": [10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55],
        "save": True,
    }
    response = browser.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["saved"] is True

    # Now verify it appears in /api/topics for imported dataset
    topics_res = browser.get(
        "/api/topics",
        params={"platform": "instagram", "dataset": "imported"},
    )
    assert topics_res.status_code == 200
    topics_list = [item["topic"] for item in topics_res.json()["items"]]
    assert "live-stream-gadgets" in topics_list

    # Delete topic
    del_res = browser.delete(
        "/api/topics/imported/instagram/live-stream-gadgets"
    )
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] == 12


def test_trending_market_endpoint(client):
    browser, _ = client
    res = browser.get("/api/trending-market")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 10
    assert len(data["categories"]) == 3
    assert len(data["countries"]) >= 8


def test_trending_market_entertainment_filter(client):
    browser, _ = client
    res = browser.get("/api/trending-market", params={"category": "entertainment"})
    assert res.status_code == 200
    data = res.json()
    assert all(item["category"] == "entertainment" for item in data["items"])
    app_names = [i["name"] for i in data["items"]]
    assert "YouTube" in app_names
    assert "TikTok" in app_names
    assert "Instagram" in app_names


def test_trending_market_news_and_country_filter(client):
    browser, _ = client
    # Filter news for India
    res_in = browser.get(
        "/api/trending-market",
        params={"category": "news", "country": "IN"},
    )
    assert res_in.status_code == 200
    items_in = res_in.json()["items"]
    assert len(items_in) >= 2
    assert all(i["country"] == "IN" and i["category"] == "news" for i in items_in)
    names_in = [i["name"] for i in items_in]
    assert "NDTV" in names_in

    # Filter news for UK
    res_uk = browser.get(
        "/api/trending-market",
        params={"category": "news", "country": "UK"},
    )
    assert res_uk.status_code == 200
    names_uk = [i["name"] for i in res_uk.json()["items"]]
    assert "BBC News" in names_uk


def test_trending_market_search(client):
    browser, _ = client
    res = browser.get("/api/trending-market", params={"search": "cnn"})
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "CNN"


