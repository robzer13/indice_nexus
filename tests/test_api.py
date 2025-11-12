from fastapi.testclient import TestClient

from stock_analysis.api import app as api_module


def test_health_endpoint() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
