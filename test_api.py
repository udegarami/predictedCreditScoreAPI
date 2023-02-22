from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome": "to the Project"}

def test_favicon():
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.json() == {"Favicon": "OK"}

def test_df():
    response = client.get("/api/v1/df")
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_fetch_prediction():
    response = client.get("/api/v1/predict/100001")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_read_image():
    response = client.get("/api/v1/image/test_image.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_fetch_characteristics():
    response = client.get("/api/v1/characteristics/100001")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_fetch_neighbors():
    response = client.get("/api/v1/neighbors/100001")
    assert response.status_code == 200
    assert isinstance(response.json(), list)