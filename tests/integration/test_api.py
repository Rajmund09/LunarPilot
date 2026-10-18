import json
import urllib.request


def test_api_endpoints_live():
    base_url = "http://localhost:8080"

    # 1. Test next_flight endpoint (Starship & Falcon)
    req = urllib.request.urlopen(f"{base_url}/api/next_flight?vehicle=starship")
    assert req.status == 200
    data = json.loads(req.read().decode())
    assert "vehicle" in data
    assert data["vehicle"] == "starship"
    assert "steps" in data
    assert len(data["steps"]) > 0

    req = urllib.request.urlopen(f"{base_url}/api/next_flight?vehicle=falcon")
    assert req.status == 200
    data = json.loads(req.read().decode())
    assert data["vehicle"] == "falcon"

    # 2. Test next_flight endpoint (Artemis)
    req = urllib.request.urlopen(f"{base_url}/api/next_flight?vehicle=artemis")
    assert req.status == 200
    data = json.loads(req.read().decode())
    assert data["vehicle"] == "artemis"

    # 3. Test simulate_rocket endpoint with Mechazilla catch and Falcon Droneship
    req = urllib.request.urlopen(f"{base_url}/api/simulate_rocket?alt=280&x=0&vx=0&vy=-35&tilt=0&vehicle=starship")
    assert req.status == 200
    data = json.loads(req.read().decode())
    assert data["is_success"] is True
    assert "MECHAZILLA CATCH" in data["outcome"]

    req_falcon = urllib.request.urlopen(f"{base_url}/api/simulate_rocket?alt=280&x=0&vx=0&vy=-35&vehicle=falcon")
    assert req_falcon.status == 200
    data_falcon = json.loads(req_falcon.read().decode())
    assert data_falcon["is_success"] is True
    assert "TOUCHDOWN" in data_falcon["outcome"]

    # 4. Test simulate_lunar endpoint
    req = urllib.request.urlopen(f"{base_url}/api/simulate_lunar?x=0.0&y=1.2&angle=0.0")
    assert req.status == 200
    data = json.loads(req.read().decode())
    assert "steps" in data
