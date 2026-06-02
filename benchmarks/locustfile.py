import random

from locust import HttpUser, between, task


class OurSystemUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://127.0.0.1:8000"

    def on_start(self):
        self.headers = {'Content-Type': 'application/json'}
        self.cached_queries = ["I love this product!", "I hate this product!", "What time is it?", "Can you tell me the weather?"]
        self.complex_queries = ["Brilliant deduction, Sherlock.", "I guess I'll just sit here and wait forever."]

    @task(3)
    def test_cached_query(self):
        query = random.choice(self.cached_queries)
        payload = {"text": query, "session_id": "locust_session", "user_id": "locust_user"}
        with self.client.post("/api/analyze?model=unified", json=payload, headers=self.headers, catch_response=True, name="[OURS] Cached") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got {response.status_code}")

    @task(1)
    def test_cold_start_query(self):
        query = random.choice(self.complex_queries)
        payload = {"text": query, "session_id": "locust_session", "user_id": "locust_user"}
        with self.client.post("/api/analyze?model=unified", json=payload, headers=self.headers, catch_response=True, name="[OURS] Complex") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got {response.status_code}")

class TheirSystemUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://127.0.0.1:8001"

    def on_start(self):
        self.headers = {'Content-Type': 'application/json'}
        self.cached_queries = ["I love this product!", "I hate this product!", "What time is it?", "Can you tell me the weather?"]
        self.complex_queries = ["Brilliant deduction, Sherlock.", "I guess I'll just sit here and wait forever."]

    @task(3)
    def test_cached_query(self):
        query = random.choice(self.cached_queries)
        payload = {"text": query, "session_id": "locust_session", "user_id": "locust_user"}
        with self.client.post("/analyze", json=payload, headers=self.headers, catch_response=True, name="[THEIRS] Cached") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got {response.status_code}")

    @task(1)
    def test_cold_start_query(self):
        query = random.choice(self.complex_queries)
        payload = {"text": query, "session_id": "locust_session", "user_id": "locust_user"}
        with self.client.post("/analyze", json=payload, headers=self.headers, catch_response=True, name="[THEIRS] Complex") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got {response.status_code}")
