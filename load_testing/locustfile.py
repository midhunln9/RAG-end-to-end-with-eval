from locust import HttpUser, task, between

class FastAPIUser(HttpUser):
    wait_time = between(0, 0.5)

    @task
    def ask_endpoint(self):
        payload = {
            "query": "What is Financial compliance and why is it important?",
            "session_id": "123456789090"
        }

        self.client.post(
            "/ask",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="POST /ask"
        )