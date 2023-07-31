from locust import HttpUser, task


class PredictionUser(HttpUser):
    @task
    def arima(self):
        self.client.get("/prediction/arima")

    @task
    def ces(self):
        self.client.get("/prediction/ces")

    @task
    def garch(self):
        self.client.get("/prediction/garch")

    @task
    def xgboost(self):
        self.client.get("/prediction/xgboost")

    @task
    def reservoir(self):
        self.client.get("/prediction/reservoir")

    @task
    def nhits(self):
        self.client.get("/prediction/nhits")
