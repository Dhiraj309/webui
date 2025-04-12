import lightning_app as L
from app_component import FlaskApp

class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.web_app = FlaskApp()

    def run(self):
        self.web_app.run()

app = L.LightningApp(RootFlow())