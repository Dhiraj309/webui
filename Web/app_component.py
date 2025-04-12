import lightning_app as L
import subprocess
import os

class FlaskApp(L.LightningWork):
    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute("cpu"), port=5000)

    def run(self):
        env = os.environ.copy()
        env["FLASK_APP"] = "app.py"
        env["FLASK_RUN_PORT"] = str(self.port)
        env["FLASK_RUN_HOST"] = "0.0.0.0"

        subprocess.run(["flask", "run"], env=env)