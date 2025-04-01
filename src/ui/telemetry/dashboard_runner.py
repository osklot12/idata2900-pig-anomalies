import time
import random

from src.schemas.metric_schema import MetricSchema
from src.schemas.signed_schema import SignedSchema
from src.ui.telemetry.rich_dashboard import RichDashboard


def run():
    dashboard = RichDashboard()
    dashboard.start()
    while True:
        schema = SignedSchema(
            issuer_id="test-component",
            schema=MetricSchema(
                metrics={
                    "inputs": random.random() * 20,
                    "outputs": random.random() * 20,
                    "usage": random.random()
                },
                timestamp=time.time()
            )
        )
        dashboard.new_schema(schema)
        time.sleep(.5)


if __name__ == "__main__":
    run()
