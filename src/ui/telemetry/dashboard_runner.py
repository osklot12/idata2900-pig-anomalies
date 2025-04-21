import time
import random

from src.schemas.converters.pressure_metric_schema_converter import PressureMetricSchemaConverter
from src.schemas.schemas.pressure_schema import PressureSchema
from src.schemas.schemas.signed_schema import SignedSchema
from src.ui.dashboard.rich_dashboard import RichDashboard


def run():
    dashboard = RichDashboard()
    dashboard.register(PressureSchema, PressureMetricSchemaConverter())
    dashboard.start()
    while True:
        schema = SignedSchema(
            signature="test-id",
            timestamp=time.time(),
            schema=PressureSchema(inputs=int(random.random() * 20), outputs=int(random.random() * 20), usage=random.random())
        )
        dashboard.new_schema(schema)
        time.sleep(.5)


if __name__ == "__main__":
    run()
