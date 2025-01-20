from lambench.databases.base_table import BaseRecord
from sqlalchemy import Column, JSON


class CalculatorRecord(BaseRecord):
    __tablename__ = "calculator"

    metrics = Column(JSON)
