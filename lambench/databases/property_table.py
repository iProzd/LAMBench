from lambench.databases.base_table import BaseRecord
from sqlalchemy import Column, String, Float

class PropertyRecord(BaseRecord):
    __tablename__ = "property"

    property_rmse = Column(Float)
    property_mae = Column(Float)
