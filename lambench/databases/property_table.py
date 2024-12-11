from lambench.databases.base_table import BaseRecord

class PropertyRecord(BaseRecord):
    __tablename__ = "property"

    task_name = Column(String(100))
    property_rmse = Column(Float)
    property_mae = Column(Float)
    