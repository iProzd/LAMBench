from lambench.databases.base_table import BaseRecord
from sqlalchemy import Column, Float

class DirectPredictRecord(BaseRecord):
    __tablename__ = "direct_predict"

    energy_rmse = Column(Float)
    energy_mae = Column(Float)
    energy_rmse_natoms = Column(Float)
    energy_mae_natoms = Column(Float)
    force_rmse = Column(Float)
    force_mae = Column(Float)
    virial_rmse = Column(Float)
    virial_mae = Column(Float)
    virial_rmse_natoms = Column(Float)
    virial_mae_natoms = Column(Float)
