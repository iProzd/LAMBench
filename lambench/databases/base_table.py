from __future__ import annotations # For class method return type hinting
import os
from typing import Sequence
from sqlalchemy import Column, Integer, String, Float, create_engine, asc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

Base = declarative_base()
load_dotenv(override=True)

db_username = os.environ.get("MYSQL_USERNAME")
db_password = os.environ.get("MYSQL_PASSWORD")
db_host = os.environ.get("MYSQL_HOST")
db_name = os.environ.get("MYSQL_DATABASE_NAME")
db = create_engine(
    "mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8"
    % (db_username, db_password, db_host, db_name)
)
Session = sessionmaker(db)

class BaseRecord(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True) # index
    model_name = Column(String(256), index=True)
    task_name = Column(String(256))
    # NOTE: record_name = model_name + "#" + task_name

    def insert(self):
        with Session() as session:
            session.add(self)
            session.commit()

    @classmethod
    def query(cls, **kwargs) -> Sequence[BaseRecord]:
        session = Session()
        records = session.query(cls).filter_by(**kwargs).all()
        session.close()
        return records

    @classmethod
    def count(cls, **kwargs) -> int:
        """Query records by keyword arguments.
        Input:
            model_name: str
            task_name: str
        Return:
            int: number of records found
        Example:
            >>> PropertyRecord.count(model_name="TEST_DP_v1", task_name="task1")
        """
        with Session() as session:
            return session.query(cls).filter_by(**kwargs).count()
