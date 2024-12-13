import os
from typing import List
from sqlalchemy import Column, Integer, String, Float, create_engine, asc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

Base = declarative_base()
load_dotenv()

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

    id = Column(Integer, primary_key=True)
    model_id = Column(String(256), index=True)
    record_name = Column(String(256))
    step = Column(Integer)
    

    def insert(self) -> int:
        session = Session()
        session.add(self)
        session.flush()
        session.commit()
        session.close()

    @classmethod
    def query(cls, **kwargs) -> List["BaseRecord"]:
        session = Session()
        records = session.query(cls).filter_by(**kwargs).order_by(asc(cls.step)).all()
        session.close()
        return records

    @classmethod
    def query_by_run(cls, model_id: str) -> List["BaseRecord"]:
        return cls.query(model_id=model_id)

    @classmethod
    def query_by_name(cls, record_name: str) -> List["BaseRecord"]:
        return cls.query(record_name=record_name)
