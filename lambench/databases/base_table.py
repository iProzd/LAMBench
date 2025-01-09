from __future__ import annotations
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
        session = Session()
        session.add(self)
        session.flush()
        session.commit()
        session.close()

    @classmethod
    def query(cls, **kwargs) -> Sequence[BaseRecord]:
        with Session() as session:
            return session.query(cls).filter_by(**kwargs).all()

    @classmethod
    def query_by_run(cls, model_name: str) -> Sequence[BaseRecord]:
        return cls.query(model_name=model_name)
