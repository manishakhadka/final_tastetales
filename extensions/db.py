import mysql.connector

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

mysql_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'tastetales'
}

engine = create_engine(
    'mysql+mysqlconnector://{user}:{password}@{host}/{database}'.format(**mysql_config))
Session = sessionmaker(bind=engine)

Base = declarative_base()
metadata = Base.metadata


def get_db():
    return mysql.connector.connect(**mysql_config)


def get_session():
    return Session()
