import os

import psycopg
import psycopg_pool


def create_pool() -> psycopg_pool.ConnectionPool:
    conninfo = psycopg.conninfo.make_conninfo(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    pool = psycopg_pool.ConnectionPool(conninfo, min_size=2, max_size=10, open=False)
    pool.open()
    return pool
