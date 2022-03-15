import sqlite3
from sqlite3 import Error
import numpy as np
import os

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def select_all_rows(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM parallax_data")

    rows = cur.fetchall()

    for row in rows:
        print(row)
    return


# Get the current working directory
cwd = os.getcwd()
database = cwd + '/whitesides_app/parallax_data.db'
conn = create_connection(database)
select_all_rows(conn)
