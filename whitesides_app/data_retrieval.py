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

def select_all_rows(conn, db):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM parallax_data")

    rows = cur.fetchall()
    # n.b. db must be list
    for row in rows:
        db.append(row[:11])
    return


# Get the current working directory
cwd = os.getcwd()
# retrieve data for subject test
directory = cwd + '/data/'
db_subjects =[]
for dbfile in os.listdir(directory):
    if not dbfile.startswith('.'):
        filename = os.fsdecode(dbfile)
        print(filename)
        database =  directory + filename
        conn = create_connection(database)
        select_all_rows(conn, db_subjects)
db_subjects = np.array(db_subjects)

# retrieve data for all attempt test
directory = cwd + '/all_attempt_data/'
db_allattempt =[]
for dbfile in os.listdir(directory):
    if not dbfile.startswith('.'):
        filename = os.fsdecode(dbfile)
        print(filename)
        database =  directory + filename
        conn = create_connection(database)
        select_all_rows(conn, db_allattempt)

db_allattempt = np.array(db_allattempt)
