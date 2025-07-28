import sqlite3

conn = sqlite3.connect('Chinooklocal.db') 

cursor= conn.cursor()

# create a tabel
table_info = """
CREATE TABLE IF NOT EXISTS hotel (
        FIND INTEGER PRIMARY KEY NOT NULL,
        FNAME TEXT NOT NULL,
        COST INTEGER NOT NULL,
        WEIGHT INTEGER
    )
"""

cursor.execute(table_info)
# Insert records
cursor.execute("INSERT INTO hotel (FIND, FNAME, COST, WEIGHT) VALUES (1, 'Cakes', 800, 10)")
cursor.execute("INSERT INTO hotel (FIND, FNAME, COST, WEIGHT) VALUES (2, 'Biscuits', 100, 20)")
cursor.execute("INSERT INTO hotel (FIND, FNAME, COST, WEIGHT) VALUES (3, 'Chocos', 1000, 30)")


cursor.execute("SELECT * FROM hotel")
rows = cursor.fetchall()
for row in rows:
    print(row)

# commit your changes to the database
conn.commit()
print("Data inserted successfully.")
conn.close()
  
  


