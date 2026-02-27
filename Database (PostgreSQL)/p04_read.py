import psycopg2

conn = psycopg2.connect(
    host="localhost",
    user="shail",
    password="shail",
    database="internship"
)

cur = conn.cursor()

cur.execute("SELECT * from students")

rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()