import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="internship",
    user="shail",
    password="shail"
)

print("✅ Connected!")
conn.close()
