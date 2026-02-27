import psycopg2

# connect to your database
conn = psycopg2.connect(
    host="localhost",
    database="internship",
    user="shail",
    password="shail"
)

cur = conn.cursor()
#You cant directly execute command in postgres so a cursor acts as an intermediary and you feed it commands so it gives them to postgres and executes them.

cur.execute("""
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    course VARCHAR(50)
)
""")

conn.commit()
print("✅ Table created successfully!")

cur.close()
conn.close()
