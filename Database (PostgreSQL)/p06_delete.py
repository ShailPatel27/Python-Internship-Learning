import p04_read
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    user="shail",
    password="shail",
    database="internship"
)

id = input("Enter id of row you want to delete: ")

cur = conn.cursor()

cur.execute("DELETE FROM students where id=%s", id)
conn.commit()

print("1 Row Deleted")

cur.close()
conn.close()