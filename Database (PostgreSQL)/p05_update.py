import p04_read
import psycopg2

id = input("Enter id of the row you want to change: ")
name = input("Enter name: ")
age = int(input("Enter age: "))
course = input("Enter course: ")

conn = psycopg2.connect(
    host="localhost",
    user="shail",
    password="shail",
    database="internship"
)

cur = conn.cursor()

cur.execute("UPDATE Students set name=%s, age=%s, course=%s where id = %s", (name, age, course, id))
conn.commit()

print("Row Updated successfully!")

cur.close()
conn.close()
