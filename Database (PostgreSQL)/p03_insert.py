import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="internship",
    user="shail",
    password="shail"
)

cur = conn.cursor()

name = input("Enter name: ")
age = int(input("Enter age: "))
course = input("Enter course: ")

try:
    cur.execute("""
            INSERT INTO students(name, age, course) values (%s, %s, %s)
            """, (name, age, course))
except Exception as e:
    print(e)
else:
    conn.commit()
    print("1 Row Inserted")

cur.close()
conn.close()