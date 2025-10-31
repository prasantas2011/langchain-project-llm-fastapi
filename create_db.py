# create_db.py
import sqlite3

# Create database
conn = sqlite3.connect("mydb.sqlite")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    country TEXT
)
""")

# Insert sample data
cursor.executemany("INSERT INTO users (name, age, country) VALUES (?, ?, ?)", [
    ("Alice", 25, "USA"),
    ("Bob", 32, "India"),
    ("Charlie", 29, "UK"),
    ("Diana", 35, "Canada"),
    ("Ethan", 28, "India")
])

conn.commit()
conn.close()

print("✅ Database created with sample data.")
