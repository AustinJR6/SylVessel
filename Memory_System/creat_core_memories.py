import sqlite3

# Connect to the database
conn = sqlite3.connect("sylana_memory.db")
cursor = conn.cursor()

# Fetch all core memories
cursor.execute("SELECT * FROM core_memories")
memories = cursor.fetchall()

# Display results
for memory in memories:
    print(memory)

conn.close()
