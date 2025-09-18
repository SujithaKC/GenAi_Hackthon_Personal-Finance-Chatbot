import sqlite3
from sqlite3 import Error

def test_sqlite_connection():
    try:
        # Replace with your database path
        db_path = "e:/COLLEGE/IBM GenAi/finance.db"
        conn = sqlite3.connect(db_path)
        print("Successfully connected to SQLite database!")
        
        # Test query to verify data access
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        print(f"Number of records in transactions table: {count}")
        
        cursor.close()
        conn.close()
        print("Connection closed successfully!")
        return True
        
    except Error as e:
        print(f"Error connecting to SQLite database: {e}")
        return False

if __name__ == "__main__":
    test_sqlite_connection()