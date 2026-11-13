import os
import psycopg

def main():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        print("FAIL: DATABASE_URL is not set")
        return

    try:
        with psycopg.connect(
            database_url,
            connect_timeout=10,
            sslmode="require"
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), NOW()")
                database_name, server_time = cursor.fetchone()

        print("SUCCESS: Connected to Supabase")
        print("Database:", database_name)
        print("Server time:", server_time)

    except Exception as error:
        print("FAIL: Supabase connection failed")
        print(type(error).__name__, str(error))

if __name__ == "__main__":
    main()