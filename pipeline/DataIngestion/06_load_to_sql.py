"""
This script loads the processed datasets into SQLite database.
"""
import pandas as pd
from sqlalchemy import create_engine, text
from paths import DATA_PROCESSED, VALIDATION_DIR

def load_to_db():
    # Setup connection
    db_path = 'hypertension_atlas.db'
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Define files to load
    files = [
        DATA_PROCESSED / "master_dataset_all_variables.csv"
    ]
    
    print("🚀 Loading master dataset into SQL...")
    for file in files:
        if file.exists():
            df = pd.read_csv(file)
            table_name = "master_dataset" 
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"✅ Loaded {table_name} into database.")
        else:
            print(f"⚠️ Warning: {file} not found. Ensure 05_merge_all.py has run.")

    # Since a master_dataset table was created,
    # the view can be a direct SELECT of that table.
    with engine.connect() as conn:
            # Drop and create views
            conn.execute(text("DROP VIEW IF EXISTS master_analysis_view"))
            conn.execute(text("DROP VIEW IF EXISTS modeling_view"))
            conn.execute(text("DROP VIEW IF EXISTS exploration_view"))
        
            conn.execute(text("CREATE VIEW master_analysis_view AS SELECT * FROM master_dataset"))
            conn.execute(text("CREATE VIEW modeling_view AS SELECT * FROM master_dataset"))
            conn.execute(text("""
                CREATE VIEW exploration_view AS
                SELECT m.*, c.*
                FROM master_dataset m
                LEFT JOIN chrr_data c ON m.fipscode = c.fipscode
            """))
            conn.commit() # Important for SQLite to persist the views
            print("✅ Views updated successfully.")

if __name__ == "__main__":
    load_to_db()