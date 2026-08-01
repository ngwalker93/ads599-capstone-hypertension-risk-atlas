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
    
    # 1. Load Master Dataset
    master_file = DATA_PROCESSED / "master_dataset_all_variables.csv"
    if master_file.exists():
        print(f"🚀 Loading master dataset into SQL...")
        df_master = pd.read_csv(master_file, dtype={'fipscode': str})
        df_master['fipscode'] = df_master['fipscode'].str.zfill(5)
        df_master.to_sql("master_dataset", engine, if_exists='replace', index=False)
        print(f"✅ Loaded master_dataset into database.")
    else:
        print(f"⚠️ Warning: {master_file} not found. Ensure 05_master_dataset.py has run.")

    # 2. Load CHRR Data for Dashboard Exploration View
    chrr_file = VALIDATION_DIR / "chrr_final_cleaned.csv"
    if chrr_file.exists():
        print(f"🚀 Loading CHR&R data into SQL...")
        df_chrr = pd.read_csv(chrr_file, dtype={'fipscode': str})
        df_chrr['fipscode'] = df_chrr['fipscode'].str.zfill(5)
        df_chrr.to_sql("chrr_data", engine, if_exists='replace', index=False)
        print(f"✅ Loaded chrr_data into database.")
    else:
        print(f"⚠️ Warning: {chrr_file} not found. Ensure 03_process_chrr_census.py has run.")

    # 3. Create Views
    with engine.connect() as conn:
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
        conn.commit()
        print("✅ Views updated successfully.")

    print("------------------------------------------------------------------")
    print("🚀 06_load_to_sql.py completed successfully. Moving to next task...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    load_to_db()