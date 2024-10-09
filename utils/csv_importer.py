import os
import csv
import sqlite3
import re
from utils import DatabaseManager

class CSVImporter:
    def __init__(self, db_path, raw_data_folder):
        self.db_manager = DatabaseManager(db_path)
        self.raw_data_folder = raw_data_folder

    def normalize_column_name(self, name):
        # Convert to lowercase and replace spaces with underscores
        name = re.sub(r'\s+', '_', name.lower())
        # Remove any non-alphanumeric characters (except underscores)
        name = re.sub(r'[^\w]+', '', name)
        # Add descriptive prefixes if needed
        if name.endswith('_id'):
            name = f"unique_{name}"
        elif name in ['name', 'title']:
            name = f"descriptive_{name}"
        elif name in ['date', 'time']:
            name = f"temporal_{name}"
        return name

    def create_table_from_csv(self, csv_file):
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            normalized_headers = [self.normalize_column_name(h) for h in headers]
            
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            create_table_sql += ",\n".join([f"{h} TEXT" for h in normalized_headers])
            create_table_sql += "\n);"
            
            self.db_manager.execute_query(create_table_sql)
            return table_name, normalized_headers

    def load_data_from_csv(self, csv_file, table_name, headers):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                insert_sql = f"INSERT INTO {table_name} ({','.join(headers)}) VALUES ({','.join(['?' for _ in headers])})"
                self.db_manager.execute_query(insert_sql, row)

    def create_relationships(self, relationships_file):
        with open(relationships_file, 'r') as f:
            for line in f:
                parent, child = line.strip().split('|')
                parent_table, parent_column = parent.split(':')
                child_table, child_column = child.split(':')
                
                # Add foreign key constraint
                alter_table_sql = f"""
                ALTER TABLE {child_table}
                ADD CONSTRAINT fk_{child_table}_{parent_table}
                FOREIGN KEY ({child_column}) REFERENCES {parent_table}({parent_column});
                """
                self.db_manager.execute_query(alter_table_sql)

    def create_sql_file(self, table_name, headers):
        sql_file = os.path.join(self.raw_data_folder, f"{table_name}.sql")
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        create_table_sql += ",\n".join([f"{h} TEXT" for h in headers])
        create_table_sql += "\n);"
        
        with open(sql_file, 'w') as f:
            f.write(create_table_sql)

    def import_csv_files(self):
        summary = []
        csv_files = [f for f in os.listdir(self.raw_data_folder) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(self.raw_data_folder, csv_file)
            table_name, headers = self.create_table_from_csv(file_path)
            self.load_data_from_csv(file_path, table_name, headers)
            self.create_sql_file(table_name, headers)
            summary.append(f"Imported {csv_file} into table {table_name}")
        
        relationships_file = os.path.join(self.raw_data_folder, 'relationships.txt')
        if os.path.exists(relationships_file):
            self.create_relationships(relationships_file)
            summary.append("Created relationships based on relationships.txt")
        
        return "\n".join(summary)

# Usage example
if __name__ == "__main__":
    db_path = "your_database.db"
    raw_data_folder = "path/to/rawdata"
    
    importer = CSVImporter(db_path, raw_data_folder)
    summary = importer.import_csv_files()
    print(summary)