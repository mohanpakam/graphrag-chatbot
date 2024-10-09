import sqlite3
from sqlite3 import Connection
from contextlib import contextmanager
import json

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def init_database(self):
        with self.connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    properties TEXT,
                    document_id INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    properties TEXT,
                    document_id INTEGER NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_id) REFERENCES nodes (id),
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            # Create indices for faster lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON documents(id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_node_document ON nodes(document_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relationship_document ON relationships(document_id)')

        print("Database initialized with required tables and indices.")

    def store_graph_document(self, doc_data):
        with self.connect() as conn:
            cursor = conn.execute('''
                INSERT INTO documents (filename, chunk_index, content)
                VALUES (?, ?, ?)
            ''', (doc_data['filename'], doc_data['chunk_index'], doc_data['content']))
            return cursor.lastrowid

    def store_node(self, node_data):
        with self.connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO nodes (id, type, properties, document_id)
                VALUES (?, ?, ?, ?)
            ''', (
                node_data['id'],
                node_data['type'],
                json.dumps(node_data['properties']),
                node_data['document_id']
            ))

    def store_relationship(self, relationship_data):
        with self.connect() as conn:
            conn.execute('''
                INSERT INTO relationships (source_id, target_id, type, properties, document_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                relationship_data['source_id'],
                relationship_data['target_id'],
                relationship_data['type'],
                json.dumps(relationship_data['properties']),
                relationship_data['document_id']
            ))

    def clear_documents(self):
        with self.connect() as conn:
            conn.execute('DELETE FROM documents')
            conn.execute('DELETE FROM nodes')
            conn.execute('DELETE FROM relationships')
        print("All documents, nodes, and relationships cleared from the database.")

    def get_subgraph_for_documents(self, doc_ids):
        with self.connect() as conn:
            placeholders = ','.join('?' for _ in doc_ids)
            nodes = conn.execute(f'''
                SELECT * FROM nodes
                WHERE document_id IN ({placeholders})
            ''', doc_ids).fetchall()
            
            relationships = conn.execute(f'''
                SELECT * FROM relationships
                WHERE document_id IN ({placeholders})
            ''', doc_ids).fetchall()
            
        return {'nodes': nodes, 'relationships': relationships}

    def get_document_content(self, doc_id):
        with self.connect() as conn:
            cursor = conn.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    def execute_query(self, query, params=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()

    def get_schema_info(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = {}
            for table in tables:
                table_name = table[0]
                #table_name starts with llm_extracted_
                if table_name.startswith('llm_extracted_'):
                    # Get the schema for the table using .schema table_name
                    cursor.execute(f".schema {table_name};")
                    table_schema = cursor.fetchall()
                    schema_info.append((table_name, table_schema))
                print(f"Schema Info for this table is {schema_info}")
        return schema_info

    def get_column_names(self, query):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [description[0] for description in cursor.description]