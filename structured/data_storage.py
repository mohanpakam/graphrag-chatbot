import sqlite3
from typing import Dict, Any, List
from config import LoggerConfig

logger = LoggerConfig.setup_logger(__name__)
config = LoggerConfig.load_config()

class StructuredDataStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_issues (
                    id INTEGER PRIMARY KEY,
                    issue_id TEXT UNIQUE,
                    severity_level TEXT,
                    root_cause TEXT,
                    impact TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_systems (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_people (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_timestamps (
                    id INTEGER PRIMARY KEY,
                    issue_id TEXT,
                    type TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_resolution_steps (
                    id INTEGER PRIMARY KEY,
                    issue_id TEXT,
                    step_number INTEGER,
                    description TEXT,
                    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_relationships (
                    id INTEGER PRIMARY KEY,
                    source TEXT,
                    relationship_type TEXT,
                    target TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_issue_systems (
                    issue_id TEXT,
                    system_id INTEGER,
                    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id),
                    FOREIGN KEY (system_id) REFERENCES llm_extracted_systems (id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_extracted_issue_people (
                    issue_id TEXT,
                    person_id INTEGER,
                    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id),
                    FOREIGN KEY (person_id) REFERENCES llm_extracted_people (id)
                )
            ''')
            conn.commit()

    def store_structured_data(self, data: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store issue
            cursor.execute('''
                INSERT OR REPLACE INTO llm_extracted_issues (issue_id, severity_level, root_cause, impact)
                VALUES (?, ?, ?, ?)
            ''', (data['issue_id'], data['severity_level'], data['root_cause'], data['impact']))
            
            # Store systems
            for system in data['systems_affected']:
                cursor.execute('INSERT OR IGNORE INTO llm_extracted_systems (name) VALUES (?)', (system,))
                cursor.execute('SELECT id FROM llm_extracted_systems WHERE name = ?', (system,))
                system_id = cursor.fetchone()[0]
                cursor.execute('INSERT INTO llm_extracted_issue_systems (issue_id, system_id) VALUES (?, ?)',
                               (data['issue_id'], system_id))
            
            # Store people
            for person in data['people_involved']:
                cursor.execute('INSERT OR IGNORE INTO llm_extracted_people (name) VALUES (?)', (person,))
                cursor.execute('SELECT id FROM llm_extracted_people WHERE name = ?', (person,))
                person_id = cursor.fetchone()[0]
                cursor.execute('INSERT INTO llm_extracted_issue_people (issue_id, person_id) VALUES (?, ?)',
                               (data['issue_id'], person_id))
            
            # Store timestamps
            for timestamp_type, timestamp in data['timestamps'].items():
                cursor.execute('''
                    INSERT INTO llm_extracted_timestamps (issue_id, type, timestamp)
                    VALUES (?, ?, ?)
                ''', (data['issue_id'], timestamp_type, timestamp))
            
            # Store resolution steps
            for i, step in enumerate(data['resolution_steps'], 1):
                cursor.execute('''
                    INSERT INTO llm_extracted_resolution_steps (issue_id, step_number, description)
                    VALUES (?, ?, ?)
                ''', (data['issue_id'], i, step))
            
            # Store relationships
            for rel in data['relationships']:
                cursor.execute('''
                    INSERT INTO llm_extracted_relationships (source, relationship_type, target)
                    VALUES (?, ?, ?)
                ''', (rel['source'], rel['type'], rel['target']))
            
            conn.commit()
        logger.info(f"Stored structured data for issue {data['issue_id']}")

    def get_issue_data(self, issue_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM llm_extracted_issues WHERE issue_id = ?', (issue_id,))
            issue = dict(cursor.fetchone())
            
            cursor.execute('''
                SELECT s.name FROM llm_extracted_systems s
                JOIN llm_extracted_issue_systems is ON s.id = is.system_id
                WHERE is.issue_id = ?
            ''', (issue_id,))
            issue['systems_affected'] = [row['name'] for row in cursor.fetchall()]
            
            cursor.execute('''
                SELECT p.name FROM llm_extracted_people p
                JOIN llm_extracted_issue_people ip ON p.id = ip.person_id
                WHERE ip.issue_id = ?
            ''', (issue_id,))
            issue['people_involved'] = [row['name'] for row in cursor.fetchall()]
            
            cursor.execute('SELECT type, timestamp FROM llm_extracted_timestamps WHERE issue_id = ?', (issue_id,))
            issue['timestamps'] = {row['type']: row['timestamp'] for row in cursor.fetchall()}
            
            cursor.execute('SELECT step_number, description FROM llm_extracted_resolution_steps WHERE issue_id = ? ORDER BY step_number', (issue_id,))
            issue['resolution_steps'] = [row['description'] for row in cursor.fetchall()]
            
            cursor.execute('SELECT source, relationship_type, target FROM llm_extracted_relationships WHERE source = ? OR target = ?', (issue_id, issue_id))
            issue['relationships'] = [dict(row) for row in cursor.fetchall()]
            
            return issue

    def get_relevant_issues(self, query: str) -> List[Dict[str, Any]]:
        # This is a simple implementation. You might want to use more sophisticated
        # matching techniques in a production environment.
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT issue_id FROM llm_extracted_issues
                WHERE root_cause LIKE ? OR impact LIKE ?
            ''', (f'%{query}%', f'%{query}%'))
            issue_ids = [row['issue_id'] for row in cursor.fetchall()]
        
        return [self.get_issue_data(issue_id) for issue_id in issue_ids]