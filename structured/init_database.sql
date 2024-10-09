CREATE TABLE IF NOT EXISTS llm_extracted_issues (
    id INTEGER PRIMARY KEY,
    issue_id TEXT UNIQUE,
    severity_level TEXT,
    root_cause TEXT,
    impact TEXT,
    llm_json_response TEXT
);

CREATE TABLE IF NOT EXISTS llm_extracted_systems (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS llm_extracted_people (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS llm_extracted_timestamps (
    id INTEGER PRIMARY KEY,
    issue_id TEXT,
    type TEXT,
    timestamp TEXT,
    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id)
);

CREATE TABLE IF NOT EXISTS llm_extracted_resolution_steps (
    id INTEGER PRIMARY KEY,
    issue_id TEXT,
    step_number INTEGER,
    description TEXT,
    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id)
);

CREATE TABLE IF NOT EXISTS llm_extracted_relationships (
    id INTEGER PRIMARY KEY,
    source TEXT,
    relationship_type TEXT,
    target TEXT
);

CREATE TABLE IF NOT EXISTS llm_extracted_issue_systems (
    issue_id TEXT,
    system_id INTEGER,
    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id),
    FOREIGN KEY (system_id) REFERENCES llm_extracted_systems (id)
);

CREATE TABLE IF NOT EXISTS llm_extracted_issue_people (
    issue_id TEXT,
    person_id INTEGER,
    FOREIGN KEY (issue_id) REFERENCES llm_extracted_issues (issue_id),
    FOREIGN KEY (person_id) REFERENCES llm_extracted_people (id)
);