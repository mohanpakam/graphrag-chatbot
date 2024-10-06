import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import markdown
from bs4 import BeautifulSoup
import argparse
import os
import pandas as pd

def preprocess_text(text: str) -> str:
    # Remove special characters and lowercase the text
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    preprocessed_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(zip(tfidf_matrix.tocsr().data, feature_names), key=lambda x: x[0], reverse=True)
    return [item[1] for item in sorted_items[:top_n]]

def markdown_to_text(markdown_content: str) -> str:
    # Convert markdown to HTML
    html = markdown.markdown(markdown_content, extensions=['tables'])
    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()

def extract_keywords_from_tables(markdown_content: str) -> List[str]:
    html = markdown.markdown(markdown_content, extensions=['tables'])
    soup = BeautifulSoup(html, features="html.parser")
    tables = soup.find_all('table')
    keywords = []
    for table in tables:
        df = pd.read_html(str(table))[0]
        keywords.extend(df.values.flatten().tolist())
    return [keyword for keyword in keywords if isinstance(keyword, str) and keyword.strip()]

def extract_keywords_from_md_file(file_path: str, top_n: int = 10) -> Dict[str, List[str]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    
    # Extract keywords from tables
    table_keywords = extract_keywords_from_tables(md_content)
    
    # If we don't have enough keywords from tables, extract from the full text
    if len(table_keywords) < top_n:
        plain_text = markdown_to_text(md_content)
        text_keywords = extract_keywords(plain_text, top_n - len(table_keywords))
        keywords = table_keywords + text_keywords
    else:
        keywords = table_keywords[:top_n]
    
    return {
        "file_name": os.path.basename(file_path),
        "keywords": keywords
    }

def main():
    parser = argparse.ArgumentParser(description="Extract keywords from a Markdown file.")
    parser.add_argument("file_path", help="Path to the Markdown file")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top keywords to extract (default: 10)")
    args = parser.parse_args()

    result = extract_keywords_from_md_file(args.file_path, args.top_n)
    print(f"Keywords extracted from {result['file_name']}:")
    for keyword in result['keywords']:
        print(f"- {keyword}")

if __name__ == "__main__":
    main()