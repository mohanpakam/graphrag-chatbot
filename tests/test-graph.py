import spacy

nlp = spacy.load("en_core_web_lg")  # Replace with your desired model

def create_graph(text):
    doc = nlp(text)

    # Create a graph representation
    graph = {}
    for token in doc:
        graph[token.text] = {}
        for child in token.children:
            graph[token.text][child.text] = child.dep_

    return graph

# Example usage
text = "The quick brown fox jumps over the lazy dog."
graph = create_graph(text)
print(graph)