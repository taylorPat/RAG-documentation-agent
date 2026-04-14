from minsearch import Index, VectorSearch

from src.embedding import Embedding

def semantic_search(embedded_chunks, chunks, query):
    search = SemanticSearch.create_from_chunks(embedded_chunks=embedded_chunks, chunks=chunks)
    return search.search(query=query)

class SemanticSearch:
    def __init__(self, index):
        self.index = index
        
    @classmethod
    def create_from_chunks(cls, embedded_chunks, chunks):
        index = VectorSearch()
        index = index.fit(embedded_chunks, chunks)
        return cls(index=index)
    
    def search(self, query: str, num_results: int = 5) -> list[any]:
        """
        Perform a semantic (embedding-based) search on the index.

        Args:
            query (str): The natural language search query.
            num_results (int, optional): Number of results to return. Defaults to 5.

        Returns:
            List[any]: Up to `num_results` semantically similar results.
        """
        embedded_query = Embedding().create(content=query)
        return self.index.search(embedded_query, num_results=num_results)

class TextSearch:
    def __init__(self, index: Index):
        self.index = index
        
    @classmethod
    def create_from_chunks(cls, text_fields, chunks):
        index = Index(text_fields=text_fields, keyword_fields=[])
        index = index.fit(chunks)
        return cls(index=index)


    def search(self, query: str, num_results: int = 5) -> list[any]:
        """
        Perform a text-based search on the FAQ index.

        Args:
            query (str): The search query string.

        Returns:
            List[any]: A list of up to 5 search results returned by the FAQ index.
        """
        return self.index.search(query, num_results=num_results)