import fitz
from tqdm.auto import tqdm
import numpy as np
import faiss
import pickle
import ollama
from spacy.lang.en import English
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass
class ChunkMetadata:
    page_number: int
    char_count: int
    token_count: float
    sentence_count: int


class RAGPipeline:
    def __init__(
        self,
        min_token_length: int = 40,
        chunk_size: int = 10,
        llm_model: str = "llama3.2:1b",
        embedding_model: str = "nomic-embed-text",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        self.min_token_length = min_token_length
        self.chunk_size = chunk_size
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.index = None
        self.chunks = []

    def process_document(self, pdf_path: str) -> None:
        """Process document and build search index."""
        # Extract content and create chunks
        pages_content = self._extract_pdf_content(pdf_path)
        self.chunks = self._create_chunks(pages_content)

        # Build search index
        self._build_faiss_index()

        print(f"Processed document with {len(self.chunks)} chunks")

    def _extract_pdf_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text and metadata from PDF."""
        doc = fitz.open(pdf_path)
        pages_content = []

        for page_num, page in tqdm(enumerate(doc), desc="Processing PDF pages"):
            text = " ".join(page.get_text().split())
            doc = self.nlp(text)
            sentences = [str(sent) for sent in doc.sents]

            pages_content.append(
                {
                    "page_number": page_num,
                    "text": text,
                    "sentences": sentences,
                }
            )
        return pages_content

    def _create_chunks(
        self, pages_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create text chunks from pages content."""
        chunks = []

        for page in tqdm(pages_content, desc="Creating chunks"):
            for i in range(0, len(page["sentences"]), self.chunk_size // 2):
                chunk_sentences = page["sentences"][i : i + self.chunk_size]
                if not chunk_sentences:
                    continue

                chunk_text = " ".join(chunk_sentences)
                token_count = len(chunk_text) / 4

                if token_count <= self.min_token_length:
                    continue

                chunks.append(
                    {
                        "page_number": page["page_number"],
                        "chunk_text": chunk_text,
                        "metadata": ChunkMetadata(
                            page_number=page["page_number"],
                            char_count=len(chunk_text),
                            token_count=token_count,
                            sentence_count=len(chunk_sentences),
                        ),
                    }
                )
        return chunks

    def _build_faiss_index(self) -> None:
        """Build FAISS index from chunks."""
        embeddings = []
        for chunk in tqdm(self.chunks, desc="Creating embeddings"):
            embedding = ollama.embed(
                model=self.embedding_model, input=chunk["chunk_text"]
            )["embeddings"]
            embeddings.append(embedding)

        embeddings_array = np.array(embeddings).astype("float32")
        embeddings_array = np.squeeze(
            embeddings_array, axis=1
        )  # Remove the extra dimension
        dimension = embeddings_array.shape[1]

        print(f"Embeddings shape: {embeddings_array.shape}")

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

    def save_state(
        self, index_path: str = "faiss_index", chunks_path: str = "chunks.pkl"
    ) -> None:
        """Save FAISS index and chunks to disk."""
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            with open(chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)

    def load_state(
        self, index_path: str = "faiss_index", chunks_path: str = "chunks.pkl"
    ) -> None:
        """Load FAISS index and chunks from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built or loaded. Process a document first.")

        # Get query embedding
        query_embedding = ollama.embed(model=self.embedding_model, input=query)[
            "embeddings"
        ]

        # Convert to numpy array and ensure correct shape
        query_vector = np.array(query_embedding).astype("float32")
        query_vector = np.squeeze(query_vector)  # Remove extra dimensions

        # Reshape to match index dimensionality (1, 768)
        query_vector = query_vector.reshape(1, -1)

        # print(f"Query vector shape: {query_vector.shape}")  # Debug print

        # Search index
        distances, indices = self.index.search(query_vector, k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append(
                {
                    "chunk": self.chunks[idx]["chunk_text"],
                    "page": self.chunks[idx]["page_number"],
                    "distance": float(distance),
                }
            )

        return results

    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using retrieved context."""
        # Prepare context string
        context_str = "\n\n".join(
            [f"[Page {c['page']}: {c['chunk']}]" for c in context]
        )

        # Prepare prompt
        prompt = f"""Use the following context to answer the question. If you cannot answer the question based on the context, say so.

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate response
        start_time = time.time()
        response = ollama.generate(
            model=self.llm_model,
            prompt=prompt,
        )
        end_time = time.time()

        return {
            "response": response["response"],
            "generation_time": end_time - start_time,
            "context_used": context,
        }

    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate."""
        # Retrieve relevant chunks
        context = self.retrieve(query, k=k)

        # Generate response
        result = self.generate(query, context)

        return result


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    rag = RAGPipeline(
        llm_model="llama3.2:1b",  # or any other model you have in Ollama
        embedding_model="nomic-embed-text",
        temperature=0.7,
        max_tokens=500,
    )

    # Either process a new document
    rag.process_document("./B5084.pdf")
    rag.save_state()  # Save for later use

    # Or load previously processed document
    # rag.load_state()

    # Example query
    query = "What is the main topic of this document?"
    result = rag.query(query)

    print("\nQuestion:", query)
    print("\nAnswer:", result["response"])
    print("\nRelevant pages:", [c["page"] for c in result["context_used"]])
    print(f"\nGeneration time: {result['generation_time']:.2f} seconds")
