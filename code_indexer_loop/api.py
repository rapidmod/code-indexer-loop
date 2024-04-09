import atexit
import os
from pathlib import Path
import chardet
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from llama_index.core import ServiceContext, VectorStoreIndex

from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from code_indexer_loop.code_splitter import CodeSplitter
from code_indexer_loop.constants import EXTENSION_TO_TREE_SITTER_LANGUAGE
from code_indexer_loop.utils import hash_md5


class CodeIndexer:
    src_dir: str
    target_chunk_tokens: int
    max_chunk_tokens: int
    enforce_max_chunk_tokens: bool
    token_model: str
    code_splitters = {}
    hash_cache = {}
    index: VectorStoreIndex = None

    def __init__(
            self,
            src_dir: str,
            target_chunk_tokens: int = 300,
            max_chunk_tokens: int = 1000,
            enforce_max_chunk_tokens: bool = False,
            coalesce: int = 50,
            token_model: str = "gpt-4",
            watch: bool = False,
            refresh: bool = True,
            db_path: str = "",  # Path to save the Chroma database, none if not persistent @rapidmod
            ignore_errors: bool = False
    ):
        self.ignore_errors = ignore_errors
        self.src_dir = src_dir
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.enforce_max_chunk_tokens = enforce_max_chunk_tokens
        self.coalesce = coalesce
        self.token_model = token_model
        self.db_path = db_path
        self._create_index()
        if refresh:
            self.refresh_nodes()

        if watch:
            self._start_watching()
            # Ensure the watcher is stopped properly.
            atexit.register(self._stop_watching)

    def query(self, query: str, k=10) -> str:
        return "\n".join(
            [node_with_score.node.text for node_with_score in
             self.index.as_retriever(similarity_top_k=k).retrieve(query)]
        )

    def query_nodes(self, query: str, k=10) -> list[NodeWithScore]:
        return self.index.as_retriever(similarity_top_k=k).retrieve(query)

    def query_documents(self, query: str, k=10) -> list[dict[str, str]]:
        nodes = self.index.as_retriever(similarity_top_k=k).retrieve(query)
        files = [node_with_score.node.metadata["file"] for node_with_score in nodes]
        # Deduplicate files, preserving order
        files = list(dict.fromkeys(files))
        # Read file contents
        contents = []
        for file in files:
            with open(file, "r") as f:
                contents.append(
                    {
                        "file": file,
                        "content": f.read(),
                    }
                )
        return contents

    def add_file(self, file: str):
        ext = Path(file).suffix
        text_splitter = self._get_code_splitter(ext)

        calculated_hash = hash_md5(file)
        if file in self.hash_cache:
            if self.hash_cache[file] == calculated_hash:
                # Skip file if it hasn't changed
                return
        else:
            self.hash_cache[file] = calculated_hash

        # Detect file encoding
        with open(file, 'rb') as f:
            raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        print(f"Detected encoding for {file}: {detected_encoding}")

        # Read the file with the detected encoding
        with open(file, "r", encoding=detected_encoding, errors='ignore') as f:
            text = f.read()
            nodes = [
                TextNode(
                    text=chunk,
                    metadata={
                        "file": file,
                    },
                )
                for chunk in text_splitter.split_text(text)
            ]

            self._remove_old_nodes(file)
            self._insert_nodes(nodes)

    def remove_file(self, file: str):
        self._remove_old_nodes(file)
        del self.hash_cache[file]

    def refresh_nodes(self):
        files = self._find_files(self.src_dir, EXTENSION_TO_TREE_SITTER_LANGUAGE)

        # Clear any files that no longer exist
        for file in list(self.hash_cache.keys()):
            if file not in files:
                del self.hash_cache[file]
                self._remove_old_nodes(file)

        # For each file, split into chunks and index
        for file in files:
            self.add_file(str(file))

    def _start_watching(self):
        event_handler = CodeChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.src_dir, recursive=True)
        self.observer.start()

    def _stop_watching(self):
        if hasattr(self, "observer"):
            self.observer.stop()
            self.observer.join()

    def _find_files(self, path, include_ext= { } ):
        """
        Recursively find all files in a given path.

        Parameters:
            path (str): The root directory to start searching from.
            include_ext (dict): A dictionary of file extensions to include
                (keys are extensions including leading period if applicable).

        Returns:
            list: A list of full file paths for each file found.
            :param path:
            :param include_ext:
            :return:
        """
        found_files = []

        for root, _, files in os.walk(path):
            for file in files:
                # Check if the file should be excluded based on its extension
                file_ext = os.path.splitext(file)[1]
                if file_ext in include_ext:
                    # Construct the full path of the file and append to list
                    full_path = Path(os.path.join(root, file)).resolve()
                    found_files.append(full_path)

        return set(found_files)

    def _get_code_splitter(self, ext) -> CodeSplitter:
        if ext not in EXTENSION_TO_TREE_SITTER_LANGUAGE:
            raise ValueError(f"Extension {ext} not supported.")
        language = EXTENSION_TO_TREE_SITTER_LANGUAGE[ext]
        if language not in self.code_splitters:
            text_splitter = CodeSplitter(
                language=language,
                target_chunk_tokens=self.target_chunk_tokens,
                max_chunk_tokens=self.max_chunk_tokens,
                enforce_max_chunk_tokens=self.enforce_max_chunk_tokens,
                coalesce=self.coalesce,
                token_model=self.token_model,
                ignore_errors=self.ignore_errors
            )
            self.code_splitters[ext] = text_splitter

        return self.code_splitters[ext]

    def _remove_old_nodes(self, file):
        # Remove existing nodes for the same file
        self.index.vector_store.client.delete(where={"file": file})

    def _insert_nodes(self, nodes):
        self.index.insert_nodes(nodes)

    def _create_index(self):
        if not self.db_path:
            print("Using ephemeral client.")
            # Use a persistent client for Chroma if a path is provided @rapidmod
            chroma_client = chromadb.EphemeralClient()
        else:
            print(f"Using persistent client with path: {self.db_path}")
            chroma_client = chromadb.PersistentClient(path=self.db_path)

        try:
            # Try to get the existing collection, or create it if it doesn't exist
            chroma_collection = chroma_client.get_or_create_collection(name="code-index")
        except Exception as e:
            print(f"Error accessing or creating the collection: {e}")
            raise

        # Define the embedding function
        embed_model = LangchainEmbedding(OpenAIEmbeddings())

        # Set up ChromaVectorStore with the persistent collection
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

        self.index = index


class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer: CodeIndexer):
        self.indexer = indexer

    def on_modified(self, event):
        if event.is_directory:
            # Directory modifications shouldn't trigger a reindex
            return
        else:
            # Update only if the changed file has a supported extension
            ext = os.path.splitext(event.src_path)[1]
            if ext in EXTENSION_TO_TREE_SITTER_LANGUAGE:
                self.indexer.add_file(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            self.indexer.refresh_nodes()
        else:
            # Update only if the changed file has a supported extension
            ext = os.path.splitext(event.src_path)[1]
            if ext in EXTENSION_TO_TREE_SITTER_LANGUAGE:
                self.indexer.add_file(event.src_path)

    def on_moved(self, event):
        self.indexer.refresh_nodes()

    def on_deleted(self, event):
        if event.is_directory:
            self.indexer.refresh_nodes()
        else:
            ext = os.path.splitext(event.src_path)[1]
            if ext in EXTENSION_TO_TREE_SITTER_LANGUAGE:
                self.indexer.remove_file(event.src_path)
