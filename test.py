import os
import sqlite3
import time

from code_indexer_loop.api import CodeIndexer
os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"
start_time_query = time.time()


path = "<directory_to_scan>"
indexer = CodeIndexer(src_dir=path, refresh=False, watch=True, db_path="<path_to_save_db>")
print("CodeIndexer took {:.2f} seconds.".format(time.time() - start_time_query))

# Perform a query
query = "select table"
starttime2 = time.time()
results = indexer.query(query)

print("Query execution took {:.2f} seconds.".format(time.time() - starttime2))

# Check if the results list is not empty and print the first 500 characters of the first result
if results:
    print(results[0:500])
else:
    print("No results found.")
print("Total time taken: {:.2f} seconds.".format(time.time() - start_time_query))