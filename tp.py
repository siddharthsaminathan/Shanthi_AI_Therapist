import pyarrow.parquet as pq

file_path = "/Users/siddharthsaminathan/Downloads/Gomathi/empathetic_dialogues_llm/data/train-00000-of-00001.parquet"
try:
    table = pq.read_table(file_path)
    print(table)
except Exception as e:
    print(f"Error: {e}")