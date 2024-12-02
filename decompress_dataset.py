
import zstandard as zstd
import os

input_file = "Dataset/lichess_db_standard_rated_2017-03.pgn.zst"
output_file = "Dataset/lichess_decompressed.pgn"

# Open and decompress the .zst file
with open(input_file, 'rb') as compressed:
    dctx = zstd.ZstdDecompressor()
    with open(output_file, 'wb') as destination:
        dctx.copy_stream(compressed, destination)

print("File  decompressed successfully.")

# Delete the original compressed file
os.remove(input_file)
