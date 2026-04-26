"""
Concatenate 2020 + 2021-2024 location parquets into african_gkg_locations_aligned.parquet
Uses pyarrow streaming to avoid loading both datasets into RAM simultaneously.
"""
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

BASE    = Path(__file__).parent.parent
INTERIM = BASE / "DATA" / "interim"
RAW     = BASE / "DATA" / "raw"
OUT     = INTERIM / "african_gkg_locations_aligned.parquet"

f1 = INTERIM / "african_gkg_locations_2020.parquet"
f2 = RAW / "african_gkg_locations_2021_2024.parquet"

print(f"Reading schema from {f2.name} ...")
schema = pq.read_schema(f2)
print(f"  {len(schema)} columns")

print(f"Writing combined parquet to {OUT} ...")
writer = None

for path, label in [(f1, "2020"), (f2, "2021-2024")]:
    print(f"  Streaming {label} ...")
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=500_000):
        table = pa.Table.from_batches([batch])
        # cast to match schema — add missing cols as null, cast types to match
        cols = []
        for col in schema.names:
            target_type = schema.field(col).type
            if col in table.schema.names:
                arr = table.column(col)
                if arr.type != target_type:
                    arr = arr.cast(target_type, safe=False)
            else:
                arr = pa.array([None] * len(table), type=target_type)
            cols.append(arr)
        table = pa.table({name: arr for name, arr in zip(schema.names, cols)}, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(OUT, schema)
        writer.write_table(table)
        n += len(table)
    print(f"    {n:,} rows written")

if writer:
    writer.close()

size_mb = OUT.stat().st_size / 1e6
print(f"\nDone -> {OUT}  ({size_mb:.0f} MB)")
