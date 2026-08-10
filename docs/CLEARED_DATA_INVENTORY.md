# Cleared data inventory

Cleared to recover disk on a volume that had reached 100 percent capacity.
Every path below is regenerable. Nothing under results figures docs src
scripts data/manifests data/processed models or data/raw/encode-foldchange
was touched.

## Removed

| path | size | restore |
|---|---|---|
| data/raw/encode-bulk | 95G | data/manifests/encode-bulk.heavy.json |
| data/raw/hg38-ref | 4.0G | reference genome download |
| data/raw/hg19-ref | 4.0G | reference genome download |
| data/raw/mm10-ref | 3.6G | reference genome download |
| data/raw/mm9-ref | 3.4G | reference genome download |
| data/raw/dbsnp-common | 1.5G | data/manifests/dbsnp-common.heavy.json |
| data/studies/training_candidate | 38G | training rerun |

## Deliberately preserved

data/raw/encode-foldchange (3.4G) is the input to the H3K27ac fold-change
enrichment in section 25 which is the only surviving positive result in this
project. It was excluded from clearing even though it is manifest-backed.

The removed paths sit under data/raw and data/studies which are gitignored
so no git history was affected. Code that reads encode-bulk lives in
src/pdac_circuit/chromatin/encode.py inventory.py data/bulk.py and
signal/precompute.py and will need the manifest refetched before it runs.
