# SMJ deep extraction: frozen supplement (2026-10-19)

Three inputs of grounding/build_grounding_inventory.py that the 2026-10-13 frozen copy (../smj-deep-frozen-2026-10-13/)
lacked, copied byte for byte from the SMJ article extraction (not distributed; see DATA.md). The source files were last
modified on 2026-07-23, before the 2026-10-13 inventory build.

- harmonize.py (name normalisation and rulebook code, imported by the builder)
- rulebook_v1.json
- name_map_v1.csv

Hashes are in SHA256SUMS.txt; the builder checks them before reading and stops if any differs. Never edit these files.
