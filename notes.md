## Known limitations

- Section headers that span PDF page boundaries are lost during text
  extraction (e.g. § 160.103, § 164.501). Their body text is retained but
  attributed to the preceding section. Affects ~5 of 266 chunks.
- Validated via automated citation checks in `src/chunk.py`; measured, not assumed.
- Future fix: preserve line structure in `clean_text.py` rather than
  collapsing to a single string.