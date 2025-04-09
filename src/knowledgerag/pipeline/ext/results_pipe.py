from docling.chunking import DocChunk

print(f"Question:\n{QUESTION}\n")
print(f"Answer:\n{rag_res['answer_builder']['answers'][0].data.strip()}\n")
print("Sources:")
sources = rag_res["answer_builder"]["answers"][0].documents
for source in sources:
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        doc_chunk = DocChunk.model_validate(source.meta["dl_meta"])
        print(f"- text: {repr(doc_chunk.text)}")
        if doc_chunk.meta.origin:
            print(f"  file: {doc_chunk.meta.origin.filename}")
        if doc_chunk.meta.headings:
            print(f"  section: {' / '.join(doc_chunk.meta.headings)}")
        bbox = doc_chunk.meta.doc_items[0].prov[0].bbox
        print(
            f"  page: {doc_chunk.meta.doc_items[0].prov[0].page_no}, "
            f"bounding box: [{int(bbox.l)}, {int(bbox.t)}, {int(bbox.r)}, {int(bbox.b)}]"
        )
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        print(repr(source.content))
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")