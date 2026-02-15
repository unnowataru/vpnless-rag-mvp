from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ingest_metadata import infer_labels_from_doc_id
from core.ingest_metadata import load_metadata_rules
from core.ingest_metadata import missing_required_metadata
from core.ingest_metadata import parse_required_metadata_fields
from core.ingest_metadata import parse_default_metadata_json
from core.ingest_metadata import resolve_document_metadata


class IngestMetadataTests(unittest.TestCase):
    def test_parse_default_metadata_json(self) -> None:
        obj = parse_default_metadata_json(
            '{"dept":"hr","labels":["policy","travel"],"confidentiality":"internal"}'
        )
        self.assertEqual(obj["dept"], "hr")
        self.assertEqual(obj["labels"], ["policy", "travel"])
        self.assertEqual(obj["label"], "policy")

    def test_load_rules_and_apply_to_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf = pdf_dir / "doc_travel_sample"
            pdf.write_bytes(b"dummy")

            rules_path = root / "rules.json"
            rules_payload = {
                "rules": [
                    {
                        "pattern": "出張|旅費",
                        "fields": {"dept": "ga", "labels": ["travel", "policy"]},
                    }
                ]
            }
            rules_path.write_text(json.dumps(rules_payload, ensure_ascii=False), encoding="utf-8")
            rules = load_metadata_rules(str(rules_path))

            metadata = resolve_document_metadata(
                pdf,
                pdf_root=pdf_dir,
                default_metadata={"confidentiality": "internal"},
                metadata_rules=rules,
                include_updated_at_from_mtime=True,
            )
            self.assertEqual(metadata["doc_id"], "doc_travel_sample")
            self.assertEqual(metadata["dept"], "ga")
            self.assertEqual(metadata["labels"], ["travel", "policy"])
            self.assertEqual(metadata["label"], "travel")
            self.assertIn("updated_at", metadata)

    def test_infer_labels_from_doc_id(self) -> None:
        labels = infer_labels_from_doc_id("doc_faq_sample")
        self.assertIn("faq", labels)
        self.assertIn("policy", labels)

    def test_parse_required_metadata_fields(self) -> None:
        fields = parse_required_metadata_fields("doc_id,dept,labels,updated_at")
        self.assertEqual(fields, ("doc_id", "dept", "labels", "updated_at"))
        with self.assertRaises(ValueError):
            parse_required_metadata_fields("doc_id,unknown_key")

    def test_missing_required_metadata(self) -> None:
        doc_meta = {"doc_id": "DOC_A", "labels": ["policy"], "updated_at": "2026-02-15"}
        missing = missing_required_metadata(doc_meta, ("doc_id", "dept", "labels", "updated_at"))
        self.assertEqual(missing, ["dept"])


if __name__ == "__main__":
    unittest.main()
