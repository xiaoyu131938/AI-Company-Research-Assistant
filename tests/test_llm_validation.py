"""Focused tests for LLM JSON validation and analysis guardrails."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_analyzer import (
    AIValidationError,
    analyze_company,
    assess_source_coverage,
    validate_ai_response,
)


def valid_payload(**overrides):
    payload = {
        "company_name": "Example Industrial Systems",
        "company_overview": (
            "Example Industrial Systems provides factory automation equipment. "
            "Its website describes software and production-line solutions."
        ),
        "core_products": [
            {"text": "Factory Control Platform", "source_ids": [1]},
            {"text": "Assembly Robot", "source_ids": [1]},
        ],
        "services_solutions": [
            {"text": "Production-line automation", "source_ids": [1]}
        ],
        "target_industries": [
            {"text": "Manufacturing", "source_ids": [1]}
        ],
        "business_keywords": ["automation", "industrial software"],
        "key_findings": [
            {
                "text": "The company combines equipment and software.",
                "source_ids": [1],
            },
            {
                "text": "Its solutions focus on production operations.",
                "source_ids": [1],
            },
        ],
        "confidence": "high",
    }
    payload.update(overrides)
    return payload


def crawl_result(content: str, *, status: str = "success"):
    warnings = []
    if status == "partial":
        warnings.append(
            {
                "type": "limited_coverage",
                "message": "Limited source coverage.",
                "details": "Only part of the website was available.",
            }
        )
    return {
        "status": status,
        "homepage": {
            "url": "https://example.com/",
            "title": "Example Industrial Systems",
            "type": "homepage",
            "content": content,
        },
        "pages": [],
        "warnings": warnings,
        "errors": [],
    }


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs):
        del kwargs
        response_text = self.responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class LLMValidationTests(unittest.TestCase):
    def test_valid_json(self) -> None:
        result = validate_ai_response(
            json.dumps(valid_payload()),
            valid_source_ids={1},
        )
        self.assertEqual(result["company_name"], "Example Industrial Systems")
        self.assertEqual(result["confidence"], "high")

    def test_markdown_code_fence_json(self) -> None:
        fenced = f"```json\n{json.dumps(valid_payload())}\n```"
        result = validate_ai_response(fenced, valid_source_ids={1})
        self.assertEqual(
            result["core_products"][0]["text"],
            "Factory Control Platform",
        )

    def test_missing_field_is_rejected(self) -> None:
        payload = valid_payload()
        del payload["key_findings"]
        with self.assertRaises(AIValidationError):
            validate_ai_response(json.dumps(payload), valid_source_ids={1})

    def test_wrong_list_field_type_is_rejected(self) -> None:
        payload = valid_payload(core_products="Assembly Robot")
        with self.assertRaises(AIValidationError):
            validate_ai_response(json.dumps(payload), valid_source_ids={1})

    def test_invalid_confidence_is_rejected(self) -> None:
        payload = valid_payload(confidence="very high")
        with self.assertRaises(AIValidationError):
            validate_ai_response(json.dumps(payload), valid_source_ids={1})

    def test_placeholder_content_is_rejected(self) -> None:
        payload = valid_payload(
            core_products=[{"text": "产品A", "source_ids": [1]}]
        )
        with self.assertRaises(AIValidationError):
            validate_ai_response(
                json.dumps(payload, ensure_ascii=False),
                valid_source_ids={1},
            )

    def test_prohibited_word_is_rejected(self) -> None:
        payload = valid_payload(business_keywords=["blocked-term"])
        with self.assertRaises(AIValidationError):
            validate_ai_response(
                json.dumps(payload),
                prohibited_words=["blocked-term"],
                valid_source_ids={1},
            )

    def test_limited_coverage_cannot_return_high_confidence(self) -> None:
        with self.assertRaises(AIValidationError):
            validate_ai_response(
                json.dumps(valid_payload(confidence="high")),
                limited_coverage=True,
                valid_source_ids={1},
            )

        result = validate_ai_response(
            json.dumps(valid_payload(confidence="medium")),
            limited_coverage=True,
            valid_source_ids={1},
        )
        self.assertEqual(result["confidence"], "medium")

    def test_missing_or_unknown_source_id_is_rejected(self) -> None:
        without_source = valid_payload(
            core_products=[{"text": "Assembly Robot", "source_ids": []}]
        )
        with self.assertRaises(AIValidationError):
            validate_ai_response(
                json.dumps(without_source),
                valid_source_ids={1},
            )

        unknown_source = valid_payload(
            core_products=[{"text": "Assembly Robot", "source_ids": [99]}]
        )
        with self.assertRaises(AIValidationError):
            validate_ai_response(
                json.dumps(unknown_source),
                valid_source_ids={1},
            )

    def test_partial_crawl_with_rich_evidence_is_sufficient(self) -> None:
        result = crawl_result("Industrial company homepage. " * 80, status="partial")
        result["warnings"] = [
            {
                "type": "unsafe_redirect",
                "message": "A non-business redirect was blocked.",
            }
        ]
        result["pages"] = [
            {
                "url": f"https://example.com/{page_type}/{index}",
                "title": f"Business page {index}",
                "type": page_type,
                "content": "Detailed product and solution evidence. " * 40,
            }
            for index, page_type in enumerate(
                [
                    "product",
                    "solution",
                    "service",
                    "business",
                    "about",
                    "product",
                    "solution",
                    "service",
                    "business",
                ],
                start=1,
            )
        ]

        coverage = assess_source_coverage(result)
        self.assertEqual(coverage["level"], "sufficient")
        self.assertEqual(coverage["successful_page_count"], 10)

        client = FakeClient([json.dumps(valid_payload(confidence="high"))])
        analysis = analyze_company(result, client=client)
        self.assertEqual(analysis["status"], "success")
        self.assertEqual(analysis["crawl_status"], "partial")
        self.assertEqual(analysis["source_coverage"]["level"], "sufficient")
        self.assertEqual(analysis["data"]["confidence"], "high")
        self.assertEqual(analysis["sources"][0]["source_id"], 1)

    def test_sparse_homepage_is_limited_coverage(self) -> None:
        result = crawl_result("Business equipment information. " * 20)
        coverage = assess_source_coverage(result)
        self.assertEqual(coverage["level"], "limited")

    def test_insufficient_source_content_skips_ai_call(self) -> None:
        client = FakeClient([json.dumps(valid_payload())])
        result = analyze_company(crawl_result("Company name and ICP record."), client=client)

        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(result["reason"], "insufficient_source_content")
        self.assertEqual(client.chat.completions.call_count, 0)

    def test_invalid_first_response_is_retried_once(self) -> None:
        client = FakeClient(
            [
                "not json",
                json.dumps(valid_payload(confidence="medium")),
            ]
        )
        result = analyze_company(
            crawl_result("Detailed website evidence. " * 20),
            client=client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(client.chat.completions.call_count, 2)

    def test_two_invalid_responses_return_manual_review(self) -> None:
        client = FakeClient(["not json", '{"company_name": "Incomplete"}'])
        result = analyze_company(
            crawl_result("Detailed website evidence. " * 20),
            client=client,
        )

        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(result["reason"], "invalid_ai_response")
        self.assertEqual(client.chat.completions.call_count, 2)

    def test_missing_api_key_returns_configuration_error(self) -> None:
        with patch.dict(os.environ, {"ZHIPU_API_KEY": ""}, clear=False):
            result = analyze_company(crawl_result("Detailed website evidence. " * 20))

        self.assertEqual(result["status"], "configuration_error")
        self.assertEqual(result["reason"], "missing_api_key")


if __name__ == "__main__":
    unittest.main()
