"""Offline nullable-schema and manager-context regressions."""
import copy
import unittest
from scripts import free_schema_compat as compat


SCHEMA = {'type': 'object', 'properties': {
    'advisory_rating': {'anyOf': [{'$ref': '#/$defs/Rating'}, {'type': 'null'}],
                        'default': None, 'description': 'Optional rating.'},
    'action': {'type': 'string', 'enum': ['BUY', 'HOLD', 'SELL']},
    'stop_loss': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None}},
    'required': ['advisory_rating', 'action', 'stop_loss'], 'additionalProperties': False,
    '$defs': {'Rating': {'type': 'string', 'enum': ['Buy', 'Hold', 'Sell'], 'title': 'Rating'}}}


class SchemaTests(unittest.TestCase):
    def test_nullable_enum_has_same_values_plus_null(self):
        result, count = compat.normalize_nullable_scalars(SCHEMA)
        field = result['properties']['advisory_rating']
        self.assertEqual(field['type'], ['string', 'null'])
        self.assertEqual(field['enum'], ['Buy', 'Hold', 'Sell', None])
        self.assertEqual(field['description'], 'Optional rating.')
        self.assertNotIn('anyOf', field)
        self.assertEqual(count, 2)

    def test_preserves_non_nullable_actions_and_object_constraints(self):
        result, count = compat.normalize_nullable_scalars(SCHEMA)
        self.assertEqual(result['properties']['action'], SCHEMA['properties']['action'])
        self.assertEqual(result['required'], SCHEMA['required'])
        self.assertFalse(result['additionalProperties'])

    def test_original_schema_not_mutated(self):
        original = copy.deepcopy(SCHEMA)
        compat.normalize_nullable_scalars(SCHEMA)
        self.assertEqual(SCHEMA, original)

    def test_not_a_general_union_rewriter(self):
        document = {'anyOf': [{'type': 'string'}, {'type': 'number'}]}
        result, count = compat.normalize_nullable_scalars(document)
        self.assertEqual(result, document)
        self.assertEqual(count, 0)

    def test_external_reference_is_left_alone(self):
        document = {'anyOf': [{'$ref': 'https://example.invalid/schema'}, {'type': 'null'}]}
        self.assertEqual(compat.normalize_nullable_scalars(document), (document, 0))

    def test_does_not_override_parent_validation_constraints(self):
        document = {'type': 'string', 'anyOf': [{'type': 'string'}, {'type': 'null'}]}
        self.assertEqual(compat.normalize_nullable_scalars(document), (document, 0))

    def test_response_format_keeps_strict_mode(self):
        payload = {'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'ResearchPlan', 'strict': True, 'schema': SCHEMA}},
            'messages': [{'role': 'user', 'content': 'Evidence'}]}
        result, count = compat.normalize_payload(payload)
        self.assertTrue(result['response_format']['json_schema']['strict'])
        self.assertEqual(result['messages'], payload['messages'])
        self.assertEqual(result['response_format']['json_schema']['name'], 'ResearchPlan')
        self.assertEqual(count, 2)

    def test_tool_requests_are_unchanged(self):
        payload = {'tools': [{'type': 'function', 'function': {'name': 'get_data'}}]}
        self.assertEqual(compat.normalize_payload(payload), (payload, 0))

    def test_final_manager_digest_removes_duplicate_history(self):
        template = '{risk_debate_digest} {history} {trader_plan} {account_status_desc} {final_format}'
        result = compat.compact_manager_history(template)
        self.assertNotIn('{history}', result)
        for field in ('risk_debate_digest', 'trader_plan', 'account_status_desc', 'final_format'):
            self.assertIn('{' + field + '}', result)

    def test_history_stays_without_a_replacement_digest(self):
        template = 'Evidence {history}'
        self.assertEqual(compat.compact_manager_history(template), template)


if __name__ == '__main__':
    unittest.main()
