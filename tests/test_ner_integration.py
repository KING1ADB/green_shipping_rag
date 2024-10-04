# tests/test_ner_integration.py

import unittest
from unittest.mock import patch
from src.ner_integration import annotate_entities, retrieve_and_generate

class TestNERIntegration(unittest.TestCase):

    @patch('src.ner_integration.openai.ChatCompletion.create')
    def test_annotate_entities(self, mock_chat):
        # Mock the API response
        mock_chat.return_value = {
            'choices': [{'message': {'content': 'Entity: Green Shipping'}}]
        }
        result = annotate_entities("What is green shipping?")
        self.assertEqual(result, 'Entity: Green Shipping')

    @patch('src.ner_integration.annotate_entities')
    @patch('src.ner_integration.openai.ChatCompletion.create')
    def test_retrieve_and_generate(self, mock_chat, mock_annotate):
        # Mock the NER response
        mock_annotate.return_value = "Entity: Green Shipping"
        mock_chat.return_value = {
            'choices': [{'message': {'content': 'Response based on entities.'}}]
        }
        result = retrieve_and_generate("What are the regulations regarding shipping emissions?")
        self.assertEqual(result, 'Response based on entities.')

if __name__ == "__main__":
    unittest.main()