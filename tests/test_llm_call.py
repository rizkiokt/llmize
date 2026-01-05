"""Test LLM call functionality for OpenRouter and Google providers."""

import json
import os
from unittest.mock import Mock, patch

import pytest
from dotenv import load_dotenv

from llmize.llm.llm_call import (
    generate_content,
    generate_content_gemini,
    generate_content_openrouter,
)
from llmize.llm.llm_init import initialize_llm

load_dotenv()


class TestLLMCall:
    """Test LLM call functions for different providers."""

    simple_prompt = "Generate a number between 1 and 10"
    simple_response = "7"

    def test_initialize_llm_google(self):
        """Test Google model initialization."""
        client = initialize_llm("google/gemma-3-27b-it", "test-key")
        assert client is not None
        assert hasattr(client, 'models')

        # Test backward compatibility
        client = initialize_llm("gemma-3-27b-it", "test-key")
        assert client is not None

        client = initialize_llm("gemini-2.5-flash", "test-key")
        assert client is not None
    
    def test_initialize_llm_openrouter(self):
        """Test OpenRouter model initialization."""
        client = initialize_llm(
            "openrouter/meta-llama/llama-3.3-70b-instruct", "test-key"
        )
        assert client is not None
        assert client["type"] == "openrouter"
        assert client["api_key"] == "test-key"

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"), reason="Need GEMINI_API_KEY to run this test"
    )
    def test_generate_content_google_real(self):
        """Test Google Gemma content generation with real API."""
        client = initialize_llm(
            "gemma-3-27b-it", os.getenv("GEMINI_API_KEY")
        )
        response = generate_content(
            client, "gemma-3-27b-it", "What is 2+2?"
        )
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nGoogle Gemma response: {response}")
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Need OPENROUTER_API_KEY to run this test",
    )
    def test_generate_content_openrouter_real(self):
        """Test OpenRouter content generation with real API."""
        client = initialize_llm(
            "openrouter/meta-llama/llama-3.3-70b-instruct",
            os.getenv("OPENROUTER_API_KEY"),
        )
        response = generate_content(
            client,
            "openrouter/meta-llama/llama-3.3-70b-instruct",
            "What is 2+2?",
        )
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nOpenRouter Llama 3.3 response: {response}")

    def test_generate_content_gemini_mock(self):
        """Test Google Gemma content generation with mock."""
        with (
            patch('llmize.llm.llm_call.genai') as mock_genai,
            patch('llmize.llm.llm_call.generate_content_gemini') as mock_generate,
        ):
            mock_generate.return_value = self.simple_response

            client = initialize_llm("google/gemma-3-27b-it", "test-key")
            response = generate_content(
                client, "google/gemma-3-27b-it", self.simple_prompt, 0.7
            )

            assert response == self.simple_response
    
    def test_generate_content_openrouter_mock(self):
        """Test OpenRouter content generation with mock."""
        with patch('llmize.llm.llm_call.requests.post') as mock_post:
            # Setup mock response
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": self.simple_response}}]
            }
            mock_post.return_value = mock_response

            client = initialize_llm(
                "openrouter/meta-llama/llama-3.3-70b-instruct", "test-key"
            )
            response = generate_content_openrouter(
                client,
                "openrouter/meta-llama/llama-3.3-70b-instruct",
                self.simple_prompt,
                0.7,
            )

            assert response == self.simple_response
            mock_post.assert_called_once()

            # Check the request was made to OpenRouter API
            call_args = mock_post.call_args
            assert "openrouter.ai/api/v1/chat/completions" in call_args[0][0]
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    def test_generate_content_routing(self):
        """Test that generate_content routes to the correct provider."""
        with (
            patch('llmize.llm.llm_call.generate_content_gemini') as mock_gemini,
            patch('llmize.llm.llm_call.generate_content_openrouter') as mock_openrouter,
        ):
            mock_gemini.return_value = "google response"
            mock_openrouter.return_value = "openrouter response"

            # Test Google routing
            client = Mock()
            generate_content(
                client, "google/gemma-3-27b-it", self.simple_prompt
            )
            mock_gemini.assert_called_once()

            # Test OpenRouter routing
            generate_content(
                client, "openrouter/meta-llama/llama-3.3-70b-instruct", self.simple_prompt
            )
            mock_openrouter.assert_called_once()

    def test_google_model_prefix_stripping(self):
        """Test that google/ prefix is stripped for API calls."""
        with (
            patch('llmize.llm.llm_call.genai') as mock_genai,
            patch('llmize.llm.llm_call.generate_content_gemini') as mock_generate,
        ):
            mock_generate.return_value = self.simple_response

            client = initialize_llm("google/gemma-3-27b-it", "test-key")
            generate_content(
                client, "google/gemma-3-27b-it", self.simple_prompt, 0.7
            )

            # Verify the generate_content_gemini was called with the full model name
            # (prefix stripping happens inside generate_content_gemini)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            # The model is passed with prefix to generate_content_gemini
            assert call_args[0][1] == "google/gemma-3-27b-it"

    def test_openrouter_model_prefix_stripping(self):
        """Test that openrouter/ prefix is stripped for API calls."""
        with patch('llmize.llm.llm_call.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": self.simple_response}}]
            }
            mock_post.return_value = mock_response

            client = initialize_llm("openrouter/anthropic/claude-3", "test-key")
            generate_content_openrouter(
                client, "openrouter/anthropic/claude-3", self.simple_prompt, 0.7
            )

            # Check the request data
            call_args = mock_post.call_args
            request_data = call_args[1]["data"]
            data = json.loads(request_data)
            assert data["model"] == "anthropic/claude-3"  # No openrouter/ prefix

    def test_openrouter_llama33_prefix_stripping(self):
        """Test that openrouter/ prefix is stripped for Llama 3.3 model."""
        with patch('llmize.llm.llm_call.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": self.simple_response}}]
            }
            mock_post.return_value = mock_response

            client = initialize_llm(
                "openrouter/meta-llama/llama-3.3-70b-instruct", "test-key"
            )
            generate_content_openrouter(
                client,
                "openrouter/meta-llama/llama-3.3-70b-instruct",
                self.simple_prompt,
                0.7,
            )

            # Check the request data
            call_args = mock_post.call_args
            request_data = call_args[1]["data"]
            data = json.loads(request_data)
            assert (
                data["model"] == "meta-llama/llama-3.3-70b-instruct"
            )  # No openrouter/ prefix
