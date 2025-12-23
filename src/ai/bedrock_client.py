"""
Amazon Bedrock Client Wrapper
=============================

Provides a unified interface to multiple foundation models available
through Amazon Bedrock:
- Anthropic Claude (3.5 Sonnet, 3 Haiku)
- Amazon Titan
- Mistral Large

Each model has different input/output formats, and this client
normalizes them into a consistent interface.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.config.settings import Settings, get_settings


logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """Supported model families."""
    CLAUDE = "claude"
    TITAN = "titan"
    MISTRAL = "mistral"


@dataclass
class ModelResponse:
    """
    Normalized response from any Bedrock model.
    
    Attributes:
        content: The generated text response
        model_id: The model that generated this response
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        latency_ms: Response latency in milliseconds
        raw_response: Original API response for debugging
    """
    content: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str = ""
    raw_response: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "model_id": self.model_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
            "timestamp": self.timestamp.isoformat(),
        }


class BedrockClient:
    """
    AWS Bedrock client supporting multiple foundation models.
    
    Provides a unified interface to invoke different models while handling
    their specific input/output formats.
    
    Usage:
        client = BedrockClient()
        response = await client.generate("Analyze this market...")
        print(response.content)
    
    Supported Models:
        - Claude 3.5 Sonnet: Best for complex reasoning
        - Claude 3 Haiku: Fast and cost-effective
        - Amazon Titan Text: AWS native model
        - Mistral Large: European model with strong reasoning
    """
    
    # Model ID mappings for different naming conventions
    # Note: Claude models use cross-region inference profiles (us. prefix)
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
    MODEL_IDS = {
        "claude-3-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3.5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "titan-text": "amazon.titan-text-express-v1",
        "titan-text-express": "amazon.titan-text-express-v1",
        "mistral-large": "mistral.mistral-large-2402-v1:0",
    }
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the Bedrock client.
        
        Args:
            settings: Optional settings object; uses default if not provided
        """
        self.settings = settings or get_settings()
        self._client = None
        self._initialized = False
        
        # Default model from settings
        self.default_model_id = self.settings.get_bedrock_model_id()
        
        # Track usage for cost monitoring
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._request_count = 0
    
    def _get_client(self):
        """
        Lazily initialize the boto3 Bedrock runtime client.
        
        Uses credentials from environment variables or settings.
        """
        if self._client is None:
            try:
                # Configure AWS credentials
                aws_config = {
                    "region_name": self.settings.env.aws_region or self.settings.ai.region,
                }
                
                # Add explicit credentials if provided
                if self.settings.env.aws_access_key_id:
                    aws_config["aws_access_key_id"] = self.settings.env.aws_access_key_id
                if self.settings.env.aws_secret_access_key:
                    aws_config["aws_secret_access_key"] = self.settings.env.aws_secret_access_key
                
                self._client = boto3.client("bedrock-runtime", **aws_config)
                self._initialized = True
                logger.info(f"Bedrock client initialized (region: {aws_config['region_name']})")
                
            except NoCredentialsError:
                logger.error("AWS credentials not found. Configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        
        return self._client
    
    def _detect_model_family(self, model_id: str) -> ModelFamily:
        """Detect which model family a model ID belongs to."""
        model_id_lower = model_id.lower()
        
        if "claude" in model_id_lower or "anthropic" in model_id_lower:
            return ModelFamily.CLAUDE
        elif "titan" in model_id_lower or "amazon" in model_id_lower:
            return ModelFamily.TITAN
        elif "mistral" in model_id_lower:
            return ModelFamily.MISTRAL
        else:
            # Default to Claude format
            logger.warning(f"Unknown model family for {model_id}, defaulting to Claude format")
            return ModelFamily.CLAUDE
    
    def _resolve_model_id(self, model_id: Optional[str]) -> str:
        """Resolve a model ID from alias or use default."""
        if model_id is None:
            return self.default_model_id
        
        # Check if it's an alias
        if model_id in self.MODEL_IDS:
            return self.MODEL_IDS[model_id]
        
        # Assume it's a full model ID
        return model_id
    
    def _build_claude_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Dict:
        """Build request body for Claude models."""
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        return body
    
    def _build_titan_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Dict:
        """Build request body for Amazon Titan models."""
        # Titan uses a different format
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\n\nAssistant:"
        
        return {
            "inputText": full_prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
            }
        }
    
    def _build_mistral_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Dict:
        """Build request body for Mistral models."""
        # Mistral uses a chat-like format
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        return {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    
    def _parse_claude_response(self, response_body: Dict) -> tuple:
        """Parse Claude response."""
        content = ""
        if "content" in response_body and response_body["content"]:
            content = response_body["content"][0].get("text", "")
        
        usage = response_body.get("usage", {})
        return (
            content,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            response_body.get("stop_reason", ""),
        )
    
    def _parse_titan_response(self, response_body: Dict) -> tuple:
        """Parse Titan response."""
        results = response_body.get("results", [{}])
        content = results[0].get("outputText", "") if results else ""
        
        # Titan doesn't provide detailed token counts
        input_tokens = response_body.get("inputTextTokenCount", 0)
        output_tokens = len(content.split()) * 1.3  # Rough estimate
        
        return (
            content.strip(),
            int(input_tokens),
            int(output_tokens),
            results[0].get("completionReason", "") if results else "",
        )
    
    def _parse_mistral_response(self, response_body: Dict) -> tuple:
        """Parse Mistral response."""
        outputs = response_body.get("outputs", [{}])
        content = outputs[0].get("text", "") if outputs else ""
        
        # Mistral token usage
        return (
            content.strip(),
            response_body.get("prompt_token_count", 0),
            response_body.get("generation_token_count", 0),
            outputs[0].get("stop_reason", "") if outputs else "",
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """
        Generate text using a Bedrock foundation model.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system instructions
            model_id: Model ID or alias (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Returns:
            ModelResponse with generated content and metadata
        """
        import time
        start_time = time.time()
        
        # Resolve model ID and get family
        resolved_model_id = self._resolve_model_id(model_id)
        model_family = self._detect_model_family(resolved_model_id)
        
        # Apply defaults from settings
        max_tokens = max_tokens or self.settings.ai.max_tokens
        temperature = temperature or self.settings.ai.temperature
        
        # Build request based on model family
        if model_family == ModelFamily.CLAUDE:
            request_body = self._build_claude_request(
                prompt, system_prompt, max_tokens, temperature
            )
        elif model_family == ModelFamily.TITAN:
            request_body = self._build_titan_request(
                prompt, system_prompt, max_tokens, temperature
            )
        elif model_family == ModelFamily.MISTRAL:
            request_body = self._build_mistral_request(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
        
        try:
            client = self._get_client()
            
            logger.debug(f"Invoking Bedrock model: {resolved_model_id}")
            
            response = client.invoke_model(
                modelId=resolved_model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            
            response_body = json.loads(response["body"].read())
            
            # Parse response based on model family
            if model_family == ModelFamily.CLAUDE:
                content, input_tokens, output_tokens, stop_reason = self._parse_claude_response(response_body)
            elif model_family == ModelFamily.TITAN:
                content, input_tokens, output_tokens, stop_reason = self._parse_titan_response(response_body)
            elif model_family == ModelFamily.MISTRAL:
                content, input_tokens, output_tokens, stop_reason = self._parse_mistral_response(response_body)
            
            # Track usage
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._request_count += 1
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Bedrock response: {input_tokens} in / {output_tokens} out tokens, "
                f"{latency_ms:.0f}ms latency"
            )
            
            return ModelResponse(
                content=content,
                model_id=resolved_model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                stop_reason=stop_reason,
                raw_response=response_body,
            )
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"Bedrock API error ({error_code}): {error_message}")
            raise
        except Exception as e:
            logger.error(f"Bedrock request failed: {e}")
            raise
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """
        Async version of generate using asyncio.
        
        Wraps the synchronous boto3 call in a thread pool executor
        for non-blocking operation.
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    
    def get_usage_stats(self) -> Dict:
        """Get cumulative usage statistics."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "request_count": self._request_count,
            "avg_tokens_per_request": (
                (self._total_input_tokens + self._total_output_tokens) / self._request_count
                if self._request_count > 0 else 0
            ),
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._request_count = 0
    
    @property
    def is_available(self) -> bool:
        """Check if Bedrock client can be initialized."""
        try:
            self._get_client()
            return True
        except Exception:
            return False


# Convenience function for quick testing
def test_bedrock_connection() -> bool:
    """
    Test if Bedrock connection works.
    
    Returns:
        True if connection is successful
    """
    try:
        client = BedrockClient()
        response = client.generate(
            prompt="Say 'Hello, I am working!' and nothing else.",
            max_tokens=50,
        )
        return "working" in response.content.lower() or "hello" in response.content.lower()
    except Exception as e:
        logger.error(f"Bedrock connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the Bedrock client
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Bedrock Client...")
    print("=" * 60)
    
    settings = get_settings()
    print(f"Model: {settings.get_bedrock_model_id()}")
    print(f"Region: {settings.ai.region}")
    print(f"AWS Credentials: {'✅ Found' if settings.has_aws_credentials() else '❌ Missing'}")
    
    if settings.has_aws_credentials():
        print("\nTesting connection...")
        if test_bedrock_connection():
            print("✅ Bedrock connection successful!")
        else:
            print("❌ Bedrock connection failed")
    else:
        print("\n⚠️ Configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to test")