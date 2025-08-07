from openai import OpenAI
import os
import dotenv
import logging
import re

# Load environment variables and define pricing constants
dotenv.load_dotenv()

# GPT model pricing per 1K tokens (in USD)
MODEL_PRICES = {
    "gpt-4.1": {"input": 0.002, "output": 0.008},  
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}, 
    "gemma3n:latest": {"input": 0.0, "output": 0.0}
}

class PricingTracker:
    """Tracks accumulated costs of LLM API usage."""
    def __init__(self):
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def add_usage(self, model, input_tokens, output_tokens):
        """Add cost for a single API call."""
        if model not in MODEL_PRICES:
            logging.warning(f"Unknown model {model}, cost tracking skipped")
            return
            
        input_cost = (input_tokens / 1000) * MODEL_PRICES[model]["input"]
        output_cost = (output_tokens / 1000) * MODEL_PRICES[model]["output"]
        
        self.total_cost += input_cost + output_cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
    def get_stats(self):
        """Get current usage statistics."""
        return {
            "total_cost_usd": self.total_cost,
            "total_input_tokens": f"{self.total_input_tokens}",
            "total_output_tokens": f"{self.total_output_tokens}"
        }

class LLMChat:
    """
    Handles chat interactions with OpenAI's LLM models.
    """
    pricing = PricingTracker()  # Class-level price tracking
    
    def __init__(self, system_prompt=None):
        """Initialize OpenAI client."""
        self.model = os.getenv("OPENAI_MODEL")
        self.base_url = os.getenv("OPENAI_BASE_URL") or None
        
        client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = OpenAI(**client_kwargs)
        self.conversation_history = []
        self.system_prompt = system_prompt
    
    def _clean_response(self, response_text):
        """
        Remove <think>...</think> tags from LLM response.
        
        Parameters:
        -----------
        response_text : str
            The raw response from the LLM
            
        Returns:
        --------
        str
            Cleaned response with thinking tags removed
        """
        return re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        
    def _create_chat_completion(self, messages, temperature=0.7):
        """
        Helper method to create chat completion using the OpenAI API.
        
        Parameters:
        -----------
        messages : list
            List of message dictionaries to send to the API
        temperature : float
            Controls randomness in the response (default: 0.7)
            
        Returns:
        --------
        str
            The LLM's response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
            
        response = self.client.chat.completions.create(**kwargs)
        
        # Track token usage
        LLMChat.pricing.add_usage(
            self.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        # Clean the response to remove thinking tags
        raw_content = response.choices[0].message.content
        return self._clean_response(raw_content)

    def _prepare_messages(self, message):
        """
        Helper method to prepare message list with system prompt if present.
        
        Parameters:
        -----------
        message : str
            The user message to include
            
        Returns:
        --------
        list
            List of message dictionaries
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": message})
        return messages

    def one_shot_chat(self, message, temperature=0.7):
        """
        Send a single message to the LLM and get a response without conversation history.
        
        Parameters:
        -----------
        message : str
            The message to send to the LLM
        temperature : float
            Controls randomness in the response (default: 0.7)
            
        Returns:
        --------
        str
            The LLM's response text
        """
        messages = self._prepare_messages(message)
        return self._create_chat_completion(messages, temperature)
        
    def send_message(self, message, temperature=0.7):
        """
        Send a message to the LLM and get a response with conversation history.
        
        Parameters:
        -----------
        message : str
            The message to send to the LLM
        temperature : float
            Controls randomness in the response (default: 0.7)
            
        Returns:
        --------
        str
            The LLM's response text
        """
        messages = self._prepare_messages(message)
        self.conversation_history.extend(messages)
        
        assistant_message = self._create_chat_completion(self.conversation_history, temperature)
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
