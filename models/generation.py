import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_length: int = 2048
    min_length: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.0
    do_sample: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    length_penalty: float = 1.0
    
class GenerationMixin:
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict:
        # Only last token for input_ids if past is not None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
        
    def _reorder_cache(self, past_key_values: Tuple, beam_idx: torch.LongTensor) -> Tuple:
        """Reorder cached past key values for beam search."""
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> torch.LongTensor:
        if generation_config is None:
            generation_config = GenerationConfig()
            
        # Set generation parameters
        max_length = generation_config.max_length
        num_beams = generation_config.num_beams
        do_sample = generation_config.do_sample
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        attention_mask = torch.ones_like(input_ids)
        
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids,
                attention_mask,
                generation_config
            )
        else:
            return self._generate_greedy_or_sampling(
                input_ids,
                attention_mask,
                generation_config
            )
            
    def _generate_beam_search(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        generation_config: GenerationConfig
    ) -> torch.LongTensor:
        batch_size = input_ids.shape[0]
        num_beams = generation_config.num_beams
        max_length = generation_config.max_length
        length_penalty = generation_config.length_penalty
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1).contiguous()
        attention_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1).contiguous()
        
        # Reshape for processing
        input_ids = input_ids.view(batch_size * num_beams, -1)
        attention_mask = attention_mask.view(batch_size * num_beams, -1)
        
        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Initialize all but first beam with -inf
        beam_scores = beam_scores.view(-1)  # Shape: (batch_size * num_beams,)
        
        # Cache for storing generated sequences
        generated_sequences = []
        past_key_values = None
        
        while len(generated_sequences) < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask
            )
            
            # Get model output - handle MoE specific output format
            outputs = self(**model_inputs)
            logits = outputs["logits"]
            
            # Get next token logits for last position
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature and repetition penalty
            next_token_logits = next_token_logits / generation_config.temperature
            
            # Calculate log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Get top-k tokens and scores
            vocab_size = next_token_scores.shape[-1]
            top_k = min(generation_config.top_k, vocab_size)
            top_scores, top_tokens = torch.topk(next_token_scores, top_k)
            
            # Expand beam paths
            beam_tokens = top_tokens.view(batch_size, num_beams * top_k)
            beam_scores_expanded = (beam_scores[:, None] + top_scores).view(
                batch_size, num_beams * top_k
            )
            
            # Get top-k beams
            topk_scores, topk_indices = torch.topk(
                beam_scores_expanded, num_beams, dim=1
            )
            beam_indices = topk_indices // top_k
            token_indices = topk_indices % top_k
            
            # Update sequences
            next_beam_tokens = torch.gather(
                beam_tokens, 1, token_indices
            )
            
            # Update beam scores
            beam_scores = topk_scores.view(-1)
            
            # Reorder sequences and past key values
            beam_indices = beam_indices + torch.arange(
                batch_size, device=beam_indices.device
            )[:, None] * num_beams
            input_ids = torch.cat([input_ids[beam_indices], next_beam_tokens.view(-1, 1)], dim=-1)
            
            if past_key_values is not None:
                past_key_values = self._reorder_cache(past_key_values, beam_indices)
                
            # Check for EOS
            if generation_config.eos_token_id in next_beam_tokens:
                break
                
        return input_ids.view(batch_size, num_beams, -1)[:, 0, :]
        
    def _generate_greedy_or_sampling(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        generation_config: GenerationConfig
    ) -> torch.LongTensor:
        max_length = generation_config.max_length
        do_sample = generation_config.do_sample
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        
        past_key_values = None
        
        while input_ids.shape[-1] < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask
            )
            
            # Get model output - handle MoE specific output format
            outputs = self(**model_inputs)
            logits = outputs["logits"]
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            if do_sample:
                # Temperature sampling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
                    
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Append next tokens
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1))
            ], dim=-1)
                
            # Check for EOS
            if generation_config.eos_token_id in next_tokens:
                break
                
        return input_ids
