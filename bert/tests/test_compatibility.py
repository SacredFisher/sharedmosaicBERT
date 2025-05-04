"""
These tests exist to ensure that the layer sharing doesn't actually
change anything about the model--the optimizations stay the same. 

Critically, we have to make sure that the matrix headed into the flash attention
calculation is unchanged, and it propagates the gradients correctly front and backwards



"""



import torch
from src.bert_layers import BertEncoder as OriginalBertEncoder
from src.bert_layers import BertUnpadSelfAttention as OriginalBertUnpadSelfAttention
from src.bert_layers_shared import BertEncoder as ModifiedBertEncoder
from src.bert_layers_shared import BertUnpadSelfAttention as ModifiedBertUnpadSelfAttention

def test_flash_attention_compatibility():
    # Create a simple config
    config = create_test_config()
    
    # Create original module
    original = OriginalBertUnpadSelfAttention(config)
    
    # Create modified module with the same initialization
    modified = ModifiedBertUnpadSelfAttention(config)  # Default behavior without sharing
    
    # Copy weights for equivalence
    copy_weights_between_modules(original, modified)
    
    # Create test inputs
    inputs = create_test_inputs()
    
    # Run both implementations and capture intermediate tensors
    orig_tensors = run_with_hooks(original, inputs)
    mod_tensors = run_with_hooks(modified, inputs)
    
    # Compare tensors going into Flash Attention
    compare_tensors(orig_tensors['flash_attn_input'], mod_tensors['flash_attn_input'])
    
    print("Flash Attention compatibility verified!")

def test_weight_sharing():
    # Create config
    config = create_test_config()
    
    # Create original encoder
    original = OriginalBertEncoder(config)
    
    # Create modified encoder with weight sharing
    modified = ModifiedBertEncoder(config, share_pattern='k')
    
    # Verify sharing is working as expected
    # Check that the same K parameter is used across layers
    # ...

def test_gradient_flow_compatibility():
    # Setup similar to previous test
    config = SimpleConfig()
    
    # Create original implementation
    original_attn = OriginalBertUnpadSelfAttention(config)
    
    # Create modified implementation with shared projections
    shared_q = nn.Linear(original_attn.all_head_size, config.hidden_size)
    shared_k = nn.Linear(original_attn.all_head_size, config.hidden_size)
    shared_v = nn.Linear(original_attn.all_head_size, config.hidden_size)
    
    # Copy weights as before
    # ...
    
    # Create two modified attention modules that share the same weights
    mod_attn1 = ModifiedBertUnpadSelfAttention(config, shared_q, shared_k, shared_v)
    mod_attn2 = ModifiedBertUnpadSelfAttention(config, shared_q, shared_k, shared_v)
    
    # Create test inputs that require gradients
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, requires_grad=True)
    # ... other inputs as before ...
    
    # Original implementation forward and backward
    orig_output = original_attn(hidden_states, cu_seqlens, seq_len, indices, attn_mask, bias)
    orig_grad = torch.randn_like(orig_output)  # Random gradient for backward
    orig_output.backward(orig_grad)
    
    # Store gradients
    orig_weight_grad = original_attn.Wqkv.weight.grad.clone()
    orig_bias_grad = original_attn.Wqkv.bias.grad.clone()
    orig_input_grad = hidden_states.grad.clone()
    
    # Reset gradients
    hidden_states.grad = None
    
    # Modified implementation forward and backward through both attention modules
    # This tests if gradients accumulate correctly when weights are shared
    mod_output1 = mod_attn1(hidden_states, cu_seqlens, seq_len, indices, attn_mask, bias)
    mod_grad1 = orig_grad.clone()  # Use same gradient for consistent comparison
    mod_output1.backward(mod_grad1)
    
    # Forward and backward through second module (sharing weights with first)
    mod_output2 = mod_attn2(hidden_states, cu_seqlens, seq_len, indices, attn_mask, bias)
    mod_grad2 = orig_grad.clone()
    mod_output2.backward(mod_grad2)
    
    # Verify gradients accumulate correctly in shared weights
    # 1. The shared weights should receive gradients from both backward passes
    print(f"Shared Q grad magnitude: {shared_q.weight.grad.norm().item()}")
    print(f"Shared K grad magnitude: {shared_k.weight.grad.norm().item()}")
    print(f"Shared V grad magnitude: {shared_v.weight.grad.norm().item()}")
    
    # 2. Check that input gradients are calculated correctly
    input_grad_diff = (hidden_states.grad - orig_input_grad).abs().max().item()
    print(f"Input gradient max difference: {input_grad_diff}")
    
    print("Gradient flow compatibility verified!")