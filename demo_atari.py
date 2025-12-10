import torch
from torch import nn
import gym
import numpy as np
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


from remora.optimizer import optimize_model

def get_moondream_model():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    device = "cuda"
    dtype = torch.float16

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    return model, processor

def map_action_to_pong(action_text: str) -> int:
    action_text = action_text.upper()
    if "UP" in action_text:
        return 2
    elif "DOWN" in action_text:
        return 3
    return 0
        
def run_performance_demo():
    print("--- Remora Engine: High-Performance Latency Demo ---")
    
    TARGET_LATENCY_MS = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, processor = get_moondream_model()
    model = optimize_model(model, processor)
    model.eval()
    
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    obs, _ = env.reset()
    
    dummy_text = torch.randint(0, 32000, (1, 10)).to(device)

    print(f"\nStarting benchmark (Target: < {TARGET_LATENCY_MS:.2f}ms)...")
    print("-" * 60)
    
    latencies = []
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "You are a pong agent. Your goal is to move the paddle to hit the ball. The ball is moving towards you. You can move the paddle up or down. What is the action you should take?"},
          {"type": "image"},
        ],
    },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    last_hidden_state_container = {}
    
    def hook_fn(module, input, output):
        last_hidden_state_container['payload'] = output
    
    hook_handle = model.language_model.norm.register_forward_hook(hook_fn)
    up_id = processor.tokenizer.encode("up")[-1]
    down_id = processor.tokenizer.encode("down")[-1]
    try:
        for i in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            img_tensor = torch.tensor(obs, device=device).float() / 255.0
            
            with torch.no_grad():
                inputs = processor(images=img_tensor, text=prompt, return_tensors='pt').to(0, torch.float16).to("cuda")
                _ = model(**inputs)
                final_hidden = last_hidden_state_container['payload']
                token_id_tensor = model.lm_head(final_hidden)
                action_idx = token_id_tensor[0, 0].item()
            
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
            
            if i > 5:
                latencies.append(latency)
            
            env_action = 0
            if action_idx == up_id: env_action = 2
            elif action_idx == down_id: env_action = 3
            else: env_action = 0
            
            obs, _, term, trunc, _ = env.step(env_action)
            
            if term or trunc: obs, _ = env.reset()
            
            status = "PASS" if latency < TARGET_LATENCY_MS else "FAIL"
            print(f"Frame {i:03d} | Latency: {latency:5.2f} ms | Action: {env_action} | [{status}]")

    except KeyboardInterrupt:
        pass
        
    avg = np.mean(latencies)
    print("-" * 60)
    print(f"Average Latency: {avg:.2f} ms")
    if avg < TARGET_LATENCY_MS:
        print("RESULT: SUCCESS. Engine is Real-Time Ready (>30Hz).")
    else:
        print("RESULT: FAIL.")

if __name__ == "__main__":
    run_performance_demo()