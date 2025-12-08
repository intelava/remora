import torch
import gymnasium as gym
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# --- Import Remora Optimizer ---
# This will allow us to patch the model for acceleration
from remora.optimizer import optimize_model

def get_moondream_model():
    """Loads the Moondream2 model and tokenizer from Hugging Face."""
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    
    print(f"Loading base model '{model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision,
        torch_dtype=torch.float16, device_map={" ": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    print("Base model loaded.")
    return model, tokenizer

def map_action_to_pong(action_text: str) -> int:
    """Converts the VLM's text output to a valid Pong action."""
    action_text = action_text.upper()
    if "UP" in action_text:
        return 2
    elif "DOWN" in action_text:
        return 3
    return 0

def run_pong_with_vlm():
    """
    Runs a live demo in the Pong environment, controlled by a VLM
    that has been accelerated by the Remora engine.
    """
    print("--- Remora-Accelerated VLM Gameplay: Moondream2 + Remora plays Pong ---")
    
    # --- Configuration ---
    TARGET_LATENCY_MS = 1000 / 30  # ~33.3ms for 30Hz
    PROMPT = "You are an expert Atari player. The image is a frame from the game Pong. Your paddle is on the right. Should you move your paddle UP or DOWN to intercept the ball? Answer with only the word UP or DOWN."

    # --- Initialization ---
    try:
        model, tokenizer = get_moondream_model()
        
        # --- Apply Remora Optimization ---
        print("\n--- Applying Remora Optimization ---")
        print("Model structure before optimization:")
        print(model)
        
        # Patch the nn.Linear layers with Remora's W8A16Linear layers
        # We skip the final lm_head to maintain full precision for logits, a common practice.
        optimize_model(model, layers_to_skip=['lm_head'])
        
        print("\nModel structure after optimization:")
        print(model)
        print("--- Remora Optimization Complete ---\n")
        
        env = gym.make("PongNoFrameskip-v4", render_mode='human')
        observation, info = env.reset(seed=42)
        
    except ImportError as e:
        print(f"\n--- DEPENDENCY ERROR ---\nError: {e}\nPlease install dependencies: pip install -r requirements.txt")
        return
    except Exception as e:
        print(f"\n--- FATAL ERROR ---\n{e}\nThis demo requires a Linux environment with a CUDA GPU and all dependencies from requirements.txt.")
        return

    print(f"Starting game loop (target < {TARGET_LATENCY_MS:.2f}ms)... Press Ctrl+C to stop.")
    latencies = []
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            start_time = time.perf_counter()
            
            pil_image = Image.fromarray(observation)
            enc_image = model.encode_image(pil_image)
            model_answer = model.answer_question(enc_image, PROMPT, tokenizer, chat_history="")
            
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start_time) * 1000
            if frame_count > 1: latencies.append(latency_ms) # Exclude first run
            
            pong_action = map_action_to_pong(model_answer)
            observation, _, terminated, truncated, _ = env.step(pong_action)
            
            status = "PASS" if latency_ms <= TARGET_LATENCY_MS else "FAIL"
            print(f"Frame {frame_count:04d}: Action='{model_answer.strip()}', Latency={latency_ms:6.2f}ms [{status}]")

            if terminated or truncated:
                observation, info = env.reset()

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during the game loop: {e}")
        
    finally:
        env.close()
        if latencies:
            latencies_arr = np.array(latencies)
            avg_latency = np.mean(latencies_arr)
            p95_latency = np.percentile(latencies_arr, 95)
            
            print("\n--- Performance Summary (Remora-Accelerated) ---")
            print(f"Average Latency (post-warmup): {avg_latency:.2f}ms")
            print(f"P95 Latency:   {p95_latency:.2f}ms")
            
            if avg_latency <= TARGET_LATENCY_MS:
                print("\nConclusion: The Remora-accelerated model met the real-time performance target!")
            else:
                print("\nConclusion: The model did not meet the performance target.")

if __name__ == "__main__":
    run_pong_with_vlm()