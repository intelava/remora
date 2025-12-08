import torch
import gymnasium as gym
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def get_moondream_model():
    """Loads the Moondream2 model and tokenizer from Hugging Face."""
    # Using a revision to ensure reproducibility, as the main branch can change.
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision,
        torch_dtype=torch.float16, device_map={"": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

def map_action_to_pong(action_text: str) -> int:
    """Converts the VLM's text output to a valid Pong action."""
    action_text = action_text.upper()
    if "UP" in action_text:
        return 2  # UP
    elif "DOWN" in action_text:
        return 3  # DOWN
    return 0  # NOOP

def run_pong_with_vlm():
    """
    Runs a live demo in the Pong environment, controlled by the Moondream2 VLM.
    """
    print("--- VLM-driven Gameplay: Moondream2 plays Pong ---")
    
    # --- Configuration ---
    TARGET_HZ = 30
    TARGET_LATENCY_MS = 1000 / TARGET_HZ
    # This prompt is crucial for guiding the VLM's decision-making.
    PROMPT = "You are an expert Atari player. The image is a frame from the game Pong. Your paddle is on the right. Should you move your paddle UP or DOWN to intercept the ball? Answer with only the word UP or DOWN."

    # --- Initialization ---
    try:
        print("Loading Moondream2 model...")
        model, tokenizer = get_moondream_model()
        
        print("Initializing Gymnasium environment...")
        env = gym.make("PongNoFrameskip-v4", render_mode='human')
        observation, info = env.reset(seed=42)
        
    except ImportError as e:
        print(f"\n--- DEPENDENCY ERROR ---\nError: {e}\nPlease install dependencies: pip install -r requirements.txt")
        return
    except Exception as e:
        print(f"\n--- FATAL ERROR ---\nFailed to initialize model or environment: {e}\nThis demo requires a CUDA GPU and all dependencies from requirements.txt.")
        return

    print(f"\nStarting game loop (target < {TARGET_LATENCY_MS:.2f}ms)... Press Ctrl+C to stop.")
    
    latencies = []
    running = True
    frame_count = 0
    
    try:
        while running:
            frame_count += 1
            
            # --- VLM Inference Step ---
            start_time = time.perf_counter()
            
            # Convert frame to PIL Image for the model's processor
            pil_image = Image.fromarray(observation)
            
            # Get the model's answer
            enc_image = model.encode_image(pil_image)
            model_answer = model.answer_question(
                enc_image,
                PROMPT,
                tokenizer,
                chat_history="" # Ensure each frame is an independent decision
            )
            
            torch.cuda.synchronize() # Ensure accurate timing
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
            # --- Environment Step ---
            pong_action = map_action_to_pong(model_answer)
            observation, reward, terminated, truncated, info = env.step(pong_action)
            
            # --- Display results ---
            status = "PASS" if latency_ms <= TARGET_LATENCY_MS else "FAIL"
            print(f"Frame {frame_count:04d}: Action='{model_answer.strip()}', Latency={latency_ms:6.2f}ms [{status}]")

            if terminated or truncated:
                print("Game over. Resetting environment.")
                observation, info = env.reset()

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during the game loop: {e}")
        
finally:
        env.close()
        if latencies:
            latencies_arr = np.array(latencies[1:]) # Exclude first run from stats (warm-up)
            if latencies_arr.size > 0:
                avg_latency = np.mean(latencies_arr)
                p95_latency = np.percentile(latencies_arr, 95)
                pass_rate = np.sum(latencies_arr <= TARGET_LATENCY_MS) / len(latencies_arr) * 100

                print("\n--- Performance Summary ---")
                print(f"Target Latency: {TARGET_LATENCY_MS:.2f}ms (~{TARGET_HZ}Hz)")
                print(f"Average Latency (post-warmup): {avg_latency:.2f}ms")
                print(f"P95 Latency:   {p95_latency:.2f}ms")
                print(f"Pass Rate (<= {TARGET_LATENCY_MS:.2f}ms): {pass_rate:.2f}%")

if __name__ == "__main__":
    run_pong_with_vlm()
