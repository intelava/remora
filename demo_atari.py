import torch
import gym
import numpy as np
import time

try:
    from remora.engine import RemoraEngine
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from remora.engine import RemoraEngine

def map_action_to_pong(action_text: str) -> int:
    """Converts the VLM's text output to a valid Pong action."""
    # Actions for "PongNoFrameskip-v4": 0:NOOP, 1:FIRE, 2:UP, 3:DOWN, 4:UP-FIRE, 5:DOWN-FIRE
    if action_text == "UP":
        return 2
    elif action_text == "DOWN":
        return 3
    return 0 # Default to NOOP

def measure_latency(engine, frame, prompt):
    """Measures the end-to-end latency of a single inference run using CUDA events."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    action_text = engine.generate(frame, prompt=prompt)
    end_event.record()
    
    torch.cuda.synchronize()
    
    latency_ms = start_event.elapsed_time(end_event)
    return action_text, latency_ms

def run_pong_demo():
    """
    Runs a live demo in the "PongNoFrameskip-v4" environment, controlled
    by the RemoraEngine, while measuring inference latency.
    """
    print("--- Remora Engine: Live Gymnasium Pong Demo ---")
    
    # --- Configuration ---
    TARGET_HZ = 30
    TARGET_LATENCY_MS = 1000 / TARGET_HZ
    PROMPT = "Track the ball and move the paddle."
    
    # --- Initialization ---
    try:
        # Initialize the high-performance RemoraEngine
        # The model is configured to output 2 actions ("UP", "DOWN")
        engine = RemoraEngine(image_size=(84, 84), hidden_dim=2048, num_layers=4, num_actions=2)
        
        # Initialize the Gymnasium environment
        env = gym.make("PongDeterministic-v4", render_mode='human')
        observation, info = env.reset(seed=42)
        
    except ImportError as e:
        print("\n--- DEPENDENCY ERROR ---")
        print(f"Error: {e}")
        print("Please install the required dependencies with: pip install -r requirements.txt")
        return
    except Exception as e:
        print("\n--- FATAL ERROR ---")
        print(f"Failed to initialize engine or environment: {e}")
        print("This demo requires a CUDA-enabled GPU with Triton and Gymnasium installed.")
        return

    print(f"\nStarting game loop (target < {TARGET_LATENCY_MS:.2f}ms latency)...")
    print("Press Ctrl+C to stop.")
    
    latencies = []
    running = True
    frame_count = 0
    
    try:
        while running:
            frame_count += 1
            
            # --- VLM Inference Step ---
            action_text, latency_ms = measure_latency(engine, observation, PROMPT)
            latencies.append(latency_ms)
            
            # --- Environment Step ---
            pong_action = map_action_to_pong(action_text)
            observation, reward, terminated, truncated, info = env.step(pong_action)
            
            # --- Display results ---
            status = "PASS" if latency_ms <= TARGET_LATENCY_MS else "FAIL"
            print(f"Frame {frame_count:04d}: Action='{action_text}', Latency={latency_ms:6.2f}ms [{status}]")

            if terminated or truncated:
                print("Game over. Resetting environment.")
                observation, info = env.reset()

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")
    except Exception as e:
        # This handles errors like the user closing the game window
        print(f"\nAn error occurred during the game loop: {e}")
        
    finally:
        env.close()
        if latencies:
            latencies_arr = np.array(latencies)
            avg_latency = np.mean(latencies_arr)
            p95_latency = np.percentile(latencies_arr, 95)
            pass_rate = np.sum(latencies_arr <= TARGET_LATENCY_MS) / len(latencies_arr) * 100

            print("\n--- Performance Summary ---")
            print(f"Target Latency: {TARGET_LATENCY_MS:.2f}ms (~{TARGET_HZ}Hz)")
            print(f"Average Latency: {avg_latency:.2f}ms")
            print(f"P95 Latency:   {p95_latency:.2f}ms")
            print(f"Pass Rate (<= {TARGET_LATENCY_MS:.2f}ms): {pass_rate:.2f}%")

            if avg_latency <= TARGET_LATENCY_MS:
                print("\nConclusion: The RemoraEngine met the real-time performance target. SUCCESS!")
            else:
                print("\nConclusion: The RemoraEngine did NOT meet the real-time performance target. FAILED.")

if __name__ == "__main__":
    run_pong_demo()