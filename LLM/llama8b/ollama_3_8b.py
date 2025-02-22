import ollama
import re


def extract_reward(self, response_text):
    """
    Extracts the first numerical reward from LLM response.
    """
    match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)  # Finds first float or integer
    if match:
        return float(match.group())  # Convert to float
    else:
        print(f"[ERROR] Could not extract number from: {response_text}")
        return 0.0  # Default fallback reward


def query_llm_for_reward(self, state, action):
    """
    Queries the locally running Llama model via Ollama to generate a reward.
    """
    prompt = f"""
    You are a reinforcement learning environment. Given the following traffic state and action, return ONLY a numerical reward between -1 and 1.

    State: {state}
    Action: {action}

    Provide ONLY a single float number as output, without explanation, formatting, or additional text.
    """

    try:
        #if model LLAMA3-8B, model = llama3:8b
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        reward_text = response.get("message", {}).get("content", "0.0")  # Get response

        # Extract numerical reward
        reward = self.extract_reward(reward_text)

        print(f"\n[LLM QUERY] - State: {state}, Action: {action}")
        print(f"[LLM RESPONSE] - Extracted Reward: {reward}\n")

        return reward

    except Exception as e:
        print(f"[ERROR] LLM query failed: {e}")
        return 0.0  # Fallback reward