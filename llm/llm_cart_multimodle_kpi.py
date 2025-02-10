import os
import openai
import pandas as pd
from dotenv import load_dotenv
import time

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

gemini_client = openai.OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Configuration dictionary for paths and parameters
CONFIG = {
    "base_dir": "/home/yuan/mybookname/Openai/Safety/output_validation",
    "timestep_interval": 1,
    "model_params": {
        "openai": {
            "model": "gpt-4o",
            "temperature": 1,
            "max_tokens": 16383,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        "gemini": {
            "model": "gemini-1.5-pro",
            "temperature": 1,
            "max_tokens": 32768,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    }
}

def set_params(engine="openai", **kwargs):
    """Set model parameters with defaults overridden by provided kwargs."""
    params = CONFIG["model_params"].get(engine, {}).copy()
    params.update(kwargs)
    return params

def get_completion(engine, params, messages, max_retries=3, retry_delay=10):
    """Attempts to get a response from the model, with retries for transient errors."""
    retries = 0
    while retries < max_retries:
        try:
            start_time = time.time()

            if engine == "openai":
                response = openai.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    top_p=params["top_p"],
                    frequency_penalty=params["frequency_penalty"],
                    presence_penalty=params["presence_penalty"],
                )
                elapsed_time = time.time() - start_time
                token_usage = response.usage.total_tokens if "usage" in response else None

            elif engine == "gemini":
                response = gemini_client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    n=1
                )
                elapsed_time = time.time() - start_time
                token_usage = None  # Gemini API might not provide token usage

            print(f"API Response Time ({engine}): {elapsed_time:.2f} seconds")
            if token_usage:
                print(f"Tokens Used ({engine}): {token_usage}")
            
            return response.choices[0].message.content, elapsed_time, token_usage

        except Exception as e:
            error_message = str(e)
            if "overloaded" in error_message.lower() or "500" in error_message:
                print(f"{engine.capitalize()} is overloaded. Retrying {retries+1}/{max_retries} in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                print(f"Error occurred while processing {engine} completion: {e}")
                return None, None, None

    print(f"Failed to get a response from {engine} after {max_retries} retries.")
    return None, None, None

def generate_safety_analysis_MDC(context):
    """Generate analysis for Minimum Distance to Collision (MDC)."""
    # Escape curly braces in the context
    escaped_context = context.replace("{", "{{").replace("}", "}}")  # Double the braces

    return [
                {
                    "role": "system",
                    "content": """You are an expert in collision analysis for autonomous driving scenarios.
                                    Your role is to evaluate the provided scenario based on the following safety metrics with scores ranging from 0 to 5 for each metric, where 0 indicates collision and 5 indicates no risk of collision:

                                    **Risk Levels and Definitions:**
                                    <Extreme Risk (Score: 1)>: Immediate collision or very high likelihood of impact. Urgent action is required.
                                    <High Risk (Score: 2>: Close to collision or highly probable collision path. Needs prompt attention and quick manoeuvring.
                                    <Medium Risk (Score: 3)>: Moderate collision risk but manageable with timely reactions and proper strategies.
                                    <Low Risk (Score: 4)>: Minimal collision risk. The situation is controllable with sufficient time to react.
                                    <Negligible Risk (Score: 5)>: No significant collision risk. Obstacles are either moving away or far enough not to interfere.

                                    **Metrics Considered:**

                                                - **Time to Collision (TTC)**: Estimated time until a potential collision.
                                                - **Minimum Distance to Collision (MDC)**: Smallest distance between the ego vehicle and obstacles before a collision occurs

                                                Provide a detailed evaluation and summarize your findings."""
                },
                {
                    "role": "user",
                                    "content": f"""Analyze the provided scenario for potential collisions:
                                            Context: {escaped_context}

                                            ### Steps:
                                            1. Identify obstacles in the same lanelet or trajectory as the ego vehicle.
                                            2. Calculate the following metrics:
                                            - **TTC**: Time to collision in both longitudinal and lateral directions.
                                            - **MDC**: Smallest distance between the ego vehicle and obstacles.
                                            3. Provide detailed reasoning for all conclusions.
                                            
                                            No matter how many obstacles are present, ensure all obstacles are included in the output with the following format:

                                            ### Safety analysis for timestep <timesteps>: Here's the evaluation of each obstacle according to the provided metrics and calculations.
                                            ### Obstacle Analysis:
                                            - Obstacle ID: <Obstacle ID>
                                            - Distance Risk reason: considering the DTClong and DTClat values and relative direction
                                            - Distance safety score: <Risk Score (0-5)>
                                            - Time Risk reason: Consider the TTClong and TTClat values and relative direction
                                            - Time safety score: <Risk Score (0-5)>
                                            - Overall Risk score: <Risk Score (0-5)>

                                            ### Summary in JSON Format: 
                                            Summarize all obstacles with collision risk which Overall Risk Score is 0 and all obstacles with extreme risk which Overall Risk Score is 1 in the following JSON format. Make sure if they don't exist, set them as `null`:
                                            ```json
                                            {{
                                                "CollisionObstacles": [
                                                    {{
                                                        "ObstacleID": "<Obstacle ID>",
                                                        "OverallRiskScore": "<Risk Score (0)>"
                                                    }}
                                                ],
                                                "ExtremeRiskObstacle": {{
                                                    "ObstacleID": "<Obstacle ID>",
                                                    "OverallRiskScore": "<Risk Score (1)>"
                                                }}
                                            }}
                                            ```"""
                }
            ]

def generate_timestep_report(ego_csv_path, obstacles_csv_path, output_txt_path, timestep_interval):
    """Generate a timestep report including positions, velocities, accelerations, and lanelet IDs."""
    try:
        ego_df = pd.read_csv(ego_csv_path)
        obstacles_df = pd.read_csv(obstacles_csv_path)

        report = []
        unique_timesteps = sorted(ego_df['timestep'].unique())
        filtered_timesteps = [t for t in unique_timesteps if t % timestep_interval == 0]

        for timestep in filtered_timesteps:
            ego_data = ego_df.query("timestep == @timestep").iloc[0]
            ego_info = (f"The position of Ego: {ego_data['ego_id']} is "
                        f"({ego_data['x_position']}, {ego_data['y_position']}), "
                        f"the orientation is {ego_data['orientation']}, "
                        f"the velocity is {ego_data['velocity']} and the acceleration is {ego_data['acceleration']}, "
                        f"current located in lanelet {ego_data['lanelet_id']}")

            obstacle_data = obstacles_df.query("timestep == @timestep")
            obstacle_info = "\n".join(
                f"The position of obstacle: {row['obstacle_id']} is "
                f"({row['x_position']}, {row['y_position']}), "
                f"the orientation is {row['orientation']}, "
                f"the velocity is {row['velocity']} and the acceleration is {row['acceleration']}, "
                f"current located in lanelet {row['lanelet_id']}"
                for _, row in obstacle_data.iterrows()
            )
            timestep = round(0.1 *timestep, 1)  # Convert to seconds
            timestep_report = f"At {timestep} seconds:\n{ego_info}\n"
            if not obstacle_data.empty:
                timestep_report += obstacle_info + "\n"
            timestep_report += "\n"
            report.append(timestep_report)

        with open(output_txt_path, 'w') as file:
            file.writelines(report)

        print(f"Report saved to {output_txt_path}")
        return report

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return []
    except KeyError as e:
        print(f"Missing column in data: {e}")
        return []

def log_performance(scenario_name, engine, timestep, response_time, token_usage):
    """Log time and token usage into a CSV file."""
    log_file = os.path.join(CONFIG["base_dir"], f"{engine}_Cart_performance_log.csv")
    log_data = {
        "Scenario Name": scenario_name,
        "Engine": engine,
        "Timestep (s)": timestep,
        "Response Time (s)": response_time,
        "Tokens Used": token_usage,
    }
    if not os.path.exists(log_file):
        pd.DataFrame([log_data]).to_csv(log_file, index=False)
    else:
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', index=False, header=False)

def process_scenario(scenario_path, timestep_interval, engine="openai"):
    Cart_dir = os.path.join(scenario_path, f"Cart/{engine}")
    ego_path = os.path.join(scenario_path, "ego_trajectory.csv")
    obstacles_path = os.path.join(scenario_path, "dynamic_obstacles.csv")
    output_txt_path = os.path.join(Cart_dir, "timestep_report.txt")
    response_txt_path = os.path.join(Cart_dir, "safety_analysis_TTC.txt")

    if not os.path.exists(Cart_dir):
        os.makedirs(Cart_dir)
        print(f"Created directory: {Cart_dir}")

    report = generate_timestep_report(ego_path, obstacles_path, output_txt_path, timestep_interval)
    if not report:
        print("No report generated for scenario:", scenario_path)
        return

    try:
        unique_timesteps = sorted(pd.read_csv(ego_path)['timestep'].unique())
        last_timestep = unique_timesteps[-1]
        formatted_timestep = last_timestep * 0.1

        start_idx = next(i for i, line in enumerate(report) if line.startswith(f"At {formatted_timestep:.1f} seconds:"))
        end_idx = next((i for i, line in enumerate(report[start_idx + 1:], start=start_idx + 1) if line.startswith("At ")), len(report))
        context = "".join(report[start_idx:end_idx])

        messages = generate_safety_analysis_MDC(context)
        response, response_time, token_usage = get_completion(engine, set_params(engine), messages)
        time.sleep(30)  # Add a small delay to avoid API rate limits
        if response:
            with open(response_txt_path, 'w') as response_file:
                response_file.write(f"Safety analysis for the last timestep ({formatted_timestep:.1f} seconds):\n")
                response_file.write(response + "\n\n")

            print(f"Safety analysis for the last timestep ({formatted_timestep:.1f} seconds) saved to {response_txt_path}")

            # Log the performance
            log_performance(os.path.basename(scenario_path), engine, formatted_timestep, response_time, token_usage)

    except (StopIteration, KeyError) as e:
        print(f"Error while processing last timestep for scenario {scenario_path}: {e}")

def main():
    base_dir = CONFIG["base_dir"]
    timestep_interval = CONFIG["timestep_interval"]

    for scenario_name in os.listdir(base_dir):
        scenario_path = os.path.join(base_dir, scenario_name)
        if os.path.isdir(scenario_path):
            print(f"Processing scenario: {scenario_name}")
            process_scenario(scenario_path, timestep_interval, engine="gemini")  # Change to "openai" for OpenAI

if __name__ == "__main__":
    main()
