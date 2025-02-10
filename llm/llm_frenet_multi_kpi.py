import os
import openai
import pandas as pd
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# Gemini API setup
gemini_client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Configuration dictionary for paths and parameters
CONFIG = {
    "base_dir": "/home/yuan/mybookname/Openai/Safety/output_validation",
    "scenario_name": "ESP_Barcelona-29_33_T-1",
    "model_params": {
        "openai": {
            "model": "gpt-4o-mini",
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
    params = CONFIG["model_params"][engine].copy()
    params.update(kwargs)
    return params

def get_completion(engine, params, messages, max_retries=3, retry_delay=10):
    """Attempts to get a response from the model, with retries for transient errors."""
    retries = 0
    while retries < max_retries:
        try:
            start_time = time.time()  # Start time tracking

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
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                return response.choices[0].message.content, elapsed_time

            elif engine == "gemini":
                response = gemini_client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    n=1
                )
                elapsed_time = time.time() - start_time
                return response.choices[0].message.content, elapsed_time

        except Exception as e:
            error_message = str(e)
            if "overloaded" in error_message.lower() or "500" in error_message:
                print(f"{engine.capitalize()} is overloaded. Retrying {retries+1}/{max_retries} in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                print(f"Error occurred while processing {engine} completion: {e}")
                return None, None

    print(f"Failed to get a response from {engine} after {max_retries} retries.")
    return None, None

def generate_safety_analysis(context):
    """Generate analysis for Minimum Distance to Collision (MDC)."""
    # Escape curly braces in the context
    # Double the braces using regex to escape them for proper formatting in the f-string
    context = re.sub(r'([{}])', r'\1\1', context)

    return [
                {
                    "role": "system",
                    "content": """You are an expert in collision analysis for autonomous driving scenarios.
                                    Your role is to evaluate the provided scenario based on the following safety metrics with scores ranging from 0 to 5 for each metric, where 0 indicates collision and 5 indicates no risk of collision:

                                    **Risk Levels and Definitions:**
                                    <Extreme Risk (Score: 1)>: Immediate collision or very high likelihood of impact. Urgent action is required.
                                    <High Risk (Score: 2)>: Close to collision or highly probable collision path. Needs prompt attention and quick manoeuvring.
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
                                            "content": f"""
                                            Based on the given scenario context:
                                            ```{context}```
                                            which describes information about obstacles at a specific timestep. Each obstacle has the following details:
                                            - Obstacle ID  
                                            - Relative Direction, which can be front, back, left, right, front-left, front-right, back-left, or back-right.
                                            - Real Distance Longitudinal (DTClong) and Lateral (DTClat) between the ego vehicle and the obstacle.
                                            - Relative Velocity Longitudinal (Vrel_long) and Lateral (Vrel_lat) between the ego vehicle and the obstacle.
                                            - Relative Acceleration Longitudinal (Arel_long) and Lateral (Arel_lat) between the ego vehicle and the obstacle.
                                            - Motion Description providing additional context about the obstacle's movement.

                                            ### Steps:
                                            1. Understand the scenario context and system message.
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

                                                                            
                                            ### Summary in JSON Format:  Summarize all obstacles with collision risk which Overall Risk Score is 0 and all obstacles with extreme risk which Overall Risk Score is 1 in the following JSON format. Make sure if they don't exist, set them as `null`:
                                            {{
                                                "CollisionObstacle": {{
                                                    "ObstacleID": "<Obstacle ID>",
                                                    "OverallRiskScore": "<0>"
                                                }},
                                                "ExtremeRiskObstacle": {{
                                                    "ObstacleID": "<Obstacle ID>",
                                                    "OverallRiskScore": "<1>"
                                                }}
                                            }}
                                            """
                }
            ]

def log_performance(scenario_name, engine, timestep, response_time):
    """Log performance metrics into a single CSV file."""
    log_file = os.path.join(CONFIG["base_dir"], f"{engine}_frenet_performance_log.csv")
    log_data = {
        "Scenario Name": scenario_name,
        "Engine": engine,
        "Timestep (s)": timestep,
        "Response Time (s)": response_time,
    }
    # Write log to CSV
    if not os.path.exists(log_file):
        pd.DataFrame([log_data]).to_csv(log_file, index=False)
    else:
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', index=False, header=False)

def process_scenarios(
    engine,
    base_dir,
    single_scenario=False,
    scenario_name=None,
    use_specific_timestep=False,
    specific_timestep=None,
    use_first_timestep=False,
    use_last_timestep=False,
):
    if single_scenario:
        if not scenario_name:
            print("Error: Single scenario processing requires a valid scenario_name.")
            return
        
        process_safety_analysis(
            engine=engine,
            base_dir=base_dir,
            scenario_name=scenario_name,
            use_specific_timestep=use_specific_timestep,
            specific_timestep=specific_timestep,
            use_first_timestep=use_first_timestep,
            use_last_timestep=use_last_timestep,
        )
    else:
        scenarios = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

        for scenario in scenarios:
            print(f"Processing scenario: {scenario}")
            process_safety_analysis(
                engine=engine,
                base_dir=base_dir,
                scenario_name=scenario,
                use_specific_timestep=use_specific_timestep,
                specific_timestep=specific_timestep,
                use_first_timestep=use_first_timestep,
                use_last_timestep=use_last_timestep,
            )

def process_safety_analysis(
    engine,
    base_dir,
    scenario_name,
    use_specific_timestep=False,
    specific_timestep=None,
    use_first_timestep=False,
    use_last_timestep=False,
):
    input_csv_path = f"{base_dir}/{scenario_name}/relative_metrics.csv"
    response_ttc_txt_path = f"{base_dir}/{scenario_name}/frenet/{engine}/safety_analysis_TTC.txt"
    context_txt_path = f"{base_dir}/{scenario_name}/frenet/{engine}/context.txt"

    os.makedirs(os.path.dirname(response_ttc_txt_path), exist_ok=True)

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: {input_csv_path} not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: {input_csv_path} is empty.")
        return

    timesteps = sorted(df['timestep'].unique())

    if use_first_timestep:
        selected_timesteps = [min(timesteps)]
    elif use_last_timestep:
        selected_timesteps = [max(timesteps)]
    elif use_specific_timestep and specific_timestep is not None:
        selected_timesteps = [specific_timestep]
    else:
        selected_timesteps = timesteps

    with open(response_ttc_txt_path, "w") as response_ttc_file, open(context_txt_path, "w") as context_file:
        for timestep in selected_timesteps:
            timestep_data = df[df['timestep'] == timestep]
            if timestep_data.empty:
                print(f"Warning: No data for timestep {timestep}. Skipping.")
                continue

            print(f"Processing timestep {0.1*timestep:.1f} seconds.")

            context = f"At {0.1 * timestep:.1f} seconds:\n"
            for _, row in timestep_data.iterrows():
                context += (
                    f"  Obstacle {row['obstacle_id']} is in the {row['relative_direction']} of the ego car. "
                    f"The real distance is longitudinal {row['adjusted_d_long']} m and lateral {row['adjusted_d_lat']} m. "
                    f"Relative velocity: longitudinal {row['v_rel_long']} m/s, lateral {row['v_rel_lat']} m/s. "
                    f"Relative acceleration: longitudinal {row['a_rel_long']} m/s², lateral {row['a_rel_lat']} m/s². "
                    f"Motion: {row['motion_description']}.\n"
                )

            context_file.write(context + "\n")
            messages = generate_safety_analysis(context)
            response, response_time = get_completion(engine, set_params(engine), messages)
            time.sleep(30)  # Adjust delay as needed

            if response:
                response_ttc_file.write(response + "\n\n")

                # Log response time
                log_performance(scenario_name, engine, timestep, response_time)

    print(f"Processed scenario {scenario_name}. Results saved to {response_ttc_txt_path}")


# Main function
def main():
    # Global flags
    engine = "gemini"  # Choose from "openai", "gemini", "deepseek"
    use_specific_timestep = False  # Change to True to run a specific timestep
    specific_timestep = 3.5  # Define the timestep of interest if use_specific_timestep=True
    use_first_timestep = False
    use_last_timestep = True  # Set True to process the last timestep
    single_scenario = False  # Set to True to process only the scenario_name below
    scenario_name = "DEU_Bremen-7_23_T-1"  # Name of the scenario to process when single_scenario is True

    base_dir = CONFIG["base_dir"]

    process_scenarios(
        engine=engine,
        base_dir=base_dir,
        single_scenario=single_scenario,
        scenario_name=scenario_name,
        use_specific_timestep=use_specific_timestep,
        specific_timestep=specific_timestep,
        use_first_timestep=use_first_timestep,
        use_last_timestep=use_last_timestep,
    )

if __name__ == "__main__":
    main()
