import os
import openai
import json
import random
import pandas as pd
from dotenv import load_dotenv
import re
import time
from openai import OpenAI  # Import Gemini client

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Gemini API setup
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# DeepSeek API setup
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
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
        },
        "deepseek": {
            "model": "deepseek-chat",
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

def get_completion(engine, params, messages):
    """Attempts to get a response from the model and measure response time."""
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
        elif engine == "gemini":
            response = gemini_client.chat.completions.create(
                model=params["model"],
                messages=messages,
                n=1
            )
        elif engine == "deepseek":
            response = deepseek_client.chat.completions.create(
                model=params["model"],
                messages=messages,
                n=1
            )

        elapsed_time = time.time() - start_time
        return response.choices[0].message.content, elapsed_time
    except Exception as e:
        print(f"Error occurred while processing {engine} completion: {e}")
        return None, None

# Function to generate safety analysis
def generate_dtc_ttc_analysis(context):
    """Generate analysis for Relative Direction and Time to Collision."""
    return [
        {
            "role": "system",
            "content": f"""
                    You are an expert in collision analysis for autonomous driving scenarios.
                    Your role is to evaluate the provided scenario based on the following safety metrics with scores ranging from 0 to 5 for each metric, where 0 indicates collision and 5 indicates no risk of collision:

                    **Risk Levels and Definitions:**
                    <Extreme Risk (Score: 1)>: Immediate very high likelihood of collision. Urgent action is required.
                    <High Risk (Score: 2)>: Close to collision or highly probable collision path. Needs prompt attention and quick manoeuvring.
                    <Medium Risk (Score: 3)>: Moderate collision risk but manageable with timely reactions and proper strategies.
                    <Low Risk (Score: 4)>: Minimal collision risk. The situation is controllable with sufficient time to react.
                    <Negligible Risk (Score: 5)>: No significant collision risk. Obstacles are either moving away or far enough not to interfere.

                    **Metrics Considered:**

                    1. **Distance to Collision (DTC):**
                        - DTClong: Longitudinal Distance to Collision.
                        - DTClat: Lateral Distance to Collision.
                        - LongDSC: Longitudinal Distance Safety Score.
                        - LatDSC: Lateral Distance Safety Score.
                        - Risk Levels Based on DTC:
                            - **Collision Risk (LongDSC = 0 or LatDSC = 0):** DTClong = 0 or DTClat = 0.
                            - **Extreme Risk (LongDSC = 1 or LatDSC = 1):** 0 <DTClong <= 0.5 or 0 <DTClat <= 0.5.
                            - **High Risk (LongDSC = 2 or LatDSC = 2):** 0.5 < DTClong <= 1 or 0.5 < DTClat <= 1.
                            - **Medium Risk (LongDSC = 3 or LatDSC = 3 ):** 1 < DTClong <= 3  or 1 < DTClat <= 3.
                            - **Low Risk (LongDSC = 4 or LatDSC = 4):**  3 < DTClong <= 5  or  3 < DTClat <= 5.
                            - **Negligible Risk (LongDSC = 5 or LatDSC = 5):** DTClong > 5 or DTClat > 5.

                        - **Weighting and Direction Adjustment:** 
                            - Overall Risk Score: DSC = LongDSC * wdominant + LatDSC * (1-wdominant),
                              where wdominant is determined by the relative direction:
                                - Front/Back: wdominant = 1.
                                - Left/Right: wdominant = 0.
                                - Other directions: wdominant = 0.5.

                    2. **Time to Collision (TTC):**
                        - TTClong: Longitudinal Time to Collision.
                        - TTClat: Lateral Time to Collision.
                        - LongTSC: Longitudinal Time Safety Score.
                        - LatTSC: Lateral Time Safety Score.
                        - Risk Levels Based on TTC:
                            - **Collision Risk (LongTSC = 0 or LatTSC = 0):** TTClong = 0 or TTClat = 0.
                            - **Extreme Risk (LongTSC = 1 or LatTSC = 1): ** TTClong <= 0.5 or TTClat <= 0.5.
                            - **High Risk (LongTSC = 2 or LatTSC = 2):** 0.5 < TTClong <= 1 or 0.5 < TTClat <= 1.
                            - **Medium Risk (LongTSC = 3 or LatTSC = 3):** 1 < TTClong <= 3 or 1 < TTClat <= 3.
                            - **Low Risk (LongTSC = 4 or LatTSC = 4):** 3 < TTClong <= 5 or 3 < TTClat <= 5.
                            - **Negligible Risk (LongTSC = 5 or LatTSC = 5):** TTClong > 5 or TTClat > 5.
                            If both are 0, the risk level should be 0 which means collision.

                        - **Weighting and Direction Adjustment:** 
                            - Overall Risk Score: TSC = LongTSC * wdominant + LatTSC * (1-wdominant),
                              where wdominant is determined by the relative direction:
                                - Front/Back: wdominant = 1.
                                - Left/Right: wdominant = 0.
                                - Other directions: wdominant = 0.5.

                    **Determining Overall Risk:**
                    The overall risk score combines DTC and TTC metrics:
                    Risk Score = 0.5 * DSC + 0.5 * TSC
                    The final risk score should be rounded to the nearest integer.
                    """
        },
                {
            "role": "user",
            "content": f"""
            Based on the given scenario context:
            ```{context}```
            which describes information about obstacles at a specific timestep. Each obstacle has the following details:
            - Obstacle ID  
            - Relative Direction, which can be front, back, left, right, front-left, front-right, back-left, or back-right.
            - Distance to Collision (DTC) in both longitudinal and lateral directions.
            - Time to Collision (TTC) in both longitudinal and lateral directions.
            - Motion Description providing additional context about the obstacle's movement.

            Steps to Follow:
            1. Use the provided relative direction to determine the dominant direction for weighting.
                - Front/Back: wdominant = 1.
                - Left/Right: wdominant = 0.
                - Other directions: wdominant = 0.5.
            2. Evaluate the DTC metrics for both longitudinal and lateral distances to determine the risk levels: LongDSC and LatDSC.
            3. Using the dominant direction weighting, calculate the overall DTC score - DSC based on the weighted combination of longitudinal and lateral risks:
            DTC Score: DSC = LongDSC * wdominant +  LatDSC * (1-wdominant).
            4. Evaluate the TTC metrics for both longitudinal and lateral times to collision to determine the risk levels: LongTSC and LongTSC.
            5. Using the dominant direction weighting, calculate the overall TTC socre - TSC based on the weighted combination of longitudinal and lateral risks:
            TTC Score: TSC =  LongTSC * wdominant + LatTSC * (1-wdominant).
            6. Calculate the overall risk score by combining the DSC and TSC.
            Risk Score = 0.5 * DSC + 0.5 * TSC, please also round the risk score down to the nearest integer, such that values less than or equal to 0.5 to 0 and greater than 0.5 to 1. 

            No matter how many obstacles are present, ensure all obstacles are included in the output with the following format:

            ### Safety analysis for timestep <timesteps>: Here's the evaluation of each obstacle according to the provided metrics and calculations.
            ### Obstacle Analysis:
                - Obstacle ID: <numeric ID>
                - Relative Direction: <Front/Back/Left/Right/front-left/front-right/back-left/back-right>
                - Distance Risk Reason: <description in context of DTClong and DTClat values and relative direction>
                - Longitudinal Distance Safety Score: <LongDSC>  
                - Lateral Distance Safety Score: <LatDSC>
                - Overall Distance Safety Score: <DSC>
                - Time Risk Reason: <description in context of TTClong and TTClat values and relative direction>
                - Longitudinal Time Safety Score: <LongTSC>
                - Lateral Time Safety Score: <LatTSC>
                - Overall Time Safety Score: <TSC>
                - Overall Risk Score: <Risk Score>

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
    return messages


# Function to process scenarios
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
        
        # Process only the specified scenario
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
        # Process all scenarios in the base directory
        scenarios = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

        for scenario_name in scenarios:
            print(f"Processing scenario: {scenario_name}")
            process_safety_analysis(
                engine=engine,
                base_dir=base_dir,
                scenario_name=scenario_name,
                use_specific_timestep=use_specific_timestep,
                specific_timestep=specific_timestep,
                use_first_timestep=use_first_timestep,
                use_last_timestep=use_last_timestep,
            )

def log_performance(scenario_name, engine, timestep, response_time):
    """Log response times into a centralized file."""
    log_file = os.path.join(CONFIG["base_dir"], f"{engine}_metrics_performance_log.csv")
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


def process_safety_analysis(
    engine,
    base_dir,
    scenario_name,
    use_specific_timestep=False,
    specific_timestep=None,
    use_first_timestep=False,
    use_last_timestep=False,
):
    output_txt_path = f"{base_dir}/{scenario_name}/output.txt"
    response_ttc_txt_path = f"{base_dir}/{scenario_name}/metrics/{engine}/safety_analysis_TTC.txt"
    context_txt_path = f"{base_dir}/{scenario_name}/metrics/{engine}/context.txt"

    # Ensure the directory for the response and context files exists
    os.makedirs(os.path.dirname(response_ttc_txt_path), exist_ok=True)

    # Open and read the output.txt file
    try:
        with open(output_txt_path, "r") as file:
            report = json.load(file)
    except FileNotFoundError:
        print(f"Error: {output_txt_path} not found.")
        return

    # Determine the timestep(s) to process
    timesteps = [
        float(re.match(r"At (\d+(\.\d+)?) seconds", timestep_key).group(1))
        for timestep_key in report
        if re.match(r"At (\d+(\.\d+)?) seconds", timestep_key)
    ]

    # Select timestep based on flags
    if use_first_timestep and timesteps:
        selected_timestep = min(timesteps)
    elif use_last_timestep and timesteps:
        selected_timestep = max(timesteps)
    elif use_specific_timestep and specific_timestep is not None:
        selected_timestep = specific_timestep
    else:
        selected_timestep = None  # Process all timesteps

    if selected_timestep is None:
        print(f"Processing all timesteps in scenario {scenario_name}.")
    else:
        print(f"Processing timestep {selected_timestep:.1f} seconds in scenario {scenario_name}.")

    with open(response_ttc_txt_path, "w") as response_ttc_file, open(context_txt_path, "w") as context_file:
        for timestep_key, obstacles_data in report.items():
            match = re.match(r"At (\d+(\.\d+)?) seconds", timestep_key)
            if not match:
                print(f"Warning: Timestep format invalid in key '{timestep_key}', skipping.")
                continue

            timestep = float(match.group(1))

            if selected_timestep is not None and timestep != selected_timestep:
                continue

            # Prepare the context for the current timestep
            context = f"Timestep {timestep:.1f} seconds:\n"  # Define outside loop
            for obstacle, details in obstacles_data.items():
                relative_direction = details.get("Relative Direction", "N/A")
                distance_to_collision = details.get("Distance to Collision", {})
                ttc = details.get("Time to Collision", {})
                motion_description = details.get("Motion Description", "No description available.")

                context += f"  {obstacle}:\n"
                context += f"    Relative Direction: {relative_direction}\n"
                context += f"    Distance to Collision:\n"
                context += f"      Longitudinal: {distance_to_collision.get('Longitudinal', 'N/A')}\n"
                context += f"      Lateral: {distance_to_collision.get('Lateral', 'N/A')}\n"
                context += f"    Time to Collision:\n"
                context += f"      Longitudinal: {ttc.get('Longitudinal', 'N/A')}\n"
                context += f"      Lateral: {ttc.get('Lateral', 'N/A')}\n"
                context += f"    Motion Description: {motion_description}\n"

            # Write the accumulated context
            context_file.write(context + "\n")

            # Generate the DTC and TTC safety analysis
            messages = generate_dtc_ttc_analysis(context)
            response, response_time = get_completion(engine, set_params(engine=engine), messages)

            # Add a sleep interval
            time.sleep(15)  # Adjust based on your API rate limit

            if response is None:
                print(f"Skipping timestep {timestep:.1f} seconds due to an error in {engine} completion.")
                continue

            # Write the response to the TTC safety analysis file
            response_ttc_file.write(f"Safety analysis for timestep {timestep:.1f} seconds:\n")
            response_ttc_file.write(response + "\n\n")

            # Log response time
            log_performance(scenario_name, engine, timestep, response_time)

    print(f"Processed scenario {scenario_name}. Results saved to:")
    print(f"  - {response_ttc_txt_path}")


# Main function
def main():
    # Global flags
    engine = "openai"  # Change to "gemini" for Gemini API
    use_specific_timestep = False  # Change to True to run a specific timestep
    specific_timestep = 3.5  # Define the timestep of interest if use_specific_timestep=True
    use_first_timestep = False
    use_last_timestep = True  # Set True to process the last timestep
    single_scenario = False  # Set to True to process only the scenario_name below
    scenario_name = "ESP_Barcelona-29_33_T-1"  # Name of the scenario to process when single_scenario is True

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
