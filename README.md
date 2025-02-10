# Risk-Aware Driving Scenario Analysis with Large Language Models

## Repository Contents

## 1. Manuscript
- The Arxiv version of the manuscript can be found here: [Paper](https://arxiv.org/abs/2502.02145)

## 2. Scenarios and Results
- In `Scnearios`, you can find a collection of **100 XML scenario files** and the related simulation results called `Simulation_scenarios_with_FrenetixMotionPlanner` which are generated using the [Frenetix Motion Planner](https://github.com/TUM-AVS/Frenetix-Motion-Planner/tree/main) within the CommonRoad framework.
- In `Results`, you can find the comparison output of **response time across models and templates** and also the outputs for each scenario across models and templates.

## 3. Code
- `Trajectory Collection` and `Safety Metrics Collection` are used to collect and caculate the related data for each template.

## Usage
### Step 1: Running the Scenarios
1. Use the provided **XML scenario files** to test the Frenetix Motion Planner in the **CommonRoad** environment.
2. The generated results should match those in `Simulation_scenarios_with_FrenetixMotionPlanner`.

### Step 2: Extracting Data & Computing Safety Metrics
- **Trajectory Collection:**
  - `main.py` (for a single scenario) or `main_multi.py` (for 100 scenarios) from `Trajectory_collection` to extract **dynamic obstacle trajectory data** from XML files and **ego trajectory data** from `Simulation_scenarios_with_FrenetixMotionPlanner`.
- **Safety Metrics Collection:**
  - `safety.py` (for a single scenario) or `safety_multi.py` (for 100 scenarios) from `Safety_metrics_collection` to compute relevant **safety metrics**. These metrics will be used in the **Frenet coordinate template** and the **safety-critical metrics template**.
- **Output Data Structure:**
  - Each scenario output will contain the following files:
    
    ```
    ├── dynamic_obstacles_with_lanelets.csv
    ├── dynamic_obstacles.csv
    ├── ego_trajectory_positions_with_lanelets.csv
    ├── ego_trajectory.csv
    ├── output.txt
    ├── relative_metrics.csv
    ```

## Paths to Modify
Ensure that the following paths in the scripts are correctly set according to your directory structure:

```python
source_dir = '/home/yuan/mybookname/Openai/Safety/collision_scenarios'  # Folder containing XML files

destination_dir = '/home/yuan/mybookname/Openai/Safety/json_scenarios'  # Folder to store converted JSON files

log_dir = '/home/yuan/mybookname/Openai/Safety/validation_scenarios'  # Folder for CommonRoad simulation results (Simulation_scenarios_with_FrenetixMotionPlanner)

output_dir = '/home/yuan/mybookname/Openai/Safety/output_validation'  # Folder for collected data, e.g., Results/output_LLMs/
```

## Requirements
- [CommonRoad](https://commonroad.in.tum.de/)
- [Frenetix Motion Planner](https://github.com/TUM-AVS/Frenetix-Motion-Planner/tree/main)
- Python 3.x

---

Ensure you have installed all dependencies before running the scripts.
  
## Abstract:
Large Language Models (LLMs) can capture
nuanced contextual relationships, reasoning, and complex
problem-solving. By leveraging their ability to process
and interpret large-scale information, LLMs have shown
potential to address domain-specific challenges, including
those in autonomous driving systems. This paper proposes
a novel framework that leverages LLMs for risk-aware
analysis of generated driving scenarios. We hypothesize that
LLMs can effectively evaluate whether driving scenarios
generated by autonomous driving testing simulators are
safety-critical. To validate this hypothesis, we conducted an
empirical evaluation to assess the effectiveness of LLMs
in performing this task. This framework will also provide
feedback to generate the new safety-critical scenario by
using adversarial method to modify existing non-critical
scenarios and test their effectiveness in validating motion
planning algorithms.

## Framework
![framework](https://github.com/user-attachments/assets/c0bb680f-c6f3-4af4-9dec-4278acaf8774)
![templates](https://github.com/user-attachments/assets/b72a45c5-fd8f-4dc3-a359-a2df21b3fac3)

## Results
<p align="center">
    <img src="https://github.com/user-attachments/assets/01f56187-626d-4764-b6bd-56172152eb41" width="45%">
    <img src="https://github.com/user-attachments/assets/7726d61d-20c3-4adc-9a9d-ac440ed1c897" width="45%">
</p>


## New generated safety-critical Scenario based on LLMs
![collision](https://github.com/user-attachments/assets/bf120d8d-8d54-4b39-abf1-b1ee7fef9be9)


## Reference
>@article{gao2025risk,
  title={Risk-Aware Driving Scenario Analysis with Large Language Models},
  author={Gao, Yuan and Piccinini, Mattia and Betz, Johannes},
  journal={arXiv preprint arXiv:2502.02145}  [Add to Citavi project by ArXiv ID] ,
  year={2025}
}
