import torch
import os
from pathlib import Path
import google.generativeai as genai
import subprocess

# Set running environment
env_name = "base"

# Keypoints Detection
command_dump = [
    "conda", "run", "-n", env_name, "python", "apply_net.py", "dump",
    "configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml",
    "densepose_rcnn_R_101_FPN_DL_s1x.pkl",
    "input/image1.jpg",
    "--output", "results/input/input_pose.pkl"
]

command_show = [
    "conda", "run", "-n", env_name, "python", "apply_net.py", "show",
    "configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml",
    "densepose_rcnn_R_101_FPN_DL_s1x.pkl",
    "input/image1.jpg",
    "dp_contour,bbox",
    "--output", "results/input/input_pose.png"
]
subprocess.run(command_dump)
subprocess.run(command_show)

#Load LLM model
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# Load Keypoint
input_keypoint = torch.load(r'results\input\input_pose.pkl')
standard_keypoint = torch.load(r'results\standard\standard_pose.pkl')

# load Image
input_image_path = Path(r"results\input\input_pose.png")
input_image_part = {
    "mime_type": "image/png",
    "data": input_image_path.read_bytes()
}

standard_image_path = Path(r"results\standard\standard_pose.png")
standard_image_part = {
    "mime_type": "image/png",
    "data": standard_image_path.read_bytes()
}

# Generate feedback
prompt_template = """
### Task:
Compare the actual plank pose with the standard plank pose using DensePose outputs and provide corrective feedback.

### Inputs:
- **Actual Plank Pose**: {input_image_part}
- **Standard Plank Pose**: {standard_image_part}

### Step-by-Step Analysis:

1. **Interpret DensePose Outputs:**:
   - Begin by interpreting the key points and body part mappings provided by the DensePose outputs for both the actual and standard plank poses.
   - Ensure that the coordinates and body part labels are correctly aligned for comparison.

2. **Compare Key Points:**:
   - Compare the positions and angles of key body parts between the actual and standard poses using the DensePose data.
   - Focus on critical areas such as the head, shoulders, hips, and legs.

3. **Evaluate Pose Deviation**:
   - Analyze any deviations in body alignment, weight distribution, and balance between the two poses based on the DensePose data.
   - Assess how these deviations might impact the effectiveness of the plank exercise.

4. **Identify Strengths in Actual Pose**:
   - Highlight areas where the actual pose aligns well with the standard pose according to the DensePose data.
   - Discuss how these strengths contribute to a proper plank posture.

5. **Identify Weaknesses in Actual Pose:**:
   - Identify significant differences or misalignments in the actual pose compared to the standard using the DensePose data.
   - Discuss potential risks or inefficiencies caused by these weaknesses, such as strain on specific muscles or joints.

6. **Provide Corrective Recommendations**
   - Suggest specific adjustments to the actual pose based on the DensePose comparison to bring it closer to the standard plank pose.
   - Explain how these corrections can improve posture, reduce injury risk, and enhance exercise effectiveness.

### Final Summary:
- Summarize the key differences between the actual and standard plank poses, including strengths and weaknesses.
- Provide a clear set of corrective actions to help achieve a proper plank pose.
"""

formatted_prompt = prompt_template.format(
    input_image_part=standard_image_part,
    standard_image_part=standard_image_part
)

feedback = model.generate_content(formatted_prompt)

# save feedback
summary_start = feedback.text.find("Final Summary")
summary_text = feedback.text[summary_start:].strip()
details_start = feedback.text.find("Evaluate Pose Deviation")
details_text = feedback.text[details_start:summary_start].strip()

with open("results/feedback/summary.txt", "w") as summary_file:
    summary_file.write(summary_text)
    
with open("results/feedback/details.txt", "w") as details_file:
    details_file.write(details_text)

print("Summary and detailed analysis have been saved.")