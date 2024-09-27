import cv2
import torch
import torchvision
import numpy as np
from torch.nn import functional as F

# Define a simplified list of action classes
labels = [
    # Basic human actions
    "walking", "running", "jumping", "sitting", "standing", "lying down",
    "climbing", "falling", "crawling", "crouching", "kneeling",
    
    # Hand and arm actions
    "clapping", "waving", "pointing", "grasping", "throwing", "catching",
    "punching", "pushing", "pulling", "lifting", "dropping",
    
    # Whole body actions
    "dancing", "stretching", "bending", "turning", "spinning", "balancing",
    
    # Daily activities
    "eating", "drinking", "talking", "laughing", "crying", "yawning", "sneezing",
    "coughing", "writing", "reading", "typing", "cooking", "cleaning", "washing",
    "dressing", "undressing", "brushing teeth", "combing hair",
    
    # Sports and exercise
    "swimming", "cycling", "playing_football", "playing_basketball", "playing_tennis",
    "playing_golf", "skiing", "skateboarding", "surfing", "yoga", "weightlifting",
    "jogging", "stretching", "push-ups", "sit-ups",
    
    # Transportation
    "driving", "riding_bicycle", "riding_motorcycle", "getting_in_car", "getting_out_car",
    
    # Work and study
    "working_on_computer", "writing_on_board", "presenting", "studying",
    
    # Social interactions
    "handshaking", "hugging", "kissing", "arguing", "fighting",
    
    # Entertainment
    "playing_instrument", "singing", "dancing", "acting", "juggling",
    
    # Technology use
    "using_smartphone", "taking_photo", "filming", "gaming",
    
    # Miscellaneous
    "shopping", "gardening", "painting", "sewing", "knitting",
    "opening_door", "closing_door", "petting_animal", "applying_makeup"
]

# Define the TSM model
class TSN(torch.nn.Module):
    def init(self, num_class, num_segments, modality, base_model='resnet50'):
        super(TSN, self).init()
        self.num_segments = num_segments
        self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, num_class)

    def forward(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = torch.mean(base_out, dim=1)
        return output

# Load the model
num_class = len(labels)
model = TSN(num_class, 8, 'RGB', base_model='resnet50')
model.eval()

# Function to preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return torch.FloatTensor(frame).unsqueeze(0)

# Open the jumping video file
video = cv2.VideoCapture('jumping.mp4')

# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
out = cv2.VideoWriter('output_jumping_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process the video
frame_buffer = []
frame_count = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = preprocess_frame(frame_rgb)
    frame_buffer.append(processed_frame)
    
    if len(frame_buffer) == 8:
        input_tensor = torch.cat(frame_buffer, dim=0).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probs, 5)
        top5_labels = [labels[idx] for idx in top5_idx[0]]
        
        # Draw the top 5 predicted actions on the frame
        for i, (label, prob) in enumerate(zip(top5_labels, top5_prob[0])):
            text = f"{i+1}. {label}: {prob.item():.2%}"
            cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)
        
        # Remove the oldest frame from the buffer
        frame_buffer.pop(0)
    
    frame_count += 1
    if frame_count % 30 == 0:  # Print progress every 30 frames
        print(f"Processed {frame_count} frames")

# Release the video capture and writer objects
video.release()
out.release()

print("Video processing complete. Output saved as 'output_jumping_video.mp4'")

# Display the first frame of the processed video
first_frame = cv2.VideoCapture('output_jumping_video.mp4')
ret, frame = first_frame.read()
if ret:
    from IPython.display import display, Image
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("First frame of processed jumping video with top 5 predicted actions")
    plt.savefig('first_frame_jumping.jpg')
    display(Image('first_frame_jumping.jpg'))
    
first_frame.release()

print("To download the processed video, look for 'output_jumping_video.mp4' in the file browser on the left.")
