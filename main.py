import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================

INPUT_FOLDER = "downloads"

MODEL_PATH = "crack_seg_cnn.pth"

IMG_SIZE = 416

MASK_FOLDER = "masks"
OVERLAY_FOLDER = "overlays"
PROB_FOLDER = "probabilities"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# MODEL
# ==========================================

class CrackSegCNN(nn.Module):

    def __init__(self):
        super(CrackSegCNN, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(
            32,
            1,
            kernel_size=2,
            stride=2
        )

    def forward_encoder(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        return e1, e2, e3

    def forward(self, x):

        e1, e2, e3 = self.forward_encoder(x)

        x = self.up1(e3)

        x = torch.cat(
            [x, e2],
            dim=1
        )

        x = self.conv1(x)

        x = self.up2(x)

        x = torch.cat(
            [x, e1],
            dim=1
        )

        x = self.conv2(x)

        x = self.up3(x)

        return torch.sigmoid(x)

# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs(
    MASK_FOLDER,
    exist_ok=True
)

os.makedirs(
    OVERLAY_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROB_FOLDER,
    exist_ok=True
)

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = CrackSegCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print(
    f"Model loaded on {DEVICE}"
)

# ==========================================
# GET IMAGES
# ==========================================

image_files = [

    f

    for f in os.listdir(
        INPUT_FOLDER
    )

    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )

]

print(
    f"\nFound "
    f"{len(image_files)} "
    f"images."
)

# ==========================================
# PROCESS IMAGES
# ==========================================

for idx, filename in enumerate(
    image_files,
    start=1
):

    print(
        f"\n[{idx}/{len(image_files)}] "
        f"{filename}"
    )

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    original = cv2.imread(
        image_path
    )

    if original is None:

        print(
            "Failed to load image."
        )

        continue

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # ======================================
    # INFERENCE
    # ======================================

    with torch.no_grad():

        pred = model(
            tensor
        )

    prob_map = (
        pred.squeeze()
        .cpu()
        .numpy()
    )

    base_name = os.path.splitext(
        filename
    )[0]

    prob_path = os.path.join(
        PROB_FOLDER,
        f"{base_name}.csv"
    )

    mask_path = os.path.join(
        MASK_FOLDER,
        f"{base_name}_mask.png"
    )

    overlay_path = os.path.join(
        OVERLAY_FOLDER,
        f"{base_name}_overlay.png"
    )

    # ======================================
    # SAVE PROBABILITIES
    # ======================================

    np.savetxt(
        prob_path,
        prob_map,
        delimiter=",",
        fmt="%.6f"
    )

    # ======================================
    # CREATE MASK
    # ======================================

    mask = (
        prob_map >= 0.5
    ).astype(np.uint8) * 255

    cv2.imwrite(
        mask_path,
        mask
    )

    # ======================================
    # CREATE OVERLAY
    # ======================================

    overlay_mask = cv2.resize(
        mask,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    overlay_mask = cv2.cvtColor(
        overlay_mask,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        original,
        0.7,
        overlay_mask,
        0.3,
        0
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    print(
        f"Saved mask: {mask_path}"
    )

    print(
        f"Saved overlay: {overlay_path}"
    )

    print(
        f"Saved probabilities: {prob_path}"
    )

# ==========================================
# DONE
# ==========================================

print(
    "\nAll images processed."
)