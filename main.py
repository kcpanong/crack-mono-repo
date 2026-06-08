import os
import cv2
import boto3
import torch
import requests
import tempfile
import firebase_admin
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms

from firebase_admin import (
    credentials,
    firestore
)

# ==========================================================
# 1. CONFIG
# ==========================================================

SERVICE_ACCOUNT_FILE = (
    "serviceAccountKey.json"
)

MODEL_PATH = (
    "crack_seg_cnn.pth"
)

IDENTITY_POOL_ID = (
    "ap-southeast-2:e70f96d9-6860-4f80-9bf8-082d2661b665"
)

REGION = (
    "ap-southeast-2"
)

BUCKET_NAME = (
    "my-angular-test-bucket-12345"
)

IMG_SIZE = 416

TEMP_DOWNLOAD_FOLDER = (
    "temp_downloads"
)

TEMP_MASK_FOLDER = (
    "temp_masks"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# 2. FIREBASE INIT
# ==========================================================

if not firebase_admin._apps:

    cred = credentials.Certificate(
        SERVICE_ACCOUNT_FILE
    )

    firebase_admin.initialize_app(
        cred
    )

db = firestore.client()

# ==========================================================
# 3. TEMP FOLDERS
# ==========================================================

os.makedirs(
    TEMP_DOWNLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    TEMP_MASK_FOLDER,
    exist_ok=True
)

# ==========================================================
# 4. MODEL
# ==========================================================

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

# ==========================================================
# 5. TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

# ==========================================================
# 6. LOAD MODEL
# ==========================================================

print("\nLoading model...")

model = CrackSegCNN().to(
    DEVICE
)

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

# ==========================================================
# 7. GET AWS TEMP CREDS
# ==========================================================

print(
    "\nGetting Cognito credentials..."
)

identity_client = boto3.client(
    "cognito-identity",
    region_name=REGION
)

identity = identity_client.get_id(
    IdentityPoolId=IDENTITY_POOL_ID
)

identity_id = identity[
    "IdentityId"
]

creds_response = (
    identity_client.get_credentials_for_identity(
        IdentityId=identity_id
    )
)

creds = creds_response[
    "Credentials"
]

print(
    "Temporary credentials acquired."
)

s3 = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=creds[
        "AccessKeyId"
    ],
    aws_secret_access_key=creds[
        "SecretKey"
    ],
    aws_session_token=creds[
        "SessionToken"
    ]
)

# ==========================================================
# 8. FIRESTORE QUERY
# ==========================================================

def get_unprocessed_crops():

    docs = (
        db.collection("images")
        .where(
            "type",
            "==",
            "cropped"
        )
        .stream()
    )

    results = []

    for doc in docs:

        data = doc.to_dict()

        if data.get(
            "retrievedForProcessing",
            False
        ):
            continue

        if not data.get(
            "storageUrl"
        ):
            continue

        results.append({

            "doc_ref":
                doc.reference,

            "sessionId":
                data.get(
                    "sessionId"
                ),

            "cropped_id":
                data.get(
                    "cropped_id"
                ),

            "filename":
                data.get(
                    "filename"
                ),

            "storageUrl":
                data.get(
                    "storageUrl"
                )
        })

    return results

# ==========================================================
# 9. DOWNLOAD
# ==========================================================

def sanitize_filename(
    filename
):

    bad_chars = (
        ':',
        '/',
        '\\',
        '*',
        '?',
        '"',
        '<',
        '>',
        '|'
    )

    for char in bad_chars:

        filename = (
            filename.replace(
                char,
                "_"
            )
        )

    return filename

def download_image(
    url,
    output_path
):

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            response.content
        )

# ==========================================================
# 10. SEGMENT
# ==========================================================

def create_mask(
    image_path,
    output_mask_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image = image.resize(
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    tensor = (
        transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        pred = model(
            tensor
        )

    prob_map = (
        pred.squeeze()
        .cpu()
        .numpy()
    )

    mask = (
        prob_map >= 0.5
    ).astype(
        np.uint8
    ) * 255

    cv2.imwrite(
        output_mask_path,
        mask
    )

# ==========================================================
# 11. S3 UPLOAD
# ==========================================================

def upload_mask(
    mask_path,
    session_id,
    cropped_id
):

    s3_key = (
        f"masks/"
        f"{session_id}/"
        f"{cropped_id}_mask.png"
    )

    s3.upload_file(
        mask_path,
        BUCKET_NAME,
        s3_key
    )

    url = (
        f"https://{BUCKET_NAME}"
        f".s3.{REGION}.amazonaws.com/"
        f"{s3_key}"
    )

    return (
        s3_key,
        url
    )

# ==========================================================
# 12. UPDATE FIRESTORE
# ==========================================================

def mark_processed(
    doc_ref,
    mask_key,
    mask_url
):

    doc_ref.update({

        "maskS3Key":
            mask_key,

        "maskS3Url":
            mask_url,

        "retrievedForProcessing":
            True,

        "processedAt":
            firestore.SERVER_TIMESTAMP
    })

# ==========================================================
# 13. PROCESS PIPELINE
# ==========================================================

def process_all_crops():

    crops = (
        get_unprocessed_crops()
    )

    print(
        f"\nFound "
        f"{len(crops)} "
        f"crops to process.\n"
    )

    processed_count = 0

    for index, crop in enumerate(
        crops,
        start=1
    ):

        print(
            f"\n[{index}/{len(crops)}]"
        )

        try:

            safe_filename = (
                sanitize_filename(
                    crop["filename"]
                )
            )

            local_image = os.path.join(
                TEMP_DOWNLOAD_FOLDER,
                safe_filename
            )

            local_mask = os.path.join(
                TEMP_MASK_FOLDER,
                (
                    crop["cropped_id"]
                    + "_mask.png"
                )
            )

            print(
                "Downloading..."
            )

            download_image(
                crop["storageUrl"],
                local_image
            )

            print(
                "Running segmentation..."
            )

            create_mask(
                local_image,
                local_mask
            )

            print(
                "Uploading mask..."
            )

            (
                mask_key,
                mask_url
            ) = upload_mask(

                local_mask,

                crop[
                    "sessionId"
                ],

                crop[
                    "cropped_id"
                ]
            )

            print(
                "Updating Firestore..."
            )

            mark_processed(

                crop[
                    "doc_ref"
                ],

                mask_key,

                mask_url
            )

            processed_count += 1

            print(
                "SUCCESS"
            )

        except Exception as e:

            print(
                f"FAILED: {e}"
            )

    result = {

        "processed":
            processed_count,

        "total":
            len(crops)

    }

    print(
        "\nFinished."
    )

    print(result)

    return result


# ==========================================================
# 14. LOCAL EXECUTION
# ==========================================================

if __name__ == "__main__":

    process_all_crops()