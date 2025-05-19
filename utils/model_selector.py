import torch
from torchvision.models import vgg16_bn
import requests
import os
from cleanlab.outlier import OutOfDistribution
import pickle as pkl

from tqdm.notebook import tqdm

# from tqdm import tqdm


def download_file_with_progress(url, output_path, desc="Downloading"):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte
    with open(output_path, "wb") as file, tqdm(
        desc=desc, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))


def model_selector(select_pretrained="rep1"):
    doi = "10.5281/zenodo.15461242"
    record_id = doi.split(".")[-1]
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(metadata_url)
    metadata = response.json()
    files = metadata["files"]

    keys = [[idx, file["key"]] for idx, file in enumerate(files)]
    selected_files = [[key[0], key[1]] for key in keys if select_pretrained in key[1]]
    OOD_file = [[file[0], file[1]] for file in selected_files if "OOD" in file[1]][0]
    weights_file = [
        [file[0], file[1]] for file in selected_files if "model" in file[1]
    ][0]
    OOD_url = files[OOD_file[0]]["links"]["self"]
    weight_url = files[weights_file[0]]["links"]["self"]
    OOD_path = OOD_file[1]
    weights_path = weights_file[1]

    if os.path.exists(weights_path):
        print(f"Weights already downloaded at {weights_path}")
    else:
        print(f"Downloading weights from {doi}...")
        download_file_with_progress(weight_url, weights_path, desc="Weights")
        print(f"Downloaded weights at {weights_path}")

    if os.path.exists(OOD_path):
        print(f"OOD already downloaded at {OOD_path}")
    else:
        print(f"Downloading OOD detector from {doi}...")
        download_file_with_progress(OOD_url, OOD_path, desc="OOD detector")
        print(f"Downloaded OOD detector at {OOD_path}")

    print(f"Loading weights and OOD detector...")

    device = "cuda" if torch.cuda.is_available() == True else "cpu"

    model_torch = vgg16_bn(weights=None)
    model_torch.classifier[6] = torch.nn.Linear(
        in_features=4096, out_features=3, bias=True
    )  # VGG16
    new_feats_list = []
    for layer in model_torch.features:
        new_feats_list.append(layer)
        if isinstance(layer, torch.nn.Conv2d):
            new_feats_list.append(torch.nn.Dropout(p=0.3))
    model_torch.features = torch.nn.Sequential(*new_feats_list)
    model_torch.classifier.add_module("7", torch.nn.Softmax(dim=1))

    state = torch.load(weights_path, weights_only=False)
    model_torch.load_state_dict(state["model_state_dict"])

    print(f"Model weigths successfully loaded...")
    with open(OOD_path, "rb") as f:
        ood_KNN = pkl.load(f)
    return model_torch.to(device), ood_KNN
