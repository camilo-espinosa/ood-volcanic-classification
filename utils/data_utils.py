from scipy import signal
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt


def preprocessing(signal_data, spect_size=(112, 112)):
    """Band-pass filter and Spectrogram calculation"""
    nperseg_ = 100
    if nperseg_ >= len(signal_data):
        raise ValueError(f"Señal muy corta, debe contener al menos {nperseg_} muestras")
    filter_but = signal.butter(
        N=5, Wn=[1.0, 15], btype="bandpass", fs=100, output="sos"
    )
    filtered_signal = signal.sosfiltfilt(filter_but, signal_data)
    f, t, Sxx = signal.spectrogram(
        filtered_signal,
        fs=100,
        window="hamming",
        nperseg=nperseg_,
        noverlap=90,
        nfft=1024,
        mode="complex",
    )
    Sxx = Sxx[0:200, :]  # Spectrogram up to 20Hz
    product = Sxx * np.conj(Sxx)
    temp = np.log10(100 * product.real + 1e-5)
    mini = temp.min()
    maxi = temp.max()
    temp = (temp - mini) / (maxi - mini)
    k = temp.min() + 0.5 * (temp.max() - temp.min())
    temp[temp <= k] = k
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    im = Image.fromarray(np.uint8(temp * 255))
    im = im.resize((spect_size))
    pix = np.array(im)
    img_RGB = getattr(cm, "jet")(pix, bytes=True)[:, :, :3]  # RGB
    return img_RGB, filtered_signal


def obtain_rep(trace_data, type_rep):
    img_RGB, filtered_signal = preprocessing(trace_data, spect_size=(150, 150))
    img_RGB = np.flip(img_RGB, axis=0)
    if type_rep == 1:
        return img_RGB
    elif type_rep == 2:
        fig = plt.figure(figsize=(15, 7.5), dpi=10)
        plt.plot(filtered_signal, color="black", lw=7)
        plt.xlim(0, 12000)  # Plots signal in 120 second- window
        plt.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        sígnal_plot = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()
        image = np.concatenate([img_RGB, sígnal_plot], axis=0)
        return image
    elif type_rep == 3:
        fig = plt.figure(figsize=(15, 7.5), dpi=10)
        plt.magnitude_spectrum(filtered_signal, Fs=100, color="black", lw=7)
        plt.xlim(0, 20)
        plt.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        spectrum_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        spectrum_img = spectrum_img.transpose((1, 0, 2))
        spectrum_img = np.flip(spectrum_img, axis=0)
        plt.close()
        image = np.concatenate([spectrum_img, img_RGB], axis=1)
        return image
    elif type_rep == 4:
        fig = plt.figure(figsize=(22.5, 7.5), dpi=10)
        plt.plot(filtered_signal, color="black", lw=7)
        plt.xlim(0, 12000)
        plt.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        sígnal_plot = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()
        # Spectrum
        fig = plt.figure(figsize=(15, 7.5), dpi=10)
        plt.magnitude_spectrum(filtered_signal, Fs=100, color="black", lw=7)
        plt.xlim(0, 20)
        plt.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        spectrum_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        spectrum_img = spectrum_img.transpose((1, 0, 2))
        spectrum_img = np.flip(spectrum_img, axis=0)
        plt.close()
        image = np.concatenate([spectrum_img, img_RGB], axis=1)
        image = np.concatenate([image, sígnal_plot], axis=0)
        return image


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


transform_ = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class CustomImageDataset(Dataset):
    def __init__(self, data_dir_df, rep_type, transform=transform_):
        self.data_info = data_dir_df
        self.transform = transform
        self.rep_type = rep_type

    def __len__(self):
        return len(self.data_info["event_class"])

    def __getitem__(self, idx):
        label = self.data_info.loc[idx]["event_class"]
        path = self.data_info.loc[idx]["path"]
        event_name = self.data_info.loc[idx]["event_name"]
        trace_data = np.load(path)
        image = obtain_rep(trace_data, self.rep_type)
        image = self.transform(image.copy())
        # return image, torch.Tensor([label])
        return image, torch.Tensor(np.array(label)).type(torch.LongTensor), event_name
