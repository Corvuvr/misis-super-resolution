import os
from pathlib import Path
from typing import Generator
import tensorflow as tf

import json
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import ticker
from matplotlib import pyplot as plt

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

def Log(allowed: bool):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(kwargs)
            model_name = kwargs["UPSCALED_PATH"].split('/')[0]
            ds_name = Path(kwargs["UPSCALED_PATH"]).stem
            METRICS_JSON = f"{model_name}/results/{ds_name}/metrics.json"
            result = function(*args, **kwargs)
            if allowed:
                with open(METRICS_JSON, 'w', encoding='utf-8') as f:
                    json.dump(list(result), f, ensure_ascii=False, indent=4)
                    print(f"LOG: saved metrics to: {METRICS_JSON}")
            return result
        return wrapper
    return decorator

def upscale_image(image, scale: int = 2):
    """
        Scales up image.
        Args:
            image: 3D tensor of preprocessed image.
    """
    return tf.convert_to_tensor(np.array(image).repeat(scale, axis=1).repeat(scale, axis=2))

def downscale_image(image, scale: int = 2):
    """
        Scales down images using bicubic downsampling.
        Args:
            image: 3D or 4D tensor of preprocessed image
    """
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = tf.squeeze(
        tf.cast(
            tf.clip_by_value(image, 0, 255), tf.uint8))

    lr_image = np.asarray(
        Image.fromarray(image.numpy()).resize(
            [image_size[0] // scale, image_size[1] // scale],
            Image.BICUBIC
        )
    )
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

def plot_image(image, title=""):
    """
        Plots images from image tensors.
        Args:
        image: 3D image tensor. [height, width, channels].
        title: Title to display in the plot.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

def plotStacked(
        images, 
        save_path: str = ""
):
    """ Creates stacked plots from given images
        Args:
        images: list of pairs (image, description).
        save_path: save plot in file with filepath - if not specified, plot won't be saved. 
    """
    plt.rcParams['figure.figsize'] = [5*len(images), 10]
    fig, axes = plt.subplots(1, len(images))
    i: int = 1
    for image, desc in images:
        fig.tight_layout()
        plt.subplot(130 + i)
        plot_image(tf.squeeze(image), desc)
        i += 1
    if save_path != "":
        plt.savefig(f"study/artifacts/plotStacked.png", bbox_inches="tight")
    plt.clf

def plotFedData(data, columns, filename: str, clip: bool = True):
    plot_info = filename.split("_")

    metric = plot_info[1]
    model_name = plot_info[0].split('/')[-1]
    match metric:
        case "PSNR":
            title = f"PSNR модели {model_name}"
            ylabel = "PSNR"
            xlabel = "Набор данных"
        case "SSIM":
            title = f"SSIM модели {model_name}"
            ylabel = "SSIM"
            xlabel = "Набор данных"
        case "TIME":
            title = f"Производительность модели {model_name}"
            ylabel = "Время (секунды)"
            xlabel = "Набор данных"
        case _:
            ...

    meta = np.array(data)
    meta = meta.transpose()
    avg = tuple(np.average(el) for el in meta)
    med = tuple(np.median(el) for el in meta)

    dataframe = pd.DataFrame(data, columns=columns)
    plot = dataframe.plot(kind="box")
    
    # Set max interval on x axis
    plot.axes.xaxis.set_major_locator(ticker.MaxNLocator(len(columns)))

    y_max = max(max(x) for x in data)
    if clip:
        y_max = int(np.ceil(y_max))
    plot.axes.set_ylim(
        bottom=0,
        top=y_max
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{filename}.png")
    print(f"LOG: Saved Plot in: {filename}.png")
    plt.clf()

    return (avg, med)

@Log(True)
def getMetrics(
    UPSCALED_PATH: str,
    VALID_HR_PATH: str = "",
    upscaled_suffix: str = "", 
    downscale_hr: int = 1
) -> Generator[tuple, None, None]:
    """ Yields pairs of PSNR and SSIM for images from given paths 
        Args:
        UPSCALED_PATH: folder to upscaled images.
        VALID_HR_PATH: folder to ground truth images.
        downscale_hr:  if you need to downscale ground truth.
    """
    # Figure out VALID path
    dataset_name = Path(UPSCALED_PATH).stem
    print(f"{str(dataset_name)}")
    if VALID_HR_PATH == "":
        VALID_HR_PATH = f"datasets/{dataset_name}/{dataset_name}"
    
    print(f"Comparing images from: {UPSCALED_PATH} and {VALID_HR_PATH}")
    psnrs = []
    ssims = []
    for filepath in Path(f"./{VALID_HR_PATH}/").rglob("*"):
        
        if filepath.suffix.lower() != '.png':
            continue
        
        IMG_NAME = filepath.stem
        try:
            SR_PATH = str(Path(f"./{UPSCALED_PATH}/{IMG_NAME}{upscaled_suffix}_out.png"))
            HR_PATH = str(Path(f"./{VALID_HR_PATH}/{IMG_NAME}.png"))
        except Exception as e:
            print(e)
            break

        sr_image = tf.image.decode_png(tf.io.read_file(SR_PATH), channels=3)
        hr_image = tf.image.decode_png(tf.io.read_file(HR_PATH), channels=3)
        
        hr_image = tf.squeeze(downscale_image(tf.squeeze(hr_image), scale=downscale_hr))
        sr_image = tf.squeeze(downscale_image(tf.squeeze(sr_image), scale=1))

        # Crop in case
        if hr_image.shape != sr_image.shape:
            # print(f"Shapes of inputs doesnt match: GT={hr_image.shape}, SR={sr_image.shape}")
            min_shape_h = min(hr_image.shape[0], sr_image.shape[0])
            min_shape_w = min(hr_image.shape[1], sr_image.shape[1])
            hr_image = tf.image.crop_to_bounding_box(
                hr_image, 0, 0,
                target_height=min_shape_h, 
                target_width=min_shape_w
            )
            sr_image = tf.image.crop_to_bounding_box(
                sr_image, 0, 0,
                target_height=min_shape_h, 
                target_width=min_shape_w
            )
        
        hr_image = tf.expand_dims(
            (tf.image.rgb_to_yuv(hr_image / 255) * 255)[:,:,0], 2
        )
        sr_image = tf.expand_dims(
            (tf.image.rgb_to_yuv(sr_image / 255) * 255)[:,:,0], 2
        )


        # Calculating PSNR wrt Original Image
        psnr = tf.image.psnr(
            tf.clip_by_value(sr_image, 0, 255),
            tf.clip_by_value(hr_image, 0, 255), 
            max_val=255
        )
        # Calculating SSIM wrt Original Image
        ssim = tf.image.ssim(
            tf.clip_by_value(sr_image, 0, 255),
            tf.clip_by_value(hr_image, 0, 255), 
            max_val=255
        )
        psnrs.append(psnr)
        ssims.append(ssim)
        yield tuple((float(psnr), float(ssim)))
    
    print(f"Avg PSNR: {sum(psnrs)/len(psnrs)}")
    print(f"Avg SSIM: {sum(ssims)/len(ssims)}")

def getComplexity(datafolder: str):
    MODEL_NAME = datafolder.split('/')[0]
    shapes: list = []
    timings: list = []
    for filepath in Path(datafolder).rglob("**/data.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            df = json.load(f)
        for record in df:
            shapes.append(record[1]*record[2])
            timings.append(record[0])
    
    np_shapes  = np.array(shapes)
    np_timings = np.array(timings)

    ids = np.argsort(np_shapes)
    np_shapes  = np_shapes[ids]
    np_timings = np_timings[ids]
    plt.title(f"Скорость обработки RGB-изображения, {MODEL_NAME}")
    plt.plot(np_shapes, np_timings)
    plt.xlabel("Кол-во пикселей")
    plt.ylabel("Время исчисления (секунды)")
    plt.savefig(f"study/artifacts/{MODEL_NAME}_PERFORMANCE.png")
    plt.clf()
    
def makeTable(filepath: dict):

    with open(filepath, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)

    dsts    = ['DIV2K', 'General100', 'BSDS100', 'set14', 'urban100']
    models  = ['Real-ESRGAN', 'RT4KSR']
    metrics = ['ssim', 'psnr', 'time']
    stats   = ['avg' , 'med']

    tbl = []
    for i in range(len(dsts)):
        tbl.append([''] * len(models) * len(metrics) * len(stats))

    for model, metric_info in data.items():
        for metric, stat_info in metric_info.items():
            for stat, dataset_info in stat_info.items():
                for dataset, value in dataset_info.items():
                    row_id = dsts.index(dataset)
                    col_id = \
                        (metrics.index(metric) * len(stats)) + \
                        (models.index(model) * len(metrics) * len(stats)) + \
                        stats.index(stat)
                    # print(f"Dataset: {dataset}, Value: {value}")
                    # print(f"Writing to table: {row_id} {col_id}")
                    tbl[row_id][col_id] = value
    pd.DataFrame(
        np.array(tbl)).to_csv(str(f"{Path(filepath).parent}/{Path(filepath).stem}") + '.csv'
    )

def main():

    models = ["Real-ESRGAN", "RT4KSR"]
    meta: dict = dict()
    for model in models:
        
        print(model)
        CALC: bool = True
        if CALC:
            # DIV2K
            getMetrics(
                UPSCALED_PATH=f"{model}/results/DIV2K", 
                VALID_HR_PATH="datasets/DIV2K/DIV2K",
                downscale_hr=1
                #upscaled_suffix="x4"
            )
            # set14
            getMetrics(
                UPSCALED_PATH=f"{model}/results/set14",
                VALID_HR_PATH="datasets/set14/set14/set14",
            )
            # General100
            getMetrics(
                UPSCALED_PATH=f"{model}/results/General100"    
            )
            # BSDS100
            getMetrics(
                UPSCALED_PATH=f"{model}/results/BSDS100"
            )
            # urban100
            getMetrics(
                UPSCALED_PATH=f"{model}/results/urban100"
            )
        
        psnr: list = []
        ssim: list = []
        time: list = []
        
        labels = ["DIV2K", "General100", "BSDS100", "set14", "urban100"]
        
        for dataset in labels:
            
            # SSIM & PSNR
            METRICS_JSON = f"{model}/results/{dataset}/metrics.json"
            with open(METRICS_JSON, 'r', encoding='utf-8') as f:
                df = json.load(f)
            metrics = list(zip(*df))
            psnr.append(metrics[0])
            ssim.append(metrics[1])

            # TIMINGS
            DATA_JSON = f"{model}/results/{dataset}/data.json"
            with open(DATA_JSON, 'r', encoding='utf-8') as f:
                df = json.load(f)
            milliscnds = list(zip(*df))
            time.append(milliscnds[0])

        # Plot
        avg_psnr, med_psnr = plotFedData(
            tuple(zip(*psnr)), labels, f"study/artifacts/{model.upper()}_PSNR_BOXPLOT")
        avg_ssim, med_ssim = plotFedData(
            tuple(zip(*ssim)), labels, f"study/artifacts/{model.upper()}_SSIM_BOXPLOT")
        avg_time, med_time = plotFedData(
            tuple(zip(*time)), labels, f"study/artifacts/{model.upper()}_TIME_BOXPLOT", clip=False)

        meta.update({
            model: {
                "psnr": {
                    "avg": dict(zip(labels, avg_psnr)), "med": dict(zip(labels, med_psnr))  
                },
                "ssim": {
                    "avg": dict(zip(labels, avg_ssim)), "med": dict(zip(labels, med_ssim)) 
                },
                "time": {
                    "avg": dict(zip(labels, avg_time)), "med": dict(zip(labels, med_time))
                }
            }
        })

        # Performance
        getComplexity(datafolder=f'{model}/results/')

    META_JSON = f"study/artifacts/meta.json"
    with open(META_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    makeTable(META_JSON)
        
    
if __name__=="__main__":
    main()