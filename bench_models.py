import torch
import timm
import gc

BATCH_SIZE = 64
WARMING_UP_BATCHES = 50
BENCHMARK_BATCHES = 100

MODEL_NAMES = {
    'resnet18': 73.3,
    'resnet34': 75.1,
    'resnet50': 79.0,
    'resnext101_32x8d': 85.1,
    'resnext50_32x4d': 83.1,
    'resnext50d_32x4d': 79.7,
    'regnety_032': 82.0,
    'rexnet_200': 81.6,
    'efficientnet_b0': 76.8,
    'efficientnet_b1': 78.8,
    'efficientnet_b2': 80.4,
    'efficientnet_b3': 81.8,
    'tf_efficientnet_b4': 83.0,
    'tf_efficientnet_b5': 83.8,
    'tf_efficientnet_b6': 84.1,
    'tf_efficientnet_b7': 84.9,
    'tf_efficientnet_b8': 85.4,
    'tf_efficientnet_l2_ns': 88.3,
    'efficientnet_lite0': 74.8,
    'tf_efficientnet_lite1': 76.7,
    'tf_efficientnet_lite2': 77.5,
    'tf_efficientnet_lite3': 79.8,
    'tf_efficientnet_lite4': 81.5,
    #'efficientnetv2_rw_t': ,
    'efficientnetv2_rw_s': 83.8,
    'efficientnetv2_rw_m': 84.8,
    'tf_efficientnetv2_s_in21ft1k': 84.9,
    'tf_efficientnetv2_m_in21ft1k': 86.2,
    'tf_efficientnetv2_l_in21ft1k': 86.8,
    #'tf_efficientnetv2_xl_in21ft1k': 87.3,
    'tf_efficientnetv2_b0': 78.4,
    'tf_efficientnetv2_b1': 79.5,
    'tf_efficientnetv2_b2': 80.2,
    'tf_efficientnetv2_b3': 82.0,
    'gernet_s': 75.7,
    'gernet_m': 80.0,
    'gernet_l': 81.3,
}

def bench(model, images):
    torch.cuda.empty_cache()

    # warming up
    for i in range(WARMING_UP_BATCHES):
        model(images)

    # bencmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for i in range(BENCHMARK_BATCHES):
        model(images)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)

    images_per_second = round(BENCHMARK_BATCHES * BATCH_SIZE / elapsed_time_ms*1000)

    return images_per_second

def _main():
    torch.backends.cudnn.benchmark = True

    images = torch.rand(BATCH_SIZE, 3, 332, 332)
    images = images.cuda()

    # fetch model weights
    print("Downloading model weights...")
    for model_name, _ in MODEL_NAMES.items():
        model = timm.create_model(model_name, pretrained=True)
    del model

    print("Batch size", images.size())
    print("Warming up batches", WARMING_UP_BATCHES)
    print("Benchmark batches", BENCHMARK_BATCHES)
    print("GPU name:", torch.cuda.get_device_name(0),"\n")

    print(f"Source   Model name                     Top1 Float32 Float16(AMP)")

    prefix = "timm"
    for model_name, acc in MODEL_NAMES.items():
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            model = timm.create_model(model_name, pretrained=True)

            model = torch.jit.script(model)

            model = model.cuda()

            model.eval()

            # float32
            images_per_second = bench(model, images)

            # Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                images_per_second_amp = bench(model, images)

            del model
            print(f"{prefix:8} {model_name:30} {acc:2.1f}% {images_per_second:5d} {images_per_second_amp:5d} img/s")

if __name__ == '__main__':
    _main()
