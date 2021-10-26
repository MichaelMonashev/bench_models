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
    gc.collect()
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

def bench_precision(model, images):

    # float32
    images_per_second = bench(model, images)

    # disable TensorFloat-32(TF32) on Ampere devices or newer
    torch.backends.cuda.matmul.allow_tf32 = False
    images_per_second_no_tf32 = bench(model, images)
    torch.backends.cuda.matmul.allow_tf32 = True # revert to defaults

    # Automatic Mixed Precision
    with torch.cuda.amp.autocast():
        images_per_second_amp = bench(model, images)

    return images_per_second, images_per_second_no_tf32, images_per_second_amp

def _main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

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

    print("Source   Model name                     Top1                          | torch.jit.script(model) | torch.jit.trace(model)  | with torch.cuda.graph()")
    print("                                              Float32 Float32 Float16 | Float32 Float32 Float16 | Float32 Float32 Float16 | Float32 Float32 Float16")
    print("                                              no TF32           (AMP) | no TF32           (AMP) | no TF32           (AMP) | no TF32           (AMP)")


    prefix = "timm"
    for model_name, acc in MODEL_NAMES.items():
        with torch.no_grad():
            model = timm.create_model(model_name, pretrained=True)

            model = model.cuda()
            model.eval()

            images_per_second, images_per_second_no_tf32, images_per_second_amp = bench_precision(model, images)

            jit_scripted_model = torch.jit.script(model)
            jit_scripted_images_per_second, jit_scripted_images_per_second_no_tf32, jit_scripted_images_per_second_amp = bench_precision(jit_scripted_model, images)
            del jit_scripted_model

            jit_traced_model = torch.jit.trace(model, (images,))
            jit_traced_images_per_second, jit_traced_images_per_second_no_tf32, jit_traced_images_per_second_amp = bench_precision(jit_traced_model, images)
            del jit_traced_model

            # CUDA Graph
            g = torch.cuda.CUDAGraph()

            # Warmup before capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    y = model(images)
            torch.cuda.current_stream().wait_stream(s)

            # Captures the graph
            # To allow capture, automatically sets a side stream as the current stream in the context
            with torch.cuda.graph(g):
                y = model(images)

            gc.collect()
            torch.cuda.empty_cache()

            # warming up
            for i in range(WARMING_UP_BATCHES):
                g.replay()

            # bencmark
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            for i in range(BENCHMARK_BATCHES):
                g.replay()

            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)

            graphed_images_per_second = round(BENCHMARK_BATCHES * BATCH_SIZE / elapsed_time_ms*1000)




            # disable TensorFloat-32(TF32) on Ampere devices or newer
            torch.backends.cuda.matmul.allow_tf32 = False
            # CUDA Graph
            g = torch.cuda.CUDAGraph()

            # Warmup before capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    y = model(images)
            torch.cuda.current_stream().wait_stream(s)

            # Captures the graph
            # To allow capture, automatically sets a side stream as the current stream in the context
            with torch.cuda.graph(g):
                y = model(images)

            gc.collect()
            torch.cuda.empty_cache()

            # warming up
            for i in range(WARMING_UP_BATCHES):
                g.replay()

            # bencmark
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            for i in range(BENCHMARK_BATCHES):
                g.replay()

            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)

            graphed_images_per_second_no_tf32 = round(BENCHMARK_BATCHES * BATCH_SIZE / elapsed_time_ms*1000)
            torch.backends.cuda.matmul.allow_tf32 = True


            # AMP
            # CUDA Graph
            g = torch.cuda.CUDAGraph()

            # Warmup before capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.cuda.amp.autocast():
                        y = model(images)
            torch.cuda.current_stream().wait_stream(s)

            # Captures the graph
            # To allow capture, automatically sets a side stream as the current stream in the context
            with torch.cuda.graph(g):
                with torch.cuda.amp.autocast():
                    y = model(images)

            gc.collect()
            torch.cuda.empty_cache()

            # warming up
            for i in range(WARMING_UP_BATCHES):
                g.replay()

            # bencmark
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            for i in range(BENCHMARK_BATCHES):
                g.replay()

            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)

            graphed_images_per_second_amp = round(BENCHMARK_BATCHES * BATCH_SIZE / elapsed_time_ms*1000)

            del g, s, y
            del model

            print(f"{prefix:8} {model_name:30} {acc:2.1f}% {images_per_second_no_tf32:7d} {images_per_second:7d} {images_per_second_amp:7d} | {jit_scripted_images_per_second_no_tf32:7d} {jit_scripted_images_per_second:7d} {jit_scripted_images_per_second_amp:7d} | {jit_traced_images_per_second_no_tf32:7d} {jit_traced_images_per_second:7d} {jit_traced_images_per_second_amp:7d} | {graphed_images_per_second_no_tf32:7d} {graphed_images_per_second:7d} {graphed_images_per_second_amp:7d} img/s")

if __name__ == '__main__':
    _main()
