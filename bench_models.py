import gc
import sys
import time
import subprocess
import platform

import torch
import timm
import cpuinfo # installation: python -m pip install -U py-cpuinfo
import distro

# ToDo: find usefull environment info here: https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py

BATCH_SIZE = 64
IMAGE_SIZE = 332

IO_WARMING_UP_BATCHES = 50
IO_BENCHMARK_BATCHES = 100

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
    'efficientnetv2_rw_t': 82.3,
    'efficientnetv2_rw_s': 83.8,
    'efficientnetv2_rw_m': 84.8,
    'tf_efficientnetv2_s_in21ft1k': 84.9,
    'tf_efficientnetv2_m_in21ft1k': 86.2,
    'tf_efficientnetv2_l_in21ft1k': 86.8,
    'tf_efficientnetv2_xl_in21ft1k': 87.3,
    'tf_efficientnetv2_b0': 78.4,
    'tf_efficientnetv2_b1': 79.5,
    'tf_efficientnetv2_b2': 80.2,
    'tf_efficientnetv2_b3': 82.0,
    'gernet_s': 75.7,
    'gernet_m': 80.0,
    'gernet_l': 81.3,

    # transformers
    'levit_128s': 76.5,
    'levit_128': 78.5,
    'levit_192': 79.9,
    'levit_256': 81.5,
    'levit_384': 82.6,
    'swin_large_patch4_window12_384': 87.1,
    'vit_small_patch32_224': 76.0,
    'vit_base_patch16_224': 84.5,
    'vit_large_patch16_384': 87.1,
    'beit_base_patch16_224': 85.2,
    'beit_large_patch16_224': 87.5,
    'beit_large_patch16_512': 88.6,
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
    torch.backends.cudnn.allow_tf32 = False
    images_per_second_no_tf32 = bench(model, images)
    torch.backends.cuda.matmul.allow_tf32 = True # revert to defaults
    torch.backends.cudnn.allow_tf32 = True

    # Automatic Mixed Precision
    with torch.cuda.amp.autocast():
        images_per_second_amp = bench(model, images)

    images_per_second_bfloat16 = bench(model.to(dtype=torch.bfloat16), images.bfloat16())

    # revert dtype
    model.float()
    images.float()

    return images_per_second, images_per_second_no_tf32, images_per_second_amp, images_per_second_bfloat16

def bench_io_(images, from_device, to_device, pin, non_blocking):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    batches = []
    sended_bytes = 0
    while sended_bytes < 500*1024*1024:
        batch = torch.randn_like(images, device=from_device)
        sended_bytes += batch.element_size() * batch.nelement()
        if pin:
            batch = batch.pin_memory()
        batches.append(batch)
        break # now usinng only first batch

    start = time.perf_counter()

    if non_blocking:
        for batch in batches:
            batch = batch.to(device=to_device, non_blocking=True)
        torch.cuda.synchronize()
    else:
        for batch in batches:
            batch = batch.to(device=to_device, non_blocking=False)

    end = time.perf_counter()

    del batches, batch

    elapsed_time = end-start

    return elapsed_time, sended_bytes

def bench_io(images, from_device, to_device, pin, non_blocking):
    # warming up
    for _ in range(IO_WARMING_UP_BATCHES):
        bench_io_(images, from_device, to_device, pin, non_blocking)

    # bencmark
    total_elapsed_time = 0
    total_sended_bytes = 0
    for _ in range(IO_BENCHMARK_BATCHES):
        elapsed_time, sended_bytes = bench_io_(images, from_device, to_device, pin, non_blocking)
        total_elapsed_time += elapsed_time
        total_sended_bytes += sended_bytes

    megabytes_per_second = round(total_sended_bytes / total_elapsed_time / 1024 / 1024)

    return megabytes_per_second

def _main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("CPU:", cpuinfo.get_cpu_info()['brand_raw'], "cores:",  cpuinfo.get_cpu_info()['count'], "\n")

    print("Motherboard vendor:", subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/board_vendor'],encoding='utf-8').strip())
    print("Motherboard model name:", subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/board_name'],encoding='utf-8').strip(), " version:", subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/board_version'],encoding='utf-8').strip(), "\n")

    print("GPU:", subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],encoding='utf-8').strip(), " " , subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],encoding='utf-8').strip())
    print("GPU driver version:", subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],encoding='utf-8').strip())
    print("PCI-E max: ", subprocess.check_output(['nvidia-smi', '--query-gpu=pcie.link.gen.max', '--format=csv,noheader,nounits'],encoding='utf-8').strip(), "@", subprocess.check_output(['nvidia-smi', '--query-gpu=pcie.link.width.max', '--format=csv,noheader,nounits'],encoding='utf-8').strip(), "x\n", sep='')

    print("OS:", distro.name(), distro.version(), distro.codename())
    print(platform.platform(),"\n")

    print("Python version:", platform.python_version())

    print("Torch version:", torch.__version__,)
    print("Torch GPU name:", torch.cuda.get_device_name(0))
    print("Torch available memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3), "Gb")
    print("Torch CUDA version:", torch.version.cuda,"\n")

    print("Sending data to GPU benchmark.")
    print("Warming up batches", IO_WARMING_UP_BATCHES)
    print("Benchmark batches", IO_BENCHMARK_BATCHES, "\n")

    print("                                       Pinned                Not pinned")
    print("Batch size                       Blocked Not blocked     Blocked Not blocked")
    # send tensor to GPU benchmark
    from_device = torch.device('cpu')
    to_device = torch.device('cuda:0')
    for tensor_size in [32,64,128,256,512,1024]:

        images = torch.empty(BATCH_SIZE, 3, tensor_size, tensor_size, dtype=torch.float)

        megabytes_per_second_pinned_not_blocking = bench_io(images, from_device, to_device, pin=True, non_blocking=True)
        megabytes_per_second_pinned_blocking = bench_io(images, from_device, to_device, pin=True, non_blocking=False)
        megabytes_per_second_not_pinned_not_blocking = bench_io(images, from_device, to_device, pin=False, non_blocking=True)
        megabytes_per_second_not_pinned_blocking = bench_io(images, from_device, to_device, pin=False, non_blocking=False)

        print(f"{str(images.size()):32} {megabytes_per_second_pinned_blocking:7d} {megabytes_per_second_pinned_not_blocking:11d} {megabytes_per_second_not_pinned_blocking:11d} {megabytes_per_second_not_pinned_not_blocking:11d} MB/s", flush=True)

    print("\n\nGetting data from GPU benchmark.")
    print("Warming up batches", IO_WARMING_UP_BATCHES)
    print("Benchmark batches", IO_BENCHMARK_BATCHES, "\n")

    print("Batch size               Blocked   Not blocked")
    # send tensor to GPU benchmark
    from_device = torch.device('cuda:0')
    to_device = torch.device('cpu')
    for tensor_size in [32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]:

        preds = torch.empty(BATCH_SIZE, tensor_size, dtype=torch.float)

        megabytes_per_second_not_blocking = bench_io(preds, from_device, to_device, pin=False, non_blocking=True)
        megabytes_per_second_blocking = bench_io(preds, from_device, to_device, pin=False, non_blocking=False)

        print(f"{str(preds.size()):24} {megabytes_per_second_blocking:7d} {megabytes_per_second_not_blocking:8d} MB/s", flush=True)


    gc.collect()
    torch.cuda.empty_cache()

    # models benchmark
    images = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    images = images.cuda()

    # round weights
    images.bfloat16()
    images.float()

    # fetch model weights
    print("\n\nDownloading model weights...")
    for model_name, _ in MODEL_NAMES.items():
        try:
            model = timm.create_model(model_name, pretrained=True)
            del model
        except KeyboardInterrupt:
            sys.exit()
        except:
            print("couldn't create a model", model_name)

    print("GPU inference benchmark (forward pass speed).")
    print("Batch size", images.size())
    print("Warming up batches", WARMING_UP_BATCHES)
    print("Benchmark batches", BENCHMARK_BATCHES, "\n")

    print("Source   Model name                     Top1                                   |     torch.jit.script(model)      |      torch.jit.trace(model)      | with torch.cuda.graph()")
    print("                                              Float32 Float32 Float16 BFloat16 | Float32 Float32 Float16 BFloat16 | Float32 Float32 Float16 BFloat16 | Float32 Float32 Float16")
    print("                                                       + TF32   (AMP)          |          + TF32   (AMP)          |          + TF32   (AMP)          |          + TF32   (AMP)")


    prefix = "timm"
    for model_name, acc in MODEL_NAMES.items():
        try:
            with torch.no_grad():
                try:
                    model = timm.create_model(model_name, img_size=IMAGE_SIZE, pretrained=True)
                except:
                    model = timm.create_model(model_name, pretrained=True)

                model = model.cuda()
                model.eval()

                for memory_format in [torch.contiguous_format, torch.channels_last]:

                    model.to(memory_format=memory_format)
                    images.to(memory_format=memory_format)

                    images_per_second, images_per_second_no_tf32, images_per_second_amp, images_per_second_bfloat16 = bench_precision(model, images)

                    jit_scripted_model = torch.jit.script(model)
                    jit_scripted_images_per_second, jit_scripted_images_per_second_no_tf32, jit_scripted_images_per_second_amp, jit_scripted_images_per_second_bfloat16 = bench_precision(jit_scripted_model, images)
                    del jit_scripted_model

                    jit_traced_model = torch.jit.trace(model, (images,))
                    jit_traced_images_per_second, jit_traced_images_per_second_no_tf32, jit_traced_images_per_second_amp, jit_traced_images_per_second_bfloat16 = bench_precision(jit_traced_model, images)
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
                    torch.backends.cudnn.allow_tf32 = False
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
                    torch.backends.cudnn.allow_tf32 = True


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


                    ## bfloat16
                    #model.bfloat16()
                    #images.bfloat16()
                    ## CUDA Graph
                    #g = torch.cuda.CUDAGraph()
                    #
                    ## Warmup before capture
                    #s = torch.cuda.Stream()
                    #s.wait_stream(torch.cuda.current_stream())
                    #with torch.cuda.stream(s):
                    #    for _ in range(3):
                    #        y = model(images)
                    #torch.cuda.current_stream().wait_stream(s)
                    #
                    ## Captures the graph
                    ## To allow capture, automatically sets a side stream as the current stream in the context
                    #with torch.cuda.graph(g):
                    #    y = model(images)
                    #
                    #gc.collect()
                    #torch.cuda.empty_cache()
                    #
                    ## warming up
                    #for i in range(WARMING_UP_BATCHES):
                    #    g.replay()
                    #
                    ## bencmark
                    #torch.cuda.synchronize()
                    #start_event = torch.cuda.Event(enable_timing=True)
                    #end_event = torch.cuda.Event(enable_timing=True)
                    #start_event.record()
                    #
                    #for i in range(BENCHMARK_BATCHES):
                    #    g.replay()
                    #
                    #end_event.record()
                    #torch.cuda.synchronize()  # Wait for the events to be recorded!
                    #elapsed_time_ms = start_event.elapsed_time(end_event)
                    #
                    #graphed_images_per_second_bfloat16 = round(BENCHMARK_BATCHES * BATCH_SIZE / elapsed_time_ms*1000)
                    #
                    ## revert dtype
                    #model.float()
                    #images.float()

                    printed_model_name = model_name
                    if memory_format == torch.channels_last:
                        printed_model_name = "  channels last"

                    print(f"{prefix:8} {printed_model_name:30} {acc:2.1f}% {images_per_second_no_tf32:7d} {images_per_second:7d} {images_per_second_amp:7d} {images_per_second_bfloat16:8d} | {jit_scripted_images_per_second_no_tf32:7d} {jit_scripted_images_per_second:7d} {jit_scripted_images_per_second_amp:7d} {jit_scripted_images_per_second_bfloat16:8d} | {jit_traced_images_per_second_no_tf32:7d} {jit_traced_images_per_second:7d} {jit_traced_images_per_second_amp:7d} {jit_traced_images_per_second_bfloat16:8d} | {graphed_images_per_second_no_tf32:7d} {graphed_images_per_second:7d} {graphed_images_per_second_amp:7d} img/s", flush=True)

                del g, s, y
                del model
        except KeyboardInterrupt:
            sys.exit()
        except:
            pass

if __name__ == '__main__':
    _main()
