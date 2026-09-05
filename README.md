# llama2 Compute Engine using Vulkan

This is a Vulkan-based implementation of the llama2 language model inference engine.

## How to use

First clone the repo and build the project using CMake. Vulkan SDK should already be installed on your system.

```bash
git clone https://github.com/rashid5925/llama-compute-kernel.git
```

Then, build the project:

```bash
cd llama-compute-kernel
mkdir build
cd build
cmake ..
make
```

Weights for a small model(TinyStories260K) are already included in the `weights` directory. You can run the inference engine with the following command:

```bash
./llama-compute-kernel
```

To run the inference engine with a custom prompt, you can use the `--prompt` argument:

```bash
./llama-compute-kernel --prompt "Once upon a time"
```

Download the model weights and tokenizer files and place them in the `weights` directory. You can use the following command to download the weights:
For TinyStories models download the weights from [here](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) and place them in the `weights` directory.

To run the inference engine with a custom model and tokenizer, you can use the `--model` and `--vocab` arguments:
```bash
./llama-compute-kernel --model "../weights/stories110M.bin" --vocab "../weights/tokenizer.bin" --prompt "Once upon a time" --temp 0.7 --top_p 0.9
```

## Running Other HuggingFace Models that use the Llama 2 architecture

(Will soon be supported in the engine without conversion, but for now you need to convert the weights and tokenizer files to the required format.)

You can load any huggingface models that use the Llama 2 architecture
To run Llama 2 models on this engine, you cannot use the raw files directly from Hugging Face. The weights and tokenizer need to be converted to flat binary format required by the engine using the conversion scripts provided by the `llama2.c` pipeline architecture.

Use this NoteBook by Andrej Karpathy to convert the official Llama 2 model weights and tokenizer to the required format for this engine: [Llama 2 Conversion Notebook](https://colab.research.google.com/github/karpathy/llama2.c/blob/master/run.ipynb#scrollTo=Bf_tj13Np5Bk)

The rest of the process is similar to running the TinyStories models. You can use the `--model` and `--vocab` arguments to specify the converted model and tokenizer files.

Model weights are loaded in raw `float32` precision.

## Performance
MacBook Air M4
16 GB
110M parameter model
FP32
**22 Tokens/sec**

## Architecture & How It Works

This project is a lightweight, pure Vulkan C++ implementation of the Llama 2 Transformer architecture. Model execution is offloaded directly to compute shaders compiled into SPIR-V.

Token ID --> [ Embedding Layer --> Transformer Blocks (Attention + Feed Forward) --> Output Layer ] --> Token ID

### Implementation Details
- Memory Management: Model weights and KV-caches are bound as Vulkan Storage Buffers (VkBuffer) mapped directly to GPU device memory.
- Execution: Commands are dispatched per-layer sequentially via command buffers using explicit Vulkan pipeline barriers to handle GPU buffer memory synchronization.
- Sampling: Softmax scaling, temperature, and Top-$P$ (nucleus) sampling run on the CPU after fetching the final logits back from the GPU.
