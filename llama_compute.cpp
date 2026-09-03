#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <assert.h>
#include <vulkan/vulkan.hpp>

struct TokenInfo {
    float score;
    std::string text;
};

// Structure matching the binary header layout of llama2.c models
struct Config {
    int dim;          // Transformer dimension
    int hidden_dim;   // For FFN layers
    int n_layers;     // Number of layers
    int n_heads;      // Number of query heads
    int n_kv_heads;   // Number of key/value heads
    int vocab_size;   // Vocabulary size
    int seq_len;      // Max sequence length
};

struct TransformerWeights {
    // 1. Token Embeddings
    float* token_embedding_table; // (vocab_size, dim)

    // 2. Layer Weights (Contiguous across all layers)
    float* rms_att_weight; // (n_layers, dim)
    float* wq;             // Query weights (n_layers, dim, n_heads * head_size)
    float* wk;             // Key weights   (n_layers, dim, n_kv_heads * head_size)
    float* wv;             // Value weights (n_layers, dim, n_kv_heads * head_size)
    float* wo;             // (n_layers, n_heads * head_size, dim)
    float* rms_ffn_weight; // (n_layers, dim)
    float* w1;             // (n_layers, dim, hidden_dim)
    float* w2;             // (n_layers, hidden_dim, dim)
    float* w3;             // (n_layers, dim, hidden_dim)

    // 3. Final Norm & Positional Embeddings
    float* rms_final_weight; // (dim)
    float* freq_cis_real;    // (seq_len, head_size / 2)
    float* freq_cis_imag;    // (seq_len, head_size / 2)

    // 4. Output Classifier
    float* wcls;             // Shared with token_embedding_table
};

struct PushConstants {
    uint32_t layer;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t pos;
    uint32_t weight_offset;
    uint32_t src_offset;   
    uint32_t dst_offset;
    uint32_t src2_offset;    // second input: hb2 for swiglu, att for attention
    uint32_t head_size;
    uint32_t n_heads;
    uint32_t kv_dim;
    uint32_t seq_len;
    uint32_t key_base;       // key_cache + l*seq_len*kv_dim
    uint32_t value_base;
};

struct StateLayout {
    uint32_t x, xb, xb2, q, k, v, att, hb, hb2, logits, key_cache, value_cache;
    uint32_t total_floats;
};

StateLayout make_layout(const Config& c) {
    uint32_t head_size = c.dim / c.n_heads;
    uint32_t kv_dim    = c.n_kv_heads * head_size;
    StateLayout L{};
    uint32_t o = 0;
    L.x      = o; o += c.dim;
    L.xb     = o; o += c.dim;
    L.xb2    = o; o += c.dim;
    L.q      = o; o += c.dim;
    L.k      = o; o += kv_dim;
    L.v      = o; o += kv_dim;
    L.att    = o; o += c.n_heads * c.seq_len;
    L.hb     = o; o += c.hidden_dim;
    L.hb2    = o; o += c.hidden_dim;
    L.logits = o; o += c.vocab_size;
    L.key_cache   = o; o += c.n_layers * c.seq_len * kv_dim;
    L.value_cache = o; o += c.n_layers * c.seq_len * kv_dim;
    L.total_floats = o;
    return L;
}

struct VulkanComputePipeline {
    VkDescriptorSetLayout setLayout0;
    VkDescriptorSetLayout setLayout1;
    VkPipelineLayout pipelineLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet set0;
    VkDescriptorSet set1;
    
    VkPipeline rmsNormPipeline;
    VkPipeline matmulPipeline;
    VkPipeline attentionPipeline;
    VkPipeline residualAddPipeline;
    VkPipeline swigluPipeline;
    VkPipeline ropePipeline;
};

std::vector<TokenInfo> read_vocab(std::ifstream& file) {
    std::vector<TokenInfo> vocabulary;
    
    // 1. Read the max token length header (4 bytes)
    int max_token_length;
    file.read(reinterpret_cast<char*>(&max_token_length), sizeof(max_token_length));
    std::cout << "Max Token Length: " << max_token_length << "\n";

    int token_id = 0;

    // 2. Loop to read up to 512 tokens
    while (file.peek() != EOF && token_id < 512) {
        float score;
        int token_length;

        // Read the token score (4-byte float)
        if (!file.read(reinterpret_cast<char*>(&score), sizeof(score))) break;

        // Read the token string length (4-byte int)
        if (!file.read(reinterpret_cast<char*>(&token_length), sizeof(token_length))) break;

        // Read the token string data
        std::string token_str(token_length, ' ');
        if (!file.read(&token_str[0], token_length)) break;

        // Store it
        vocabulary.push_back({score, token_str});
        
        token_id++;
    }

    std::cout << "Successfully parsed " << vocabulary.size() << " tokens.\n";
    std::cout << "------------------------------------\n";

    return vocabulary;
}

static bool starts_with(const std::string& str,
                        size_t pos,
                        const std::string& prefix)
{
    return pos + prefix.size() <= str.size() &&
           str.compare(pos, prefix.size(), prefix) == 0;
}

std::vector<int> tokenize(const std::string& text, const std::vector<TokenInfo>& vocabulary) {
    std::vector<int> tokens;

    if (text.empty())
        return tokens;

    /*
     * ------------------------------------------------------------------
     * 1. Build vocabulary lookup
     * ------------------------------------------------------------------
     */
    std::unordered_map<std::string, int> vocab_lookup;
    vocab_lookup.reserve(vocabulary.size());

    for (int i = 0; i < static_cast<int>(vocabulary.size()); ++i) {
        vocab_lookup[vocabulary[i].text] = i;
    }

    /*
     * ------------------------------------------------------------------
     * 2. Convert input into initial byte tokens
     * ------------------------------------------------------------------
    */
    std::vector<int> pieces;

    for (unsigned char byte : text) {
        std::string byte_string(1, static_cast<char>(byte));

        auto it = vocab_lookup.find(byte_string);

        if (it != vocab_lookup.end()) {
            pieces.push_back(it->second);
        } else {
            int byte_token = -1;

            for (int id = 0;
                 id < static_cast<int>(vocabulary.size());
                 ++id)
            {
                const std::string& token = vocabulary[id].text;

                if (token.size() == 1 &&
                    static_cast<unsigned char>(token[0]) == byte)
                {
                    byte_token = id;
                    break;
                }
            }

            if (byte_token == -1) {
                return {};
            }

            pieces.push_back(byte_token);
        }
    }

    /*
     * ------------------------------------------------------------------
     * 3. BPE merge loop
     * ------------------------------------------------------------------
     *
     * At every iteration:
     *
     *     pair = token[i] + token[i+1]
     *
     * is looked up in the vocabulary.
     *
     * Among all possible pairs, choose the pair with the HIGHEST
     * vocabulary score.
     */
    while (pieces.size() >= 2) {

        float best_score = -1e10f;
        int best_token_id = -1;
        size_t best_position = 0;

        for (size_t i = 0; i + 1 < pieces.size(); ++i) {

            int token_a = pieces[i];
            int token_b = pieces[i + 1];

            const std::string& a = vocabulary[token_a].text;
            const std::string& b = vocabulary[token_b].text;

            std::string merged = a + b;

            auto it = vocab_lookup.find(merged);

            if (it == vocab_lookup.end())
                continue;

            int merged_id = it->second;

            float score = vocabulary[merged_id].score;

            /*
             * llama2.c selects the highest-scoring merge.
             */
            if (score > best_score) {
                best_score = score;
                best_token_id = merged_id;
                best_position = i;
            }
        }

        /*
         * No more merges possible.
         */
        if (best_token_id == -1)
            break;

        /*
        Replace:
         
             A B
        
        with:
        
             AB
        */
        pieces[best_position] = best_token_id;

        pieces.erase(pieces.begin() + best_position + 1);
    }

    return pieces;
}

bool load_model_weights(std::ifstream& file, Config& config, TransformerWeights& weights, std::vector<float>& weight_buffer) {
    // Read Header
    file.read(reinterpret_cast<char*>(&config), sizeof(Config));

    int head_size = config.dim / config.n_heads;

    std::cout << "--- Model Architecture Config ---\n";
    std::cout << "Embedding Dim (dim) : " << config.dim << "\n";
    std::cout << "Hidden Dim          : " << config.hidden_dim << "\n";
    std::cout << "Layers              : " << config.n_layers << "\n";
    std::cout << "Heads               : " << config.n_heads << "\n";
    std::cout << "KV Heads            : " << config.n_kv_heads << "\n";
    std::cout << "Vocab Size          : " << config.vocab_size << "\n";
    std::cout << "Seq Length          : " << config.seq_len << "\n";
    std::cout << "----------------------------------\n";

    // Calculate total float elements in file
    size_t total_floats = 0;
    total_floats += config.vocab_size * config.dim; // token_embedding_table
    total_floats += config.n_layers * config.dim;   // rms_att_weight
    total_floats += config.n_layers * config.dim * (config.n_heads * head_size); // wq
    total_floats += config.n_layers * config.dim * (config.n_kv_heads * head_size); // wk
    total_floats += config.n_layers * config.dim * (config.n_kv_heads * head_size); // wv
    total_floats += config.n_layers * (config.n_heads * head_size) * config.dim; // wo
    total_floats += config.n_layers * config.dim;   // rms_ffn_weight
    total_floats += config.n_layers * config.dim * config.hidden_dim; // w1
    total_floats += config.n_layers * config.hidden_dim * config.dim; // w2
    total_floats += config.n_layers * config.dim * config.hidden_dim; // w3
    total_floats += config.dim; // rms_final_weight
    total_floats += config.seq_len * (head_size / 2); // freq_cis_real
    total_floats += config.seq_len * (head_size / 2); // freq_cis_imag

    // Read all binary float weights in a single I/O pass
    weight_buffer.resize(total_floats);
    file.read(reinterpret_cast<char*>(weight_buffer.data()), total_floats * sizeof(float));

    // Map struct pointers into contiguous memory buffer
    float* ptr = weight_buffer.data();

    weights.token_embedding_table = ptr; ptr += config.vocab_size * config.dim;
    weights.rms_att_weight        = ptr; ptr += config.n_layers * config.dim;
    weights.wq                    = ptr; ptr += config.n_layers * config.dim * (config.n_heads * head_size);
    weights.wk                    = ptr; ptr += config.n_layers * config.dim * (config.n_kv_heads * head_size);
    weights.wv                    = ptr; ptr += config.n_layers * config.dim * (config.n_kv_heads * head_size);
    weights.wo                    = ptr; ptr += config.n_layers * (config.n_heads * head_size) * config.dim;
    weights.rms_ffn_weight        = ptr; ptr += config.n_layers * config.dim;
    weights.w1                    = ptr; ptr += config.n_layers * config.dim * config.hidden_dim;
    weights.w2                    = ptr; ptr += config.n_layers * config.hidden_dim * config.dim;
    weights.w3                    = ptr; ptr += config.n_layers * config.dim * config.hidden_dim;
    weights.rms_final_weight      = ptr; ptr += config.dim;
    weights.freq_cis_real         = ptr; ptr += config.seq_len * (head_size / 2);
    weights.freq_cis_imag         = ptr; ptr += config.seq_len * (head_size / 2);

    // Tied weights check: point classifier weights back to embeddings
    weights.wcls = weights.token_embedding_table;

    return true;
}

std::string get_string_from_token_ids(const std::vector<TokenInfo>& vocabulary, std::vector<int> token_ids) {
    std::string output = "";
    for (int id : token_ids) {
        output += vocabulary[id].text;
    }
    return output;
}

void initialize_vulkan(VkInstance &instance, VkApplicationInfo &appInfo, VkInstanceCreateInfo &createInfo, VkPhysicalDevice &physicalDevice, VkPhysicalDeviceProperties &deviceProperties, VkPhysicalDeviceFeatures &deviceFeatures, uint32_t *queueFamilyIndex, VkResult &res) {
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "llama Compute Kernel";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = layers;

    res = vkCreateInstance(&createInfo, nullptr, &instance);
    assert(res == VK_SUCCESS);

    uint32_t physicalDevicesCount = 0;
    res = vkEnumeratePhysicalDevices(instance, &physicalDevicesCount, nullptr);
    assert(res == VK_SUCCESS);
    
    std::vector<VkPhysicalDevice> physicalDevices(physicalDevicesCount);
    res = vkEnumeratePhysicalDevices(instance, &physicalDevicesCount, physicalDevices.data());
    std::cout << "Number of physical devices: " << physicalDevicesCount << std::endl;
    assert(res == VK_SUCCESS);

    physicalDevice = physicalDevices[0];
    
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

    std::cout << "Physical Device apiVersion: " << deviceProperties.apiVersion << std::endl;
    std::cout << "Physical Device driverVersion: " << deviceProperties.driverVersion << std::endl;
    std::cout << "Physical Device deviceName: " << deviceProperties.deviceName << std::endl;
    std::cout << "Physical Device VkPhysicalDeviceType: " << deviceProperties.deviceType << std::endl;

    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    std::cout << "Queue Family Count: " << queueFamilyPropertyCount << std::endl;
    std::cout << "------------------------------------\n";
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());
    for (uint32_t i = 0; i < queueFamilyPropertyCount; ++i) {
        if (
            queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT &&
            queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT
        ) {
            *queueFamilyIndex = i;
            break;
        }
    }
}

void create_device(VkDeviceQueueCreateInfo& queueCreateInfo, VkDeviceCreateInfo& deviceCreateInfo, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, VkPhysicalDeviceFeatures deviceFeatures, VkDevice& device, VkResult res) {
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.00f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = 0;
    deviceCreateInfo.ppEnabledExtensionNames = nullptr;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    res = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    assert(res == VK_SUCCESS);
}

void create_buffer(VkBufferCreateInfo& bufferCreateInfo, VkBufferUsageFlags bufferUsage, VkBuffer& stagingBuffer, VkDevice device, VkDeviceSize bufferSize, VkResult res) {
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = bufferUsage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    res = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &stagingBuffer);
    assert(res == VK_SUCCESS);
}

void create_memory(VkDevice device, VkPhysicalDevice& physicalDevice, VkBuffer &stagingBuffer, VkDeviceMemory& stagingMemory, VkResult res) {
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memoryRequirements);

    VkMemoryAllocateInfo localMemoryAllocateInfo{};
    localMemoryAllocateInfo.allocationSize = memoryRequirements.size;
    localMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    localMemoryAllocateInfo.memoryTypeIndex = -1;
    VkPhysicalDeviceMemoryProperties localMemoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &localMemoryProperties);
    VkMemoryPropertyFlags localRequired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < localMemoryProperties.memoryTypeCount; ++i) {
        if (
            (memoryRequirements.memoryTypeBits & (1 << i)) &&
            ((localMemoryProperties.memoryTypes[i].propertyFlags & localRequired) == localRequired)
        ) {
            localMemoryAllocateInfo.memoryTypeIndex = i;
            break;
        }
    }
    assert(localMemoryAllocateInfo.memoryTypeIndex != UINT32_MAX);

    res = vkAllocateMemory(device, &localMemoryAllocateInfo, nullptr, &stagingMemory);
    assert(res == VK_SUCCESS);
}

std::vector<uint32_t> read_spirv_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& spirvCode, VkResult& res) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();

    VkShaderModule shaderModule;
    res = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    assert(res == VK_SUCCESS);
    return shaderModule;
}

VkPipeline create_pipeline (VkDevice device, VulkanComputePipeline& pipelineState, const std::vector<uint32_t>& code, VkResult& res) {
    VkShaderModule mod = create_shader_module(device, code, res);
    VkComputePipelineCreateInfo computeInfo{};
    computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeInfo.stage.module = mod;
    computeInfo.stage.pName = "main";
    computeInfo.layout = pipelineState.pipelineLayout;

    VkPipeline pipe;
    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &pipe);
    assert(res == VK_SUCCESS);
    vkDestroyShaderModule(device, mod, nullptr);
    return pipe;
};

void vulkan_pipeline(
    VkDevice device, 
    VkBuffer weightBuffer, size_t weightSizeBytes,
    VkBuffer buffer_activation,
    size_t stateSizeBytes,
    const std::vector<uint32_t>& rmsSpirv,
    const std::vector<uint32_t>& matmulSpirv,
    const std::vector<uint32_t>& ropeSpirv,
    const std::vector<uint32_t>& attentionSpirv,
    const std::vector<uint32_t>& residualaddSpirv,
    const std::vector<uint32_t>& swigluSpirv,
    VulkanComputePipeline& pipelineState,
    VkResult &res
) {
    // Create Descriptor Set Layout 0 for weight and caches
    VkDescriptorSetLayoutBinding weightBinding{};
    weightBinding.binding = 0;
    weightBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    weightBinding.descriptorCount = 1;
    weightBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo0{};
    layoutInfo0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo0.bindingCount = 1;
    layoutInfo0.pBindings = &weightBinding;
    vkCreateDescriptorSetLayout(device, &layoutInfo0, nullptr, &pipelineState.setLayout0);

    // Create Descriptor Set Layout 1 for actiations/sate
    std::vector<VkDescriptorSetLayoutBinding> stateBinding(1);
    stateBinding[0].binding = 0;
    stateBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    stateBinding[0].descriptorCount = 1;
    stateBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo1{};
    layoutInfo1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo1.bindingCount = 1;
    layoutInfo1.pBindings = stateBinding.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo1, nullptr, &pipelineState.setLayout1);

    // Create Pipeline Layout with push constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkDescriptorSetLayout layouts[] = { pipelineState.setLayout0, pipelineState.setLayout1 };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineState.pipelineLayout);

    // Descriptor Pool and Allocation
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &pipelineState.descriptorPool);

    VkDescriptorSetAllocateInfo allocInfo0{};
    allocInfo0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo0.descriptorPool = pipelineState.descriptorPool;
    allocInfo0.descriptorSetCount = 1;
    allocInfo0.pSetLayouts = &pipelineState.setLayout0;
    vkAllocateDescriptorSets(device, &allocInfo0, &pipelineState.set0);

    VkDescriptorSetAllocateInfo allocInfo1{};
    allocInfo1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo1.descriptorPool = pipelineState.descriptorPool;
    allocInfo1.descriptorSetCount = 1;
    allocInfo1.pSetLayouts = &pipelineState.setLayout1;
    vkAllocateDescriptorSets(device, &allocInfo1, &pipelineState.set1);

    // Write Descriptor Set 0 for weight buffer
    VkDescriptorBufferInfo weightBufInfo{ weightBuffer, 0, weightSizeBytes };
    VkWriteDescriptorSet writeSet0{};
    writeSet0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet0.dstSet = pipelineState.set0;
    writeSet0.dstBinding = 0;
    writeSet0.descriptorCount = 1;
    writeSet0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeSet0.pBufferInfo = &weightBufInfo;
    vkUpdateDescriptorSets(device, 1, &writeSet0, 0, nullptr);

    // Write Descriptor Set 1 for state buffers
    VkDescriptorBufferInfo stateBufInfo = { buffer_activation, 0, stateSizeBytes };

    VkWriteDescriptorSet writeSet = {};
    writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.dstSet = pipelineState.set1;
    writeSet.dstBinding = 0;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeSet.pBufferInfo = &stateBufInfo;
    vkUpdateDescriptorSets(device, 1, &writeSet, 0, nullptr);

    // Create Compute Pipelines 
    pipelineState.rmsNormPipeline = create_pipeline(device, pipelineState, rmsSpirv, res);
    pipelineState.matmulPipeline  = create_pipeline(device, pipelineState, matmulSpirv, res);
    pipelineState.ropePipeline = create_pipeline(device, pipelineState, ropeSpirv, res);
    pipelineState.attentionPipeline = create_pipeline(device, pipelineState, attentionSpirv, res);
    pipelineState.residualAddPipeline = create_pipeline(device, pipelineState, residualaddSpirv, res);
    pipelineState.swigluPipeline = create_pipeline(device, pipelineState, swigluSpirv, res);
}

void forward_pass(
    VkCommandBuffer cmd,
    VulkanComputePipeline& pipe,
    const Config& config,
    const StateLayout& layout,
    uint32_t current_pos
) {
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipe.pipelineLayout, 0, 1, &pipe.set0, 0, nullptr
    );
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipe.pipelineLayout, 1, 1, &pipe.set1, 0, nullptr
    );
    uint32_t head_size = config.dim / config.n_heads;
    uint32_t kv_dim = config.n_kv_heads * head_size;   

    uint32_t rms_att_base = config.vocab_size * config.dim;
    uint32_t wq_base = rms_att_base + config.n_layers * config.dim;
    uint32_t wk_base = wq_base + config.n_layers * config.dim * config.dim;      // wq is dim x dim
    uint32_t wv_base = wk_base + config.n_layers * config.dim * kv_dim;         // wk is dim x kv_dim
    uint32_t wo_base = wv_base + config.n_layers * config.dim * kv_dim;         // wv is dim x kv_dim
    uint32_t rms_ffn_base = wo_base + config.n_layers * config.dim * config.dim;
    uint32_t w1_base = rms_ffn_base + config.n_layers * config.dim;
    uint32_t w2_base = w1_base + config.n_layers * config.dim * config.hidden_dim;
    uint32_t w3_base = w2_base + config.n_layers * config.dim * config.hidden_dim;
    uint32_t rms_final_base = w3_base + config.n_layers * config.dim * config.hidden_dim;
    uint32_t freq_cis_real_base = rms_final_base + config.dim;
    uint32_t freq_cis_imag_base = freq_cis_real_base + config.seq_len * (head_size / 2);
    
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };

    for (uint32_t l = 0; l < config.n_layers; ++l) {

        // 1. Dispatch RMSNorm (Attention Norm)
        const uint32_t rms_weight_offset = rms_att_base + (l * config.dim);
        PushConstants pc_rms{ 
            l, 
            static_cast<uint32_t>(config.dim), 
            0, 
            current_pos, 
            rms_weight_offset,
            layout.x,
            layout.xb
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_rms);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.rmsNormPipeline);
        vkCmdDispatch(cmd, 1, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 2. Dispatch MatMul (Query Projection Q)
        const uint32_t q_weight_offset = wq_base + (l * config.dim * config.dim);
        PushConstants pc_q{ 
            l, 
            static_cast<uint32_t>(config.dim),      // in_dim = dim
            static_cast<uint32_t>(config.dim),      // out_dim = dim
            current_pos, 
            q_weight_offset,
            layout.xb,
            layout.q
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_q);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 3. Dispatch MatMul (Key Projection K)
        const uint32_t k_weight_offset = wk_base + (l * config.dim * kv_dim);
        PushConstants pc_k{ 
            l, 
            static_cast<uint32_t>(config.dim),      // in_dim = dim
            static_cast<uint32_t>(config.n_kv_heads * head_size),      // out_dim = dim
            current_pos, 
            k_weight_offset,
            layout.xb,
            layout.key_cache + (l * config.seq_len + current_pos) * kv_dim
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_k);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (kv_dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 4. Dispatch MatMul (Value Projection V)
        const uint32_t v_weight_offset = wv_base + (l * config.dim * kv_dim);
        PushConstants pc_v{ 
            l, 
            static_cast<uint32_t>(config.dim),      // in_dim = dim
            static_cast<uint32_t>(config.n_kv_heads * head_size),      // out_dim = dim
            current_pos, 
            v_weight_offset,
            layout.xb,
            layout.value_cache + (l * config.seq_len + current_pos) * kv_dim
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_v);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (kv_dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 5. Dispatch RoPE (Rotary Positional Encoding)
        const uint32_t freq_cis_real_offset = freq_cis_real_base;
        PushConstants pc_rope{
            l,
            static_cast<uint32_t>(config.dim),
            0,
            current_pos,
            0,

            layout.q,
            layout.q,
            
            0,
            head_size
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_rope);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.ropePipeline);
        vkCmdDispatch(cmd, (config.dim /2 + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        const uint32_t k_slot = layout.key_cache + (l * config.seq_len + current_pos) * kv_dim;
        PushConstants pc_rope_k{
            l, kv_dim, 0, current_pos,
            0,          // weight_offset unused
            k_slot,     // src_offset
            k_slot,     // dst_offset
            0,          // src2_offset unused
            head_size
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_rope_k);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.ropePipeline);
        vkCmdDispatch(cmd, ((kv_dim / 2) + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 6. Dispatch Attention
        PushConstants pc_att{
            l,
            static_cast<uint32_t>(config.dim),
            static_cast<uint32_t>(config.dim),
            current_pos,
            0, // weight_offset not used for attention

            layout.q,
            layout.xb,
            layout.att,
            head_size,
            static_cast<uint32_t>(config.n_heads),
            kv_dim,
            static_cast<uint32_t>(config.seq_len),
            layout.key_cache   + l * config.seq_len * kv_dim,
            layout.value_cache + l * config.seq_len * kv_dim
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_att);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.attentionPipeline);
        vkCmdDispatch(cmd, config.n_heads, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 7. Dispatch MatMul 
        PushConstants pc_matmul{
            l,
            static_cast<uint32_t>(config.n_heads * head_size),
            static_cast<uint32_t>(config.dim),
            current_pos,
            wo_base + (l * config.dim * config.dim),

            layout.xb,
            layout.xb2
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_matmul);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 8. Dispatch Residual Addition
        PushConstants pc_residual{
            l,
            static_cast<uint32_t>(config.dim),
            static_cast<uint32_t>(config.dim),
            current_pos,
            0, // weight_offset not used for residual addition
            layout.xb2,
            layout.x
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_residual);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.residualAddPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 9. Dispatch RMSNorm (FFN Norm)
        const uint32_t rms_ffn_weight_offset = rms_ffn_base + (l * config.dim);
        PushConstants pc_rms_ffn{
            l,
            static_cast<uint32_t>(config.dim),
            0, // out_dim not used for RMSNorm
            current_pos,
            rms_ffn_weight_offset,
            layout.x,
            layout.xb
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_rms_ffn);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.rmsNormPipeline);
        vkCmdDispatch(cmd, 1, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 10. Dispatch MatMul (FFN w1)
        const uint32_t w1_weight_offset = w1_base + (l * config.dim * config.hidden_dim);
        PushConstants pc_w1{
            l,
            static_cast<uint32_t>(config.dim),
            static_cast<uint32_t>(config.hidden_dim),
            current_pos,
            w1_weight_offset,
            layout.xb,
            layout.hb
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_w1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.hidden_dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 11. Dispatch MatMul (FFN w2)
        const uint32_t w3_weight_offset = w3_base + (l * config.hidden_dim * config.dim);
        PushConstants pc_w2{
            l,
            static_cast<uint32_t>(config.dim),
            static_cast<uint32_t>(config.hidden_dim),
            current_pos,
            w3_weight_offset,
            layout.xb,
            layout.hb2
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_w2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.hidden_dim + 63) / 64, 1, 1);   

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 12. Dispatch swiglu (FFN activation)
        PushConstants pc_swiglu{
            l,
            static_cast<uint32_t>(config.hidden_dim),
            static_cast<uint32_t>(config.hidden_dim),
            current_pos,
            0, // weight_offset not used for swiglu
            layout.hb,
            layout.hb,
            layout.hb2,
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_swiglu);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.swigluPipeline);
        vkCmdDispatch(cmd, (config.hidden_dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 13. Dispatch MatMul (FFN w3)
        const uint32_t w2_weight_offset = w2_base + (l * config.hidden_dim * config.dim);
        PushConstants pc_w3{
            l,
            static_cast<uint32_t>(config.hidden_dim),
            static_cast<uint32_t>(config.dim),
            current_pos,
            w2_weight_offset,
            layout.hb,
            layout.xb
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_w3);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);


        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 14. Dispatch Residual Addition (FFN)
        PushConstants pc_residual_ffn{
            l,
            static_cast<uint32_t>(config.dim),
            static_cast<uint32_t>(config.dim),
            current_pos,
            0, // weight_offset not used for residual addition
            layout.xb,
            layout.x
        }; 
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_residual_ffn);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.residualAddPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    // 16. Dispatch RMSNorm (Final Norm)
    PushConstants pc_rms_final{
        0,
        static_cast<uint32_t>(config.dim),
        0, // out_dim not used for RMSNorm
        current_pos,
        rms_final_base,
        layout.x,
        layout.xb
    };
    vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_rms_final);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.rmsNormPipeline);
    vkCmdDispatch(cmd, 1, 1, 1);

    // Memory Barrier between Operations
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // 17. Dispatch MatMul (Final Projection)
    PushConstants pc_final_matmul{
        0,
        static_cast<uint32_t>(config.dim),
        static_cast<uint32_t>(config.vocab_size),
        current_pos,
        0, // weight_offset not used for final projection
        layout.xb,
        layout.logits
    };
    vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_final_matmul);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
    vkCmdDispatch(cmd, (config.vocab_size + 63) / 64, 1, 1);

    // Memory Barrier between Operations
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

int sample_next_token(const std::vector<float>& logits) {
    // Find the index of the maximum logit
    auto max_iter = std::max_element(logits.begin(), logits.end());
    int max_index = std::distance(logits.begin(), max_iter);
    return max_index;
}

int main() {
    VkResult res;
    VkInstance instance;
    VkApplicationInfo appInfo{};
    VkInstanceCreateInfo createInfo{};
    VkPhysicalDeviceProperties deviceProperties{};
    VkPhysicalDeviceFeatures deviceFeatures{};
    VkPhysicalDevice physicalDevice{};
    uint32_t queueFamilyIndex = 0;
    VkDeviceQueueCreateInfo queueCreateInfo{};    
    VkDeviceCreateInfo deviceCreateInfo{};
    VkDevice device;
    VkQueue queue{};
    VkBufferCreateInfo bufferCreateInfo{};
    initialize_vulkan(instance, appInfo, createInfo, physicalDevice, deviceProperties, deviceFeatures, &queueFamilyIndex, res);
    create_device(queueCreateInfo, deviceCreateInfo, physicalDevice, queueFamilyIndex, deviceFeatures, device, res);
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    std::ifstream file_tok("weights/tok512.bin", std::ios::binary);
    if (!file_tok) {
        std::cerr << "Failed to open weights/tok512.bin\n";
        return 1;
    }
    std::vector<TokenInfo> vocabulary = read_vocab(file_tok);

    std::ifstream file_model("weights/stories260K.bin", std::ios::binary);
    if (!file_model) {
        std::cerr << "Failed to open weights/stories260K.bin\n";
        return 1;
    }
    Config config;
    TransformerWeights weights;
    std::vector<float> weight_buffer;

    if (!load_model_weights(file_model, config, weights, weight_buffer)) {
        std::cerr << "Failed to load model weights\n";
        return 1;
    }

    std::string input_text = " Once upon a time";
    std::vector<int> token_ids = tokenize(input_text, vocabulary);
    token_ids.insert(token_ids.begin(), 1);
    const uint32_t prompt_length = static_cast<uint32_t>(token_ids.size());

    VkDeviceSize total_weight_bytes = weight_buffer.size() * sizeof(float);
    StateLayout layout = make_layout(config);
    VkDeviceSize activation_bytes = layout.total_floats * sizeof(float);
    uint32_t head_size = config.dim / config.n_heads;
    uint32_t kv_dim = config.n_kv_heads * head_size;  

    VkBuffer gpu_weight_buffer{};
    VkDeviceMemory gpu_weight_memory{};
    VkBuffer buffer_activation{};
    VkDeviceMemory memory_activation{};

    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    VkCommandPool commandPool{};
    VkCommandBuffer cmd{};
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    VulkanComputePipeline pipelineState{};
    VkFence fence{};
    

    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    res = vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool);
    assert(res == VK_SUCCESS);

    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &cmd);
    assert(res == VK_SUCCESS);
    
    // Buffer Input Activations (x)
    create_buffer(bufferCreateInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer_activation, device, activation_bytes, res);
    create_memory(device, physicalDevice, buffer_activation, memory_activation, res);
    res = vkBindBufferMemory(device, buffer_activation, memory_activation, 0);
    assert(res == VK_SUCCESS);

    // GPU weight buffer
    create_buffer(bufferCreateInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gpu_weight_buffer, device, total_weight_bytes, res);
    create_memory(device, physicalDevice, gpu_weight_buffer, gpu_weight_memory, res);
    res = vkBindBufferMemory(device, gpu_weight_buffer, gpu_weight_memory, 0);
    assert(res == VK_SUCCESS);

    // CPU memcpy into GPU storage buffer
    void* data_ptr = nullptr;
    vkMapMemory(device, gpu_weight_memory, 0, total_weight_bytes, 0, &data_ptr);
    std::memcpy(data_ptr, weight_buffer.data(), total_weight_bytes);
    vkUnmapMemory(device, gpu_weight_memory);
    
    std::vector<uint32_t> rmsSpirv = read_spirv_file("shaders/rmsnorm.spv");
    std::vector<uint32_t> matmulSpirv = read_spirv_file("shaders/matmul.spv");
    std::vector<uint32_t> ropeSpirv = read_spirv_file("shaders/rope.spv");
    std::vector<uint32_t> attentionSpirv = read_spirv_file("shaders/attention.spv");
    std::vector<uint32_t> residualaddSpirv = read_spirv_file("shaders/residualadd.spv");
    std::vector<uint32_t> swigluSpirv = read_spirv_file("shaders/swiglu.spv");

    vulkan_pipeline(
        device, 
        gpu_weight_buffer, total_weight_bytes, 
        buffer_activation, 
        activation_bytes, 
        rmsSpirv, 
        matmulSpirv, 
        ropeSpirv, 
        attentionSpirv, 
        residualaddSpirv, 
        swigluSpirv,
        pipelineState, 
        res
    );
    assert(res == VK_SUCCESS);

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
    assert(res == VK_SUCCESS);
    
    for (int i = 0; i < 32; ++i) {
        vkResetCommandBuffer(cmd, 0);
        vkResetFences(device, 1, &fence);

        // Copy embedding vector 'x' into buffer_activation 
        void* x;
        vkMapMemory(device, memory_activation, 0, activation_bytes, 0, &x);
        int id = token_ids[i];
        size_t start_index = static_cast<size_t>(id) * config.dim;
        float* embedding_ptr = weights.token_embedding_table + start_index;
        memcpy((char*)x, embedding_ptr, config.dim * sizeof(float));
        vkUnmapMemory(device, memory_activation);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        res = vkBeginCommandBuffer(cmd, &beginInfo);
        assert(res == VK_SUCCESS);

        forward_pass(cmd, pipelineState, config, layout, i);

        res = vkEndCommandBuffer(cmd);
        assert(res == VK_SUCCESS);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        res = vkQueueSubmit(queue, 1, &submitInfo, fence);
        assert(res == VK_SUCCESS);
        res = vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
        assert(res == VK_SUCCESS);

        // Read back logits from buffer_activation
        void* logits_ptr;
        vkMapMemory(device, memory_activation, 0, activation_bytes, 0, &logits_ptr);
        std::vector<float> logits(config.vocab_size);
        const float* s = static_cast<const float*>(logits_ptr);
        memcpy(logits.data(), s + layout.logits, config.vocab_size * sizeof(float));

        if (i >= static_cast<int>(prompt_length) - 1) {
            int next_token_id = sample_next_token(logits);
            token_ids.push_back(next_token_id);
        } 
        vkUnmapMemory(device, memory_activation);

        void* p = nullptr;
        vkMapMemory(device, memory_activation, 0, activation_bytes, 0, &p);
        const float* s2 = static_cast<const float*>(p);

        auto dump = [&](const char* name, uint32_t off, uint32_t n) {
            std::cout << name << ": ";
            for (uint32_t i = 0; i < n; ++i) std::cout << s2[off + i] << " ";
            std::cout << "\n";
        };

        dump("xb (attn nrm)", layout.xb,  8);
        dump("q (post-rope)", layout.q,   8);
        dump("k (post-rope)", layout.key_cache + (0*config.seq_len + i)*kv_dim, 8);
        dump("att h0",        layout.att, i + 1);
        dump("hb (swiglu)",   layout.hb,  8);
        dump("x (post-ffn)",  layout.x,   8);
        dump("logits",        layout.logits, 8);
        std::cout << "-----------------------------------------\n";
        

        vkUnmapMemory(device, memory_activation);
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    std::cout << "Original String Back from Token IDs: ";
    std::string output = get_string_from_token_ids(vocabulary, token_ids);
    std::cout << output << "\n";

    return 0;
}
