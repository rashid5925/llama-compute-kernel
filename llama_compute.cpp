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
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t pos;
    uint32_t weight_offset;
};

struct VulkanComputePipeline {
    VkDescriptorSetLayout setLayout0;
    VkDescriptorSetLayout setLayout1;
    VkPipelineLayout pipelineLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet set0;
    VkDescriptorSet set1;
    
    VkPipeline rmsNormPipeline;
    VkPipeline matmulPipeline;
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

std::vector<int> tokenize(const std::string& text, const std::vector<TokenInfo>& vocabulary) {
    std::vector<int> tokens;
    
    // 1. Llama tokenizer prepends the special meta-space character
    // UTF-8 representation of the BPE meta-space ' '
    std::string processed_text = "\xe2\x96\x81" + text; 

    size_t i = 0;
    while (i < processed_text.length()) {
        int best_token_id = -1;
        size_t best_match_len = 0;

        // 2. Greedy Search: Look for the longest substring that matches a token
        for (size_t id = 0; id < vocabulary.size(); ++id) {
            const std::string& token_str = vocabulary[id].text;
            size_t len = token_str.length();

            // Skip special control tokens that shouldn't match raw user text
            if (id == 0 || id == 1 || id == 2) continue; 

            // Check if token_str matches the current window of processed_text
            if (i + len <= processed_text.length() && 
                processed_text.compare(i, len, token_str) == 0) {
                
                // We want the LONGEST matching token string
                if (len > best_match_len) {
                    best_match_len = len;
                    best_token_id = id;
                }
            }
        }

        // 3. Byte Fallback Handling
        if (best_token_id == -1) {
            // If no token matched, fallback to the raw single byte representation
            unsigned char raw_byte = static_cast<unsigned char>(processed_text[i]);
            
            // Byte tokens (0x00 to 0xFF) are shifted into indices 3 to 258
            best_token_id = 3 + raw_byte; 
            best_match_len = 1;
        }

        // Append the token ID and advance our string index window
        tokens.push_back(best_token_id);
        i += best_match_len;
    }

    return tokens;
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

    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;

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

void create_buffer(VkBufferCreateInfo& bufferCreateInfo, VkBufferUsageFlags bufferUsage, VkBuffer& stagingBuffer, VkDevice device, uint32_t bufferSize, VkResult res) {
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
    VkBuffer stateBuffer, size_t stateSizeBytes,
    const std::vector<uint32_t>& rmsSpirv,
    const std::vector<uint32_t>& matmulSpirv,
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
    std::vector<VkDescriptorSetLayoutBinding> stateBinding(5);
    for (uint32_t i = 0; i < 5; ++i) {
        stateBinding[i].binding = i;
        stateBinding[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        stateBinding[i].descriptorCount = 1;
        stateBinding[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo1{};
    layoutInfo1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo1.bindingCount = 5;
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

    // Write Descriptor Set 1 for state buffer
    VkDescriptorBufferInfo stateBufInfo{ stateBuffer, 0, stateSizeBytes };
    std::vector<VkWriteDescriptorSet> writeSets(5);
    for (uint32_t i = 0; i < 5; ++i) {
        writeSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeSets[i].dstSet = pipelineState.set1;
        writeSets[i].dstBinding = i;
        writeSets[i].descriptorCount = 1;
        writeSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeSets[i].pBufferInfo = &stateBufInfo;
    }
    vkUpdateDescriptorSets(device, 5, writeSets.data(), 0, nullptr);

    // Create Compute Pipelines 
    pipelineState.rmsNormPipeline = create_pipeline(device, pipelineState, rmsSpirv, res);
    pipelineState.matmulPipeline  = create_pipeline(device, pipelineState, matmulSpirv, res);
}

void forward_pass(
    VkCommandBuffer cmd,
    VulkanComputePipeline& pipe,
    const Config& config,
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
            rms_weight_offset 
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
            q_weight_offset 
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_q);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 3. Dispatch MatMul (Key Projection K)
        const uint32_t k_weight_offset = wk_base + (l * config.dim * config.dim);
        PushConstants pc_k{ 
            l, 
            static_cast<uint32_t>(config.dim),      // in_dim = dim
            static_cast<uint32_t>(config.n_kv_heads),      // out_dim = dim
            current_pos, 
            k_weight_offset 
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_k);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // 4. Dispatch MatMul (Value Projection V)
        const uint32_t v_weight_offset = wv_base + (l * config.dim * config.dim);
        PushConstants pc_v{ 
            l, 
            static_cast<uint32_t>(config.dim),      // in_dim = dim
            static_cast<uint32_t>(config.n_kv_heads),      // out_dim = dim
            current_pos, 
            v_weight_offset 
        };
        vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_v);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.matmulPipeline);
        vkCmdDispatch(cmd, (config.dim + 63) / 64, 1, 1);

        // Memory Barrier between Operations
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Insert additional barriers & dispatches for RoPE, Attention, Output MatMul, FFN, and Residual addition.

    }
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

    std::string input_text = "Once upon a time";
    std::vector<int> token_ids = tokenize(input_text, vocabulary);
    


    const uint32_t total_weight_bytes = weight_buffer.size() * sizeof(float);
    uint32_t num_tokens = token_ids.size();
    uint32_t activation_bytes = num_tokens * config.dim * sizeof(float);

    VkBuffer gpu_weight_buffer{};
    VkDeviceMemory gpu_weight_memory{};
    VkBuffer buffer_x{};
    VkDeviceMemory memory_x{};
    VkBuffer buffer_out{};
    VkDeviceMemory memory_out{};
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    VkCommandPool commandPool{};
    VkCommandBuffer cmd{};
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    VulkanComputePipeline pipelineState{};
    

    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    res = vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool);
    assert(res == VK_SUCCESS);

    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &cmd);
    assert(res == VK_SUCCESS);
    
    // Buffer Input Activations (x)
    create_buffer(bufferCreateInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer_x, device, activation_bytes, res);
    create_memory(device, physicalDevice, buffer_x, memory_x, res);
    res = vkBindBufferMemory(device, buffer_x, memory_x, 0);
    assert(res == VK_SUCCESS);

    // GPU weight buffer
    create_buffer(bufferCreateInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gpu_weight_buffer, device, total_weight_bytes, res);
    create_memory(device, physicalDevice, gpu_weight_buffer, gpu_weight_memory, res);
    res = vkBindBufferMemory(device, gpu_weight_buffer, gpu_weight_memory, 0);
    assert(res == VK_SUCCESS);

    // Buffer Output Activations (output)
    create_buffer(bufferCreateInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, buffer_out, device, activation_bytes, res);
    create_memory(device, physicalDevice, buffer_out, memory_out, res);
    res = vkBindBufferMemory(device, buffer_out, memory_out, 0);
    assert(res == VK_SUCCESS);

    // CPU memcpy into GPU storage buffer
    void* data_ptr = nullptr;
    vkMapMemory(device, gpu_weight_memory, 0, total_weight_bytes, 0, &data_ptr);
    std::memcpy(data_ptr, weight_buffer.data(), total_weight_bytes);
    vkUnmapMemory(device, gpu_weight_memory);


    // Copy embedding vector 'x' into buffer_x 
    void* x;
    vkMapMemory(device, memory_x, 0, activation_bytes, 0, &x);
    size_t offset = 0;
    for (int id : token_ids) {
        size_t start_index = static_cast<size_t>(id) * config.dim;
        float* embedding_ptr = weights.token_embedding_table + start_index;
        memcpy((char*)x + offset, embedding_ptr, config.dim * sizeof(float));
        offset += config.dim * sizeof(float);
    }
    vkUnmapMemory(device, memory_x);

    std::vector<uint32_t> rmsSpirv = read_spirv_file("shaders/rmsnorm.spv");
    std::vector<uint32_t> matmulSpirv = read_spirv_file("shaders/matmul.spv");
    vulkan_pipeline(device, gpu_weight_buffer, total_weight_bytes, buffer_x, activation_bytes, rmsSpirv, matmulSpirv, pipelineState, res);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(cmd, &beginInfo);
    assert(res == VK_SUCCESS);

    forward_pass(cmd, pipelineState, config, 0);

    res = vkEndCommandBuffer(cmd);
    assert(res == VK_SUCCESS);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence{};
    res = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
    assert(res == VK_SUCCESS);

    res = vkQueueSubmit(queue, 1, &submitInfo, fence);
    assert(res == VK_SUCCESS);
    res = vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
    assert(res == VK_SUCCESS);

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    std::cout << "Original String Back from Token IDs: ";
    std::string output = get_string_from_token_ids(vocabulary, token_ids);
    std::cout << output << "\n";

    return 0;
}
