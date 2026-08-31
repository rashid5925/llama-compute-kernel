#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

struct TokenInfo {
    float score;
    std::string text;
};

std::vector<TokenInfo> readVocab(const std::string file_name) {
    std::vector<TokenInfo> vocabulary;
    std::ifstream file(file_name, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << file_name << "\n";
        return vocabulary;
    }

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

// Tokenizes input text using a greedy longest-match approach
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

int main() {
    std::vector<TokenInfo> vocabulary = readVocab("weights/tok512.bin");
    if (vocabulary.size() == 0) {
        return 1;
    }

    // Example usage of the tokenizer
    std::string input_text = "Hello, world!";
    std::cout << "Input Text: " << input_text << std::endl;
    std::vector<int> token_ids = tokenize(input_text, vocabulary);
    std::cout << "Token IDs: ";
    for (int id : token_ids) {
        std::cout << id << " ";
    }
    std::cout << "\n";

    std::cout << "Original String Back from Token IDs: ";
    for (int id : token_ids) {
        std::cout << vocabulary[id].text;
    }

    std::cout << "\n";

    return 0;
}