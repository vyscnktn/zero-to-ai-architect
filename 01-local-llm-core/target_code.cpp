#include <iostream>
#include <cstring>

// Vulnerable function mimicking poor memory handling
void process_user_input(char* input_data) {
    char internal_buffer[8]; 
    // SECURITY FLAW: strcpy does not check bounds. 
    // Passing input larger than 8 bytes will corrupt the stack pointer.
    std::strcpy(internal_buffer, input_data); 
    std::cout << "Data successfully copied to internal buffer." << std::endl;
}

int main() {
    // Malicious payload designed to trigger a buffer overflow condition
    char malicious_payload[128] = "AttackPayloadExceedingTheEightByteAllocationLimitSimulatingAnExploit...";
    process_user_input(malicious_payload);
    return 0;
}