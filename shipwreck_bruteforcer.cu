#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        cudaDeviceReset(); \
        exit(EXIT_FAILURE); \
    } \
}

// =======================================================================
// 1. Core Data Structures & Enums (Host and Device)
// =======================================================================

enum class BlockRotation {
    NONE = 0, CLOCKWISE_90 = 1, CLOCKWISE_180 = 2, COUNTERCLOCKWISE_90 = 3
};

enum class StructureType {
    INVALID = -1,
    RIGHTSIDEUP_BACKHALF, RIGHTSIDEUP_BACKHALF_DEGRADED,
    RIGHTSIDEUP_FRONTHALF, RIGHTSIDEUP_FRONTHALF_DEGRADED,
    RIGHTSIDEUP_FULL, RIGHTSIDEUP_FULL_DEGRADED,
    SIDEWAYS_BACKHALF, SIDEWAYS_BACKHALF_DEGRADED,
    SIDEWAYS_FRONTHALF, SIDEWAYS_FRONTHALF_DEGRADED,
    SIDEWAYS_FULL, SIDEWAYS_FULL_DEGRADED,
    UPSIDEDOWN_BACKHALF, UPSIDEDOWN_BACKHALF_DEGRADED,
    UPSIDEDOWN_FRONTHALF, UPSIDEDOWN_FRONTHALF_DEGRADED,
    UPSIDEDOWN_FULL, UPSIDEDOWN_FULL_DEGRADED,
    WITH_MAST, WITH_MAST_DEGRADED,
};

struct ShipwreckConstraint {
    int32_t chunkX, chunkZ;
    BlockRotation requiredRotation;
    StructureType requiredType;
    bool isBeached;
};

// =======================================================================
// 2. Minecraft LCG & Constants
// =======================================================================

__constant__ int32_t SHIPWRECK_SPACING = 24;
__constant__ int32_t SHIPWRECK_SEPARATION = 4;
__constant__ int32_t SHIPWRECK_SALT = 165745295;
__constant__ int64_t MULT_A = 341873128712LL;
__constant__ int64_t MULT_B = 132897987541LL;
__constant__ int64_t LCG_MULT = 25214903917LL;
__constant__ int64_t LCG_ADD = 11LL;
__constant__ int64_t XOR_MASK = 25214903917LL;
__constant__ int64_t MASK_48 = (1LL << 48) - 1;
__constant__ int64_t LCG_MULT_INV = 246154705703781LL; // Specific to Reversing Kernel
__constant__ int32_t OCEAN_TYPE_COUNT = 20;
__constant__ int32_t BEACHED_TYPE_COUNT = 11;
__constant__ StructureType d_STRUCTURE_LOCATION_OCEAN[20];
__constant__ StructureType d_STRUCTURE_LOCATION_BEACHED[11];

// =======================================================================
// 3. Device-Side Logic & Kernels
// =======================================================================

__device__ inline int32_t floorDiv(int32_t a, int32_t n) {
    int32_t r = a / n;
    if ((a % n != 0) && ((a < 0) != (n < 0))) {
        r--;
    }
    return r;
}

struct StandaloneChunkRand {
private:
    int64_t seed;
public:
    __device__ void setSeed(int64_t s) {
        seed = (s ^ 0x5DEECE66DLL) & MASK_48;
    }
    __device__ int32_t next(int32_t bits) {
        seed = (seed * 0x5DEECE66DLL + 0xBLL) & MASK_48;
        return (int32_t)((uint64_t)seed >> (48 - bits));
    }
    __device__ int32_t nextInt(int32_t bound) {
        if (bound <= 0) return 0;
        if ((bound & -bound) == bound) return (int32_t)((bound * (int64_t)next(31)) >> 31);
        int32_t bits, val;
        do { bits = next(31); val = bits % bound; } while (bits - val + (bound - 1) < 0);
        return val;
    }
    __device__ int64_t nextLong() {
        return ((int64_t)next(32) << 32) + next(32);
    }
    __device__ void setRegionSeed(int64_t structureSeed, int32_t regionX, int32_t regionZ) {
        int64_t s = (long long)regionX * MULT_A + (long long)regionZ * MULT_B + structureSeed + SHIPWRECK_SALT;
        setSeed(s);
    }
    __device__ void setCarverSeed(int64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
        setSeed(worldSeed);
        long long a = nextLong();
        long long b = nextLong();
        setSeed((long long)chunkX * a ^ (long long)chunkZ * b ^ worldSeed);
    }
};

// --- STAGE 1 KERNEL (COMMON TO BOTH APPROACHES) ---

__device__ bool canGenerate_20bit_fast_filter(uint32_t lower20bits_of_world_seed, int32_t chunkX, int32_t chunkZ) {
    int32_t regX = floorDiv(chunkX, SHIPWRECK_SPACING);
    int32_t regZ = floorDiv(chunkZ, SHIPWRECK_SPACING);
    uint32_t regionalSeed32 = (uint32_t)(((long long)lower20bits_of_world_seed + (long long)regX * MULT_A + (long long)regZ * MULT_B + (long long)SHIPWRECK_SALT) ^ XOR_MASK);
    regionalSeed32 = (uint32_t)((long long)regionalSeed32 * LCG_MULT + LCG_ADD);
    uint32_t xCheck = (regionalSeed32 >> 17) & 3; 
    regionalSeed32 = (uint32_t)((long long)regionalSeed32 * LCG_MULT + LCG_ADD);
    uint32_t zCheck = (regionalSeed32 >> 17) & 3;
    return xCheck == (chunkX & 3) && zCheck == (chunkZ & 3);
}

__global__ void findLower20BitSeeds_kernel(const ShipwreckConstraint* d_constraints, int num_constraints, uint32_t* d_results, uint32_t* d_result_count) {
    uint32_t lower20bits = blockIdx.x * blockDim.x + threadIdx.x;
    if (lower20bits >= (1 << 20)) return;
    for (int i = 0; i < num_constraints; ++i) {
        if (!canGenerate_20bit_fast_filter(lower20bits, d_constraints[i].chunkX, d_constraints[i].chunkZ)) {
            return;
        }
    }
    uint32_t index = atomicAdd(d_result_count, 1);
    d_results[index] = lower20bits;
}


// --- STAGE 2 KERNEL (REVERSING APPROACH for 1-2 shipwrecks) ---

__global__ void reverseAndCheck_kernel(
    const uint32_t* d_valid_lower20bits, uint32_t num_valid_lower20bits,
    const ShipwreckConstraint* d_primary_anchor,
    const ShipwreckConstraint* d_secondary_anchor, int num_secondary_anchors,
    const ShipwreckConstraint* d_validators, int num_validators,
    int64_t* d_found_seeds, uint32_t* d_found_count
) {
    uint32_t lower20_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lower20_idx >= num_valid_lower20bits) return;

    uint32_t lower20bit_seed = d_valid_lower20bits[lower20_idx];
    
    StandaloneChunkRand rand;

    int32_t regX = floorDiv(d_primary_anchor->chunkX, SHIPWRECK_SPACING);
    int32_t regZ = floorDiv(d_primary_anchor->chunkZ, SHIPWRECK_SPACING);
    int32_t gen_regionsize = SHIPWRECK_SPACING - SHIPWRECK_SEPARATION;
    int32_t expectedRelX = ((d_primary_anchor->chunkX % SHIPWRECK_SPACING) + SHIPWRECK_SPACING) % SHIPWRECK_SPACING;
    int32_t expectedRelZ = ((d_primary_anchor->chunkZ % SHIPWRECK_SPACING) + SHIPWRECK_SPACING) % SHIPWRECK_SPACING;

    int64_t term_x = (int64_t)regX * MULT_A;
    int64_t term_z = (int64_t)regZ * MULT_B;
    uint64_t u_initial_part = (uint64_t)lower20bit_seed + (uint64_t)term_x + (uint64_t)term_z + (uint64_t)SHIPWRECK_SALT;
    
    uint64_t u_state0 = (u_initial_part ^ (uint64_t)XOR_MASK) & MASK_48;
    uint64_t u_state1 = (u_state0 * (uint64_t)LCG_MULT + (uint64_t)LCG_ADD) & MASK_48;
    uint64_t u_state2 = (u_state1 * (uint64_t)LCG_MULT + (uint64_t)LCG_ADD) & MASK_48;
    
    uint32_t finalLower20LCG = (uint32_t)(u_state2 & 0xFFFFF);
    uint32_t base_Z_contrib_for_lcg_val = finalLower20LCG >> 17;

    uint32_t u_mod_5_solutions[5];
    int num_u_mod_5_solutions = 0;
    for (uint32_t test_u = 0; test_u < 5; test_u++) {
        if ((((test_u << 3) + base_Z_contrib_for_lcg_val) % gen_regionsize) == expectedRelZ) {
            u_mod_5_solutions[num_u_mod_5_solutions++] = test_u;
        }
    }

    if (num_u_mod_5_solutions == 0) return;

    for (int i = 0; i < num_u_mod_5_solutions; i++) {
        for (long j = 0; ; j++) {
            uint64_t upper28LCG = 5 * j + u_mod_5_solutions[i];
            if (upper28LCG >= (1ULL << 28)) break;

            uint64_t u_currentLCGState = (upper28LCG << 20) | finalLower20LCG;
            uint64_t u_lcgStateForX = ((u_currentLCGState - (uint64_t)LCG_ADD) * (uint64_t)LCG_MULT_INV) & MASK_48;
            
            if (((u_lcgStateForX >> 17) % gen_regionsize) == expectedRelX) {
                uint64_t u_lcgStateInitial = ((u_lcgStateForX - (uint64_t)LCG_ADD) * (uint64_t)LCG_MULT_INV) & MASK_48;
                uint64_t u_scrambled = u_lcgStateInitial ^ (uint64_t)XOR_MASK;
                uint64_t u_result = u_scrambled - (uint64_t)term_x - (uint64_t)term_z - (uint64_t)SHIPWRECK_SALT;
                int64_t full48BitStructureSeed = (int64_t)(u_result & MASK_48);
                
                bool is_seed_valid = true;
                
                rand.setCarverSeed(full48BitStructureSeed, d_primary_anchor->chunkX, d_primary_anchor->chunkZ);
                if (static_cast<BlockRotation>(rand.nextInt(4)) != d_primary_anchor->requiredRotation) is_seed_valid = false;
                if (is_seed_valid) {
                    StructureType type;
                    if (d_primary_anchor->isBeached) type = d_STRUCTURE_LOCATION_BEACHED[rand.nextInt(BEACHED_TYPE_COUNT)];
                    else type = d_STRUCTURE_LOCATION_OCEAN[rand.nextInt(OCEAN_TYPE_COUNT)];
                    if (type != d_primary_anchor->requiredType) is_seed_valid = false;
                }

                if (is_seed_valid && num_secondary_anchors > 0) {
                    int32_t s_regX = floorDiv(d_secondary_anchor->chunkX, SHIPWRECK_SPACING);
                    int32_t s_regZ = floorDiv(d_secondary_anchor->chunkZ, SHIPWRECK_SPACING);
                    rand.setRegionSeed(full48BitStructureSeed, s_regX, s_regZ);
                    if ((s_regX * SHIPWRECK_SPACING + rand.nextInt(gen_regionsize) != d_secondary_anchor->chunkX) ||
                        (s_regZ * SHIPWRECK_SPACING + rand.nextInt(gen_regionsize) != d_secondary_anchor->chunkZ)) {
                        is_seed_valid = false;
                    }
                    if (is_seed_valid) {
                        rand.setCarverSeed(full48BitStructureSeed, d_secondary_anchor->chunkX, d_secondary_anchor->chunkZ);
                        if (static_cast<BlockRotation>(rand.nextInt(4)) != d_secondary_anchor->requiredRotation) is_seed_valid = false;
                        if (is_seed_valid) {
                           StructureType type;
                           if (d_secondary_anchor->isBeached) type = d_STRUCTURE_LOCATION_BEACHED[rand.nextInt(BEACHED_TYPE_COUNT)];
                           else type = d_STRUCTURE_LOCATION_OCEAN[rand.nextInt(OCEAN_TYPE_COUNT)];
                           if (type != d_secondary_anchor->requiredType) is_seed_valid = false;
                        }
                    }
                }
                
                if (is_seed_valid && num_validators > 0) {
                     // The reversing approach is only used for 1 or 2 constraints, so this validator loop will not run.
                     // It is kept for structural completeness but is not performance-critical.
                }

                if (is_seed_valid) {
                    uint32_t idx = atomicAdd(d_found_count, 1);
                    d_found_seeds[idx] = full48BitStructureSeed;
                }
            }
        }
    }
}


// --- STAGE 2 KERNEL (BRUTE-FORCE APPROACH for 3+ shipwrecks) ---

__device__ bool check_shipwreck_properties(int64_t structureSeed, const ShipwreckConstraint& constraint, StandaloneChunkRand& rand) {
    int32_t regX = floorDiv(constraint.chunkX, SHIPWRECK_SPACING);
    int32_t regZ = floorDiv(constraint.chunkZ, SHIPWRECK_SPACING);

    rand.setRegionSeed(structureSeed, regX, regZ);
    
    int32_t offset = SHIPWRECK_SPACING - SHIPWRECK_SEPARATION;
    if (regX * SHIPWRECK_SPACING + rand.nextInt(offset) != constraint.chunkX) return false;
    if (regZ * SHIPWRECK_SPACING + rand.nextInt(offset) != constraint.chunkZ) return false;

    rand.setCarverSeed(structureSeed, constraint.chunkX, constraint.chunkZ);
    if (static_cast<BlockRotation>(rand.nextInt(4)) != constraint.requiredRotation) return false;
    
    StructureType type;
    if (constraint.isBeached) {
        type = d_STRUCTURE_LOCATION_BEACHED[rand.nextInt(BEACHED_TYPE_COUNT)];
    } else {
        type = d_STRUCTURE_LOCATION_OCEAN[rand.nextInt(OCEAN_TYPE_COUNT)];
    }

    return type == constraint.requiredType;
}

__global__ void bruteforceStructureSeeds_kernel(
    const uint32_t* d_valid_lower20bits, uint32_t num_valid_lower20bits,
    const ShipwreckConstraint* d_constraints, int num_constraints,
    int64_t* d_found_seeds, uint32_t* d_found_count
) {
    uint64_t num_upper_bits_to_check = 1ULL << 28;
    uint64_t total_tasks = (uint64_t)num_valid_lower20bits * num_upper_bits_to_check;

    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    StandaloneChunkRand rand;

    for (uint64_t global_idx = thread_id; global_idx < total_tasks; global_idx += stride) {
        uint32_t lower20_idx = global_idx / num_upper_bits_to_check;
        uint32_t upper28_bits = global_idx % num_upper_bits_to_check;
        
        uint32_t lower20_val = d_valid_lower20bits[lower20_idx];
        int64_t candidateSeed = ((int64_t)upper28_bits << 20) | lower20_val;
        
        bool valid_for_all_constraints = true;
        for (int i = 0; i < num_constraints; i++) {
            // Pass the thread-local rand object to the check function to avoid re-instantiating it.
            if (!check_shipwreck_properties(candidateSeed, d_constraints[i], rand)) {
                valid_for_all_constraints = false;
                break;
            }
        }
        
        if (valid_for_all_constraints) {
            uint32_t result_idx = atomicAdd(d_found_count, 1);
            d_found_seeds[result_idx] = candidateSeed;
        }
    }
}


// =======================================================================
// 4. Host-Side Setup and Main Logic
// =======================================================================
std::map<std::string, StructureType> nameToTypeEnum;
std::map<std::string, BlockRotation> nameToRotEnum;

void initialize_host_maps() {
    nameToTypeEnum["rightsideup_backhalf"] = StructureType::RIGHTSIDEUP_BACKHALF; nameToTypeEnum["rightsideup_backhalf_degraded"] = StructureType::RIGHTSIDEUP_BACKHALF_DEGRADED;
    nameToTypeEnum["rightsideup_fronthalf"] = StructureType::RIGHTSIDEUP_FRONTHALF; nameToTypeEnum["rightsideup_fronthalf_degraded"] = StructureType::RIGHTSIDEUP_FRONTHALF_DEGRADED;
    nameToTypeEnum["rightsideup_full"] = StructureType::RIGHTSIDEUP_FULL; nameToTypeEnum["rightsideup_full_degraded"] = StructureType::RIGHTSIDEUP_FULL_DEGRADED;
    nameToTypeEnum["sideways_backhalf"] = StructureType::SIDEWAYS_BACKHALF; nameToTypeEnum["sideways_backhalf_degraded"] = StructureType::SIDEWAYS_BACKHALF_DEGRADED;
    nameToTypeEnum["sideways_fronthalf"] = StructureType::SIDEWAYS_FRONTHALF; nameToTypeEnum["sideways_fronthalf_degraded"] = StructureType::SIDEWAYS_FRONTHALF_DEGRADED;
    nameToTypeEnum["sideways_full"] = StructureType::SIDEWAYS_FULL; nameToTypeEnum["sideways_full_degraded"] = StructureType::SIDEWAYS_FULL_DEGRADED;
    nameToTypeEnum["upsidedown_backhalf"] = StructureType::UPSIDEDOWN_BACKHALF; nameToTypeEnum["upsidedown_backhalf_degraded"] = StructureType::UPSIDEDOWN_BACKHALF_DEGRADED;
    nameToTypeEnum["upsidedown_fronthalf"] = StructureType::UPSIDEDOWN_FRONTHALF; nameToTypeEnum["upsidedown_fronthalf_degraded"] = StructureType::UPSIDEDOWN_FRONTHALF_DEGRADED;
    nameToTypeEnum["upsidedown_full"] = StructureType::UPSIDEDOWN_FULL; nameToTypeEnum["upsidedown_full_degraded"] = StructureType::UPSIDEDOWN_FULL_DEGRADED;
    nameToTypeEnum["with_mast"] = StructureType::WITH_MAST; nameToTypeEnum["with_mast_degraded"] = StructureType::WITH_MAST_DEGRADED;
    
    nameToRotEnum["NONE"] = BlockRotation::NONE;
    nameToRotEnum["CLOCKWISE_90"] = BlockRotation::CLOCKWISE_90;
    nameToRotEnum["CLOCKWISE_180"] = BlockRotation::CLOCKWISE_180;
    nameToRotEnum["COUNTERCLOCKWISE_90"] = BlockRotation::COUNTERCLOCKWISE_90;
}

void initialize_device_constants() {
    std::vector<StructureType> ocean_types = {
        nameToTypeEnum["with_mast"], nameToTypeEnum["upsidedown_full"], nameToTypeEnum["upsidedown_fronthalf"], nameToTypeEnum["upsidedown_backhalf"],
        nameToTypeEnum["sideways_full"], nameToTypeEnum["sideways_fronthalf"], nameToTypeEnum["sideways_backhalf"], nameToTypeEnum["rightsideup_full"],
        nameToTypeEnum["rightsideup_fronthalf"], nameToTypeEnum["rightsideup_backhalf"], nameToTypeEnum["with_mast_degraded"],
        nameToTypeEnum["upsidedown_full_degraded"], nameToTypeEnum["upsidedown_fronthalf_degraded"], nameToTypeEnum["upsidedown_backhalf_degraded"],
        nameToTypeEnum["sideways_full_degraded"], nameToTypeEnum["sideways_fronthalf_degraded"], nameToTypeEnum["sideways_backhalf_degraded"],
        nameToTypeEnum["rightsideup_full_degraded"], nameToTypeEnum["rightsideup_fronthalf_degraded"], nameToTypeEnum["rightsideup_backhalf_degraded"]
    };
    std::vector<StructureType> beached_types = {
        nameToTypeEnum["with_mast"], nameToTypeEnum["sideways_full"], nameToTypeEnum["sideways_fronthalf"], nameToTypeEnum["sideways_backhalf"],
        nameToTypeEnum["rightsideup_full"], nameToTypeEnum["rightsideup_fronthalf"], nameToTypeEnum["rightsideup_backhalf"],
        nameToTypeEnum["with_mast_degraded"], nameToTypeEnum["rightsideup_full_degraded"],
        nameToTypeEnum["rightsideup_fronthalf_degraded"], nameToTypeEnum["rightsideup_backhalf_degraded"]
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_STRUCTURE_LOCATION_OCEAN, ocean_types.data(), ocean_types.size() * sizeof(StructureType)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_STRUCTURE_LOCATION_BEACHED, beached_types.data(), beached_types.size() * sizeof(StructureType)));
}

bool parse_input_file(const std::string& filename, std::vector<ShipwreckConstraint>& constraints) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return false;
    }
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        if (line.empty() || line[0] == '#') continue; 

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> parts;
        while(std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            parts.push_back(token);
        }

        if (parts.size() != 5) {
             std::cerr << "Warning: Malformed line " << line_num << ". Skipping." << std::endl;
             continue;
        }

        try {
            ShipwreckConstraint c;
            c.chunkX = std::stoi(parts[0]);
            c.chunkZ = std::stoi(parts[1]);

            if (nameToRotEnum.find(parts[2]) == nameToRotEnum.end()) {
                std::cerr << "Warning: Invalid rotation on line " << line_num << ". Skipping." << std::endl; continue;
            }
            c.requiredRotation = nameToRotEnum.at(parts[2]);

            if (nameToTypeEnum.find(parts[3]) == nameToTypeEnum.end()) {
                 std::cerr << "Warning: Invalid structure type on line " << line_num << ". Skipping." << std::endl; continue;
            }
            c.requiredType = nameToTypeEnum.at(parts[3]);
            
            std::transform(parts[4].begin(), parts[4].end(), parts[4].begin(), ::tolower);
            if (parts[4] == "beached") c.isBeached = true;
            else if (parts[4] == "ocean") c.isBeached = false;
            else {
                 std::cerr << "Warning: Invalid biome type on line " << line_num << " (must be 'Ocean' or 'Beached'). Skipping." << std::endl; continue;
            }
            constraints.push_back(c);
        } catch (...) {
            std::cerr << "Warning: Error parsing values on line " << line_num << ". Skipping." << std::endl;
        }
    }
    return !constraints.empty();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <constraints_file.txt>" << std::endl;
        std::cerr << "File format (one per line):\n";
        std::cerr << "ChunkX, ChunkZ, ROTATION, type_name, Ocean|Beached\n";
        std::cerr << "Example: -54, -14, COUNTERCLOCKWISE_90, sideways_fronthalf, Ocean\n";
        return 1;
    }
    auto startTime = std::chrono::high_resolution_clock::now();
    initialize_host_maps();
    initialize_device_constants();
    
    std::vector<ShipwreckConstraint> h_constraints;
    if (!parse_input_file(argv[1], h_constraints)) {
        std::cerr << "No valid constraints found in file. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Parsed " << h_constraints.size() << " shipwreck constraints.\n";

    ShipwreckConstraint* d_all_constraints;
    CUDA_CHECK(cudaMalloc(&d_all_constraints, h_constraints.size() * sizeof(ShipwreckConstraint)));
    CUDA_CHECK(cudaMemcpy(d_all_constraints, h_constraints.data(), h_constraints.size() * sizeof(ShipwreckConstraint), cudaMemcpyHostToDevice));

    // --- STAGE 1 (COMMON) ---
    std::cout << "\n--- Stage 1: Filtering Lower 20-bit seed patterns ---\n";
    std::vector<uint32_t> h_valid_lower20bits;
    uint32_t* d_lower20_results, *d_lower20_count;
    CUDA_CHECK(cudaMalloc(&d_lower20_results, (1 << 20) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lower20_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_lower20_count, 0, sizeof(uint32_t)));
    
    int threads = 256;
    int blocks = ((1 << 20) + threads - 1) / threads;
    findLower20BitSeeds_kernel<<<blocks, threads>>>(d_all_constraints, h_constraints.size(), d_lower20_results, d_lower20_count);
    
    uint32_t h_lower20_count;
    CUDA_CHECK(cudaMemcpy(&h_lower20_count, d_lower20_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    if (h_lower20_count == 0) {
        std::cout << "No seed candidates found in Stage 1. Exiting.\n";
        CUDA_CHECK(cudaFree(d_all_constraints));
        CUDA_CHECK(cudaFree(d_lower20_results));
        CUDA_CHECK(cudaFree(d_lower20_count));
        return 0;
    }
    h_valid_lower20bits.resize(h_lower20_count);
    CUDA_CHECK(cudaMemcpy(h_valid_lower20bits.data(), d_lower20_results, h_lower20_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Found " << h_lower20_count << " potential 20-bit candidates.\n";
    CUDA_CHECK(cudaFree(d_lower20_results));
    CUDA_CHECK(cudaFree(d_lower20_count));

    // --- STAGE 2 (CONDITIONAL LOGIC) ---
    uint32_t* d_valid_lower20bits_gpu;
    CUDA_CHECK(cudaMalloc(&d_valid_lower20bits_gpu, h_lower20_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_valid_lower20bits_gpu, h_valid_lower20bits.data(), h_lower20_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    int64_t* d_found_seeds_gpu;
    uint32_t* d_found_count_ptr;
    const uint32_t results_buffer_size = 20000000; // Large buffer for results
    CUDA_CHECK(cudaMalloc(&d_found_seeds_gpu, results_buffer_size * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_found_count_ptr, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_found_count_ptr, 0, sizeof(uint32_t)));

    if (h_constraints.size() <= 2) {
        // --- Use Reversing Approach ---
        std::cout << "\n--- Stage 2: Using REVERSING Approach (1-2 constraints) ---\n";
        ShipwreckConstraint* d_primary_anchor = d_all_constraints;
        ShipwreckConstraint* d_secondary_anchor = (h_constraints.size() > 1) ? d_all_constraints + 1 : nullptr;
        int num_secondary_anchors = (h_constraints.size() > 1) ? 1 : 0;
        
        blocks = (h_lower20_count + threads - 1) / threads;
        std::cout << "Launching reversal kernel for " << h_lower20_count << " candidates...\n";
        
        reverseAndCheck_kernel<<<blocks, threads>>>(
            d_valid_lower20bits_gpu, h_lower20_count,
            d_primary_anchor,
            d_secondary_anchor, num_secondary_anchors,
            nullptr, 0, // Validators not used in this path
            d_found_seeds_gpu, d_found_count_ptr);

    } else {
        // --- Use Brute-force Approach ---
        std::cout << "\n--- Stage 2: Using BRUTE-FORCE Approach (3+ constraints) ---\n";
        blocks = 32768; // Launch many blocks for good occupancy
        uint64_t total_tasks = (uint64_t)h_lower20_count * (1ULL << 28);
        std::cout << "Launching bruteforce kernel to check " << total_tasks << " total seed candidates...\n";

        bruteforceStructureSeeds_kernel<<<blocks, threads>>>(
            d_valid_lower20bits_gpu, h_lower20_count,
            d_all_constraints, h_constraints.size(),
            d_found_seeds_gpu, d_found_count_ptr
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Process Results (COMMON) ---
    uint32_t h_found_count;
    CUDA_CHECK(cudaMemcpy(&h_found_count, d_found_count_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    std::vector<int64_t> h_found_seeds;
    if (h_found_count > 0) {
        if (h_found_count > results_buffer_size) {
             std::cerr << "\nFATAL ERROR: Found " << h_found_count << " seeds, which exceeds the allocated buffer size of " 
                       << results_buffer_size << ". Results are incomplete." << std::endl;
             h_found_count = results_buffer_size;
        }
        h_found_seeds.resize(h_found_count);
        CUDA_CHECK(cudaMemcpy(h_found_seeds.data(), d_found_seeds_gpu, h_found_count * sizeof(int64_t), cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_all_constraints));
    CUDA_CHECK(cudaFree(d_valid_lower20bits_gpu));
    CUDA_CHECK(cudaFree(d_found_seeds_gpu));
    CUDA_CHECK(cudaFree(d_found_count_ptr));
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "\n--- Search Complete in " << std::chrono::duration<double>(endTime - startTime).count() << " seconds ---\n";
    
    if (h_found_seeds.empty()) {
        std::cout << "No structure seeds found.\n";
    } else {
        std::cout << "Found " << h_found_seeds.size() << " valid seeds. Writing to found_seeds.txt...\n";
        std::sort(h_found_seeds.begin(), h_found_seeds.end());
        std::ofstream outfile("found_seeds.txt");
        for (const auto& seed : h_found_seeds) outfile << seed << "\n";
        outfile.close();
        std::cout << "Done.\n";
    }
    
    return 0;
}