#include <random>
#include <chrono>
#include "./rand.h"
#include "pcg_random.hpp"

int sample(double* probs, int size){
    pcg64 rng( std::random_device{}() );
    std::discrete_distribution<int> distribution(probs, probs + size);
    int sampledIndex = distribution(rng);
    return sampledIndex;
}

double gen_uni_rand(){
    pcg64 rng( std::random_device{}() );
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double out = distribution(rng);
    return out;
}
