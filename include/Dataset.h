#pragma once

#include <cstddef>
#include <vector>

struct Sample
{
    float x;
    float y;
    float z;
    int label;
};

class SyntheticDataset
{
private:
    std::vector<Sample> samples_;

public:
    explicit SyntheticDataset(
        std::size_t count,
        unsigned int seed = 42
    );

    const std::vector<Sample>& samples() const;

    std::vector<Sample> trainingSet(
        float trainRatio = 0.8f
    ) const;

    std::vector<Sample> testSet(
        float trainRatio = 0.8f
    ) const;
};