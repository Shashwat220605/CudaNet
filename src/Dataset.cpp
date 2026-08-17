#include "Dataset.h"

#include <algorithm>
#include <random>
#include <stdexcept>


SyntheticDataset::SyntheticDataset(
    std::size_t count,
    unsigned int seed)
{
    if (count < 2)
    {
        throw std::invalid_argument(
            "Dataset must contain at least 2 samples."
        );
    }

    samples_.reserve(count);

    std::mt19937 rng(seed);

    std::normal_distribution<float> noise(
        0.0f,
        0.35f
    );

    std::uniform_real_distribution<float> thirdFeature(
        -1.0f,
        1.0f
    );

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        Sample sample{};

        if (i % 2 == 0)
        {
            sample.x = -1.0f + noise(rng);
            sample.y = -1.0f + noise(rng);
            sample.label = 0;
        }
        else
        {
            sample.x = 1.0f + noise(rng);
            sample.y = 1.0f + noise(rng);
            sample.label = 1;
        }

        sample.z = thirdFeature(rng);

        samples_.push_back(sample);
    }

    std::shuffle(
        samples_.begin(),
        samples_.end(),
        rng
    );
}


const std::vector<Sample>&
SyntheticDataset::samples() const
{
    return samples_;
}


std::vector<Sample>
SyntheticDataset::trainingSet(
    float trainRatio) const
{
    if (trainRatio <= 0.0f ||
        trainRatio >= 1.0f)
    {
        throw std::invalid_argument(
            "trainRatio must be between 0 and 1."
        );
    }

    std::size_t trainCount =
        static_cast<std::size_t>(
            samples_.size() * trainRatio
        );

    return std::vector<Sample>(
        samples_.begin(),
        samples_.begin() + trainCount
    );
}


std::vector<Sample>
SyntheticDataset::testSet(
    float trainRatio) const
{
    if (trainRatio <= 0.0f ||
        trainRatio >= 1.0f)
    {
        throw std::invalid_argument(
            "trainRatio must be between 0 and 1."
        );
    }

    std::size_t trainCount =
        static_cast<std::size_t>(
            samples_.size() * trainRatio
        );

    return std::vector<Sample>(
        samples_.begin() + trainCount,
        samples_.end()
    );
}
