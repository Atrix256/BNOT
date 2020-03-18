#include <array>
#include <random>
#include <stdio.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DETERMINISTIC() true

using Vec2 = std::array<float, 2>;

template <typename T>
T Clamp(T value, T min, T max)
{
    if (value <= min)
        return min;
    else if (value >= max)
        return max;
    else
        return value;
}

struct DensityImage
{
    int width = 0;
    int height = 0;
    std::vector<float> densities;

    bool Load(const char* fileName)
    {
        int components;
        unsigned char* pixels = stbi_load(fileName, &width, &height, &components, 4);
        if (!pixels)
            return false;

        densities.resize(width * height);
        std::fill(densities.begin(), densities.end(), 0.0f);

        unsigned char* pixel = pixels;
        for (float& density : densities)
        {
            float r = float(pixel[0]) / 255.0f;
            float g = float(pixel[1]) / 255.0f;
            float b = float(pixel[2]) / 255.0f;

            density = 1.0f - (r * 0.3f + g * 0.59f + b * 0.11f);
            pixel += 4;
        }

        stbi_image_free(pixels);

        return true;
    }

    bool Save(const char* fileName) const
    {
        std::vector<unsigned char> pixels(width * height);
        unsigned char* pixel = pixels.data();

        for (float density : densities)
        {
            *pixel = (unsigned char)Clamp(density * 256.0f, 0.0f, 255.0f);
            pixel++;
        }

        return stbi_write_png(fileName, width, height, 1, pixels.data(), width) != 0;
    }

    float& GetDensity(int x, int y)
    {
        return densities[y * width + x];
    }

    float GetDensity(int x, int y) const
    {
        return densities[y * width + x];
    }

    void MakeImageFromPoints(int w, int h, const std::vector<Vec2>& points)
    {
        width = w;
        height = h;
        densities.resize(width * height);
        std::fill(densities.begin(), densities.end(), 1.0f);

        for (const Vec2& point : points)
        {
            int x = Clamp(int(point[0] * float(width)), 0, width - 1);
            int y = Clamp(int(point[1] * float(height)), 0, height - 1);
            GetDensity(x, y) = 0.0f;
        }
    }
};

std::mt19937 GetRNG(uint32_t index)
{
#if DETERMINISTIC()
    std::seed_seq seq({ index, (unsigned int)0x65cd8674, (unsigned int)0x7952426c, (unsigned int)0x2a816f2c, (unsigned int)0x689dbc5f, (unsigned int)0xe138d1e5, (unsigned int)0x91da7241, (unsigned int)0x57f2d0e0, (unsigned int)0xed41c211 });
    std::mt19937 rng(seq);
#else
    std::random_device rd;
    std::mt19937 rng(rd());
#endif
    return rng;
}

void GenerateBlueNoisePoints(const char* baseFileName, const size_t c_numPoints, const DensityImage& densityImage)
{
    std::mt19937 rng = GetRNG(0);

    // initialize point locations conforming to the density function passed in, using rejection sampling.
    std::vector<Vec2> points;
    {
        std::uniform_real_distribution<float> distDensity(0.0f, 1.0f);
        std::uniform_int_distribution<int> distWidth(0, densityImage.width - 1);
        std::uniform_int_distribution<int> distHeight(0, densityImage.height - 1);

        while (points.size() < c_numPoints)
        {
            int x = distWidth(rng);
            int y = distHeight(rng);

            if (distDensity(rng) > densityImage.GetDensity(x, y))
                continue;

            float u = float(x) / float(densityImage.width - 1);
            float v = float(y) / float(densityImage.height - 1);

            points.push_back({ u, v });
        }
    }

    // save an image of the initial points generated
    {
        char buffer[1024];
        sprintf_s(buffer, "%s.begin.png", baseFileName);

        DensityImage image;
        image.MakeImageFromPoints(densityImage.width, densityImage.height, points);
        image.Save(buffer);
    }

    // iteratively optimize the points into blue noise
    {
    }

    // save an image of the final points generated
    {
        char buffer[1024];
        sprintf_s(buffer, "%s.end.png", baseFileName);

        DensityImage image;
        image.MakeImageFromPoints(densityImage.width, densityImage.height, points);
        image.Save(buffer);
    }
}

int main(int argc, char** argv)
{
    {
        DensityImage image;
        image.Load("images/puppy.png");
        //image.Save("out/puppy.png");
        GenerateBlueNoisePoints("out/puppy", 100000, image);
    }

    return 0;
}

/*

TODO:
- should this code be generalized to arbitrary dimensions? could maybe try it out. dunno if useful.
- make a set of points from a procedural density function, like maybe a good blue noise DFT lol.
- could show comparisons vs white noise (which is initial point set)
- in the paper they rasterize literal spheres onto the image where the sample points go.  You could do that too w/ a bounding box and SDF.
- could do a special version with no density function (should be faster??)


Blue Noise Through Optimal Transport
https://graphics.stanford.edu/~kbreeden/pub/dGBOD12.pdf
What i wanted to implement initially. There are some math things I need to learn to do it though.
CCVT (Balzer 2009) has same quality looks like, and i think is simpler, but takes longer to generate.  I think i'll go with that.
That is "Capacity-Constrained Point Distributions: A Variant of Lloyd’s Method.


Capacity-Constrained Point Distributions: A Variant of Lloyd’s Method:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.6047&rep=rep1&type=pdf
yes, adapts to arbitrary density functions.



*/