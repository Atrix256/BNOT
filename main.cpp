#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DETERMINISTIC() true

#define SHOW_VORONOI_EVOLUTION() true
#define SHOW_VORONOI_WITH_POINTS() true

static const float c_goldenRatioConjugate = 0.61803398875f;  // 1 / goldenRatio

static const float c_antiAliasWidth = 1.2f;

using Vec2 = std::array<float, 2>;
using Vec3 = std::array<float, 3>;

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

template <size_t N>
float ToroidalDistanceSquared(const std::array<float, N>& A, const std::array<float, N>& B)
{
    float ret = 0.0f;
    for (size_t index = 0; index < N; ++index)
    {
        float diff = fabs(A[index] - B[index]);
        if (diff > 0.5f)
            diff = 1.0f - diff;

        ret += diff * diff;
    }
    return ret;
}

template <size_t N>
std::array<float, N> operator + (const std::array<float, N>& A, const std::array<float, N>& B)
{
    std::array<float, N> ret;
    for (size_t index = 0; index < N; ++index)
        ret[index] = A[index] + B[index];
    return ret;
}

template <size_t N>
std::array<float, N> operator * (const std::array<float, N>& A, float B)
{
    std::array<float, N> ret;
    for (size_t index = 0; index < N; ++index)
        ret[index] = A[index] * B;
    return ret;
}

template <size_t N>
std::array<float, N> operator / (const std::array<float, N>& A, float B)
{
    std::array<float, N> ret;
    for (size_t index = 0; index < N; ++index)
        ret[index] = A[index] / B;
    return ret;
}

inline float SmoothStep(float value, float min, float max)
{
    float x = (value - min) / (max - min);
    x = std::min(x, 1.0f);
    x = std::max(x, 0.0f);

    return 3.0f * x * x - 2.0f * x * x * x;
}

float Lerp(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

float LinearToSRGB(float x)
{
    x = Clamp(x, 0.0f, 1.0f);
    if (x < 0.0031308f)
        return x * 12.92f;
    else
        return pow(x * 1.055f, 1.0f / 2.4f) - 0.055f;
}

float SRGBToLinear(float x)
{
    x = Clamp(x, 0.0f, 1.0f);
    if (x < 0.04045f)
        return x / 12.92f;
    else
        return pow((x + 0.055f) / 1.055f, 2.4f);
}

float Fract(float x)
{
    return x - floor(x);
}

Vec3 IndexSVToRGB(size_t index, float s, float v)
{
    float h = Fract(float(index) * c_goldenRatioConjugate);

    int h_i = int(h * 6);
    float f = Fract(h * 6);
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (h_i)
    {
        case 0: return Vec3{ v, t, p };
        case 1: return Vec3{ q, v, p };
        case 2: return Vec3{ p, v, t };
        case 3: return Vec3{ p, q, v };
        case 4: return Vec3{ t, p, v };
        case 5: return Vec3{ v, p, q };
    }
    return Vec3{ 0.0f, 0.0f, 0.0f };
}

struct DensityImage
{
    int width = 0;
    int height = 0;
    std::vector<float> densities;

    float totalDensity = 0.0;

    bool Load(const char* fileName)
    {
        int components;
        unsigned char* pixels = stbi_load(fileName, &width, &height, &components, 4);
        if (!pixels)
            return false;

        densities.resize(width * height);
        std::fill(densities.begin(), densities.end(), 0.0f);

        unsigned char* pixel = pixels;
        size_t pixelIndex = 0;
        for (float& density : densities)
        {
            float r = float(pixel[0]) / 255.0f;
            float g = float(pixel[1]) / 255.0f;
            float b = float(pixel[2]) / 255.0f;

            density = r * 0.3f + g * 0.59f + b * 0.11f;

            density = 1.0f - LinearToSRGB(density);

            totalDensity += density;

            pixel += 4;
            pixelIndex++;
        }

        stbi_image_free(pixels);

        return true;
    }

    bool Save(const char* fileName, bool invertDensity = true) const
    {
        std::vector<unsigned char> pixels(width * height);
        unsigned char* pixel = pixels.data();

        for (float density : densities)
        {
            if (invertDensity)
                density = 1.0f - density;

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

    void MakeImageFromPoints(int w, int h, const std::vector<Vec2>& points, float dotRadius)
    {
        width = w;
        height = h;
        densities.resize(width * height);
        std::fill(densities.begin(), densities.end(), 0.0f);

        if (dotRadius <= 0.0f)
        {
            for (const Vec2& point : points)
            {
                int x = Clamp(int(point[0] * float(width)), 0, width - 1);
                int y = Clamp(int(point[1] * float(height)), 0, height - 1);
                GetDensity(x, y) = 1.0f;
            }
        }
        else
        {
            for (const Vec2& point : points)
            {
                int x = Clamp(int(point[0] * float(width)), 0, width - 1);
                int y = Clamp(int(point[1] * float(height)), 0, height - 1);

                int x1 = Clamp(x - int(ceil(dotRadius + c_antiAliasWidth)), 0, width - 1);
                int x2 = Clamp(x + int(ceil(dotRadius + c_antiAliasWidth)), 0, width - 1);

                int y1 = Clamp(y - int(ceil(dotRadius + c_antiAliasWidth)), 0, height - 1);
                int y2 = Clamp(y + int(ceil(dotRadius + c_antiAliasWidth)), 0, height - 1);

                for (int iy = y1; iy <= y2; ++iy)
                {
                    float disty = float(iy) - float(y);
                    float* pixel = &densities[iy * width + x1];
                    for (int ix = x1; ix <= x2; ++ix)
                    {
                        float distx = float(ix) - float(x);
                        float distance = sqrtf(distx * distx + disty * disty);
                        distance -= dotRadius;
                        distance = SmoothStep(distance, c_antiAliasWidth, 0.0f);
                        *pixel = std::max(*pixel, distance);
                        ++pixel;
                    }
                }
            }
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

void SaveVoronoi(const char* prefix, const std::vector<std::vector<size_t>>& cellMembers, const std::vector<Vec2>& points, size_t width, size_t height, size_t index)
{
    char buffer[256];
    sprintf_s(buffer, "out/voronoi_%s_%zu.png", prefix, index);

    static const float c_dotRadius = 2.0f;

    std::vector<unsigned char> pixels(width * height * 3, 0);

    // draw the pixel membership
    for (size_t cellIndex = 0; cellIndex < cellMembers.size(); ++cellIndex)
    {
        Vec3 color = IndexSVToRGB(cellIndex, 0.7f, 1.0f);
        unsigned char R = (unsigned char)Clamp(LinearToSRGB(color[0]) * 256.0f, 0.0f, 255.0f);
        unsigned char G = (unsigned char)Clamp(LinearToSRGB(color[1]) * 256.0f, 0.0f, 255.0f);
        unsigned char B = (unsigned char)Clamp(LinearToSRGB(color[2]) * 256.0f, 0.0f, 255.0f);

        for (size_t pixelIndex : cellMembers[cellIndex])
        {
            pixels[pixelIndex * 3 + 0] = R;
            pixels[pixelIndex * 3 + 1] = G;
            pixels[pixelIndex * 3 + 2] = B;
        }
    }

    // draw the location of the sample points
    for (size_t cellIndex = 0; cellIndex < cellMembers.size(); ++cellIndex)
    {
        Vec3 color = IndexSVToRGB(cellIndex, 1.0f, 1.0f);
        unsigned char R = (unsigned char)Clamp(LinearToSRGB(color[0]) * 256.0f, 0.0f, 255.0f);
        unsigned char G = (unsigned char)Clamp(LinearToSRGB(color[1]) * 256.0f, 0.0f, 255.0f);
        unsigned char B = (unsigned char)Clamp(LinearToSRGB(color[2]) * 256.0f, 0.0f, 255.0f);

        int x = Clamp<int>(int(points[cellIndex][0] * float(width)), 0, (int)width - 1);
        int y = Clamp<int>(int(points[cellIndex][1] * float(height)), 0, (int)height - 1);

        int x1 = Clamp(x - int(ceil(c_dotRadius)), 0, (int)width - 1);
        int x2 = Clamp(x + int(ceil(c_dotRadius)), 0, (int)width - 1);

        int y1 = Clamp(y - int(ceil(c_dotRadius)), 0, (int)height - 1);
        int y2 = Clamp(y + int(ceil(c_dotRadius)), 0, (int)height - 1);

        for (int iy = y1; iy <= y2; ++iy)
        {
            float disty = float(iy) - float(y);
            unsigned char* pixel = &pixels[(iy * width + x1) * 3];
            for (int ix = x1; ix <= x2; ++ix)
            {
                float distx = float(ix) - float(x);
                float distance = sqrtf(distx * distx + disty * disty);
                distance -= c_dotRadius;
                if (distance < 0.0f)
                {
                    pixel[0] = R;
                    pixel[1] = G;
                    pixel[2] = B;
                }
                pixel += 3;
            }
        }
    }

    stbi_write_png(buffer, (int)width, (int)height, 3, pixels.data(), (int)width * 3);
}

void MakeCapacityConstraintedVoronoiTessellation(std::mt19937& rng, std::vector<Vec2>& points, const DensityImage& densityImage, size_t iteration)
{
    // This is "Algorithm 1" in the paper, which is used for step 1 of the "Capacity-Constrained Method"

    // TODO: capacity maybe doesn't need to be stored on points, but can be it's own array in here? see if it needs to survive the function call or not. probably doesn't.
    // TODO: rethink structures in here a bit after it's working. like maybe voronoiCellCapacity and members can be in a struct

    const size_t c_numCells = points.size();

    // TODO: do we need to know members globally, or do we put voronoiCellMembers into the block that begins with the PixelsAndDensities struct

    // initialize Voronoi cells to have a capacity of 0 and no members
    std::vector<float> voronoiCellCapacity(c_numCells, 0.0f );
    std::vector<std::vector<size_t>> voronoiCellMembers(c_numCells);

    // Make a randomized map of density image pixels to Voronoi cells (points).
    // The capacity of all cells must be (roughly!) equal.
    {
        // Get pixel density, sorted from smallest to largest, so we can pop the largest ones off the back and
        // put it into a voronoi cell with the lowest capacity.
        struct PixelsAndDensities
        {
            size_t pixelIndex;
            float density;
        };
        std::vector<PixelsAndDensities> sortedPixelDensities(densityImage.densities.size());
        {
            for (size_t index = 0; index < densityImage.densities.size(); ++index)
            {
                sortedPixelDensities[index].pixelIndex = index;
                sortedPixelDensities[index].density = densityImage.densities[index];
            }
            std::sort(sortedPixelDensities.begin(), sortedPixelDensities.end(),
                [](const PixelsAndDensities& A, const PixelsAndDensities& B)
                {
                    return A.density < B.density;
                }
            );
        }

        // Assign these pixel densities to cells randomly, but as evenly as possible
        {
            // randomize the order of the Voronoi cells for getting density from pixels
            struct VoronoiCellOrder
            {
                size_t cellIndex;
                float capacity;
            };
            std::vector<VoronoiCellOrder> voronoiCellOrder(c_numCells);
            for (size_t cellIndex = 0; cellIndex < voronoiCellOrder.size(); ++cellIndex)
            {
                voronoiCellOrder[cellIndex].cellIndex = cellIndex;
                voronoiCellOrder[cellIndex].capacity = 0.0f;
            }
            std::shuffle(voronoiCellOrder.begin(), voronoiCellOrder.end(), rng);

            // add each pixel to the Voronoi cell with the lowest capacity, otherwise preserving the randomized order from the shuffle.
            while (!sortedPixelDensities.empty())
            {
                const PixelsAndDensities& pixelAndDensity = sortedPixelDensities.back();

                // make a Voronoi cell entry with updated data
                const VoronoiCellOrder& oldCellOrder = voronoiCellOrder.back();
                VoronoiCellOrder updatedCellOrder;
                updatedCellOrder.cellIndex = oldCellOrder.cellIndex;
                updatedCellOrder.capacity = oldCellOrder.capacity + pixelAndDensity.density;

                // find where this updated cell would go
                size_t newIndex = std::lower_bound(
                    voronoiCellOrder.begin(),
                    voronoiCellOrder.end(),
                    updatedCellOrder,
                    [=](const VoronoiCellOrder& A, const VoronoiCellOrder& B)
                    {
                        return A.capacity >= B.capacity;
                    }
                ) - voronoiCellOrder.begin();

                // calculate how many cells need to shift to the right to make room and do it if there are any
                size_t numToShift = voronoiCellOrder.size() - 1 - newIndex;
                if (numToShift > 0)
                    memmove(&voronoiCellOrder[newIndex + 1], &voronoiCellOrder[newIndex], sizeof(VoronoiCellOrder)* numToShift);

                // set the updated cell data
                voronoiCellOrder[newIndex] = updatedCellOrder;

                // store external data
                voronoiCellCapacity[updatedCellOrder.cellIndex] = updatedCellOrder.capacity;
                voronoiCellMembers[updatedCellOrder.cellIndex].push_back(pixelAndDensity.pixelIndex);

                // remove the pixel from the list now that we've placed it in a voronoi cell
                sortedPixelDensities.pop_back();
            }
        }
    }

    if (iteration == 1 && SHOW_VORONOI_EVOLUTION())
        SaveVoronoi("evolution", voronoiCellMembers, points, densityImage.width, densityImage.height, 0);

    // The "Voronoi" cells now have roughly equal capacity, but they contain random pixels - not the pixels they should.
    // We now look at each pair of Voronoi cells and see if they have any pixels that want to swap for better results.
    // We do this until it stabilizes and no pixels want to swap.
    printf("0%%");
    bool stable = false;
    int loopCount = 0;
    while (!stable)
    {
        stable = true;
        loopCount++;

        int swapCount = 0;

        // TODO: i think this loop parallelizable with OMP. Actually no. need to modify the membership of both cells involved. one cell may be involved in a different operation at the same time.
        //  * Could try a mutex if it's probably not going to happen? actually not so easy because of read access too.
        // TODO: profile and optimize if it's too slow. or maybe try with fewer points for now.

        struct HeapItem
        {
            float key;
            size_t pixelIndex;

            bool operator < (const HeapItem& B)
            {
                return key < B.key;
            }
        };

        // for each pair of cells
        for (size_t cellj = 1; cellj < c_numCells; ++cellj)
        {
            std::vector<HeapItem> Hi, Hj;
            int percent = int(100.0f * float(cellj) / float(c_numCells - 1));
            printf("\r                                     \rattempt %i %i%%", loopCount, percent);
            for (size_t celli = 0; celli < cellj; ++celli)
            {
                Hi.clear();
                Hj.clear();

                // for each point belonging to cell i, calculate how much "energy" would be saved by switching to the other cell
                for (size_t pixelIndex : voronoiCellMembers[celli])
                {
                    Vec2 pixelUV;
                    pixelUV[0] = float(pixelIndex % densityImage.width) / float(densityImage.width);
                    pixelUV[1] = float(pixelIndex / densityImage.width) / float(densityImage.height);

                    HeapItem heapItem;
                    heapItem.key = ToroidalDistanceSquared(pixelUV, points[celli]) - ToroidalDistanceSquared(pixelUV, points[cellj]);
                    heapItem.key *= densityImage.densities[pixelIndex];
                    heapItem.pixelIndex = pixelIndex;

                    Hi.push_back(heapItem);
                    std::push_heap(Hi.begin(), Hi.end());
                }

                // for each point belonging to cell j, calculate how much "energy" would be saved by switching to the other cell
                for (size_t pixelIndex : voronoiCellMembers[cellj])
                {
                    Vec2 pixelUV;
                    pixelUV[0] = float(pixelIndex % densityImage.width) / float(densityImage.width);
                    pixelUV[1] = float(pixelIndex / densityImage.width) / float(densityImage.height);

                    HeapItem heapItem;
                    heapItem.key = ToroidalDistanceSquared(pixelUV, points[cellj]) - ToroidalDistanceSquared(pixelUV, points[celli]);
                    heapItem.key *= densityImage.densities[pixelIndex];
                    heapItem.pixelIndex = pixelIndex;

                    Hj.push_back(heapItem);
                    std::push_heap(Hj.begin(), Hj.end());
                }

                // while we have more pixels to swap, and swapping results in a net energy savings
                while (!Hi.empty() && !Hj.empty() && (Hi[0].key + Hj[0].key) > 0.0f)
                {
                    // get the pixel index of the largest value from each heap
                    std::pop_heap(Hi.begin(), Hi.end());
                    std::pop_heap(Hj.begin(), Hj.end());
                    size_t pixelIndex_i = Hi.back().pixelIndex;
                    size_t pixelIndex_j = Hj.back().pixelIndex;
                    Hi.pop_back();
                    Hj.pop_back();

                    // swap membership to decrease overall energy
                    for (size_t& pixelIndex : voronoiCellMembers[celli])
                    {
                        if (pixelIndex == pixelIndex_i)
                        {
                            pixelIndex = pixelIndex_j;
                            break;
                        }
                    }
                    for (size_t& pixelIndex : voronoiCellMembers[cellj])
                    {
                        if (pixelIndex == pixelIndex_j)
                        {
                            pixelIndex = pixelIndex_i;
                            break;
                        }
                    }

                    swapCount++;

                    stable = false;
                }

            }
        }

        printf("\r                                     \rattempt %i 100%% - %i swaps\n", loopCount, swapCount);

        if (iteration == 1 && SHOW_VORONOI_EVOLUTION())
            SaveVoronoi("evolution", voronoiCellMembers, points, densityImage.width, densityImage.height, loopCount);
    }

    // Now move each point to the center of mass of all the points in it's Voronoi diagram
    {
        for (size_t cellIndex = 0; cellIndex < c_numCells; ++cellIndex)
        {
            float totalWeight = 0.0f;
            Vec2 centerOfMass = { 0.0f, 0.0f };
            for (size_t pixelIndex : voronoiCellMembers[cellIndex])
            {
                Vec2 pixelUV;
                pixelUV[0] = float(pixelIndex % densityImage.width) / float(densityImage.width);
                pixelUV[1] = float(pixelIndex / densityImage.width) / float(densityImage.height);

                float weight = densityImage.densities[pixelIndex];

                centerOfMass = centerOfMass + pixelUV * weight;
                totalWeight += weight;
            }

            centerOfMass = centerOfMass / totalWeight;

            points[cellIndex] = centerOfMass;
        }
    }

    if (SHOW_VORONOI_WITH_POINTS())
    {
        SaveVoronoi("points", voronoiCellMembers, points, densityImage.width, densityImage.height, iteration);
    }

    // TODO: make a pixelIndex to UV function? and make it handle half pixel offsets!
    // TODO: and a UV to pixelIndex function.
}

bool GenerateBlueNoisePoints(const char* baseFileName, const size_t c_numPoints, const size_t c_numIterations, const int c_pointImageWidth, const int c_pointImageHeight, float dotRadius)
{
    // load the image if we can
    DensityImage densityImage;
    char buffer[4096];
    sprintf_s(buffer, "images/%s.png", baseFileName);
    if (!densityImage.Load(buffer))
        return false;

    // save the starting image
    sprintf_s(buffer, "out/%s.png", baseFileName);
    printf("Saving starting image as %s\n\n", buffer);
    densityImage.Save(buffer);

    // get a random number generator
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

            // TODO: if no density image, we can just select u and v and put a point there. This is the same as not doing the density check at all.
            if (distDensity(rng) > densityImage.GetDensity(x, y))
                continue;

            float u = float(x) / float(densityImage.width - 1);
            float v = float(y) / float(densityImage.height - 1);

            points.push_back({u, v});
        }
    }

    // save an image of the initial points generated
    {
        sprintf_s(buffer, "out/%s.0.png", baseFileName);
        printf("Saving %s\n\n", buffer);

        DensityImage image;
        image.MakeImageFromPoints(c_pointImageWidth, c_pointImageHeight, points, dotRadius);
        image.Save(buffer);
    }

    // iteratively optimize the points into blue noise
    {
        for (int i = 1; i <= c_numIterations; ++i)
        {
            MakeCapacityConstraintedVoronoiTessellation(rng, points, densityImage, i);

            sprintf_s(buffer, "out/%s.%i.png", baseFileName, i);
            printf("Saving %s\n\n", buffer);

            DensityImage image;
            image.MakeImageFromPoints(c_pointImageWidth, c_pointImageHeight, points, dotRadius);
            image.Save(buffer);
        }
    }

    // TODO: write the final points out to a csv, or txt file, or .h or something
    // TODO: DFT of each step to see it evolving

    return true;
}

int main(int argc, char** argv)
{

    //GenerateBlueNoisePoints("white", 10, 10, 512, 512, 3.0f);

    GenerateBlueNoisePoints("puppysmall", 10, 5, 512, 512, 3.0f);

    //GenerateBlueNoisePoints("mountains", 1000000);

    return 0;
}

/*

Currently:
 * want to visualize the voronoi that goes with each point set, but also for debugging probably want to look at how it evolves.
! When SHOW_VORONOI_EVOLUTION() is true, there are some pixels that never seem to collect around their cells and float by themselves. some bug somewhere or a logic problem.



TODO: need to multiply distance by density i think? not real sure though... UPDATE: i did this. it wasn't the fix. should re-read paper
* show dft of each step

Question:
* the energy switch between voronoi doesn't take into account density of the points which seems wrong.
 * is the algorithm listing incorrect?

TODO:
* show how long it took total.
* draw circles for points, not dots. How to specify circle size? cause it kinda depends on how many circles there are.
- should this code be generalized to arbitrary dimensions? could maybe try it out. dunno if useful.
- make a set of points from a procedural density function, like maybe a good blue noise DFT lol.
- could show comparisons vs white noise (which is initial point set)
- in the paper they rasterize literal spheres onto the image where the sample points go.  You could do that too w/ a bounding box and SDF.
- could do a special version with no density function (should be faster??)
- probably should convert from sRGB to linear.
? should the num samples parameter be a multiplier for pixel count so it doesn't need to know resolution when choosing it?
- may not need to explicitly store capacity? not sure...

NOTES:
* brightness of resulting image is proportional to input image, not supposed to be exact
* BNOT seems to have basically same quality result, just runs more quickly and is more difficult to implement.
* CCVT
 * they say "assign points randomly to a cell, but make sure the cells have even weights". They don't say how they solve this "pluralized knapsack problem".
 * you can't always divide the capacity up evenly. if you have "a lot" of points (whatever that is), the total density of the image divided by cell count will be < 1, but you can have individual pixels with density of 1.
 * TODO: what did you do to solve it?

* generating colors programatically: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/

Papers:

Blue Noise Through Optimal Transport
https://graphics.stanford.edu/~kbreeden/pub/dGBOD12.pdf
What i wanted to implement initially. There are some math things I need to learn to do it though.
CCVT (Balzer 2009) has same quality looks like, and i think is simpler, but takes longer to generate.  I think i'll go with that.
That is "Capacity-Constrained Point Distributions: A Variant of Lloyd’s Method.


Capacity-Constrained Point Distributions: A Variant of Lloyd’s Method:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.6047&rep=rep1&type=pdf
yes, adapts to arbitrary density functions.



Should read this from 2015.
"A Survey of Blue-Noise Sampling and Its Applications"
https://www.researchgate.net/publication/276513263_A_Survey_of_Blue-Noise_Sampling_and_Its_Applications

*/