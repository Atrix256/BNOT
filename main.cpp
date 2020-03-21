#include <algorithm>
#include <array>
#include <chrono>
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

static const float c_goldenRatioConjugate = 0.61803398875f;  // 1 / goldenRatio

static const float c_antiAliasWidth = 1.2f;

using Vec2 = std::array<float, 2>;
using Vec3 = std::array<float, 3>;

struct ScopedTimer
{
    ScopedTimer(const char* label)
    {
        m_label = label;
        m_start = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer()
    {
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - m_start);
        printf("%s completed in %0.2f seconds\n\n", m_label.c_str(), time_span.count());
    }

    std::string m_label;
    std::chrono::high_resolution_clock::time_point m_start;
};

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
inline float ToroidalDistanceSquared(const std::array<float, N>& A, const std::array<float, N>& B)
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

struct Parameters
{
    struct Debug
    {
        bool showVoronoiEvolution = false;
    };

    Debug debug;
};

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

struct SiteInfo
{
    std::vector<size_t> members;
    size_t revisionNumber = 0;
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

void SaveVoronoi(const char* baseFileName, const std::vector<SiteInfo>& sitesInfo, const std::vector<Vec2>& sites, const std::vector<Vec2>& points, size_t width, size_t height, float dotRadius, size_t index)
{
    char buffer[256];
    sprintf_s(buffer, "out/%s.voronoi_%zu.png", baseFileName, index);

    std::vector<unsigned char> pixels(width * height * 3, 0);

    // draw the location of the points
    size_t siteIndex = 0;
    for (const SiteInfo& siteInfo : sitesInfo)
    {
        Vec3 color = IndexSVToRGB(siteIndex, 0.9f, 0.125f);
        unsigned char R = (unsigned char)Clamp(LinearToSRGB(color[0]) * 256.0f, 0.0f, 255.0f);
        unsigned char G = (unsigned char)Clamp(LinearToSRGB(color[1]) * 256.0f, 0.0f, 255.0f);
        unsigned char B = (unsigned char)Clamp(LinearToSRGB(color[2]) * 256.0f, 0.0f, 255.0f);

        for (size_t pointIndex : siteInfo.members)
        {
            const float c_dotRadius = dotRadius * 0.5f;
            const Vec2& point = points[pointIndex];

            int x = Clamp<int>(int(point[0] * float(width)), 0, (int)width - 1);
            int y = Clamp<int>(int(point[1] * float(height)), 0, (int)height - 1);

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
        siteIndex++;
    }

    // draw the location of the sites
    siteIndex = 0;
    for (const Vec2& point : sites)
    {
        const float c_dotRadius = dotRadius;

        Vec3 color = IndexSVToRGB(siteIndex, 0.9f, 0.95f);
        unsigned char R = (unsigned char)Clamp(LinearToSRGB(color[0]) * 256.0f, 0.0f, 255.0f);
        unsigned char G = (unsigned char)Clamp(LinearToSRGB(color[1]) * 256.0f, 0.0f, 255.0f);
        unsigned char B = (unsigned char)Clamp(LinearToSRGB(color[2]) * 256.0f, 0.0f, 255.0f);

        int x = Clamp<int>(int(point[0] * float(width)), 0, (int)width - 1);
        int y = Clamp<int>(int(point[1] * float(height)), 0, (int)height - 1);

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

        siteIndex++;
    }

    stbi_write_png(buffer, (int)width, (int)height, 3, pixels.data(), (int)width * 3);
}

// returns true if it did any swaps
void MakeCapacityConstraintedVoronoiTessellation(const char* baseFileName, std::mt19937& rng, const std::vector<Vec2>& points, std::vector<Vec2>& sites, size_t imageWidth, size_t imageHeight, float dotRadius, const Parameters& parameters)
{
    // This is "Algorithm 1" in the paper, which is used for step 1 of the "Capacity-Constrained Method"

    const size_t c_numCells = sites.size();

    // Make a randomized map of points to sites. Each site must have the same number of points.
    std::vector<SiteInfo> sitesInfo(c_numCells);
    {
        std::vector<size_t> pointIndices(points.size());
        std::iota(pointIndices.begin(), pointIndices.end(), (size_t)0);
        std::shuffle(pointIndices.begin(), pointIndices.end(), rng);

        for (size_t index = 0; index < pointIndices.size(); ++index)
            sitesInfo[index % sites.size()].members.push_back(pointIndices[index]);
    }

    if (parameters.debug.showVoronoiEvolution)
        SaveVoronoi(baseFileName, sitesInfo, sites, points, imageWidth, imageHeight, dotRadius, 0);

    // The sites now have equal capacity, but they contain random pixels - not the pixels they should.
    // We now look at each pair of Voronoi cells and see if they have any pixels that want to swap for better results.
    // We do this until it stabilizes and no pixels want to swap.
    struct CellPairRevisionNumbers
    {
        size_t revisionNumber_i = (size_t)-1;
        size_t revisionNumber_j = (size_t)-1;
    };
    const size_t c_numCellPairs = c_numCells * (c_numCells + 1) / 2 - 1; // the number of pairs we are checking
    const size_t c_numCellPairs1Percent = std::max<size_t>(size_t(float(c_numCellPairs) / float(100.0f)), 1);
    std::vector<CellPairRevisionNumbers> cellPairRevisions(c_numCellPairs);

    printf("0%%");
    bool stable = false;
    int loopCount = 0;
    while (!stable)
    {
        stable = true;
        loopCount++;

        int swapCount = 0;

        struct HeapItem
        {
            float key;
            size_t pointIndex;

            bool operator < (const HeapItem& B)
            {
                return key < B.key;
            }
        };

        // for each pair of cells
        size_t cellPairIndex = 0;
        for (size_t cellj = 1; cellj < c_numCells; ++cellj)
        {
            std::vector<HeapItem> Hi, Hj;
            for (size_t celli = 0; celli < cellj; ++celli)
            {
                if (celli % c_numCellPairs1Percent == 0)
                {
                    int percent = int(100.0f * float(cellPairIndex) / float(c_numCellPairs - 1));
                    printf("\r                                     \riteration %i %i%%", loopCount, percent);
                }

                // don't check these pairs of cells if we have already checked them previously and nothing has changed.
                CellPairRevisionNumbers& cellPairRevisionNumbers = cellPairRevisions[cellPairIndex];
                cellPairIndex++;
                if (cellPairRevisionNumbers.revisionNumber_i == sitesInfo[celli].revisionNumber && cellPairRevisionNumbers.revisionNumber_j == sitesInfo[cellj].revisionNumber)
                    continue;

                Hi.clear();
                Hj.clear();

                // for each point belonging to cell i, calculate how much energy would be saved by switching to the other cell
                for (size_t pointIndex : sitesInfo[celli].members)
                {
                    const Vec2& point = points[pointIndex];

                    HeapItem heapItem;
                    heapItem.key = ToroidalDistanceSquared(point, points[celli]) - ToroidalDistanceSquared(point, points[cellj]);
                    heapItem.pointIndex = pointIndex;

                    Hi.push_back(heapItem);
                }
                std::make_heap(Hi.begin(), Hi.end());

                // for each point belonging to cell j, calculate how much energy would be saved by switching to the other cell
                for (size_t pointIndex : sitesInfo[cellj].members)
                {
                    const Vec2& point = points[pointIndex];

                    HeapItem heapItem;
                    heapItem.key = ToroidalDistanceSquared(point, points[cellj]) - ToroidalDistanceSquared(point, points[celli]);
                    heapItem.pointIndex = pointIndex;

                    Hj.push_back(heapItem);
                }
                std::make_heap(Hj.begin(), Hj.end());

                // while we have more pixels to swap, and swapping results in a net energy savings
                while (!Hi.empty() && !Hj.empty() && (Hi[0].key + Hj[0].key) > 0.0f)
                {
                    // get the pixel index of the largest value from each heap
                    std::pop_heap(Hi.begin(), Hi.end());
                    std::pop_heap(Hj.begin(), Hj.end());
                    size_t pointIndex_i = Hi.back().pointIndex;
                    size_t pointIndex_j = Hj.back().pointIndex;
                    Hi.pop_back();
                    Hj.pop_back();

                    sitesInfo[celli].revisionNumber++;
                    sitesInfo[cellj].revisionNumber++;

                    // swap membership to decrease overall energy
                    for (size_t& pixelIndex : sitesInfo[celli].members)
                    {
                        if (pixelIndex == pointIndex_i)
                        {
                            pixelIndex = pointIndex_j;
                            break;
                        }
                    }
                    for (size_t& pixelIndex : sitesInfo[cellj].members)
                    {
                        if (pixelIndex == pointIndex_j)
                        {
                            pixelIndex = pointIndex_i;
                            break;
                        }
                    }

                    swapCount++;

                    stable = false;
                }

                // update revision numbers in case they changed
                cellPairRevisionNumbers.revisionNumber_i = sitesInfo[celli].revisionNumber;
                cellPairRevisionNumbers.revisionNumber_j = sitesInfo[cellj].revisionNumber;
            }
        }

        printf("\r                                     \riteration %i 100%% - %i swaps\n", loopCount, swapCount);

        if (parameters.debug.showVoronoiEvolution)
            SaveVoronoi(baseFileName, sitesInfo, sites, points, imageWidth, imageHeight, dotRadius, loopCount);
    }

    // TODO: do this for each cell when they have swaps occur
    // Now move each point to the center of mass of all the points in it's Voronoi diagram
    {
        for (size_t cellIndex = 0; cellIndex < c_numCells; ++cellIndex)
        {
            float totalWeight = 0.0f;
            Vec2 centerOfMass = { 0.0f, 0.0f };
            for (size_t pointIndex : sitesInfo[cellIndex].members)
            {
                Vec2 point = points[pointIndex];

                // make sure the point is the closest version of the point to the site, torroidally
                {
                    float distx = fabs(sites[cellIndex][0] - point[0]);
                    if (distx > 0.5f)
                    {
                        if (sites[cellIndex][0] < 0.5f)
                            point[0] -= 1.0f;
                        else
                            point[0] += 1.0f;
                    }

                    float disty = fabs(sites[cellIndex][1] - point[1]);
                    if (disty > 0.5f)
                    {
                        if (sites[cellIndex][1] < 0.5f)
                            point[1] -= 1.0f;
                        else
                            point[1] += 1.0f;
                    }
                }

                centerOfMass = centerOfMass + point;
                totalWeight += 1.0f;
            }

            centerOfMass = centerOfMass / totalWeight;

            sites[cellIndex] = centerOfMass;
        }
    }
}

void GeneratePointsFromImage(const DensityImage& densityImage, std::vector<Vec2>& points, size_t c_numPoints, std::mt19937& rng)
{
    std::uniform_real_distribution<float> distDensity(0.0f, 1.0f);
    std::uniform_int_distribution<int> distWidth(0, densityImage.width - 1);
    std::uniform_int_distribution<int> distHeight(0, densityImage.height - 1);

    while (points.size() < c_numPoints)
    {
        // TODO: should try using a low discrepancy sampling here - like blue noise or sobol or something. maybe even regular sampling?

        int x = distWidth(rng);
        int y = distHeight(rng);

        if (distDensity(rng) > densityImage.GetDensity(x, y))
            continue;

        float u = float(x) / float(densityImage.width - 1);
        float v = float(y) / float(densityImage.height - 1);

        points.push_back({ u, v });
    }
}

template <typename GENERATE_POINTS_LAMBDA>
void GenerateBlueNoisePoints(const char* baseFileName, const size_t c_numPoints, const size_t c_numSites, const int c_siteImageWidth, const int c_siteImageHeight, float dotRadius, const Parameters& params, const GENERATE_POINTS_LAMBDA& generatePointsLambda)
{
    ScopedTimer timer(baseFileName);

    // get a random number generator
    std::mt19937 rng = GetRNG(0);

    // Initialize point locations conforming to the density function
    std::vector<Vec2> points;
    generatePointsLambda(points, c_numPoints, rng);

    // initialize site locations
    std::vector<Vec2> sites(c_numSites);
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (Vec2& s : sites)
        {
            s[0] = dist(rng);
            s[1] = dist(rng);
        }
    }

    // save an image of the initial points generated
    char buffer[4096];
    {
        sprintf_s(buffer, "out/%s.points.png", baseFileName);
        printf("Saving %s\n\n", buffer);

        DensityImage image;
        image.MakeImageFromPoints(c_siteImageWidth, c_siteImageHeight, points, dotRadius);
        image.Save(buffer);
    }

    // save an image of the initial site locations
    {
        sprintf_s(buffer, "out/%s.0.png", baseFileName);
        printf("Saving %s\n\n", buffer);

        DensityImage image;
        image.MakeImageFromPoints(c_siteImageWidth, c_siteImageHeight, sites, dotRadius);
        image.Save(buffer);
    }

    // iteratively optimize the points into blue noise
    {
        MakeCapacityConstraintedVoronoiTessellation(baseFileName, rng, points, sites, c_siteImageWidth, c_siteImageHeight, dotRadius, params);

        // TODO: move this into MakeCapacityConstraintedVoronoiTessellation() i guess
        int iteration = 1;
        sprintf_s(buffer, "out/%s.%i.png", baseFileName, iteration);
        printf("Saving %s\n\n", buffer);

        DensityImage image;
        image.MakeImageFromPoints(c_siteImageWidth, c_siteImageHeight, sites, dotRadius);
        image.Save(buffer);
    }

    // TODO: write the final points out to a csv, or txt file, or .h or something?
    // TODO: DFT of each step to see it evolving
    // TODO: maybe show how to do it for a mathematical density function (including one that returns constant density)
}

int main(int argc, char** argv)
{
    // TODO: puppysmall.points.png has lines in it. quantization / conversion problem?
    // TODO: does the real ccvt code go until convergence?
    // TODO: real ccvt code has numPoints = numSites * 1024;
    // TODO: uniform test failed. maybe has to do with torroidal centroid?
    // TODO: special mode for vornoi in classic mode where it loops through pixels instead of doing floating point conversion stuff?

    // This is the "classic" algorithm
    {
        static const size_t c_gridSize = 256;

        static const char* baseFileName = "uniform";
        static const size_t c_numSites = 10;
        static const size_t c_numPoints = c_gridSize * c_gridSize;
        static const size_t c_imageSize = c_gridSize;
        static const float c_dotRadius = 1.0f;

        Parameters params;
        params.debug.showVoronoiEvolution = true;

        GenerateBlueNoisePoints(baseFileName, c_numPoints, c_numSites, c_imageSize, c_imageSize, c_dotRadius, params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                std::uniform_real_distribution<float> dist;
                points.resize(0);
                points.reserve(numPoints);

                size_t n = (size_t)sqrt(float(numPoints));
                for (size_t iy = 0; iy < n; ++iy)
                    for (size_t ix = 0; ix < n; ++ix)
                        points.push_back({ float(ix) / float(n), float(iy) / float(n) });
            }
        );
    }

    // TODO: temp!
    return 0;

    {
        static const char* baseFileName = "puppysmall";
        static const size_t c_numSites = 1000;
        static const size_t c_numPoints = c_numSites * 10;// *1024;
        static const size_t c_imageSize = 256;
        static const float c_dotRadius = 1.0f;

        Parameters params;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.png", baseFileName);
        printf("Saving starting image as %s\n\n", buffer);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(baseFileName, c_numPoints, c_numSites, c_imageSize, c_imageSize, c_dotRadius, params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng);
            }
        );
    }

    {
        static const char* baseFileName = "mountains";
        static const size_t c_numSites = 10000;
        static const size_t c_numPoints = c_numSites * 10;// *1024;
        static const size_t c_imageSize = 512;
        static const float c_dotRadius = 1.0f;

        Parameters params;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.png", baseFileName);
        printf("Saving starting image as %s\n\n", buffer);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(baseFileName, c_numPoints, c_numSites, c_imageSize, c_imageSize, c_dotRadius, params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng);
            }
        );
    }

    {
        static const char* baseFileName = "puppy";
        static const size_t c_numSites = 10000;
        static const size_t c_numPoints = c_numSites * 10;// *1024;
        static const size_t c_imageSize = 512;
        static const float c_dotRadius = 1.0f;

        Parameters params;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.png", baseFileName);
        printf("Saving starting image as %s\n\n", buffer);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(baseFileName, c_numPoints, c_numSites, c_imageSize, c_imageSize, c_dotRadius, params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng);
            }
        );
    }

    return 0;
}

/*

! ccvt centroidalizes after every swap. makes sense, you should try that too.
 * they only do "1 iteration" btw. they go til it converges then leave it there.

! oh man... they don't use a greyscale image to start out. they use a binary one representative of the greyscale image, using white noise to generate it.
 * maybe try using a blue noise texture or IGN and show how that affects quality.
 * using one type of blue noise to make another - WTF? :P
 * also use IGN. could even use sobol...
 * Masks: mask the image for thresholding, then use the results as points
 * 2d points: use them as the points

 * the CCVT code repo says it's GPL but i don't think it is. verify & fix that!
 https://github.com/Atrix256/ccvt

 ? show a power diagram test w/ uniform (classic mode)?

 DFT! compare with mitchell's best candidate.

 Talk about how to make progressive?

Currently:
 * want to visualize the voronoi that goes with each point set, but also for debugging probably want to look at how it evolves.
! When SHOW_VORONOI_EVOLUTION() is true, there are some pixels that never seem to collect around their cells and float by themselves. some bug somewhere or a logic problem.


* calculating centroid doesn't account for wrap around torroidal space. need to think about that.
 * they also have code for centroid calculation
     inline Point2 centroid(const Point2& center, const Point2::Vector& points) const {
      Point2 centroid;
      int pointsSize = static_cast<int>(points.size());
      for (int j = 0; j < pointsSize; ++j) {
        double p = points[j].x;
        if (fabs(center.x - p) > size.x / 2) {
          if (center.x < size.x / 2) {
            p -= size.x;
          } else {
            p += size.x;
          }
        }
        centroid.x += p;
        p = points[j].y;
        if (fabs(center.y - p) > size.y / 2) {
          if (center.y < size.y / 2) {
            p -= size.y;
          } else {
            p += size.y;
          }
        }
        centroid.y += p;
      }
      centroid.x /= pointsSize;
      centroid.y /= pointsSize;
      if (centroid.x < 0) {
        centroid.x += size.x;
      }
      if (centroid.x >= size.x) {
        centroid.x -= size.x;
      }
      if (centroid.y < 0) {
        centroid.y += size.y;
      }
      if (centroid.y >= size.y) {
        centroid.y -= size.y;
      }
      return centroid;
    }


* I think the thing is "how much energy is gained by switching". Both points have to agree its good to swap.
 * And it does energy squared apparently.
 * Energy is integrating density over the cell, minus the amount of capacity each cell should have.
 * If it's negative though??

! could speed things up by keeping a "revision #" per cell, and incrementing it when it changes.
 * also keep a struct for each pair of cells that stores the revision number for each. Only go through the check if the revision numbers are different than they used to be.


? if a cell hasn't changed it's revision number since last loop through, i think we can ignore it completely? in both the inner and outer loop. should help perf.
 * actually no, i don't think this is true

TODO: need to multiply distance by density i think? not real sure though... UPDATE: i did this. it wasn't the fix. should re-read paper
* show dft of each step

* in blog, show the full story of each algorithm in pictures

Question:
* the energy switch between voronoi doesn't take into account density of the points which seems wrong.
 * is the algorithm listing incorrect?

TODO:
* better anti aliased dots in output!
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
* site = a sample point in the results.  point = a density function sample point that is used to make the results.
* brightness of resulting image is proportional to input image, not supposed to be exact
* BNOT seems to have basically same quality result, just runs more quickly and is more difficult to implement.
* CCVT
 * they say "assign points randomly to a cell, but make sure the cells have even weights". They don't say how they solve this "pluralized knapsack problem".
 * you can't always divide the capacity up evenly. if you have "a lot" of points (whatever that is), the total density of the image divided by cell count will be < 1, but you can have individual pixels with density of 1.
 * TODO: what did you do to solve it?
* mention your CCVT "optimization" with cell revision numbers. retime when everything is working correctly. (before then it went from 10 seconds to 6, so 40% time savings)
* they use white noise to generate the binary sample points.
* i originally thought this algorithm used float density pixel values and was getting bad results.
* generating colors programatically: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
* a white image won't give you good results for ccvt because it'll sample it with white noise and get a white noise result. you want to sample it with low discrepancy sequence to get best results, which will also give the same results for a constant color texture vs making a grid of constant density.

Links:

I resurrected the code they made to go with the paper.  It was on google code.
https://github.com/Atrix256/ccvt


Papers:








==================== LANDFILL ====================

Blue Noise Through Optimal Transport
https://graphics.stanford.edu/~kbreeden/pub/dGBOD12.pdf
What i wanted to implement initially. There are some math things I need to learn to do it though.
CCVT (Balzer 2009) has same quality looks like, and i think is simpler, but takes longer to generate.  I think i'll go with that.
That is "Capacity-Constrained Point Distributions: A Variant of Lloyd�s Method.


Capacity-Constrained Point Distributions: A Variant of Lloyd�s Method:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.6047&rep=rep1&type=pdf
yes, adapts to arbitrary density functions.

Prior paper to the variant of lloyds method: CAPACITY-CONSTRAINED VORONOI DIAGRAMS IN FINITE SPACES
https://pdfs.semanticscholar.org/50c6/47450bb252ed0a3286316b9f8e486c1da0d2.pdf

Should read this from 2015.
"A Survey of Blue-Noise Sampling and Its Applications"
https://www.researchgate.net/publication/276513263_A_Survey_of_Blue-Noise_Sampling_and_Its_Applications


From Bruno Levy
https://members.loria.fr/Bruno.Levy/papers/CPD_SIGASIA_2016.pdf


video and post release paper here (if you have acm access)
https://dl.acm.org/doi/10.1145/1576246.1531392


*/
