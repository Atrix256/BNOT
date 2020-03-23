#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>
#include "dft.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DETERMINISTIC() true

// The real "voronoi" diagrams show the sparse points in the color of their site.
// This is hard to visually make sense of so when this define is true, it will
// make a real Voronoi diagram, not rely on the underlying points.
#define VORONOI_IS_SOLID() true

static const float c_goldenRatioConjugate = 0.61803398875f;  // 1 / goldenRatio

static const float c_pi = 3.14159265359f;

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

float VanDerCorput(size_t index, size_t base)
{
    float ret = 0.0f;
    float denominator = float(base);
    while (index > 0)
    {
        size_t multiplier = index % base;
        ret += float(multiplier) / denominator;
        index = index / base;
        denominator *= base;
    }
    return ret;
}

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

float LinearToSRGB(float x)
{
    x = Clamp(x, 0.0f, 1.0f);
    if (x == 0.0f || x == 1.0f)
        return x;
    if (x < 0.0031308f)
        return x * 12.92f;
    else
        return pow(x * 1.055f, 1.0f / 2.4f) - 0.055f;
}

float SRGBToLinear(float x)
{
    x = Clamp(x, 0.0f, 1.0f);
    if (x == 0.0f || x == 1.0f)
        return x;
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
    // Main functionality parameters
    const char* baseFileName = nullptr;
    size_t numSites = 1000;
    size_t numPoints = numSites * 1024;
    size_t siteImageWidth = 256;
    size_t siteImageHeight = 256;
    float siteImageDotRadius = 1.0f;

    // Debug parameters
    size_t DFTImageWidth = 256;
    size_t DFTImageHeight = 256;

    bool showVoronoiEvolution = false;
    bool showDFTEvolution = false;
    bool showSiteEvolution = false;

    bool showVoronoiFinal = true;
    bool showDFTFinal = true;
};

struct DensityImage
{
    int width = 0;
    int height = 0;
    std::vector<float> densities;

    float totalDensity = 0.0;

    bool Load(const char* fileName, bool normalize = false)
    {
        int components;
        unsigned char* pixels = stbi_load(fileName, &width, &height, &components, 4);
        if (!pixels)
            return false;

        densities.resize(width * height);
        std::fill(densities.begin(), densities.end(), 0.0f);

        float minDensity = FLT_MAX;
        float maxDensity = -FLT_MAX;

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

            minDensity = std::min(minDensity, density);
            maxDensity = std::max(maxDensity, density);

            pixel += 4;
            pixelIndex++;
        }

        // normalize the densities if we should.
        // This can help with the generation of the points, and also when you have a solid white image.
        if (normalize)
        {
            if (minDensity == maxDensity)
            {
                std::fill(densities.begin(), densities.end(), 1.0f);
                totalDensity = float(densities.size());
            }
            else
            {
                totalDensity = 0.0f;
                for (float& density : densities)
                {
                    density = (density - minDensity) / (maxDensity - minDensity);
                    totalDensity += density;
                }
            }
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

#if VORONOI_IS_SOLID()
    size_t siteIndex = 0;
    unsigned char* pixel = pixels.data();
    for (size_t iy = 0; iy < height; ++iy)
    {
        for (size_t ix = 0; ix < width; ++ix)
        {
            // find out which site this pixel is closest to
            float u = (float(ix) + 0.5f) / float(width);
            float v = (float(iy) + 0.5f) / float(height);

            size_t closestSiteIndex = 0;
            float closestSiteDistance = ToroidalDistanceSquared(Vec2{ u,v }, sites[0]);

            for (size_t siteIndex = 1; siteIndex < sites.size(); ++siteIndex)
            {
                float distance = ToroidalDistanceSquared(Vec2{ u,v }, sites[siteIndex]);
                if (distance < closestSiteDistance)
                {
                    closestSiteIndex = siteIndex;
                    closestSiteDistance = distance;
                }
            }

            Vec3 color = IndexSVToRGB(closestSiteIndex, 0.9f, 0.125f);
            unsigned char R = (unsigned char)Clamp(LinearToSRGB(color[0]) * 256.0f, 0.0f, 255.0f);
            unsigned char G = (unsigned char)Clamp(LinearToSRGB(color[1]) * 256.0f, 0.0f, 255.0f);
            unsigned char B = (unsigned char)Clamp(LinearToSRGB(color[2]) * 256.0f, 0.0f, 255.0f);

            pixel[0] = R;
            pixel[1] = G;
            pixel[2] = B;

            pixel += 3;
        }
    }

#else
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
#endif

    // draw the location of the sites
    siteIndex = 0;
    for (const Vec2& point : sites)
    {
        const float c_dotRadius = dotRadius*2;

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

void SaveDFT(const char* baseFileName, const std::vector<Vec2>& points, size_t width, size_t height, size_t index)
{
    std::vector<float> pointsForDFT(width * height, 0.0f);

    for (const Vec2& point : points)
    {
        int x = Clamp<int>(int(point[0] * float(width)), 0, (int)width - 1);
        int y = Clamp<int>(int(point[1] * float(height)), 0, (int)height - 1);

        pointsForDFT[y * width + x] = 1.0f;
    }

    std::vector<float> DFTMagnitudes;
    DFTPeriodogram(pointsForDFT, DFTMagnitudes, width, height);

    float maxMag = GetMaxMagnitudeDFT(DFTMagnitudes);
    for (float& f : DFTMagnitudes)
        f /= maxMag;

    std::vector<unsigned char> DFTPixels(width * height);
    for (size_t index = 0; index < width * height; ++index)
        DFTPixels[index] = (unsigned char)Clamp(DFTMagnitudes[index] * 256.0f, 0.0f, 255.0f);

    char buffer[256];
    sprintf_s(buffer, "out/%s.DFT_%zu.png", baseFileName, index);
    stbi_write_png(buffer, (int)width, (int)height, 1, DFTPixels.data(), (int)width);
}

Vec2 CalculateCentroid(const Vec2& site, const SiteInfo& siteInfo, const std::vector<Vec2>& points)
{
    Vec2 centerOfMass = { 0.0f, 0.0f };
    for (size_t pointIndex : siteInfo.members)
    {
        Vec2 point = points[pointIndex];

        // make sure the point is the closest version of the point to the site, torroidally
        {
            float distx = fabs(site[0] - point[0]);
            if (distx > 0.5f)
            {
                if (site[0] < 0.5f)
                    point[0] -= 1.0f;
                else
                    point[0] += 1.0f;
            }

            float disty = fabs(site[1] - point[1]);
            if (disty > 0.5f)
            {
                if (site[1] < 0.5f)
                    point[1] -= 1.0f;
                else
                    point[1] += 1.0f;
            }
        }

        centerOfMass = centerOfMass + point;
    }

    centerOfMass = centerOfMass / float(siteInfo.members.size());
    centerOfMass[0] = Fract(centerOfMass[0]);
    centerOfMass[1] = Fract(centerOfMass[1]);

    return centerOfMass;
}

void MakeCapacityConstraintedVoronoiTessellation(std::mt19937& rng, const std::vector<Vec2>& points, std::vector<Vec2>& sites, const Parameters& params)
{
    // This is "Algorithm 1" in the paper, which is used for step 1 of the "Capacity-Constrained Method"

    const size_t c_numCells = sites.size();
    char buffer[4096];

    // Make a randomized map of points to sites. Each site must have the same number of points.
    std::vector<SiteInfo> sitesInfo(c_numCells);
    {
        std::vector<size_t> pointIndices(points.size());
        std::iota(pointIndices.begin(), pointIndices.end(), (size_t)0);
        std::shuffle(pointIndices.begin(), pointIndices.end(), rng);

        for (size_t index = 0; index < pointIndices.size(); ++index)
            sitesInfo[index % sites.size()].members.push_back(pointIndices[index]);
    }

    // Save the initial state of things
    {
        if (params.showSiteEvolution)
        {
            sprintf_s(buffer, "out/%s.%i.png", params.baseFileName, 0);

            DensityImage image;
            image.MakeImageFromPoints((int)params.siteImageWidth, (int)params.siteImageHeight, sites, params.siteImageDotRadius);
            image.Save(buffer);
        }

        if (params.showSiteEvolution)
            SaveVoronoi(params.baseFileName, sitesInfo, sites, points, params.siteImageWidth, params.siteImageHeight, params.siteImageDotRadius, 0);

        if (params.showDFTEvolution)
            SaveDFT(params.baseFileName, sites, params.DFTImageWidth, params.DFTImageHeight, 0);
    }

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

    bool stable = false;
    int loopCount = 0;
    while (!stable)
    {
        stable = true;
        loopCount++;

        int swapCount = 0;
        int unstableCellPairCount = 0;
        int cellPairCheckCount = 0;

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
                bool didASwap = false;

                if (celli % c_numCellPairs1Percent == 0)
                {
                    int percent = int(100.0f * float(cellPairIndex) / float(c_numCellPairs - 1));
                    printf("\r                                     \riteration %i: %i%%", loopCount, percent);
                }

                // don't check these pairs of cells if we have already checked them previously and nothing has changed.
                CellPairRevisionNumbers& cellPairRevisionNumbers = cellPairRevisions[cellPairIndex];
                cellPairIndex++;
                if (cellPairRevisionNumbers.revisionNumber_i == sitesInfo[celli].revisionNumber && cellPairRevisionNumbers.revisionNumber_j == sitesInfo[cellj].revisionNumber)
                    continue;
                cellPairCheckCount++;

                // for each point belonging to cell i, calculate how much energy would be saved by switching to the other cell
                size_t itemIndex = 0;
                Hi.resize(sitesInfo[celli].members.size());
                for (size_t pointIndex : sitesInfo[celli].members)
                {
                    const Vec2& point = points[pointIndex];

                    HeapItem& heapItem = Hi[itemIndex];
                    heapItem.key = ToroidalDistanceSquared(point, sites[celli]) - ToroidalDistanceSquared(point, sites[cellj]);
                    heapItem.pointIndex = pointIndex;

                    itemIndex++;
                }
                std::make_heap(Hi.begin(), Hi.end());

                // for each point belonging to cell j, calculate how much energy would be saved by switching to the other cell
                itemIndex = 0;
                Hj.resize(sitesInfo[cellj].members.size());
                for (size_t pointIndex : sitesInfo[cellj].members)
                {
                    const Vec2& point = points[pointIndex];

                    HeapItem& heapItem = Hj[itemIndex];
                    heapItem.key = ToroidalDistanceSquared(point, sites[cellj]) - ToroidalDistanceSquared(point, sites[celli]);
                    heapItem.pointIndex = pointIndex;

                    itemIndex++;
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

                    didASwap = true;

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
                }

                if (didASwap)
                {
                    stable = false;

                    // update revision numbers
                    sitesInfo[celli].revisionNumber++;
                    sitesInfo[cellj].revisionNumber++;

                    // move the sites to the centroid of the cell
                    sites[celli] = CalculateCentroid(sites[celli], sitesInfo[celli], points);
                    sites[cellj] = CalculateCentroid(sites[cellj], sitesInfo[cellj], points);

                    unstableCellPairCount++;
                }

                // update revision number whether it changed or not
                cellPairRevisionNumbers.revisionNumber_i = sitesInfo[celli].revisionNumber;
                cellPairRevisionNumbers.revisionNumber_j = sitesInfo[cellj].revisionNumber;
            }
        }

        printf("\r                                     \riteration %i: %i swaps, %i unstable cell pairs, %i cell pairs checked\n", loopCount, swapCount, unstableCellPairCount, cellPairCheckCount);

        // Save the state of things at the end of this iteration
        {
            if (params.showSiteEvolution)
            {
                sprintf_s(buffer, "out/%s.%i.png", params.baseFileName, loopCount);

                DensityImage image;
                image.MakeImageFromPoints((int)params.siteImageWidth, (int)params.siteImageHeight, sites, params.siteImageDotRadius);
                image.Save(buffer);
            }

            if (params.showVoronoiEvolution || (params.showVoronoiFinal && stable))
                SaveVoronoi(params.baseFileName, sitesInfo, sites, points, params.siteImageWidth, params.siteImageHeight, params.siteImageDotRadius, loopCount);

            if (params.showDFTEvolution || (params.showDFTFinal && stable))
                SaveDFT(params.baseFileName, sites, params.DFTImageWidth, params.DFTImageHeight, loopCount);
        }
    }
}

template <typename LAMBDA>
void GeneratePointsFromFunction(std::vector<Vec2>& points, size_t c_numPoints, std::mt19937& rng, bool lowDiscrepancy, const LAMBDA& lambda)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t attemptIndex = 0;
    while (points.size() < c_numPoints)
    {
        float x, y;
        if (lowDiscrepancy)
        {
            x = VanDerCorput(attemptIndex + 1, 2);
            y = VanDerCorput(attemptIndex + 1, 3);
        }
        else
        {
            x = dist(rng);
            y = dist(rng);
        }
        attemptIndex++;

        if (dist(rng) >= lambda(x, y))
            continue;

        points.push_back({ x, y });
    }
}

void GeneratePointsFromImage(const DensityImage& densityImage, std::vector<Vec2>& points, size_t c_numPoints, std::mt19937& rng, bool lowDiscrepancy)
{
    GeneratePointsFromFunction(points, c_numPoints, rng, lowDiscrepancy,
        [&] (float u, float v)
        {
            int x = (int)Clamp(u * float(densityImage.width), 0.0f, float(densityImage.width - 1));
            int y = (int)Clamp(v * float(densityImage.height), 0.0f, float(densityImage.height - 1));
            return densityImage.GetDensity(x, y);
        }
    );
}

template <typename GENERATE_POINTS_LAMBDA>
void GenerateBlueNoisePoints(const Parameters& params, const GENERATE_POINTS_LAMBDA& generatePointsLambda)
{
    ScopedTimer timer(params.baseFileName);

    printf("%s...\n", params.baseFileName);

    // get a random number generator
    std::mt19937 rng = GetRNG(0);

    // Initialize point locations conforming to the density function
    std::vector<Vec2> points;
    generatePointsLambda(points, params.numPoints, rng);

    // initialize site locations
    std::vector<Vec2> sites(params.numSites);
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (Vec2& s : sites)
        {
            s[0] = dist(rng);
            s[1] = dist(rng);
        }
    }

    // save an image of the initial points generated. This is the density distribution function and is what the algorithm uses as input along
    // with the initial site locations.
    char buffer[4096];
    {
        sprintf_s(buffer, "out/%s.in.png", params.baseFileName);
        DensityImage image;
        image.MakeImageFromPoints((int)params.siteImageWidth, (int)params.siteImageHeight, points, params.siteImageDotRadius);
        image.Save(buffer);
    }

    // iteratively optimize the points into blue noise
    MakeCapacityConstraintedVoronoiTessellation(rng, points, sites, params);

    // save an image of the final site locations.  This is the results
    {
        sprintf_s(buffer, "out/%s.out.png", params.baseFileName);
        DensityImage image;
        image.MakeImageFromPoints((int)params.siteImageWidth, (int)params.siteImageHeight, sites, params.siteImageDotRadius);
        image.Save(buffer);
    }
}

int main(int argc, char** argv)
{
    bool doTest_Uniform64   = false;
    bool doTest_Uniform1k   = false;
    bool doTest_Procedural  = false;
    bool doTest_SolidLDS    = true;  // TODO: this and the next are for showing halton doing better than white
    bool doTest_GradientLDS = false;  // TODO: try and make the difference between white and LDS bigger in both solid and gradient
    bool doTest_Gradient    = false;
    bool doTest_PuppySmall  = false;
    bool doTest_Puppy       = false;
    bool doTest_Mountain    = false;

    // TODO: note regarding halton. missing sequences can hurt the sampling pattern. unsure if it's a net gain or not yet (try and find out?)
    // is there an LDS where when values random drop out, the remaining ones are decently lds? might be a good thing to add to the list.

    // This is the "classic" algorithm making regular blue noise.
    // The "classic" algorithm works off of pixels, and this would probably be more efficiently implemented that way instead
    // of having sparse points.
    {
        if (doTest_Uniform64)
        {
            static const size_t c_gridSize = 256;

            Parameters params;
            params.baseFileName = "uniform64";
            params.numSites = 64;
            params.numPoints = c_gridSize * c_gridSize;
            params.siteImageWidth = params.siteImageHeight = c_gridSize;
            params.DFTImageWidth = params.DFTImageHeight = 32;

            GenerateBlueNoisePoints(params,
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

        if (doTest_Uniform1k)
        {
            static const size_t c_gridSize = 256;

            Parameters params;
            params.baseFileName = "uniform1k";
            params.numSites = 1024;
            params.numPoints = c_gridSize * c_gridSize;
            params.siteImageWidth = params.siteImageHeight = c_gridSize;
            params.DFTImageWidth = params.DFTImageHeight = 256;

            GenerateBlueNoisePoints(params,
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
    }

    // Procedural density function
    if (doTest_Procedural)
    {
        Parameters params;
        params.baseFileName = "procedural";
        params.numSites = 1000;
        params.numPoints = params.numSites * 100;

        GenerateBlueNoisePoints(params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromFunction(points, numPoints, rng, false,
                    [](float x, float y)
                    {
                        return (sin(x * 2.0f * c_pi * 4.0f) * 0.5f + 0.5f) * (sin(y * 2.0f * c_pi * 4.0f) * 0.5f + 0.5f);
                    }
                );
            }
        );
    }

    if (doTest_SolidLDS)
    {
        Parameters params;
        params.baseFileName = "solid";
        params.numSites = 4096;
        params.numPoints = params.numSites * 5;
        params.siteImageDotRadius = 0.5f;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", params.baseFileName);
        if (!densityImage.Load(buffer, true))
            return 1;

        params.siteImageWidth = densityImage.width;
        params.siteImageHeight = densityImage.height;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.source.png", params.baseFileName);
        densityImage.Save(buffer);

        // white noise to init the initial distribution
        {
            params.baseFileName = "solid_white";

            GenerateBlueNoisePoints(params,
                [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
                {
                    GeneratePointsFromImage(densityImage, points, numPoints, rng, false);
                }
            );
        }

        // halton to init the initial distribution
        {
            params.baseFileName = "solid_halton";

            GenerateBlueNoisePoints(params,
                [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
                {
                    GeneratePointsFromImage(densityImage, points, numPoints, rng, true);
                }
            );
        }
    }

    if (doTest_GradientLDS)
    {
        Parameters params;
        params.baseFileName = "gradient";
        params.numSites = 1024;
        params.numPoints = params.numSites * 5;
        params.siteImageDotRadius = 0.5f;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", params.baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        params.siteImageWidth = densityImage.width;
        params.siteImageHeight = densityImage.height;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.source.png", params.baseFileName);
        densityImage.Save(buffer);

        // white noise to init the initial distribution
        {
            params.baseFileName = "gradient_white";

            GenerateBlueNoisePoints(params,
                [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
                {
                    GeneratePointsFromImage(densityImage, points, numPoints, rng, false);
                }
            );
        }

        // halton to init the initial distribution
        {
            params.baseFileName = "gradient_halton";

            GenerateBlueNoisePoints(params,
                [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
                {
                    GeneratePointsFromImage(densityImage, points, numPoints, rng, true);
                }
            );
        }
    }

    if (doTest_Gradient)
    {
        Parameters params;
        params.baseFileName = "gradient";
        params.numSites = 1024;
        params.numPoints = params.numSites * 100;
        params.siteImageDotRadius = 0.5f;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", params.baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        params.siteImageWidth = densityImage.width;
        params.siteImageHeight = densityImage.height;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.source.png", params.baseFileName);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng, false);
            }
        );
    }

    if (doTest_PuppySmall)
    {
        Parameters params;
        params.baseFileName = "puppysmall";
        params.numSites = 1024;
        params.numPoints = params.numSites * 100;
        params.siteImageDotRadius = 0.5f;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", params.baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        params.siteImageWidth = densityImage.width;
        params.siteImageHeight = densityImage.height;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.source.png", params.baseFileName);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng, false);
            }
        );
    }

    if(doTest_Puppy)
    {
        Parameters params;
        params.baseFileName = "puppy";
        params.numSites = 1024;
        params.numPoints = params.numSites * 100;
        params.siteImageDotRadius = 2.0f;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", params.baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        params.siteImageWidth = densityImage.width;
        params.siteImageHeight = densityImage.height;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.source.png", params.baseFileName);
        printf("Saving starting image as %s\n\n", buffer);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng, false);
            }
        );
    }

    if (doTest_Mountain)
    {
        static const char* baseFileName = "mountains";

        Parameters params;
        params.baseFileName = "mountains";
        params.numSites = 10240;
        params.numPoints = params.numSites * 10;
        params.siteImageDotRadius = 5.0f;

        // load the image if we can
        char buffer[1024];
        DensityImage densityImage;
        sprintf_s(buffer, "images/%s.png", baseFileName);
        if (!densityImage.Load(buffer))
            return 1;

        params.siteImageWidth = densityImage.width;
        params.siteImageHeight = densityImage.height;

        // save the starting image (we made it greyscale)
        sprintf_s(buffer, "out/%s.png", baseFileName);
        densityImage.Save(buffer);

        GenerateBlueNoisePoints(params,
            [&](std::vector<Vec2>& points, size_t numPoints, std::mt19937& rng)
            {
                GeneratePointsFromImage(densityImage, points, numPoints, rng, false);
            }
        );
    }

    return 0;
}

/*

* LDS vs white noise for that sampling stuff for initial image distribution. or maybe just talk about it and how i'll bet it's better.
 * could just do halton for example.
 * do this on a white image?

* could make a csv of swaps over time to show the convergence rate


NOTES:
* in blog, show the full story of the algorithms talked about in pictures
* Talk about how to make progressive? with void and cluster idea.
* site = a sample point in the results.  point = a density function sample point that is used to make the results.
* brightness of resulting image is proportional to input image, not exact without tuning of point count and point size
* BNOT seems to have basically same quality result, just runs more quickly and is more difficult to implement.
* mention your CCVT "optimization" with cell revision numbers. re-time when everything is working correctly. (before then it went from 10 seconds to 6, so 40% time savings)
* they use white noise to generate the binary sample points, what?!
* i originally thought this algorithm used float density pixel values and was getting bad results.
* generating colors programatically: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
* a white image won't give you good results for ccvt because it'll sample it with white noise and get a white noise result. you want to sample it with low discrepancy sequence to get best results, which will also give the same results for a constant color texture vs making a grid of constant density.
 * could try it out w/ halton vs white noise on a white image!
 * this matters cause there could be images with large regions of similar colors (densities)
? i wonder how bilinear interpolation would look, in making the original point set, instead of making it stuck to a grid?
* the official implementation for the paper has numpoints = numsites*1024, but this is tuneable for quality vs speed
* i made voronoi's solid, but in reality they aren't (could show)
* i'm sure there's some good optimizations to be done to the algorithm. 






*/
