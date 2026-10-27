#include "SimulationEngine.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define SLEEP_MS(ms) Sleep(ms)
#else
#include <unistd.h>
#define SLEEP_MS(ms) usleep((ms) * 1000)
#endif

// ANSI Color helper codes
namespace Color {
    const char* RESET   = "\033[0m";
    const char* RED     = "\033[1;31m";
    const char* GREEN   = "\033[1;32m";
    const char* YELLOW  = "\033[1;33m";
    const char* BLUE    = "\033[1;34m";
    const char* MAGENTA = "\033[1;35m";
    const char* CYAN    = "\033[1;36m";
    const char* WHITE   = "\033[1;37m";
    const char* GRAY    = "\033[0;90m";
}

SimulationEngine::SimulationEngine(const SimulationConfig& config)
    : m_config(config),
      m_httpClient(std::make_unique<HttpClient>(config.backendHost, config.backendPort)),
      m_running(false)
{
}

void SimulationEngine::addNode(std::unique_ptr<EdgeNode> node) {
    if (node) {
        m_nodes.push_back(std::move(node));
    }
}

void SimulationEngine::setupDefaultSwarm() {
    m_nodes.clear();

    // Default deployment zone: Amazon basin bioreserve perimeter
    struct NodeDef {
        const char* id;
        double lat;
        double lon;
        double radiusKm;
    };

    std::vector<NodeDef> defaults = {
        {"NODE-ALPHA",    -3.4461, -62.2104, 1.8},
        {"NODE-JULIETT",  -3.4498, -62.2288, 1.8},
        {"NODE-NOVEMBER", -3.4102, -62.2196, 1.8},
        {"NODE-HOTEL",    -3.4217, -62.1828, 1.8},
        {"NODE-INDIA",    -3.4369, -62.2357, 1.8},
        {"NODE-CHARLIE",  -3.4470, -62.2660, 1.8},
        {"NODE-FOXTROT",  -3.4723, -62.1828, 2.0},
        {"NODE-GOLF",     -3.4663, -62.2196, 1.8},
        {"NODE-KILO",     -3.4323, -62.2550, 1.8},
        {"NODE-OSCAR",    -3.4571, -62.2136, 2.0},
        {"NODE-DELTA",    -3.4484, -62.2449, 1.8},
        {"NODE-LIMA",     -3.4709, -62.2702, 2.0},
        {"NODE-ECHO",     -3.4718, -62.2426, 2.0},
        {"NODE-BRAVO",    -3.4194, -62.2624, 2.0},
        {"NODE-PAPA",     -3.4056, -62.1529, 2.0},
        {"NODE-MIKE",     -3.4553, -62.1888, 2.0}
    };

    int countToCreate = std::min(m_config.nodeCount, static_cast<int>(defaults.size()));
    for (int i = 0; i < countToCreate; ++i) {
        m_nodes.push_back(std::make_unique<EdgeNode>(
            defaults[i].id,
            defaults[i].lat,
            defaults[i].lon,
            defaults[i].radiusKm
        ));
    }
}

void SimulationEngine::run() {
    if (m_nodes.empty()) {
        setupDefaultSwarm();
    }

    m_running = true;
    std::cout << Color::CYAN << "\n=================================================================" << Color::RESET << std::endl;
    std::cout << Color::WHITE << "  EchoGuard Bioacoustic Edge Node Simulator running..." << Color::RESET << std::endl;
    std::cout << Color::GRAY << "  Target: http://" << m_config.backendHost << ":" << m_config.backendPort << "/api/v1/telemetry/" << Color::RESET << std::endl;
    std::cout << Color::GRAY << "  Active Sensor Nodes: " << m_nodes.size() << " | Interval: " << m_config.intervalMs << "ms" << Color::RESET << std::endl;
    std::cout << Color::CYAN << "=================================================================\n" << Color::RESET << std::endl;

    // Check backend health
    bool reachable = m_httpClient->testConnection("/health");
    if (reachable) {
        std::cout << Color::GREEN << "[INIT] Backend connection verified at http://" << m_config.backendHost << ":" << m_config.backendPort << Color::RESET << "\n" << std::endl;
    } else {
        std::cout << Color::YELLOW << "[WARN] Backend /health not responding at http://" << m_config.backendHost << ":" << m_config.backendPort << ". Alerts will still attempt dispatch." << Color::RESET << "\n" << std::endl;
    }

    int iteration = 0;
    while (m_running) {
        processIteration(++iteration);

        if (m_config.maxIterations > 0 && iteration >= m_config.maxIterations) {
            std::cout << Color::CYAN << "\n[SIM] Reached max iterations (" << m_config.maxIterations << "). Stopping.\n" << Color::RESET;
            break;
        }

        SLEEP_MS(m_config.intervalMs);
    }
}

void SimulationEngine::processIteration(int iterationIndex) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> chanceDist(0.0, 1.0);

    for (const auto& node : m_nodes) {
        if (!node || !node->isOperational()) continue;

        double roll = chanceDist(gen);
        if (roll < m_config.anomalyProbability) {
            // Threat detected!
            ThreatEvent event = node->detectRandomThreat();
            HttpResponse resp = m_httpClient->sendTelemetryAlert(event);
            logEvent(event, resp);
        }
    }
}

void SimulationEngine::runSingleShot(const std::string& nodeId, ThreatType threatType, double confidence) {
    std::cout << Color::CYAN << "[SINGLE-SHOT] Dispatching alert for node " << nodeId << "..." << Color::RESET << std::endl;

    double lat = -3.4653;
    double lon = -62.2159;

    // Search if node exists
    for (const auto& n : m_nodes) {
        if (n && n->getId() == nodeId) {
            lat = n->getLatitude();
            lon = n->getLongitude();
            break;
        }
    }

    ThreatEvent event = AcousticSignature::generateEvent(
        nodeId,
        threatType,
        lat,
        lon,
        confidence,
        confidence
    );

    HttpResponse resp = m_httpClient->sendTelemetryAlert(event);
    logEvent(event, resp);
}

void SimulationEngine::runBurstScenario(int burstCount, ThreatType primaryThreat) {
    if (m_nodes.empty()) setupDefaultSwarm();

    std::cout << Color::RED << "\n[BURST SCENARIO] Initiating threat surge: " 
              << burstCount << " rapid alerts (" 
              << AcousticSignature::threatTypeToString(primaryThreat) << ")..." 
              << Color::RESET << std::endl;

    for (int i = 0; i < burstCount; ++i) {
        size_t nodeIdx = i % m_nodes.size();
        ThreatEvent event = m_nodes[nodeIdx]->detectThreat(primaryThreat);
        HttpResponse resp = m_httpClient->sendTelemetryAlert(event);
        logEvent(event, resp);
        SLEEP_MS(300); // Fast burst cadence
    }
}

void SimulationEngine::logEvent(const ThreatEvent& event, const HttpResponse& response) {
    // Select color based on threat category
    const char* threatColor = Color::YELLOW;
    if (event.threatName == "CHAINSAW" || event.threatName == "GUNSHOT" || event.threatName == "BLAST_FISHING") {
        threatColor = Color::RED;
    } else if (event.threatName == "ILLEGAL_VESSEL" || event.threatName == "VEHICLE_ENGINE") {
        threatColor = Color::MAGENTA;
    }

    std::cout << Color::GRAY << "[" << event.timestampIso << "] " << Color::RESET
              << Color::WHITE << "[" << std::setw(12) << std::left << event.deviceId << "] " << Color::RESET
              << threatColor << std::setw(16) << std::left << event.threatName << Color::RESET
              << " | Conf: " << Color::WHITE << std::fixed << std::setprecision(2) << event.confidenceScore << Color::RESET
              << " | Lat: " << std::fixed << std::setprecision(4) << event.latitude 
              << " Lon: " << std::fixed << std::setprecision(4) << event.longitude;

    if (response.success) {
        std::cout << " -> " << Color::GREEN << "[HTTP " << response.statusCode << " CREATED]" << Color::RESET << std::endl;
    } else {
        std::cout << " -> " << Color::RED << "[FAILED: " << response.errorMessage << "]" << Color::RESET << std::endl;
    }
}

void SimulationEngine::stop() {
    m_running = false;
}

bool SimulationEngine::isRunning() const {
    return m_running;
}
