// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – verbose file logger
// Activated by environment variable: DEFLATTEN_LOG_FILE=<path>
#pragma once
#include <windows.h>
#include <string>
#include <fstream>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>

class Logger {
public:
    static Logger& Instance() {
        static Logger s;
        return s;
    }

    // Call once during DLL attach; reads DEFLATTEN_LOG_FILE env var.
    void Init(const wchar_t* dllPath);

    bool IsEnabled() const { return m_enabled; }

    template<typename... Args>
    void Log(const char* level, Args&&... args) {
        if (!m_enabled) return;
        std::ostringstream oss;
        oss << Timestamp() << " [" << level << "] ";
        (oss << ... << std::forward<Args>(args));
        oss << "\n";
        Write(oss.str());
    }

    void LogFmt(const char* level, const char* fmt, ...);

private:
    Logger() = default;
    std::string Timestamp();
    void        Write(const std::string& line);

    bool          m_enabled = false;
    std::mutex    m_mtx;
    std::ofstream m_file;
};

// ── Convenience macros ────────────────────────────────────────────────────────
#define LOG_INFO(...) Logger::Instance().Log("INFO ", __VA_ARGS__)
#define LOG_WARN(...) Logger::Instance().Log("WARN ", __VA_ARGS__)
#define LOG_ERR(...)  Logger::Instance().Log("ERROR", __VA_ARGS__)
#define LOG_DBG(...)  Logger::Instance().Log("DEBUG", __VA_ARGS__)
