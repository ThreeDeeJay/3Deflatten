// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten - verbose file logger
// Activated by environment variable: DEFLATTEN_LOG_FILE=<path>
#pragma once
#include <windows.h>
#include <string>
#include <fstream>
#include <mutex>
#include <sstream>
#include <cwchar>

class Logger {
public:
    static Logger& Instance() {
        static Logger s;
        return s;
    }

    void Init(const wchar_t* dllPath);
    bool IsEnabled() const { return m_enabled; }

    // stream_arg: writes one argument into oss, with special handling for
    // wide strings (which ostringstream cannot accept directly).
    static void stream_arg(std::ostringstream& oss, const std::wstring& w) {
        oss << std::string(w.begin(), w.end());
    }
    static void stream_arg(std::ostringstream& oss, const wchar_t* w) {
        if (w) oss << std::string(w, w + std::wcslen(w));
    }
    template<typename T>
    static void stream_arg(std::ostringstream& oss, const T& v) {
        oss << v;
    }

    template<typename... Args>
    void Log(const char* level, Args&&... args) {
        if (!m_enabled) return;
        std::ostringstream oss;
        oss << "[" << level << "] ";
        int dummy[] = { 0, (stream_arg(oss, args), 0)... };
        (void)dummy;
        oss << "\n";
        Write(oss.str());
    }

    void LogFmt(const char* level, const char* fmt, ...);

private:
    Logger() = default;
    void Write(const std::string& line);

    bool          m_enabled = false;
    std::mutex    m_mtx;
    std::ofstream m_file;
};

#define LOG_INFO(...) Logger::Instance().Log("INFO ", __VA_ARGS__)
#define LOG_WARN(...) Logger::Instance().Log("WARN ", __VA_ARGS__)
#define LOG_ERR(...)  Logger::Instance().Log("ERROR", __VA_ARGS__)
#define LOG_DBG(...)  Logger::Instance().Log("DEBUG", __VA_ARGS__)
