// SPDX-License-Identifier: GPL-3.0-or-later
#include "logger.h"
#include <cstdarg>
#include <ctime>

void Logger::Init(const wchar_t* /*dllPath*/) {
    wchar_t envBuf[MAX_PATH] = {};
    if (!GetEnvironmentVariableW(L"DEFLATTEN_LOG_FILE", envBuf, MAX_PATH))
        return;

    std::lock_guard<std::mutex> lk(m_mtx);
    m_file.open(envBuf, std::ios::app);
    if (!m_file.is_open()) return;

    m_enabled = true;
    m_file << "\n====== 3Deflatten log opened ======\n";
    m_file.flush();
}

std::string Logger::Timestamp() {
    return {};  // timestamps removed -- log entries are sequential
}

void Logger::Write(const std::string& line) {
    std::lock_guard<std::mutex> lk(m_mtx);
    if (!m_file.is_open()) return;
    m_file << line;
    m_file.flush();
}

void Logger::LogFmt(const char* level, const char* fmt, ...) {
    if (!m_enabled) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    Log(level, buf);
}
