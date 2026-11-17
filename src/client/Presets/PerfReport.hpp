/*
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// perfreport: runs the sweeps described by an INI file (PERFREPORT_INI), or the default
// tests below, and writes the results as JSON into one folder (PERFREPORT_DIR, default
// ./perfreport_<YYYYmmdd-HHMMSS>): run.json, plan.json, platform.json, platform/,
// results.jsonl and input.ini. PerfReport.html, next to this file, displays them.

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <pwd.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "Topology.hpp"

extern char** environ;

namespace PerfReport
{
  using StrPairs = std::vector<std::pair<std::string, std::string>>;

  // An INI file, or the default tests, as written: every section's keys and values in order.
  struct Input
  {
    StrPairs                                      globals;
    StrPairs                                      tests;
    std::vector<std::pair<std::string, StrPairs>> sweeps;
  };

  //==========================================================================================
  // Default tests, used when PERFREPORT_INI is not set. Each function adds tests and sweeps
  // with the same keys and values an INI file uses, for example:
  //   in.tests.push_back({"H2D", "1 {cus} (P0->G0->G0)"});
  //   in.sweeps.push_back({"h2d-cus", {{"tests", "H2D"}, {"cus", "1, 8, 32"}}});
  //==========================================================================================
  void AddFanoutTests ([[maybe_unused]] Input& in, [[maybe_unused]] int numGpus) {}

  void AddDefaultTests(Input& in, int numGpus)
  {
    AddFanoutTests(in, numGpus);
  }

  char const* const kVersion = "1";

  //==========================================================================================
  // Text helpers
  //==========================================================================================
  std::string Trim(std::string const& s)
  {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
  }

  std::string Squeeze(std::string const& s)
  {
    std::istringstream words(s);
    std::string word, out;
    while (words >> word) out += (out.empty() ? "" : " ") + word;
    return out;
  }

  std::vector<std::string> Split(std::string const& s, char sep)
  {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
      size_t pos = s.find(sep, start);
      out.push_back(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
      if (pos == std::string::npos) break;
      start = pos + 1;
    }
    return out;
  }

  std::string Lower(std::string s)
  {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)tolower(c); });
    return s;
  }

  bool IsName(std::string const& s)
  {
    if (s.empty() || !(isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
    for (char c : s) if (!(isalnum((unsigned char)c) || c == '_')) return false;
    return true;
  }

  bool IsTestName(std::string const& s)
  {
    if (s.empty()) return false;
    for (char c : s) if (!(isalnum((unsigned char)c) || c == '_' || c == '-')) return false;
    return true;
  }

  std::string const* Find(StrPairs const& kv, std::string const& key)
  {
    for (auto const& p : kv) if (p.first == key) return &p.second;
    return nullptr;
  }

  std::string ReadFile(std::string const& path, bool* ok = nullptr)
  {
    std::ifstream f(path, std::ios::binary);
    if (ok) *ok = (bool)f;
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }

  // Written to a temporary name and renamed, so readers never see half a file.
  bool WriteFile(std::string const& path, std::string const& text)
  {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return false;
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    ok = (fclose(f) == 0) && ok;
    return ok && rename(tmp.c_str(), path.c_str()) == 0;
  }

  bool MakeDirs(std::string const& path)
  {
    std::string cur;
    for (auto const& part : Split(path, '/')) {
      cur += part + "/";
      if (part.empty()) continue;
      if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return true;
  }

  // The standard output of a shell command; empty if it can't run.
  std::string Capture(std::string const& cmd, bool* ok = nullptr)
  {
    std::string out;
    FILE* p = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!p) {
      if (ok) *ok = false;
      return out;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
    int rc = pclose(p);
    if (ok) *ok = (rc == 0);
    return out;
  }

  std::string NowIso()
  {
    time_t t = time(nullptr);
    struct tm lt;
    localtime_r(&t, &lt);
    char buf[40];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &lt);
    std::string s = buf;
    if (s.size() >= 5) s.insert(s.size() - 2, ":");
    return s;
  }

  std::string Stamp()
  {
    time_t t = time(nullptr);
    struct tm lt;
    localtime_r(&t, &lt);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &lt);
    return buf;
  }

  std::string RealPath(std::string const& path)
  {
    char buf[PATH_MAX];
    return realpath(path.c_str(), buf) ? std::string(buf) : path;
  }

  std::string SelfExe()
  {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, n) : "";
  }

  std::string FmtDuration(double s)
  {
    char buf[32];
    if (s < 60) snprintf(buf, sizeof(buf), "%.1f s", s);
    else if (s < 3600) snprintf(buf, sizeof(buf), "%dm%02ds", (int)s / 60, (int)s % 60);
    else snprintf(buf, sizeof(buf), "%dh%02dm", (int)s / 3600, (int)s % 3600 / 60);
    return buf;
  }

  //==========================================================================================
  // JSON writing and reading
  //==========================================================================================
  std::string Q(std::string const& s)
  {
    std::string out = "\"";
    for (unsigned char ch : s) {
      switch (ch) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (ch < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", ch);
          out += buf;
        } else {
          out += char(ch);
        }
      }
    }
    return out + "\"";
  }

  std::string Num(double v)
  {
    if (!std::isfinite(v)) return "null";
    char buf[32];
    snprintf(buf, sizeof(buf), "%.12g", v);
    return buf;
  }

  std::string StrList(std::vector<std::string> const& v)
  {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + Q(v[i]);
    return s + "]";
  }

  std::string StrMap(StrPairs const& kv)
  {
    std::string s = "{";
    for (size_t i = 0; i < kv.size(); i++) s += (i ? "," : "") + Q(kv[i].first) + ":" + Q(kv[i].second);
    return s + "}";
  }

  struct JVal
  {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    std::string                               text;   // a string, or a number as written
    bool                                      flag = false;
    std::vector<JVal>                         items;
    std::vector<std::pair<std::string, JVal>> fields;

    JVal const* Get(std::string const& key) const
    {
      if (kind != OBJ) return nullptr;
      for (auto const& f : fields) if (f.first == key) return &f.second;
      return nullptr;
    }
  };

  class JReader
  {
  public:
    explicit JReader(std::string const& s) : s_(s) {}

    bool Parse(JVal& v)
    {
      Skip();
      if (!Value(v, 0)) return false;
      Skip();
      return p_ == s_.size();
    }

  private:
    std::string const& s_;
    size_t             p_ = 0;

    void Skip() { while (p_ < s_.size() && isspace((unsigned char)s_[p_])) p_++; }

    bool Lit(char const* w)
    {
      size_t n = strlen(w);
      if (s_.compare(p_, n, w) != 0) return false;
      p_ += n;
      return true;
    }

    bool Str(std::string& out)
    {
      if (p_ >= s_.size() || s_[p_] != '"') return false;
      p_++;
      while (p_ < s_.size()) {
        char c = s_[p_++];
        if (c == '"') return true;
        if (c != '\\') {
          out += c;
          continue;
        }
        if (p_ >= s_.size()) return false;
        char e = s_[p_++];
        switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          if (p_ + 4 > s_.size()) return false;
          unsigned cp = (unsigned)strtoul(s_.substr(p_, 4).c_str(), nullptr, 16);
          p_ += 4;
          if (cp < 0x80) {
            out += char(cp);
          } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
          } else {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
          }
          break;
        }
        default: out += e;
        }
      }
      return false;
    }

    bool Value(JVal& v, int depth)
    {
      if (depth > 64 || p_ >= s_.size()) return false;
      char c = s_[p_];
      if (c == '{' || c == '[') {
        bool obj = (c == '{');
        v.kind = obj ? JVal::OBJ : JVal::ARR;
        p_++;
        Skip();
        if (p_ < s_.size() && s_[p_] == (obj ? '}' : ']')) {
          p_++;
          return true;
        }
        while (true) {
          Skip();
          std::string key;
          if (obj) {
            if (!Str(key)) return false;
            Skip();
            if (p_ >= s_.size() || s_[p_] != ':') return false;
            p_++;
            Skip();
          }
          JVal item;
          if (!Value(item, depth + 1)) return false;
          if (obj) v.fields.emplace_back(key, std::move(item));
          else     v.items.push_back(std::move(item));
          Skip();
          if (p_ >= s_.size()) return false;
          if (s_[p_] == ',') {
            p_++;
            continue;
          }
          if (s_[p_] == (obj ? '}' : ']')) {
            p_++;
            return true;
          }
          return false;
        }
      }
      if (c == '"') {
        v.kind = JVal::STR;
        return Str(v.text);
      }
      if (Lit("true"))  { v.kind = JVal::BOOL; v.flag = true;  return true; }
      if (Lit("false")) { v.kind = JVal::BOOL; v.flag = false; return true; }
      if (Lit("null"))  { v.kind = JVal::NUL; return true; }
      size_t start = p_;
      while (p_ < s_.size() && strchr("+-0123456789.eE", s_[p_])) p_++;
      if (p_ == start) return false;
      v.kind = JVal::NUM;
      v.text = s_.substr(start, p_ - start);
      return true;
    }
  };

  std::string JWrite(JVal const& v)
  {
    switch (v.kind) {
    case JVal::NUL:  return "null";
    case JVal::BOOL: return v.flag ? "true" : "false";
    case JVal::NUM:  return v.text;
    case JVal::STR:  return Q(v.text);
    case JVal::ARR: {
      std::string s = "[";
      for (size_t i = 0; i < v.items.size(); i++) s += (i ? "," : "") + JWrite(v.items[i]);
      return s + "]";
    }
    case JVal::OBJ: {
      std::string s = "{";
      for (size_t i = 0; i < v.fields.size(); i++) s += (i ? "," : "") + Q(v.fields[i].first) + ":" + JWrite(v.fields[i].second);
      return s + "}";
    }
    }
    return "null";
  }

  // Serial numbers, UUIDs and FRU ids are removed from every saved file, because reports get shared.
  bool IsIdentifier(std::string const& key)
  {
    std::string k = Lower(key);
    return k.find("serial") != std::string::npos || k.find("uuid") != std::string::npos ||
           k.find("fru") != std::string::npos;
  }

  void Scrub(JVal& v)
  {
    if (v.kind == JVal::OBJ) {
      std::vector<std::pair<std::string, JVal>> kept;
      for (auto& f : v.fields) {
        if (IsIdentifier(f.first)) continue;
        Scrub(f.second);
        kept.push_back(std::move(f));
      }
      v.fields.swap(kept);
    } else if (v.kind == JVal::ARR) {
      for (auto& item : v.items) Scrub(item);
    }
  }

  std::string ScrubText(std::string const& text)
  {
    std::istringstream lines(text);
    std::string line, out;
    while (std::getline(lines, line)) {
      std::string low = Lower(line);
      size_t k = std::min(low.find("uuid"), low.find("serial"));
      if (k != std::string::npos) {
        size_t sep = line.find_first_of(":=", k);
        if (sep != std::string::npos) line = line.substr(0, sep + 1) + " (removed)";
      }
      out += line + "\n";
    }
    return out;
  }

  // The JSON of a value, with amd-smi's {"value": ...} wrappers and "N/A" undone.
  std::string JText(JVal const* v)
  {
    if (!v) return "null";
    if (v->kind == JVal::OBJ) {
      JVal const* inner = v->Get("value");
      return inner ? JText(inner) : "null";
    }
    if (v->kind == JVal::STR && v->text == "N/A") return "null";
    return JWrite(*v);
  }

  JVal const* At(JVal const* v, std::vector<std::string> const& keys)
  {
    for (auto const& k : keys) {
      if (!v) return nullptr;
      v = v->Get(k);
    }
    return v;
  }

  // amd-smi gives {"gpu_data": [...]} for some commands and a bare list for others.
  std::vector<JVal> const* GpuList(JVal const& v)
  {
    if (v.kind == JVal::OBJ) {
      JVal const* d = v.Get("gpu_data");
      return d && d->kind == JVal::ARR ? &d->items : nullptr;
    }
    return v.kind == JVal::ARR ? &v.items : nullptr;
  }

  //==========================================================================================
  // INI files (the same rules as an INI v1 file for tb-run)
  //==========================================================================================
  char const* const kRunnerSettings[] = {"tb_source", "lib_path", "output_dir"};
  std::map<std::string, std::string> const kLegacyKeys = {
    {"tb_bin",    "perfreport runs inside TransferBench, so it needs no binary"},
    {"envs",      "write one env.NAME = values line per variable"},
    {"executors", "use an axis such as exe = G, D and put {exe} in the test lines"},
    {"se",        "use an axis such as cus = 8, 32 and put {cus} in the test lines"},
    {"size_min",  "use size = low..high"},
    {"size_max",  "use size = low..high"},
  };
  StrPairs const kLegacyLine = {
    {"->*",  "put a placeholder such as {exe} where the executor goes"},
    {"<SE>", "put a placeholder such as {cus} where the CU count goes"},
    {";",    "split parts into separate tests, or sweep a placeholder such as {root}"},
  };

  bool IsRunnerSetting(std::string const& key)
  {
    for (char const* s : kRunnerSettings) if (key == s) return true;
    return false;
  }

  bool ReadIni(std::string const& path, Input& in, std::string& err)
  {
    std::ifstream f(path);
    if (!f) {
      err = "cannot read the INI file " + path;
      return false;
    }
    struct Section { std::string name; StrPairs kv; };
    std::vector<Section> secs;
    std::string line;
    int lineNo = 0, keyIndent = -1;
    bool inValue = false;
    while (std::getline(f, line)) {
      lineNo++;
      std::string const where = "line " + std::to_string(lineNo) + ": ";
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t first = line.find_first_not_of(" \t");
      std::string text = Trim(line);
      if (text.empty() || text[0] == '#') {
        inValue = false;
        continue;
      }
      int indent = (int)first;
      if (inValue && indent > keyIndent) {
        secs.back().kv.back().second += "\n" + text;
        continue;
      }
      if (text.front() == '[' && text.back() == ']') {
        std::string name = text.substr(1, text.size() - 2);
        for (auto const& s : secs) {
          if (s.name == name) {
            err = where + "section [" + name + "] appears twice";
            return false;
          }
        }
        secs.push_back({name, {}});
        inValue = false;
        continue;
      }
      if (secs.empty()) {
        err = where + "text before the first [section]";
        return false;
      }
      size_t eq = text.find('=');
      if (eq == std::string::npos) {
        err = where + "'" + text + "' is not key = value";
        return false;
      }
      std::string key = Trim(text.substr(0, eq)), value = Trim(text.substr(eq + 1));
      if (key.empty()) {
        err = where + "a value without a key";
        return false;
      }
      if (Find(secs.back().kv, key)) {
        err = where + "[" + secs.back().name + "] sets " + key + " twice";
        return false;
      }
      secs.back().kv.emplace_back(key, value);
      keyIndent = indent;
      inValue = true;
    }

    bool haveTests = false;
    for (auto& s : secs) {
      if (s.name == "globals") {
        in.globals = s.kv;
      } else if (s.name == "tests") {
        in.tests = s.kv;
        haveTests = true;
      } else if (s.name.rfind("sweep:", 0) == 0 && s.name.size() > 6) {
        in.sweeps.emplace_back(s.name.substr(6), s.kv);
      } else {
        err = "unknown section [" + s.name + "]; the sections are [globals], [tests] and [sweep:<name>]";
        return false;
      }
    }
    if (!haveTests) {
      err = "no [tests] section";
      return false;
    }
    return true;
  }

  std::string ToIni(Input const& in)
  {
    auto put = [](StrPairs const& kv) {
      std::string s;
      for (auto const& p : kv) {
        std::string v = p.second;
        for (size_t i = 0; (i = v.find('\n', i)) != std::string::npos; i += 5) v.replace(i, 1, "\n    ");
        s += p.first + " = " + v + "\n";
      }
      return s;
    };
    std::string s;
    if (!in.globals.empty()) s += "[globals]\n" + put(in.globals) + "\n";
    s += "[tests]\n" + put(in.tests);
    for (auto const& sw : in.sweeps) s += "\n[sweep:" + sw.first + "]\n" + put(sw.second);
    return s;
  }

  bool ToInt(std::string tok, unsigned long long& out, std::string& err)
  {
    tok = Trim(tok);
    unsigned long long mult = 1;
    if (!tok.empty()) {
      char u = (char)toupper((unsigned char)tok.back());
      if (u == 'K') mult = 1ULL << 10;
      if (u == 'M') mult = 1ULL << 20;
      if (u == 'G') mult = 1ULL << 30;
    }
    std::string digits = mult > 1 ? tok.substr(0, tok.size() - 1) : tok;
    bool ok = !digits.empty() && digits.size() <= 15;
    for (char c : digits) ok = ok && isdigit((unsigned char)c);
    if (!ok) {
      err = "'" + tok + "' is not a whole number";
      return false;
    }
    out = strtoull(digits.c_str(), nullptr, 10) * mult;
    return true;
  }

  bool ParseValues(std::string const& key, std::string const& text, std::vector<std::string>& values,
                   std::string& err)
  {
    values.clear();
    for (std::string tok : Split(text, ',')) {
      tok = Trim(tok);
      if (tok.empty()) continue;
      size_t dots = tok.find("..");
      if (dots != std::string::npos) {
        unsigned long long lo, hi;
        if (!ToInt(tok.substr(0, dots), lo, err) || !ToInt(tok.substr(dots + 2), hi, err)) return false;
        if (lo < 1 || (lo & (lo - 1)) || (hi & (hi - 1)) || lo > hi) {
          err = key + " = " + text + ": '" + tok + "' must be powers of two, low..high";
          return false;
        }
        for (unsigned long long v = lo; v <= hi; v *= 2) values.push_back(std::to_string(v));
      } else if (key == "size") {
        unsigned long long v;
        if (!ToInt(tok, v, err)) return false;
        values.push_back(std::to_string(v));
      } else {
        values.push_back(tok);
      }
    }
    if (values.empty()) {
      err = key + " has no values";
      return false;
    }
    if (std::set<std::string>(values.begin(), values.end()).size() < values.size()) {
      err = key + " = " + text + ": a value appears twice";
      return false;
    }
    return true;
  }

  bool CheckKey(std::string const& key, std::string const& where, std::string& err)
  {
    auto legacy = kLegacyKeys.find(key);
    if (legacy != kLegacyKeys.end()) {
      err = "[" + where + "] " + key + " is no longer used: " + legacy->second;
      return false;
    }
    std::string name = key.rfind("env.", 0) == 0 ? key.substr(4) : key;
    if (!IsName(name)) {
      err = "[" + where + "] '" + key + "' is not a valid axis name";
      return false;
    }
    return true;
  }

  bool CheckTimeout(std::string const& text, std::string const& where, int& out, std::string& err)
  {
    bool ok = !text.empty() && text.size() <= 9;
    for (char c : text) ok = ok && isdigit((unsigned char)c);
    if (!ok || atoi(text.c_str()) < 1) {
      err = "[" + where + "] timeout must be a positive number of seconds";
      return false;
    }
    out = atoi(text.c_str());
    return true;
  }

  std::vector<std::string> Placeholders(std::string const& line)
  {
    std::vector<std::string> out;
    for (size_t i = 0; i < line.size(); i++) {
      if (line[i] != '{') continue;
      size_t j = line.find('}', i + 1);
      if (j == std::string::npos) break;
      std::string name = line.substr(i + 1, j - i - 1);
      if (IsName(name) && std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
    }
    return out;
  }

  std::string Fill(std::string const& line, std::map<std::string, std::string> const& values)
  {
    std::string out;
    for (size_t i = 0; i < line.size(); i++) {
      if (line[i] == '{') {
        size_t j = line.find('}', i + 1);
        if (j != std::string::npos) {
          auto it = values.find(line.substr(i + 1, j - i - 1));
          if (it != values.end()) {
            out += it->second;
            i = j;
            continue;
          }
        }
      }
      out += line[i];
    }
    return out;
  }

  struct Sweep
  {
    std::string                                                  name, note;
    std::vector<std::string>                                     tests;
    int                                                          timeout = 300;
    std::vector<std::pair<std::string, std::vector<std::string>>> axes;
  };

  struct Run
  {
    std::string         id, sweep, test, line;
    StrPairs            settings;
    std::vector<size_t> sizes;
    int                 timeout = 300;
  };

  struct Plan
  {
    StrPairs                 tests;
    std::vector<Sweep>       sweeps;
    std::vector<Run>         runs;
    std::vector<std::string> warnings;
  };

  using Axes = std::vector<std::pair<std::string, std::vector<std::string>>>;

  void SetAxis(Axes& axes, std::string const& key, std::vector<std::string> const& values)
  {
    for (auto& a : axes) {
      if (a.first == key) {
        a.second = values;
        return;
      }
    }
    axes.emplace_back(key, values);
  }

  std::vector<std::string> const* AxisOf(Axes const& axes, std::string const& key)
  {
    for (auto const& a : axes) if (a.first == key) return &a.second;
    return nullptr;
  }

  // Expand tests and sweeps into runs: one run per test and combination of the axes that
  // apply to it, every env. axis plus each placeholder in the test's line.
  bool Expand(Input const& in, bool fromIni, Plan& plan, std::string& err)
  {
    Axes defaults;
    std::vector<std::string> values;
    if (!ParseValues("size", "1K..512M", values, err)) return false;
    defaults.emplace_back("size", values);
    std::string globalTimeout = "300";
    int unused;
    for (auto const& kv : in.globals) {
      if (kv.first == "timeout") {
        if (!CheckTimeout(kv.second, "globals", unused, err)) return false;
        globalTimeout = kv.second;
      } else if (kv.first == "tests" || kv.first == "note") {
        err = "[globals] " + kv.first + " belongs in a [sweep:<name>] section";
        return false;
      } else if (IsRunnerSetting(kv.first)) {
        plan.warnings.push_back("[globals] " + kv.first + " is a tb-run setting; perfreport ignores it" +
                                (kv.first == "output_dir" ? " (set PERFREPORT_DIR instead)" : ""));
      } else {
        if (!CheckKey(kv.first, "globals", err) || !ParseValues(kv.first, kv.second, values, err)) return false;
        SetAxis(defaults, kv.first, values);
      }
    }

    for (auto const& kv : in.tests) {
      if (!IsTestName(kv.first)) {
        err = "[tests] '" + kv.first + "': test names use letters, digits, '_' and '-'";
        return false;
      }
      for (auto const& legacy : kLegacyLine) {
        if (kv.second.find(legacy.first) != std::string::npos) {
          err = "[tests] " + kv.first + ": '" + legacy.first + "' is from the old format; " + legacy.second;
          return false;
        }
      }
      plan.tests.emplace_back(kv.first, Squeeze(kv.second));
    }

    for (auto const& sec : in.sweeps) {
      std::string const where = "sweep:" + sec.first;
      std::string const* testList = Find(sec.second, "tests");
      if (!testList) {
        err = "[" + where + "] needs a 'tests =' line";
        return false;
      }
      Sweep sw;
      sw.name = sec.first;
      for (std::string pat : Split(*testList, ',')) {
        pat = Trim(pat);
        if (pat.empty()) continue;
        bool hit = false;
        for (auto const& t : plan.tests) {
          if (fnmatch(pat.c_str(), t.first.c_str(), 0) != 0) continue;
          hit = true;
          if (std::find(sw.tests.begin(), sw.tests.end(), t.first) == sw.tests.end()) sw.tests.push_back(t.first);
        }
        if (!hit) {
          err = "[" + where + "] tests: nothing in [tests] matches '" + pat + "'";
          return false;
        }
      }
      std::string const* timeout = Find(sec.second, "timeout");
      if (!CheckTimeout(timeout ? *timeout : globalTimeout, where, sw.timeout, err)) return false;
      std::string const* note = Find(sec.second, "note");
      sw.note = note ? Squeeze(*note) : "";

      Axes own, axes = defaults;
      for (auto const& kv : sec.second) {
        if (kv.first == "tests" || kv.first == "timeout" || kv.first == "note") continue;
        if (IsRunnerSetting(kv.first)) {
          err = "[" + where + "] " + kv.first + " can only be set in [globals]";
          return false;
        }
        if (!CheckKey(kv.first, where, err) || !ParseValues(kv.first, kv.second, values, err)) return false;
        own.emplace_back(kv.first, values);
        SetAxis(axes, kv.first, values);
      }
      std::vector<std::string> envs;
      for (auto const& a : axes) if (a.first.rfind("env.", 0) == 0) envs.push_back(a.first);
      std::sort(envs.begin(), envs.end());
      std::vector<size_t> sizes;
      for (auto const& v : *AxisOf(axes, "size")) sizes.push_back(strtoull(v.c_str(), nullptr, 10));

      std::vector<std::string> used;
      for (auto const& test : sw.tests) {
        std::string const& line = *Find(plan.tests, test);
        std::vector<std::string> holes = Placeholders(line);
        for (auto const& h : holes) {
          if (!AxisOf(axes, h)) {
            err = "[" + where + "] test " + test + " uses {" + h + "}, but neither [" + where + "] nor [globals] sets " + h;
            return false;
          }
          if (std::find(used.begin(), used.end(), h) == used.end()) used.push_back(h);
        }
        std::vector<std::string> names = holes;
        names.insert(names.end(), envs.begin(), envs.end());
        std::vector<std::vector<std::string> const*> lists;
        for (auto const& n : names) lists.push_back(AxisOf(axes, n));
        std::vector<size_t> pick(names.size(), 0);
        for (int k = 0;; k++) {
          Run r;
          char num[16];
          snprintf(num, sizeof(num), "%03d", k);
          r.id = sw.name + "/" + test + "/" + num;
          r.sweep = sw.name;
          r.test = test;
          std::map<std::string, std::string> fill;
          for (size_t i = 0; i < names.size(); i++) {
            r.settings.emplace_back(names[i], (*lists[i])[pick[i]]);
            fill[names[i]] = (*lists[i])[pick[i]];
          }
          r.line = Fill(line, fill);
          r.sizes = sizes;
          r.timeout = sw.timeout;
          plan.runs.push_back(r);
          size_t i = names.size();
          while (i > 0 && ++pick[i - 1] == lists[i - 1]->size()) pick[--i] = 0;
          if (i == 0) break;
        }
      }
      for (auto const& a : own) {
        if (a.first != "size" && a.first.rfind("env.", 0) != 0 &&
            std::find(used.begin(), used.end(), a.first) == used.end())
          plan.warnings.push_back("[" + where + "] sets " + a.first + ", but none of its tests has {" + a.first + "}");
      }
      for (auto const& u : used) sw.axes.emplace_back(u, *AxisOf(axes, u));
      for (auto const& e : envs) sw.axes.emplace_back(e, *AxisOf(axes, e));
      sw.axes.emplace_back("size", *AxisOf(axes, "size"));
      plan.sweeps.push_back(sw);
    }
    if (plan.sweeps.empty() && fromIni) {
      err = "no [sweep:<name>] section, so nothing would run";
      return false;
    }

    // The same test line with the same env settings in two sweeps, at overlapping sizes.
    std::map<std::string, std::vector<std::pair<std::string, std::set<size_t>>>> seen;
    std::vector<std::pair<std::string, int>> repeats;
    for (auto const& r : plan.runs) {
      std::string key = r.line;
      for (auto const& s : r.settings) if (s.first.rfind("env.", 0) == 0) key += "\n" + s.first + "=" + s.second;
      std::set<size_t> mine(r.sizes.begin(), r.sizes.end());
      for (auto const& prev : seen[key]) {
        bool overlap = false;
        for (size_t s : mine) overlap = overlap || prev.second.count(s);
        if (!overlap) continue;
        std::string what = r.test + ": run(s) in [sweep:" + r.sweep + "] repeat runs of [sweep:" + prev.first + "]";
        auto it = std::find_if(repeats.begin(), repeats.end(), [&](auto const& p) { return p.first == what; });
        if (it == repeats.end()) repeats.emplace_back(what, 1);
        else it->second++;
      }
      seen[key].emplace_back(r.sweep, mine);
    }
    for (auto const& r : repeats) {
      size_t colon = r.first.find(':');
      plan.warnings.push_back(r.first.substr(0, colon) + ": " + std::to_string(r.second) + r.first.substr(colon + 1) +
                              " at the same sizes");
    }
    return true;
  }

  std::string PlanJson(Plan const& plan, std::string const& label)
  {
    std::string s = "{\"format\":\"perfreport plan v1\",\"label\":" + Q(label) + ",\n\"tests\":" + StrMap(plan.tests) +
                    ",\n\"sweeps\":[";
    for (size_t i = 0; i < plan.sweeps.size(); i++) {
      Sweep const& sw = plan.sweeps[i];
      s += std::string(i ? ",\n " : "\n ") + "{\"name\":" + Q(sw.name) + ",\"note\":" + Q(sw.note) +
           ",\"tests\":" + StrList(sw.tests) + ",\"timeout\":" + std::to_string(sw.timeout) + ",\"axes\":{";
      for (size_t a = 0; a < sw.axes.size(); a++) s += (a ? "," : "") + Q(sw.axes[a].first) + ":" + StrList(sw.axes[a].second);
      s += "}}";
    }
    s += "],\n\"runs\":[";
    for (size_t i = 0; i < plan.runs.size(); i++) {
      Run const& r = plan.runs[i];
      std::string sizes = "[";
      for (size_t k = 0; k < r.sizes.size(); k++) sizes += (k ? "," : "") + std::to_string(r.sizes[k]);
      s += std::string(i ? ",\n " : "\n ") + "{\"id\":" + Q(r.id) + ",\"sweep\":" + Q(r.sweep) + ",\"test\":" + Q(r.test) +
           ",\"settings\":" + StrMap(r.settings) + ",\"line\":" + Q(r.line) + ",\"timeout\":" + std::to_string(r.timeout) +
           ",\"sizes\":" + sizes + "]}";
    }
    return s + "]}\n";
  }

  //==========================================================================================
  // TransferBench settings
  //==========================================================================================
  // TransferBench settings that become ConfigOptions fields: set in the environment, read
  // by a fresh EnvVars as TransferBench itself would, then read back from ConfigOptions.
  struct Setting
  {
    char const*                                      name;
    std::function<std::string(ConfigOptions const&)> get;
  };

  template <typename T>
  std::string Join(std::vector<T> const& v, char const* fmt)
  {
    std::string s;
    char buf[64];
    for (size_t i = 0; i < v.size(); i++) {
      snprintf(buf, sizeof(buf), fmt, v[i]);
      s += (i ? "," : "") + std::string(buf);
    }
    return s;
  }

#define PERFREPORT_INT_SETTING(NAME, EXPR) {NAME, [](ConfigOptions const& c) { return std::to_string(EXPR); }}

  std::vector<Setting> const& Settings()
  {
    static std::vector<Setting> const settings = {
      PERFREPORT_INT_SETTING("NUM_ITERATIONS",     c.general.numIterations),
      PERFREPORT_INT_SETTING("NUM_SUBITERATIONS",  c.general.numSubIterations),
      PERFREPORT_INT_SETTING("NUM_WARMUPS",        c.general.numWarmups),
      PERFREPORT_INT_SETTING("PNP_FLAG_BUFFER",    c.general.pingpongFlagBuffer),
      PERFREPORT_INT_SETTING("PNP_SPIN_LIMIT",     c.general.pingpongSpinLimit),
      PERFREPORT_INT_SETTING("PNP_STRIDE",         c.general.pingpongStride),
      PERFREPORT_INT_SETTING("SHOW_ITERATIONS",    c.general.recordPerIteration),
      PERFREPORT_INT_SETTING("SHOW_PERCENTILES",   c.general.recordPerIteration),
      PERFREPORT_INT_SETTING("USE_HIP_EVENTS",     c.general.useHipEvents),
      PERFREPORT_INT_SETTING("USE_SINGLE_STREAM",  !c.general.useMultiStream),
      PERFREPORT_INT_SETTING("ALWAYS_VALIDATE",    c.data.alwaysValidate),
      PERFREPORT_INT_SETTING("BLOCK_BYTES",        c.data.blockBytes),
      PERFREPORT_INT_SETTING("BYTE_OFFSET",        c.data.byteOffset),
      {"FILL_COMPRESS", [](ConfigOptions const& c) { return Join(c.data.fillCompress, "%d"); }},
      {"FILL_PATTERN",  [](ConfigOptions const& c) { return Join(c.data.fillPattern, "%g"); }},
      PERFREPORT_INT_SETTING("VALIDATE_DIRECT",    c.data.validateDirect),
      PERFREPORT_INT_SETTING("VALIDATE_ON_DEVICE", c.data.validateOnDevice),
      PERFREPORT_INT_SETTING("VALIDATE_SOURCE",    c.data.validateSource),
      PERFREPORT_INT_SETTING("USE_HSA_DMA",        c.dma.useHsaCopy),
      PERFREPORT_INT_SETTING("GFX_BLOCK_ORDER",    c.gfx.blockOrder),
      PERFREPORT_INT_SETTING("GFX_BLOCK_SIZE",     c.gfx.blockSize),
      {"CU_MASK",       [](ConfigOptions const& c) { return Join(c.gfx.cuMask, "0x%x"); }},
      PERFREPORT_INT_SETTING("GFX_KERNEL",         c.gfx.gfxKernel),
      {"XCC_PREF_TABLE", [](ConfigOptions const& c) {
         std::string s;
         for (size_t i = 0; i < c.gfx.prefXccTable.size(); i++) s += (i ? ";" : "") + Join(c.gfx.prefXccTable[i], "%d");
         return s; }},
      PERFREPORT_INT_SETTING("GFX_SE_TYPE",        c.gfx.seType),
      PERFREPORT_INT_SETTING("GFX_TEMPORAL",       c.gfx.temporalMode),
      PERFREPORT_INT_SETTING("GFX_UNROLL",         c.gfx.unrollFactor),
      PERFREPORT_INT_SETTING("GFX_SINGLE_TEAM",    c.gfx.useSingleTeam),
      PERFREPORT_INT_SETTING("GFX_WAVE_ORDER",     c.gfx.waveOrder),
      PERFREPORT_INT_SETTING("GFX_WORD_SIZE",      c.gfx.wordSize),
      PERFREPORT_INT_SETTING("NIC_CHUNK_BYTES",    c.nic.chunkBytes),
      PERFREPORT_INT_SETTING("NIC_CQ_POLL_BATCH",  c.nic.cqPollBatch),
      PERFREPORT_INT_SETTING("IB_GID_INDEX",       c.nic.ibGidIndex),
      PERFREPORT_INT_SETTING("IB_PORT_NUMBER",     c.nic.ibPort),
      PERFREPORT_INT_SETTING("IP_ADDRESS_FAMILY",  c.nic.ipAddressFamily),
      PERFREPORT_INT_SETTING("NIC_RELAX_ORDER",    c.nic.useRelaxedOrder),
      PERFREPORT_INT_SETTING("NIC_SERVICE_LEVEL",  c.nic.serviceLevel),
      PERFREPORT_INT_SETTING("NIC_TRAFFIC_CLASS",  c.nic.trafficClass),
      PERFREPORT_INT_SETTING("ROCE_VERSION",       c.nic.roceVersion),
      PERFREPORT_INT_SETTING("TDM_BLOCK_ORDER",    c.tdm.blockOrder),
      PERFREPORT_INT_SETTING("TDM_BLOCK_SIZE",     c.tdm.blockSize),
      PERFREPORT_INT_SETTING("TDM_LDS_BYTES",      c.tdm.ldsBytes),
    };
    return settings;
  }

#undef PERFREPORT_INT_SETTING

  // TransferBench client settings with no effect here: display options, the client's own
  // size sweep, variable sub-executor search and interactive pauses.
  char const* const kNoEffect[] = {
    "HIDE_ENV", "OUTPUT_TO_CSV", "SHOW_BORDERS", "SAMPLING_FACTOR", "SWEEP_MIN_POW2",
    "SWEEP_MAX_POW2", "MIN_VAR_SUBEXEC", "MAX_VAR_SUBEXEC", "USE_INTERACTIVE",
  };

  Setting const* FindSetting(std::string const& name)
  {
    for (auto const& s : Settings()) if (name == s.name) return &s;
    return nullptr;
  }

  bool HasNoEffect(std::string const& name)
  {
    for (char const* n : kNoEffect) if (name == n) return true;
    return false;
  }

  //==========================================================================================
  // Platform record
  //==========================================================================================
  struct GpuUser { int pid; std::string user, comm; };

  // Other processes with the GPUs open. The kernel lists a process for a moment after it
  // exits, so entries without a /proc directory are skipped.
  std::vector<GpuUser> OtherGpuUsers()
  {
    std::vector<GpuUser> out;
    DIR* d = opendir("/sys/class/kfd/kfd/proc");
    if (!d) return out;
    int self = (int)getpid();
    while (dirent* e = readdir(d)) {
      if (!isdigit((unsigned char)e->d_name[0])) continue;
      int pid = atoi(e->d_name);
      if (pid == self) continue;
      std::string proc = "/proc/" + std::to_string(pid);
      struct stat st;
      if (stat(proc.c_str(), &st) != 0) continue;
      GpuUser u{pid, std::to_string(st.st_uid), Trim(ReadFile(proc + "/comm"))};
      if (passwd* pw = getpwuid(st.st_uid)) u.user = pw->pw_name;
      if (u.comm.empty()) u.comm = "?";
      out.push_back(u);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), [](GpuUser const& a, GpuUser const& b) { return a.pid < b.pid; });
    return out;
  }

  std::string UsersJson(std::vector<GpuUser> const& users)
  {
    std::string s = "[";
    for (size_t i = 0; i < users.size(); i++)
      s += std::string(i ? "," : "") + "{\"pid\":" + std::to_string(users[i].pid) + ",\"user\":" + Q(users[i].user) +
           ",\"comm\":" + Q(users[i].comm) + "}";
    return s + "]";
  }

  std::string TbVersion()
  {
    return std::string(TransferBench::VERSION) + "." + CLIENT_VERSION;
  }

  std::string TbBranch()
  {
#ifdef TB_GIT_BRANCH
    return TB_GIT_BRANCH;
#else
    return "unknown";
#endif
  }

  std::string TbCommit()
  {
#ifdef TB_GIT_COMMIT
    return TB_GIT_COMMIT;
#else
    return "unknown";
#endif
  }

  struct Engine { std::string path, sha256; };

  Engine ThisEngine()
  {
    Engine e;
    e.path = SelfExe();
    std::string sum = Capture("sha256sum '" + e.path + "'");
    e.sha256 = sum.substr(0, sum.find(' '));
    return e;
  }

  std::string EngineJson(Engine const& e)
  {
    return "{\"path\":" + Q(e.path) + ",\"sha256\":" + Q(e.sha256) + ",\"tb_version\":" + Q(TbVersion()) +
           ",\"tb_branch\":" + Q(TbBranch()) + ",\"tb_commit\":" + Q(TbCommit()) + ",\"tb_path\":null}";
  }

  // TransferBench's own topology and settings printout, as the client shows them.
  void SaveTopology(EnvVars const& ev, std::string const& path)
  {
    fflush(stdout);
    int saved = dup(1);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (saved < 0 || fd < 0) {
      if (saved >= 0) close(saved);
      if (fd >= 0) close(fd);
      return;
    }
    dup2(fd, 1);
    DisplayTopology(false, true);
    ev.DisplayEnvVars();
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    close(fd);
    WriteFile(path, ScrubText(ReadFile(path)));
  }

  std::string PlatformJson(std::string const& dir, EnvVars const& ev, Engine const& engine)
  {
    std::string raw = dir + "/platform/";
    auto save = [&](std::string const& name, std::string const& text, bool ok) {
      WriteFile(raw + name, ok && !text.empty() ? text : "(not available)\n");
    };
    auto json = [&](std::string const& name, std::string const& cmd, JVal& v) {
      bool ok;
      std::string text = Capture(cmd, &ok);
      bool parsed = ok && JReader(text).Parse(v);
      if (parsed) Scrub(v);
      save(name, parsed ? JWrite(v) + "\n" : "", parsed);
      return parsed;
    };

    JVal smiStatic, firmware, topology, lscpu;
    json("amd-smi-static.json", "timeout 120 amd-smi static --asic --vbios --board --limit --partition --json", smiStatic);
    json("amd-smi-firmware.json", "timeout 120 amd-smi firmware --json", firmware);
    json("amd-smi-topology.json", "timeout 120 amd-smi topology -t --json", topology);
    bool ok;
    std::string smiVersion = Trim(Capture("timeout 60 amd-smi version", &ok));
    save("amd-smi-version.txt", smiVersion + "\n", ok && !smiVersion.empty());
    std::string rocminfo = Capture("timeout 120 rocminfo", &ok);
    save("rocminfo.txt", ScrubText(rocminfo), ok);
    std::string lscpuText = Capture("lscpu -J", &ok);
    save("lscpu.json", lscpuText, ok);
    bool haveLscpu = ok && JReader(lscpuText).Parse(lscpu);
    std::string freeText = Capture("free -b", &ok);
    save("free.txt", freeText, ok);
    std::string unameText = Capture("uname -a", &ok);
    save("uname.txt", unameText, ok);
    std::string osRelease = ReadFile("/etc/os-release", &ok);
    save("os-release", osRelease, ok);
    std::string amdgpu = Trim(ReadFile("/sys/module/amdgpu/version", &ok));
    save("amdgpu-version.txt", amdgpu + "\n", ok && !amdgpu.empty());
    struct stat st;
    bool haveRocm = stat("/opt/rocm", &st) == 0;
    std::string rocmPath = haveRocm ? RealPath("/opt/rocm") : "";
    std::string rocmVersion = Trim(Split(ReadFile("/opt/rocm/.info/version"), '\n')[0]);
    save("rocm.txt", "path=" + rocmPath + "\nversion=" + rocmVersion + "\n", true);
    SaveTopology(ev, raw + "engine.txt");

    auto str = [](std::string const& v) { return v.empty() ? std::string("null") : Q(v); };

    std::map<std::string, std::string> fwByGpu, bdfByGpu;
    if (auto const* list = GpuList(firmware)) {
      for (auto const& g : *list) {
        std::string fw = "{";
        if (JVal const* fl = g.Get("fw_list")) {
          for (auto const& item : fl->items)
            fw += (fw.size() > 1 ? "," : "") + Q(item.Get("fw_id") ? item.Get("fw_id")->text : "?") + ":" +
                  JText(item.Get("fw_version"));
        }
        fwByGpu[JText(g.Get("gpu"))] = fw + "}";
      }
    }
    std::string links = "[";
    if (auto const* list = GpuList(topology)) {
      bool firstRow = true;
      for (auto const& g : *list) {
        bdfByGpu[JText(g.Get("gpu"))] = JText(g.Get("bdf"));
        std::string row = "[";
        if (JVal const* ls = g.Get("links"))
          for (size_t i = 0; i < ls->items.size(); i++) row += (i ? "," : "") + JText(ls->items[i].Get("link_type"));
        links += (firstRow ? "" : ",") + row + "]";
        firstRow = false;
      }
    }
    links += "]";
    std::string gpus = "[";
    if (auto const* list = GpuList(smiStatic)) {
      for (size_t i = 0; i < list->size(); i++) {
        JVal const& g = (*list)[i];
        std::string idx = JText(g.Get("gpu"));
        JVal const* ppt0 = At(&g, {"limit", "ppt0"});
        std::string power = JText(At(ppt0, {"socket_power_limit"}));
        if (power == "null") power = JText(At(ppt0, {"max_power_limit"}));
        gpus += std::string(i ? "," : "") + "{\"index\":" + idx +
                ",\"model\":" + JText(At(&g, {"asic", "market_name"})) +
                ",\"arch\":" + JText(At(&g, {"asic", "target_graphics_version"})) +
                ",\"device_id\":" + JText(At(&g, {"asic", "device_id"})) +
                ",\"rev_id\":" + JText(At(&g, {"asic", "rev_id"})) +
                ",\"cus\":" + JText(At(&g, {"asic", "num_compute_units"})) +
                ",\"vbios\":" + JText(At(&g, {"vbios", "version"})) +
                ",\"vbios_part_number\":" + JText(At(&g, {"vbios", "part_number"})) +
                ",\"power_limit_w\":" + power +
                ",\"partition\":{\"compute\":" + JText(At(&g, {"partition", "accelerator_partition"})) +
                ",\"memory\":" + JText(At(&g, {"partition", "memory_partition"})) + "}" +
                ",\"bdf\":" + (bdfByGpu.count(idx) ? bdfByGpu[idx] : "null") +
                ",\"firmware\":" + (fwByGpu.count(idx) ? fwByGpu[idx] : "{}") + "}";
      }
    }
    gpus += "]";

    std::map<std::string, std::string> cpuFields;
    std::function<void(JVal const&)> walk = [&](JVal const& list) {
      for (auto const& e : list.items) {
        JVal const* field = e.Get("field");
        JVal const* data = e.Get("data");
        if (field && data && data->kind == JVal::STR) {
          std::string name = field->text;
          while (!name.empty() && name.back() == ':') name.pop_back();
          cpuFields.emplace(name, data->text);
        }
        if (JVal const* kids = e.Get("children")) walk(*kids);
      }
    };
    if (haveLscpu && lscpu.Get("lscpu")) walk(*lscpu.Get("lscpu"));
    auto cpu = [&](std::string const& name, bool number) {
      auto it = cpuFields.find(name);
      if (it == cpuFields.end()) return std::string("null");
      bool digits = !it->second.empty() && std::all_of(it->second.begin(), it->second.end(), ::isdigit);
      return number && digits ? it->second : Q(it->second);
    };
    std::string memory = "null";
    for (auto const& line : Split(freeText, '\n')) {
      if (line.rfind("Mem:", 0) == 0) {
        std::istringstream w(line);
        std::string label, total;
        w >> label >> total;
        if (!total.empty()) memory = total;
      }
    }
    std::string pretty;
    for (auto const& line : Split(osRelease, '\n')) {
      if (line.rfind("PRETTY_NAME=", 0) == 0) {
        pretty = line.substr(12);
        if (pretty.size() >= 2 && pretty.front() == '"' && pretty.back() == '"') pretty = pretty.substr(1, pretty.size() - 2);
      }
    }
    std::string kernel;
    {
      std::istringstream w(unameText);
      std::string a, b;
      w >> a >> b >> kernel;
    }
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);

    int numGpus = GetNumExecutors(EXE_GPU_GFX);
    std::string cus = "[";
    for (int g = 0; g < numGpus; g++) cus += (g ? "," : "") + std::to_string(GetNumSubExecutors({EXE_GPU_GFX, g}));
    cus += "]";

    return "{\"host\":" + Q(host) + ",\"os\":" + str(pretty) + ",\"kernel\":" + str(kernel) +
           ",\n\"cpu\":{\"model\":" + cpu("Model name", false) + ",\"sockets\":" + cpu("Socket(s)", true) +
           ",\"cores_per_socket\":" + cpu("Core(s) per socket", true) + ",\"threads\":" + cpu("CPU(s)", true) +
           ",\"numa_nodes\":" + cpu("NUMA node(s)", true) + "},\"memory_bytes\":" + memory +
           ",\n\"rocm\":{\"path\":" + str(rocmPath) + ",\"version\":" + str(rocmVersion) + "},\"amdgpu_driver\":" + str(amdgpu) +
           ",\"amd_smi\":" + str(smiVersion) + ",\n\"gpus\":" + gpus + ",\n\"links\":" + links +
           ",\n\"transferbench\":{\"version\":" + Q(TbVersion()) + ",\"branch\":" + Q(TbBranch()) + ",\"commit\":" + Q(TbCommit()) +
           ",\"path\":null,\"engine\":" + Q(engine.path) + ",\"engine_sha256\":" + Q(engine.sha256) +
           ",\"gpus\":" + std::to_string(numGpus) + ",\"cus\":" + cus + "}}\n";
  }

  //==========================================================================================
  // Running
  //==========================================================================================
  char const* ExeTypeName(ExeType t)
  {
    switch (t) {
    case EXE_CPU:         return "CPU";
    case EXE_GPU_GFX:     return "GFX";
    case EXE_GPU_DMA:     return "DMA";
    case EXE_NIC:         return "NIC";
    case EXE_NIC_NEAREST: return "NIC_NEAREST";
    case EXE_GPU_BDMA:    return "BDMA";
    case EXE_GPU_TDM:     return "TDM";
    default:              return "?";
    }
  }

  std::string TransferDesc(Transfer const& t)
  {
    std::string src, dst;
    for (size_t i = 0; i < t.srcs.size(); i++) src += (i ? "," : "") + Utils::MemDeviceToStr(t.srcs[i]);
    for (size_t i = 0; i < t.dsts.size(); i++) dst += (i ? "," : "") + Utils::MemDeviceToStr(t.dsts[i]);
    std::string exe = Utils::ExeDeviceToStr(t.exeDevice, t.exeSubIndex, t.exeSubSlot);
    return (src.empty() ? "N" : src) + " -> " + exe + ":" + std::to_string(t.numSubExecs) + " -> " + (dst.empty() ? "N" : dst);
  }

  struct Outcome
  {
    std::string              status, reason;
    std::string              measurements;   // JSON fields, without braces, for PASS
    std::vector<std::string> warnings, errors;
  };

  Outcome Fail(std::string const& reason)
  {
    Outcome o;
    o.status = "FAIL";
    o.reason = reason;
    o.errors.push_back(reason);
    return o;
  }

  Outcome RunSize(ConfigOptions const& cfg, std::vector<Transfer> const& parsed, size_t size)
  {
    std::vector<Transfer> transfers = parsed;
    for (auto& t : transfers) {
      if (t.numSubExecs == 0) return Fail("variable sub-executor counts (0) are not supported by perfreport");
      if (t.numBytes == 0) t.numBytes = size;
    }
    if (size % 4) return Fail("the size must be a multiple of 4 bytes");

    Outcome o;
    TestResults res;
    bool ok = RunTransfers(cfg, transfers, res);
    for (auto const& e : res.errResults) {
      if (e.errType == ERR_FATAL) o.errors.push_back(e.errMsg);
      else if (e.errType == ERR_WARN) o.warnings.push_back(e.errMsg);
    }
    if (!ok) {
      o.status = "FAIL";
      o.reason = o.errors.empty() ? "RunTransfers failed without an error message" : o.errors[0];
      return o;
    }

    std::string exes = "[";
    bool first = true;
    for (auto const& kv : res.exeResults) {
      ExeDevice const& d = kv.first;
      ExeResult const& r = kv.second;
      std::string idx = "[";
      for (size_t i = 0; i < r.transferIdx.size(); i++) idx += (i ? "," : "") + std::to_string(r.transferIdx[i]);
      exes += std::string(first ? "" : ",") + "{\"exe\":" + Q(Utils::ExeDeviceToStr(d)) +
              ",\"type\":" + Q(ExeTypeName(d.exeType)) + ",\"index\":" + std::to_string(d.exeIndex) +
              ",\"gbps\":" + Num(r.avgBandwidthGbPerSec) + ",\"ms\":" + Num(r.avgDurationMsec) +
              ",\"bytes\":" + std::to_string(r.numBytes) + ",\"sum_gbps\":" + Num(r.sumBandwidthGbPerSec) +
              ",\"transfers\":" + idx + "]}";
      first = false;
    }
    exes += "]";

    std::string tfrs = "[";
    for (size_t i = 0; i < transfers.size() && i < res.tfrResults.size(); i++) {
      Transfer const& t = transfers[i];
      TransferResult const& r = res.tfrResults[i];
      std::vector<std::string> srcs, dsts;
      for (auto const& m : t.srcs) srcs.push_back(Utils::MemDeviceToStr(m));
      for (auto const& m : t.dsts) dsts.push_back(Utils::MemDeviceToStr(m));
      tfrs += std::string(i ? "," : "") + "{\"index\":" + std::to_string(i) + ",\"desc\":" + Q(TransferDesc(t)) +
              ",\"src\":" + StrList(srcs) + ",\"exe\":" + Q(Utils::ExeDeviceToStr(t.exeDevice, t.exeSubIndex, t.exeSubSlot)) +
              ",\"dst\":" + StrList(dsts) + ",\"cus\":" + std::to_string(t.numSubExecs) +
              ",\"gbps\":" + Num(r.avgBandwidthGbPerSec) + ",\"ms\":" + Num(r.avgDurationMsec) +
              ",\"bytes\":" + std::to_string(r.numBytes);
      if (!r.perIterMsec.empty()) {
        tfrs += ",\"per_iter_ms\":[";
        for (size_t k = 0; k < r.perIterMsec.size(); k++) tfrs += (k ? "," : "") + Num(r.perIterMsec[k]);
        tfrs += "]";
      }
      tfrs += "}";
    }
    tfrs += "]";

    o.status = "PASS";
    o.measurements = "\"iterations\":" + std::to_string(res.numTimedIterations) + ",\"executors\":" + exes +
                     ",\"transfers\":" + tfrs + ",\"cpu_aggregate\":{\"gbps\":" + Num(res.avgTotalBandwidthGbPerSec) +
                     ",\"ms\":" + Num(res.avgTotalDurationMsec) + ",\"bytes\":" + std::to_string(res.totalBytesTransferred) +
                     ",\"overhead_ms\":" + Num(res.overheadMsec) + "}";
    return o;
  }

  std::string RecordJson(std::string const& run, size_t size, int attempt, bool final, Outcome const& o,
                         std::string const& time, double seconds, std::vector<GpuUser> const& users,
                         std::string const& settingsUsed)
  {
    std::string s = "{\"run\":" + Q(run) + ",\"size\":" + std::to_string(size) + ",\"attempt\":" + std::to_string(attempt) +
                    ",\"final\":" + (final ? "true" : "false") + ",\"status\":" + Q(o.status) + ",\"reason\":" + Q(o.reason) +
                    ",\"time\":" + Q(time) + ",\"duration_s\":" + Num(seconds) + ",\"other_gpu_users\":" + UsersJson(users) +
                    ",\"settings_used\":" + settingsUsed;
    if (!o.measurements.empty()) s += "," + o.measurements;
    return s + ",\"warnings\":" + StrList(o.warnings) + ",\"errors\":" + StrList(o.errors) + "}";
  }

  // Ends the process when one size takes longer than its run's timeout: a transfer stuck on
  // the GPU can't be stopped from inside the process.
  class Watchdog
  {
  public:
    Watchdog() : thread_([this] { Loop(); }) {}

    ~Watchdog()
    {
      {
        std::lock_guard<std::mutex> g(mu_);
        quit_ = true;
      }
      cv_.notify_all();
      thread_.join();
    }

    void Arm(int seconds, std::function<void()> onTimeout)
    {
      std::lock_guard<std::mutex> g(mu_);
      armed_ = true;
      deadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
      onTimeout_ = std::move(onTimeout);
      cv_.notify_all();
    }

    void Disarm()
    {
      std::lock_guard<std::mutex> g(mu_);
      armed_ = false;
    }

  private:
    std::mutex                            mu_;
    std::condition_variable               cv_;
    bool                                  armed_ = false;
    bool                                  quit_ = false;
    std::chrono::steady_clock::time_point deadline_;
    std::function<void()>                 onTimeout_;
    std::thread                           thread_;

    void Loop()
    {
      std::unique_lock<std::mutex> lock(mu_);
      while (!quit_) {
        if (!armed_) {
          cv_.wait(lock);
          continue;
        }
        cv_.wait_until(lock, deadline_);
        if (!quit_ && armed_ && std::chrono::steady_clock::now() >= deadline_) {
          std::function<void()> fire = onTimeout_;
          lock.unlock();
          fire();
          fflush(stdout);
          _exit(124);
        }
      }
    }
  };

  struct Counts { int pass = 0, fail = 0, timeout = 0, skip = 0; };

  std::string CountsJson(Counts const& c, int missing)
  {
    return "{\"PASS\":" + std::to_string(c.pass) + ",\"FAIL\":" + std::to_string(c.fail) + ",\"TIMEOUT\":" +
           std::to_string(c.timeout) + ",\"SKIP\":" + std::to_string(c.skip) + ",\"missing\":" + std::to_string(missing) + "}";
  }

  void Add(Counts& c, std::string const& status)
  {
    if (status == "PASS") c.pass++;
    else if (status == "FAIL") c.fail++;
    else if (status == "TIMEOUT") c.timeout++;
    else if (status == "SKIP") c.skip++;
  }

  int Main(EnvVars& ev, size_t const numBytesPerTransfer, bool const bytesSpecified)
  {
    bool const out = Utils::RankDoesOutput();
    std::string const iniPath = EnvVars::GetEnvVar("PERFREPORT_INI", std::string(""));
    std::string dir = EnvVars::GetEnvVar("PERFREPORT_DIR", std::string(""));
    int const numGpus = GetNumExecutors(EXE_GPU_GFX);

    Input in;
    std::string label = "default", err;
    if (!iniPath.empty()) {
      if (!ReadIni(iniPath, in, err)) {
        Utils::Print("[ERROR] perfreport: %s: %s\n", iniPath.c_str(), err.c_str());
        return 1;
      }
      label = iniPath.substr(iniPath.find_last_of('/') + 1);
      label = label.substr(0, label.find_last_of('.'));
    } else {
      AddDefaultTests(in, numGpus);
    }
    Plan plan;
    if (!Expand(in, !iniPath.empty(), plan, err)) {
      Utils::Print("[ERROR] perfreport: %s: %s\n", iniPath.empty() ? "default tests" : iniPath.c_str(), err.c_str());
      return 1;
    }
    if (bytesSpecified) {
      for (auto& r : plan.runs) r.sizes = {numBytesPerTransfer};
      for (auto& sw : plan.sweeps) SetAxis(sw.axes, "size", {std::to_string(numBytesPerTransfer)});
    }
    std::set<std::string> warned;
    for (auto const& r : plan.runs) {
      for (auto const& s : r.settings) {
        if (s.first.rfind("env.", 0) != 0 || !HasNoEffect(s.first.substr(4)) || !warned.insert(s.first).second) continue;
        plan.warnings.push_back(s.first + " has no effect in perfreport (a TransferBench client display or "
                                "size-sweep setting), so runs that differ only in it are identical");
      }
    }
    for (auto const& w : plan.warnings) Utils::Print("[WARN] perfreport: %s\n", w.c_str());

    size_t planned = 0;
    for (auto const& r : plan.runs) planned += r.sizes.size();
    if (dir.empty()) dir = "perfreport_" + Stamp();
    if (out) {
      struct stat st;
      if (stat((dir + "/run.json").c_str(), &st) == 0) {
        Utils::Print("[ERROR] perfreport: %s already holds a perfreport run; set PERFREPORT_DIR to a new folder\n", dir.c_str());
        return 1;
      }
      if (!MakeDirs(dir + "/platform")) {
        Utils::Print("[ERROR] perfreport: cannot create %s: %s\n", dir.c_str(), strerror(errno));
        return 1;
      }
    }
    std::string const absDir = RealPath(dir);
    Utils::Print("perfreport: %s, %zu run(s), %zu size(s), writing to %s\n",
                 iniPath.empty() ? "default tests" : iniPath.c_str(), plan.runs.size(), planned, absDir.c_str());
    if (plan.runs.empty() && iniPath.empty())
      Utils::Print("perfreport: no default tests are defined yet; set PERFREPORT_INI to an INI file to run sweeps\n");

    // The process environment at start: the baseline every run starts from.
    std::set<std::string> touched;
    for (auto const& r : plan.runs)
      for (auto const& s : r.settings)
        if (s.first.rfind("env.", 0) == 0) touched.insert(s.first.substr(4));
    std::map<std::string, std::pair<bool, std::string>> baseline;
    for (auto const& name : touched) {
      char const* v = getenv(name.c_str());
      baseline[name] = {v != nullptr, v ? v : ""};
    }
    StrPairs environment;
    for (char** e = environ; e && *e; e++) {
      std::string kv = *e;
      size_t eq = kv.find('=');
      if (eq == std::string::npos) continue;
      std::string name = kv.substr(0, eq);
      bool gpu = name == "LD_LIBRARY_PATH" || name.rfind("PERFREPORT_", 0) == 0;
      for (char const* p : {"HSA_", "HIP_", "ROC", "AMD_", "GPU_", "HCC_"}) gpu = gpu || name.rfind(p, 0) == 0;
      if (gpu || FindSetting(name) || HasNoEffect(name)) environment.emplace_back(name, kv.substr(eq + 1));
    }
    std::sort(environment.begin(), environment.end());

    std::vector<std::string> command;
    for (auto const& a : Split(ReadFile("/proc/self/cmdline"), '\0')) if (!a.empty()) command.push_back(a);
    Engine const engine = ThisEngine();
    std::string const started = NowIso();
    std::string const iniAbs = iniPath.empty() ? "" : RealPath(iniPath);
    auto runJson = [&](std::string const& status, std::string const& reason, std::string const& finished,
                       Counts const& c, int missing) {
      return "{\"format\":\"perfreport run v1\",\"label\":" + Q(label) + ",\"ini\":" + (iniAbs.empty() ? "null" : Q(iniAbs)) +
             ",\"command\":" + StrList(command) + ",\"runner\":{\"name\":\"perfreport\",\"version\":" + Q(kVersion) + "}" +
             ",\"engine\":" + EngineJson(engine) + ",\"environment\":" + StrMap(environment) +
             ",\"size_override\":" + (bytesSpecified ? std::to_string(numBytesPerTransfer) : "null") +
             ",\"started\":" + Q(started) + ",\"finished\":" + (finished.empty() ? "null" : Q(finished)) +
             ",\"resumed\":[],\"status\":" + Q(status) + ",\"stop_reason\":" + Q(reason) +
             ",\"counts\":" + CountsJson(c, missing) + "}\n";
    };

    FILE* results = nullptr;
    if (out) {
      WriteFile(dir + "/input.ini", iniPath.empty() ? ToIni(in) : ReadFile(iniPath));
      WriteFile(dir + "/plan.json", PlanJson(plan, label));
      Utils::Print("perfreport: collecting the platform record\n");
      WriteFile(dir + "/platform.json", PlatformJson(dir, ev, engine));
      WriteFile(dir + "/run.json", runJson("running", "", "", Counts(), (int)planned));
      results = fopen((dir + "/results.jsonl").c_str(), "a");
      if (!results) {
        Utils::Print("[ERROR] perfreport: cannot write %s/results.jsonl: %s\n", dir.c_str(), strerror(errno));
        return 1;
      }
    }
    std::mutex outMu;
    Counts counts;
    size_t finals = 0;
    auto record = [&](std::string const& line, bool final, std::string const& status) {
      std::lock_guard<std::mutex> g(outMu);
      if (final) {
        Add(counts, status);
        finals++;
      }
      if (!results) return;
      fputs((line + "\n").c_str(), results);
      fflush(results);
    };

    auto const clockStart = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t0) {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    Watchdog watchdog;
    for (size_t ri = 0; ri < plan.runs.size(); ri++) {
      Run const& run = plan.runs[ri];
      auto const runStart = std::chrono::steady_clock::now();

      // Every run starts from the baseline environment, then sets its own env. values. HIP and
      // ROCm read their own variables only at startup, so changing those here may have no effect.
      for (auto const& b : baseline) {
        if (b.second.first) setenv(b.first.c_str(), b.second.second.c_str(), 1);
        else unsetenv(b.first.c_str());
      }
      for (auto const& s : run.settings)
        if (s.first.rfind("env.", 0) == 0) setenv(s.first.substr(4).c_str(), s.second.c_str(), 1);
      EnvVars evRun;
      ConfigOptions const cfg = evRun.ToConfigOptions();
      StrPairs used;
      for (auto const& s : run.settings) {
        if (s.first.rfind("env.", 0) != 0) continue;
        if (Setting const* setting = FindSetting(s.first.substr(4))) used.emplace_back(s.first.substr(4), setting->get(cfg));
      }
      std::string const settingsUsed = StrMap(used);

      std::vector<Transfer> transfers;
      std::string lineError;
      ErrResult parseErr = ParseTransfers(run.line, transfers);
      if (parseErr.errType == ERR_FATAL) lineError = "test line rejected: " + parseErr.errMsg;
      else if (transfers.empty()) lineError = "the test line has no transfers";

      Counts runCounts;
      std::string lastFail, skipReason;
      int sameFails = 0;
      for (size_t size : run.sizes) {
        if (!skipReason.empty()) {
          Outcome o;
          o.status = "SKIP";
          o.reason = skipReason;
          record(RecordJson(run.id, size, 1, true, o, NowIso(), 0, {}, settingsUsed), true, o.status);
          Add(runCounts, o.status);
          continue;
        }
        Outcome o;
        int attempt = 1;
        while (true) {
          std::string const time = NowIso();
          std::vector<GpuUser> users = OtherGpuUsers();
          auto const t0 = std::chrono::steady_clock::now();
          if (!lineError.empty()) {
            o = Fail(lineError);
          } else {
            watchdog.Arm(run.timeout, [&, time, size, attempt, t0]() {
              Outcome t;
              t.status = "TIMEOUT";
              t.reason = "no result within " + std::to_string(run.timeout) + " s";
              record(RecordJson(run.id, size, attempt, true, t, time, since(t0), OtherGpuUsers(), settingsUsed), true, t.status);
              Utils::Print("[ERROR] perfreport: %s at %zu bytes took longer than %d s; stopping\n",
                           run.id.c_str(), size, run.timeout);
              if (out) {
                std::lock_guard<std::mutex> g(outMu);
                WriteFile(dir + "/run.json", runJson("stopped", run.id + " at " + std::to_string(size) + " bytes: " + t.reason,
                                                     NowIso(), counts, (int)(planned - finals)));
              }
            });
            o = RunSize(cfg, transfers, size);
            watchdog.Disarm();
          }
          for (auto const& u : OtherGpuUsers())
            if (std::none_of(users.begin(), users.end(), [&](GpuUser const& x) { return x.pid == u.pid; })) users.push_back(u);
          bool const retry = o.status == "FAIL" && attempt == 1;
          record(RecordJson(run.id, size, attempt, !retry, o, time, since(t0), users, settingsUsed), !retry, o.status);
          if (!retry) break;
          attempt = 2;
        }
        Add(runCounts, o.status);
        if (o.status == "FAIL") {
          sameFails = (o.reason == lastFail) ? sameFails + 1 : 1;
          lastFail = o.reason;
          if (sameFails >= 2) skipReason = "not run: the previous two sizes failed with the same error";
        } else {
          sameFails = 0;
          lastFail.clear();
        }
      }
      std::string summary = std::to_string(runCounts.pass) + " PASS";
      if (runCounts.fail)    summary += ", " + std::to_string(runCounts.fail) + " FAIL";
      if (runCounts.timeout) summary += ", " + std::to_string(runCounts.timeout) + " TIMEOUT";
      if (runCounts.skip)    summary += ", " + std::to_string(runCounts.skip) + " SKIP";
      Utils::Print("[%zu/%zu] %s: %s in %s\n", ri + 1, plan.runs.size(), run.id.c_str(), summary.c_str(),
                   FmtDuration(since(runStart)).c_str());
    }
    for (auto const& b : baseline) {
      if (b.second.first) setenv(b.first.c_str(), b.second.second.c_str(), 1);
      else unsetenv(b.first.c_str());
    }

    int const missing = (int)(planned - finals);
    if (results) fclose(results);
    if (out) WriteFile(dir + "/run.json", runJson("complete", "", NowIso(), counts, missing));
    Utils::Print("perfreport: %d PASS, %d FAIL, %d TIMEOUT, %d SKIP in %s; results in %s\n", counts.pass, counts.fail,
                 counts.timeout, counts.skip, FmtDuration(since(clockStart)).c_str(), absDir.c_str());
    return 0;
  }
}

int PerfReportPreset(EnvVars&                          ev,
                     size_t      const                 numBytesPerTransfer,
                     [[maybe_unused]] std::string const presetName,
                     bool        const                 bytesSpecified)
{
  return PerfReport::Main(ev, numBytesPerTransfer, bytesSpecified);
}
