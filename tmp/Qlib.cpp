# include <cstring>
# include <cstdlib>
# include <cctype>
# include "Qlib.hpp"

template <> int strto<int>(const char* str) { return std::atoi(str); }
template <> int strto<int>(const char* str, char*& str_end) { return (int)std::strtol(str, &str_end, 0); }
template <> int strto<int>(const char* str, int base) { return (int)std::strtol(str, NULL, base); }
template <> int strto<int>(const char* str, char*& str_end, int base) { return (int)std::strtol(str, &str_end, base); }
template <> long strto<long>(const char* str) { return std::strtol(str, NULL, 0); }
template <> long strto<long>(const char* str, char*& str_end) { return std::strtol(str, &str_end, 0); }
template <> long strto<long>(const char* str, int base) { return std::strtol(str, NULL, base); }
template <> long strto<long>(const char* str, char*& str_end, int base) { return std::strtol(str, &str_end, base); }
template <> long long strto<long long>(const char* str) { return std::strtoll(str, NULL, 0); }
template <> long long strto<long long>(const char* str, char*& str_end) { return std::strtoll(str, &str_end, 0); }
template <> long long strto<long long>(const char* str, int base) { return std::strtoll(str, NULL, base); }
template <> long long strto<long long>(const char* str, char*& str_end, int base) { return std::strtoll(str, &str_end, base); }
template <> unsigned int strto<unsigned int>(const char* str) { return (unsigned int)std::strtoul(str, NULL, 0); }
template <> unsigned int strto<unsigned int>(const char* str, char*& str_end) { return (unsigned int)std::strtoul(str, &str_end, 0); }
template <> unsigned int strto<unsigned int>(const char* str, int base) { return (unsigned int)std::strtoul(str, NULL, base); }
template <> unsigned int strto<unsigned int>(const char* str, char*& str_end, int base) { return (unsigned int)std::strtoul(str, &str_end, base); }
template <> unsigned long strto<unsigned long>(const char* str) { return std::strtoul(str, NULL, 0); }
template <> unsigned long strto<unsigned long>(const char* str, char*& str_end) { return std::strtoul(str, &str_end, 0); }
template <> unsigned long strto<unsigned long>(const char* str, int base) { return std::strtoul(str, NULL, base); }
template <> unsigned long strto<unsigned long>(const char* str, char*& str_end, int base) { return std::strtoul(str, &str_end, base); }
template <> unsigned long long strto<unsigned long long>(const char* str) { return std::strtoull(str, NULL, 0); }
template <> unsigned long long strto<unsigned long long>(const char* str, char*& str_end) { return std::strtoull(str, &str_end, 0); }
template <> unsigned long long strto<unsigned long long>(const char* str, int base) { return std::strtoull(str, NULL, base); }
template <> unsigned long long strto<unsigned long long>(const char* str, char*& str_end, int base) { return std::strtoull(str, &str_end, base); }
template <> float strto<float>(const char* str) { return std::strtof(str, NULL); }
template <> float strto<float>(const char* str, char*& str_end) { return std::strtof(str, &str_end); }
template <> double strto<double>(const char* str) { return std::strtod(str, NULL); }
template <> double strto<double>(const char* str, char*& str_end) { return std::strtod(str, &str_end); }
template <> long double strto<long double>(const char* str) { return std::strtold(str, NULL); }
template <> long double strto<long double>(const char* str, char*& str_end) { return std::strtold(str, &str_end); }

template <> bool strto<bool>(const char* str) {
    const char* i;
    for (i = str; i != '\0' && std::isspace(*i); ++i);
    if (std::isdigit(*i)) return std::atoi(str);
    else if (!std::strcmp(i, "true") || !std::strcmp(i, "True") || !std::strcmp(i, "TRUE"))
        return true;
    else return false;
}

char* strtostr(char* str) {
    char* begin, * end;
    for (begin = str; std::isspace(*begin) && *begin != '\0'; ++begin);
    if (*begin == '\0') return NULL;
    for (end = begin + 1; !std::isspace(*end) && *end != '\0'; ++end);
    *end = '\0';
    return begin;
}


