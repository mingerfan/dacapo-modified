#pragma once
// CKKSPL01 / CKKSMF01 version 1, shared wire specification with ckks-runtime.
// Each record comes from the same builder as JSON. No complete JSON or typed
// plan is retained: a temporary record file lets the string table go first.
#include "nlohmann/json.hpp"
#include "llvm/ADT/StringRef.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <type_traits>
#include <vector>

namespace {
class BinaryFields {
public:
  explicit BinaryFields(std::ostream &output) : output(output) {}
  void bytes(const char *data, size_t size) {
    output.write(data, static_cast<std::streamsize>(size));
    if (!output)
      fail("failed to write binary artifact");
  }
  template <class T> void number(T value) {
    using U = std::make_unsigned_t<T>;
    U bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    char data[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i)
      data[i] = static_cast<char>(bits >> (8 * i));
    bytes(data, sizeof(T));
  }
  void text(const std::string &value) {
    if (value.size() > 1024 * 1024) {
      fail("binary string exceeds 1 MiB");
      return;
    }
    number<uint32_t>(value.size());
    bytes(value.data(), value.size());
  }
  void header(const char *magic, const nlohmann::json &metadata) {
    bytes(magic, 8);
    number<uint32_t>(1);
    text(metadata.dump());
  }
  void digest(const std::string &content) {
    if (content.size() != 71 || content.substr(0, 7) != "sha256:") {
      fail("invalid binary content ID");
      return;
    }
    char raw[32];
    const auto digit = [&](char c) -> unsigned {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      fail("invalid binary content ID");
      return 0;
    };
    for (size_t i = 0; i < 32; ++i)
      raw[i] = static_cast<char>(digit(content[7 + 2 * i]) * 16 +
                                 digit(content[8 + 2 * i]));
    bytes(raw, 32);
  }
  void fail(const std::string &message) {
    if (error.empty())
      error = message;
    output.setstate(std::ios::failbit);
  }
  std::string error;

private:
  std::ostream &output;
};

class BinaryPlanWriter {
  using J = nlohmann::json;

public:
  BinaryPlanWriter(std::ostream &output, const std::filesystem::path &temporary)
      : output(output), temporary(temporary),
        records(temporary, std::ios::binary | std::ios::in | std::ios::out |
                               std::ios::trunc),
        wire(records) {
    if (!records)
      wire.fail("cannot open binary record file: " + temporary.string());
  }
  ~BinaryPlanWriter() {
    const bool owned = records.is_open();
    records.close();
    if (owned) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
  }
  void field(llvm::StringRef name, const J &value) {
    if (name == "finalization") {
      beginArray(name);
      endArray();
    } else
      metadata[name.str()] = value;
  }
  void beginArray(llvm::StringRef name) {
    static constexpr const char *order[] = {"values",         "external_inputs",
                                            "initialization", "execution",
                                            "finalization",   "final_outputs"};
    if (section >= 6 || name != order[section++]) {
      wire.fail("binary section order mismatch");
      return;
    }
    array = name.str();
    metadata[array] = J::array();
    count = 0;
    countPosition = records.tellp();
    wire.number<uint64_t>(0);
  }
  void endArray() {
    const auto end = records.tellp();
    records.seekp(countPosition);
    wire.number<uint64_t>(count);
    records.seekp(end);
    if (!records)
      wire.fail("failed to patch binary record count");
  }
  void record(const J &r) {
    ++count;
    if (array == "values") {
      id(r.at("id"));
      kind(r.at("kind"));
      place(r.at("place"));
      stringRef(r.at("context"));
      i32(r.at("level"));
      i32(r.at("scale_log2"));
      wire.number<uint8_t>(r.at("ntt").get<bool>());
      i32(r.at("components"));
    } else if (array == "external_inputs" || array == "final_outputs")
      id(r);
    else
      instruction(r);
  }
  void finish() {
    if (section != 6) {
      wire.fail("incomplete binary plan");
      return;
    }
    if (!wire.error.empty())
      return;
    records.flush();
    records.seekg(0);
    if (!records) {
      wire.fail("failed to rewind binary records");
      return;
    }
    BinaryFields result(output);
    result.header("CKKSPL01", metadata);
    result.number<uint64_t>(strings.size());
    for (const auto &value : strings)
      result.text(value);
    std::array<char, 1024 * 1024> buffer;
    while (records.read(buffer.data(), buffer.size()) || records.gcount())
      result.bytes(buffer.data(), static_cast<size_t>(records.gcount()));
    if (records.bad())
      wire.fail("failed to read binary records");
    if (!result.error.empty())
      wire.fail(result.error);
  }
  bool good() const { return wire.error.empty() && output.good(); }
  std::string error() const {
    return wire.error.empty() ? "failed to write binary plan" : wire.error;
  }
  double serializationSeconds = 0;

private:
  void id(const J &v) {
    uint64_t value = 0;
    const auto text = v.get<std::string>();
    if (llvm::StringRef(text).getAsInteger(10, value)) {
      wire.fail("invalid binary ID");
      return;
    }
    wire.number<uint64_t>(value);
  }
  void i32(const J &v) {
    const auto n = v.get<int64_t>();
    if (n < INT32_MIN || n > INT32_MAX) {
      wire.fail("binary int32 out of range");
      return;
    }
    wire.number<int32_t>(static_cast<int32_t>(n));
  }
  void stringRef(const J &v) {
    auto text = v.get<std::string>();
    auto entry = indices.find(text);
    if (entry == indices.end()) {
      if (strings.size() >= UINT32_MAX) {
        wire.fail("too many binary strings");
        return;
      }
      entry =
          indices.emplace(text, static_cast<uint32_t>(strings.size())).first;
      strings.push_back(std::move(text));
    }
    wire.number<uint32_t>(entry->second);
  }
  void kind(const J &v) {
    wire.number<uint8_t>(choice(v, {"plaintext", "ciphertext"}));
  }
  uint8_t choice(const J &v, std::initializer_list<const char *> choices) {
    const auto value = v.get<std::string>();
    uint8_t i = 0;
    for (const auto *candidate : choices) {
      if (value == candidate)
        return i;
      ++i;
    }
    wire.fail("unknown binary enum: " + value);
    return 0;
  }
  void place(const J &v) {
    wire.number<uint8_t>(choice(v.at("kind"), {"host", "device"}));
    i32(v.at("rank"));
    i32(v.value("index", J(0)));
  }
  template <class Visit> void sequence(const J &v, Visit visit) {
    wire.number<uint64_t>(v.size());
    for (const auto &item : v)
      visit(item);
  }
  void instruction(const J &r) {
    wire.number<uint64_t>(r.at("ordinal").get<uint64_t>());
    const auto tag = choice(
        r.at("kind"), {"encode", "compute", "transfer", "release", "fence", "prepare_plaintext"});
    wire.number<uint8_t>(tag);
    if (tag == 0 || tag == 5) {
      id(r.at("output"));
      const auto &p = r.at("payload");
      const auto payload = choice(p.at("kind"), {"inline", "bundle"});
      wire.number<uint8_t>(payload);
      if (payload)
        stringRef(p.at("content"));
      else
        sequence(p.at("values"), [&](const J &v) {
          const double value = v.get<double>();
          uint64_t bits = 0;
          std::memcpy(&bits, &value, 8);
          wire.number(bits);
        });
    } else if (tag == 1) {
      const auto op =
          choice(r.at("op"), {"add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc",
                              "mul_cp", "negate", "rotate", "rescale",
                              "mod_switch", "relinearize", "boot"});
      wire.number<uint8_t>(op);
      sequence(r.at("inputs"), [&](const J &v) { id(v); });
      id(r.at("output"));
      place(r.at("place"));
      const uint8_t attrs = op == 7    ? 1
                            : op == 8  ? 2
                            : op == 9  ? 3
                            : op == 11 ? 4
                                       : 0;
      wire.number<uint8_t>(attrs);
      if (attrs) {
        const auto &a = r.at("attrs");
        if (attrs == 1)
          i32(a.at("steps"));
        else {
          i32(a.at("target_level"));
          if (attrs == 2 || attrs == 4)
            i32(a.at("target_scale_log2"));
          if (attrs == 4) {
            i32(a.at("target_components"));
            stringRef(a.at("operator_profile"));
            wire.number<uint8_t>(choice(a.at("implementation"),
                                        {"native", "decrypt_reencrypt"}));
          }
        }
      }
      wire.number<uint8_t>(r.contains("reuse_input"));
      if (r.contains("reuse_input"))
        wire.number<uint64_t>(r.at("reuse_input").get<uint64_t>());
    } else if (tag == 2) {
      id(r.at("transfer_id"));
      wire.number<uint8_t>(0);
      wire.number<uint8_t>(
          choice(r.at("hint"), {"auto", "point_to_point", "broadcast", "tree",
                                "ring", "host_staged"}));
      sequence(r.at("inputs"), [&](const J &v) { id(v); });
      sequence(r.at("outputs"), [&](const J &v) { id(v); });
      sequence(r.at("sources"), [&](const J &v) { place(v); });
      sequence(r.at("destinations"), [&](const J &v) { place(v); });
      sequence(r.at("output_kinds"), [&](const J &v) { kind(v); });
    } else if (tag == 3)
      id(r.at("value"));
  }
  std::ostream &output;
  std::filesystem::path temporary;
  std::fstream records;
  BinaryFields wire;
  J metadata;
  std::string array;
  size_t section = 0;
  uint64_t count = 0;
  std::streampos countPosition;
  std::map<std::string, uint32_t> indices;
  std::vector<std::string> strings;
};
} // namespace
