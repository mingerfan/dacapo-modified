#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/CKKS/Transforms/RuntimePlanUtils.h"
#include "RuntimePlanBinaryWriter.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <string>
#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_EMITRUNTIMEPLAN
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;
namespace ckks = hecate::ckks;
namespace dist = hecate::dist;

namespace {

using Json = nlohmann::json;

bool isCanonicalId(llvm::StringRef value) {
  if (value.empty() || (value.size() > 1 && value.front() == '0'))
    return false;
  for (char c : value)
    if (c < '0' || c > '9')
      return false;
  uint64_t parsed = 0;
  return !value.getAsInteger(10, parsed);
}

bool isSha256(llvm::StringRef value) {
  if (!value.consume_front("sha256:") || value.size() != 64)
    return false;
  for (char c : value)
    if (!llvm::isDigit(c) && (c < 'a' || c > 'f'))
      return false;
  return true;
}

Json placeJson(int64_t rank, int64_t device) {
  if (device == -1)
    return {{"kind", "host"}, {"rank", rank}};
  return {{"kind", "device"}, {"rank", rank}, {"index", device}};
}

std::string sha256(llvm::StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  const auto digest = hasher.final();
  return "sha256:" + llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

LogicalResult writeFile(const std::filesystem::path &path,
                        llvm::StringRef bytes, Operation *op) {
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return op->emitError("cannot open output file: ") << path.string();
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output)
    return op->emitError("failed to write output file: ") << path.string();
  return success();
}

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

uint64_t residentBytes() {
#ifdef __linux__
  std::ifstream status("/proc/self/statm");
  uint64_t total = 0, resident = 0;
  status >> total >> resident;
  return resident * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}
uint64_t peakResidentBytes() {
#ifdef __linux__
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#else
  return 0;
#endif
}

// nlohmann's stream serializer writes small pieces. Buffer them before file
// I/O and hashing, so each output byte enters the digest exactly once.
class JsonFileBuffer : public std::streambuf {
public:
  JsonFileBuffer(const std::filesystem::path &path, bool profile, bool hashSource = true)
      : file(path, std::ios::binary | std::ios::trunc), profile(profile), hashSource(hashSource) {
    setp(buffer.data(), buffer.data() + buffer.size());
  }
  bool isOpen() const { return file.is_open(); }
  bool close() {
    if (closed)
      return good;
    good = flush();
    const auto start = profile ? Clock::now() : Clock::time_point{};
    file.close();
    if (profile)
      writeSeconds += elapsed(start);
    good &= !file.fail();
    closed = true;
    if (good && hashSource) {
      const auto digest = hash.final();
      digestString =
          "sha256:" + llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
    }
    return good;
  }
  uint64_t bytes = 0;
  double writeSeconds = 0, hashSeconds = 0;
  std::string digestString;

private:
  bool flush() {
    const auto count = pptr() - pbase();
    if (!count)
      return !file.fail();
    auto start = profile ? Clock::now() : Clock::time_point{};
    file.write(pbase(), count);
    if (profile)
      writeSeconds += elapsed(start);
    if (!file)
      return false;
    start = profile ? Clock::now() : Clock::time_point{};
    if (hashSource) hash.update(llvm::StringRef(pbase(), count));
    if (profile && hashSource)
      hashSeconds += elapsed(start);
    bytes += static_cast<uint64_t>(count);
    setp(buffer.data(), buffer.data() + buffer.size());
    return true;
  }
  int sync() override { return flush() ? 0 : -1; }
  int_type overflow(int_type c) override {
    if (!flush())
      return traits_type::eof();
    if (!traits_type::eq_int_type(c, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(c);
      pbump(1);
    }
    return traits_type::not_eof(c);
  }
  std::streamsize xsputn(const char *data, std::streamsize count) override {
    std::streamsize written = 0;
    while (written < count) {
      if (pptr() == epptr() && !flush())
        break;
      const auto size = std::min(count - written, epptr() - pptr());
      std::memcpy(pptr(), data + written, static_cast<size_t>(size));
      pbump(static_cast<int>(size));
      written += size;
    }
    return written;
  }
  std::ofstream file;
  std::array<char, 64 * 1024> buffer;
  llvm::SHA256 hash;
  bool profile, hashSource, closed = false, good = true;
};

class JsonArtifact {
private:
  std::filesystem::path path, temporary;
  bool opened = false, published = false;

public:
  JsonArtifact(std::filesystem::path path, bool profile, bool hashSource = true)
      : path(std::move(path)), temporary(this->path.string() + ".tmp"),
        buffer(temporary, profile, hashSource), output(&buffer) {}
  ~JsonArtifact() {
    buffer.close();
    if (!published && opened) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
  }
  LogicalResult checkOpen(Operation *op) {
    opened = buffer.isOpen();
    if (!opened)
      return op->emitError("cannot open output file: ") << temporary.string();
    return success();
  }
  LogicalResult finish(Operation *op) {
    const bool streamGood = output.good();
    if (!buffer.close() || !streamGood)
      return op->emitError("failed to write output file: ")
             << temporary.string();
    return success();
  }
  LogicalResult publish(Operation *op) {
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error)
      return op->emitError("cannot publish output file: ")
             << path.string() << ": " << error.message();
    published = true;
    return success();
  }
  JsonFileBuffer buffer;
  std::ostream output;
};

// Adds the enclosing indentation to a single record's ordinary JSON
// serializer. Only the explicit diagnostic mode takes this character path.
class IndentBuffer : public std::streambuf {
public:
  IndentBuffer(std::ostream &output, unsigned spaces)
      : output(output), spaces(spaces) {}

private:
  int_type overflow(int_type c) override {
    if (traits_type::eq_int_type(c, traits_type::eof()))
      return traits_type::not_eof(c);
    const char value = traits_type::to_char_type(c);
    output.put(value);
    if (value == '\n')
      for (unsigned i = 0; i < spaces; ++i)
        output.put(' ');
    return output ? c : traits_type::eof();
  }
  std::ostream &output;
  unsigned spaces;
};

class JsonStreamWriter {
public:
  JsonStreamWriter(std::ostream &output, bool pretty, bool profile)
      : output(output), pretty(pretty), profile(profile) {
    output << '{';
  }
  void field(llvm::StringRef name, const Json &record) {
    key(name);
    value(record, 2);
  }
  void beginArray(llvm::StringRef name) {
    key(name);
    output << '[';
    firstRecord = true;
  }
  void record(const Json &record) {
    if (!firstRecord)
      output << ',';
    if (pretty)
      output << "\n    ";
    firstRecord = false;
    value(record, 4);
  }
  void endArray() {
    if (pretty && !firstRecord)
      output << "\n  ";
    output << ']';
  }
  void finish() { output << (pretty ? "\n}\n" : "}\n"); }
  bool good() const { return output.good(); }
  std::string error() const { return "failed to write JSON plan"; }
  double serializationSeconds = 0;

private:
  void key(llvm::StringRef name) {
    if (!firstField)
      output << ',';
    if (pretty)
      output << "\n  ";
    firstField = false;
    output << Json(name.str()) << (pretty ? ": " : ":");
  }
  void value(const Json &record, unsigned indent) {
    const auto start = profile ? Clock::now() : Clock::time_point{};
    if (pretty) {
      IndentBuffer buffer(output, indent);
      std::ostream indented(&buffer);
      indented << std::setw(2) << record;
      if (!indented)
        output.setstate(std::ios::badbit);
    } else
      output << record;
    if (profile)
      serializationSeconds += elapsed(start);
  }
  std::ostream &output;
  bool pretty, profile, firstField = true, firstRecord = true;
};

class PlaintextBundleWriter {
public:
  PlaintextBundleWriter(std::filesystem::path directory, std::string id,
                        bool profile, bool packed, bool binaryManifest = false)
      : directory(std::move(directory)),
        staging(this->directory.string() + ".tmp"), id(std::move(id)),
        profile(profile), packed(packed), binaryManifest(binaryManifest) {}

  ~PlaintextBundleWriter() {
    if (pack.is_open())
      pack.close();
    if (stagingOwned) {
      std::error_code ignored;
      std::filesystem::remove_all(staging, ignored);
    }
  }
  LogicalResult initialize(Operation *op) {
    std::error_code error;
    if (!std::filesystem::create_directory(staging, error))
      return op->emitError(
                 "cannot create RuntimePlan bundle staging directory: ")
             << staging.string() << ": "
             << (error ? error.message() : "already exists");
    stagingOwned = true;
    if (packed) {
      pack.open(staging / "data.bin", std::ios::binary | std::ios::trunc);
      if (!pack)
        return op->emitError("cannot open RuntimePlan bundle pack");
    } else if (!std::filesystem::create_directory(staging / "data", error))
      return op->emitError("cannot create RuntimePlan bundle data directory: ")
             << error.message();
    return success();
  }

  Json statistics() const {
    return {{"constant_conversion_seconds", conversionSeconds},
            {"constant_hash_seconds", hashSeconds},
            {"constant_write_seconds", writeSeconds},
            {"bundle_bytes", blobBytes},
            {"bundle_format_version", packed ? 2 : 1},
            {"unique_blobs", blobs.size()},
            {"duplicate_hits", duplicateHits},
            {"manifest_bytes", manifestBytes},
            {"manifest_seconds", manifestSeconds},
            {"bundle_reuse_verify_seconds", reuseSeconds}};
  }

  bool used() const { return !blobs.empty(); }

  FailureOr<Json> externalize(DenseElementsAttr dense, Operation *op) {
    // Dense attributes are immutable and shared within the MLIR context.
    // Repeated Encode operations can reuse the already validated blob.
    if (auto it = exported.find(dense); it != exported.end()) {
      ++duplicateHits;
      return Json{{"kind", "bundle"}, {"content", it->second}};
    }
    if (!dense.getElementType().isa<FloatType>()) {
      op->emitError(
          "RuntimePlan bundle externalization requires a float payload");
      return failure();
    }

    const auto conversionStart = profile ? Clock::now() : Clock::time_point{};
    std::string bytes;
    const int64_t elements = dense.getNumElements();
    constexpr uint64_t maxBundleBytes = (1ULL << 53) - 1;
    if (elements <= 0 || static_cast<uint64_t>(elements) > maxBundleBytes / 8) {
      op->emitError("RuntimePlan bundle payload byte length is out of range");
      return failure();
    }
    bytes.reserve(static_cast<size_t>(elements) * 8);
    for (const APFloat &value : dense.getValues<APFloat>()) {
      const double converted = value.convertToDouble();
      if (!std::isfinite(converted)) {
        op->emitError("RuntimePlan Encode payload must contain finite values");
        return failure();
      }
      uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(converted), "float64 is required");
      std::memcpy(&bits, &converted, sizeof(bits));
      for (int byte = 0; byte < 8; ++byte)
        bytes.push_back(static_cast<char>((bits >> (byte * 8)) & 0xff));
    }

    if (profile)
      conversionSeconds += elapsed(conversionStart);
    const auto hashStart = profile ? Clock::now() : Clock::time_point{};
    const std::string content = sha256(bytes);
    if (profile)
      hashSeconds += elapsed(hashStart);
    const auto [it, inserted] = blobs.emplace(
        content, Blob{static_cast<uint64_t>(bytes.size()), blobBytes});
    if (inserted) {
      const auto writeStart = profile ? Clock::now() : Clock::time_point{};
      if (blobBytes > maxBundleBytes - bytes.size()) {
        op->emitError("RuntimePlan bundle pack byte length is out of range");
        return failure();
      }
      if (packed) {
        pack.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!pack) {
          op->emitError("failed to write RuntimePlan bundle pack");
          return failure();
        }
      } else {
        const auto blobPath = staging / "data" / (content.substr(7) + ".bin");
        if (failed(writeFile(blobPath, bytes, op)))
          return failure();
      }
      if (profile)
        writeSeconds += elapsed(writeStart);
      blobBytes += bytes.size();
    } else if (it->second.length != bytes.size()) {
      op->emitError("RuntimePlan bundle SHA-256 collision");
      return failure();
    }
    if (!inserted)
      ++duplicateHits;
    exported.try_emplace(dense, content);
    return Json{{"kind", "bundle"}, {"content", content}};
  }

  FailureOr<Json> writeManifest(Operation *op) {
    const auto start = profile ? Clock::now() : Clock::time_point{};
    if (packed) {
      const auto writeStart = profile ? Clock::now() : Clock::time_point{};
      pack.close();
      if (profile)
        writeSeconds += elapsed(writeStart);
      if (!pack) {
        op->emitError("failed to close RuntimePlan bundle pack");
        return failure();
      }
    }
    JsonArtifact artifact(staging / (binaryManifest ? "manifest.bin" : "manifest.json"), profile, !binaryManifest);
    if (failed(artifact.checkOpen(op)))
      return failure();
    if (binaryManifest) {
      BinaryFields writer(artifact.output);
      writer.header("CKKSMF01", Json{{"bundle_format_version", 2}, {"bundle_id", id},
                                   {"version", 1}, {"pack_byte_length", blobBytes}, {"blobs", Json::array()}});
      writer.number<uint64_t>(blobs.size());
      for (const auto &[content, blob] : blobs) {
        writer.digest(content); writer.number(blob.offset); writer.number(blob.length);
      }
    } else {
      JsonStreamWriter writer(artifact.output, false, profile);
      writer.field("bundle_format_version", packed ? 2 : 1);
      writer.field("bundle_id", id);
      writer.field("version", 1);
      if (packed)
        writer.field("pack_byte_length", blobBytes);
      writer.beginArray("blobs");
      for (const auto &[content, blob] : blobs) {
        Json entry{{"content", content}, {"byte_length", blob.length}};
        if (packed)
          entry["offset"] = blob.offset;
        writer.record(entry);
      }
      writer.endArray();
      writer.finish();
    }
    if (failed(artifact.finish(op)) || failed(artifact.publish(op)))
      return failure();
    manifestDigest = artifact.buffer.digestString;
    manifestBytes = artifact.buffer.bytes;
    if (profile)
      manifestSeconds = elapsed(start);
    if (binaryManifest) return Json{{"id", id}, {"version", 1}, {"manifest_format", "binary"}};
    return Json{
        {"id", id}, {"version", 1}, {"manifest_sha256", manifestDigest}};
  }

  LogicalResult publish(Operation *op) {
    if (!used())
      return success();
    std::error_code error;
    const bool exists = std::filesystem::exists(directory, error);
    if (error)
      return op->emitError("cannot inspect bundle directory: ")
             << error.message();
    if (exists) {
      // Reusing a published bundle requires its exact manifest and every blob
      // to match. Replacing a different bundle requires a new output prefix.
      const auto start = profile ? Clock::now() : Clock::time_point{};
      FailureOr<std::string> existingDigest = std::string{};
      if (!binaryManifest) {
        existingDigest = fileDigest(directory / "manifest.json", op);
        if (failed(existingDigest)) return failure();
      }
      if (binaryManifest) {
        std::ifstream oldManifest(directory / "manifest.bin", std::ios::binary),
            newManifest(staging / "manifest.bin", std::ios::binary);
        const std::string oldBytes((std::istreambuf_iterator<char>(oldManifest)), {}),
            newBytes((std::istreambuf_iterator<char>(newManifest)), {});
        if (!oldManifest || !newManifest || oldBytes != newBytes)
          return op->emitError("existing binary manifest differs; use a new prefix");
      } else if (*existingDigest != manifestDigest)
        return op->emitError("existing plaintext bundle has a different "
                             "manifest; use a new prefix");
      if (packed) {
        const auto path = directory / "data.bin";
        const auto size = std::filesystem::file_size(path, error);
        if (error || size != blobBytes)
          return op->emitError("existing bundle pack byte length mismatch");
        // Compare the published pack with the newly emitted bytes once, using
        // bounded buffers. Content IDs were computed during emission.
        std::ifstream oldPack(path, std::ios::binary),
            newPack(staging / "data.bin", std::ios::binary);
        if (!oldPack || !newPack)
          return op->emitError("cannot open existing bundle pack");
        std::array<char, 64 * 1024> left, right;
        for (uint64_t position = 0; position < blobBytes;) {
          const auto count =
              std::min<uint64_t>(left.size(), blobBytes - position);
          oldPack.read(left.data(), count);
          newPack.read(right.data(), count);
          if (!oldPack || !newPack ||
              std::memcmp(left.data(), right.data(), count))
            return op->emitError("existing bundle pack bytes mismatch");
          position += count;
        }
      } else
        for (const auto &[content, blob] : blobs) {
          const auto path = directory / "data" / (content.substr(7) + ".bin");
          const auto size = std::filesystem::file_size(path, error);
          if (error || size != blob.length)
            return op->emitError("existing bundle blob byte length mismatch: ")
                   << path.string();
          auto digest = fileDigest(path, op);
          if (failed(digest))
            return failure();
          if (*digest != content)
            return op->emitError("existing bundle blob SHA-256 mismatch: ")
                   << path.string();
        }
      if (profile)
        reuseSeconds = elapsed(start);
      return success();
    }
    std::filesystem::rename(staging, directory, error);
    if (error)
      return op->emitError("cannot publish plaintext bundle: ")
             << error.message();
    stagingOwned = false;
    publishedNew = true;
    return success();
  }
  void rollback() {
    if (publishedNew) {
      std::error_code ignored;
      std::filesystem::remove_all(directory, ignored);
    }
  }

private:
  static FailureOr<std::string> fileDigest(const std::filesystem::path &path,
                                           Operation *op) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      op->emitError("cannot open existing bundle file: ") << path.string();
      return failure();
    }
    llvm::SHA256 hash;
    std::array<char, 64 * 1024> buffer;
    while (input.read(buffer.data(), buffer.size()) || input.gcount())
      hash.update(llvm::StringRef(buffer.data(), input.gcount()));
    if (input.bad()) {
      op->emitError("failed to read existing bundle file: ") << path.string();
      return failure();
    }
    const auto digest = hash.final();
    return "sha256:" + llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
  }
  std::filesystem::path directory, staging;
  std::string id, manifestDigest;
  bool profile, packed, binaryManifest, stagingOwned = false, publishedNew = false;
  std::ofstream pack;
  struct Blob {
    uint64_t length, offset;
  };
  double conversionSeconds = 0, hashSeconds = 0, writeSeconds = 0,
         manifestSeconds = 0, reuseSeconds = 0;
  uint64_t blobBytes = 0, manifestBytes = 0, duplicateHits = 0;
  std::map<std::string, Blob> blobs;
  llvm::DenseMap<Attribute, std::string> exported;
};

FailureOr<Json> encodePayload(Attribute payload, Operation *op,
                              PlaintextBundleWriter &bundle,
                              uint64_t inlinePayloadMaxBytes) {
  auto dense = payload.dyn_cast<DenseElementsAttr>();
  if (!dense) {
    op->emitError("RuntimePlan Encode requires a DenseElementsAttr; "
                  "legacy .cst indices are not supported");
    return failure();
  }

  const int64_t elements = dense.getNumElements();
  if (elements <= 0 || static_cast<uint64_t>(elements) >
                           std::numeric_limits<uint64_t>::max() / 8) {
    op->emitError("RuntimePlan Encode payload size is out of range");
    return failure();
  }
  if (static_cast<uint64_t>(elements) * 8 > inlinePayloadMaxBytes)
    return bundle.externalize(dense, op);

  Json values = Json::array();
  auto elementType = dense.getElementType();
  if (elementType.isa<FloatType>()) {
    for (const APFloat &value : dense.getValues<APFloat>()) {
      double converted = value.convertToDouble();
      if (!std::isfinite(converted)) {
        op->emitError("RuntimePlan Encode payload must contain finite values");
        return failure();
      }
      values.push_back(converted);
    }
  } else if (elementType.isa<IntegerType>()) {
    for (const APInt &value : dense.getValues<APInt>()) {
      if (value.getBitWidth() > 64) {
        op->emitError("RuntimePlan Encode integer payload exceeds int64");
        return failure();
      }
      values.push_back(value.getSExtValue());
    }
  } else {
    op->emitError("RuntimePlan Encode payload must contain integers or floats");
    return failure();
  }

  if (values.empty()) {
    op->emitError("RuntimePlan Encode payload cannot be empty");
    return failure();
  }
  return Json{{"kind", "inline"}, {"values", std::move(values)}};
}

struct RuntimePlanBuilder {
  RuntimePlanBuilder(func::FuncOp func, llvm::StringRef contextId, bool ntt,
                     llvm::StringRef bootProfile,
                     llvm::StringRef bootImplementation,
                     PlaintextBundleWriter &bundle,
                     uint64_t inlinePayloadMaxBytes, bool profile)
      : func(func), contextId(contextId), ntt(ntt), bootProfile(bootProfile),
        bootImplementation(bootImplementation), bundle(bundle),
        inlinePayloadMaxBytes(inlinePayloadMaxBytes), profile(profile) {}

  template<class Writer> LogicalResult write(Writer &writer) {
    if (!func.getBody().hasOneBlock())
      return func.emitError(
          "RuntimePlan export currently requires a single-block function");
    const auto layoutStart = profile ? Clock::now() : Clock::time_point{};
    if (failed(assignIds()))
      return failure();
    if (profile)
      layoutSeconds = elapsed(layoutStart);
    writer.beginArray("values");
    for (const auto &[id, value] : valuesById) {
      auto description = buildValueDesc(value, id);
      if (failed(description))
        return failure();
      writer.record(*description);
    }
    writer.endArray();
    writer.beginArray("external_inputs");
    for (BlockArgument argument : func.getArguments())
      writer.record(std::to_string(ids.lookup(argument)));
    writer.endArray();
    // Layout owns both numbering and phase order. Never assign export-local IDs
    // or move V3's online Encode operations into initialization.
    for (bool initialization : {true, false}) {
      writer.beginArray(initialization ? "initialization" : "execution");
      for (const auto &scheduled : instructions) {
        if (scheduled.initialization != initialization)
          continue;
        auto instruction = buildInstruction(scheduled.operation);
        if (failed(instruction))
          return failure();
        (*instruction)["ordinal"] = scheduled.ordinal;
        writer.record(*instruction);
      }
      writer.endArray();
    }
    writer.field("finalization", Json::array());
    auto returnOp =
        dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp)
      return func.emitError("RuntimePlan export requires func.return");
    if (returnOp.getNumOperands() == 0)
      return func.emitError("RuntimePlan requires at least one final output");
    writer.beginArray("final_outputs");
    for (Value output : returnOp.getOperands()) {
      auto it = ids.find(output);
      if (it == ids.end())
        return returnOp.emitError("final output has no RuntimePlan ValueId");
      writer.record(std::to_string(it->second));
    }
    writer.endArray();
    return success();
  }
  double layoutSeconds = 0;
  std::size_t valueCount() const { return valuesById.size(); }
  std::size_t instructionCount() const { return instructions.size(); }

private:
  FailureOr<Json> buildInstruction(Operation *operation) {
    Operation &op = *operation;
    if (isa<dist::FenceOp>(op))
      return Json{{"kind", "fence"}};
    if (auto release = dyn_cast<dist::ReleaseOp>(op)) {
      auto id = valueId(release.getInput(), &op);
      if (failed(id))
        return failure();
      return Json{{"kind", "release"}, {"value", *id}};
    }
    if (op.getNumRegions() != 0) {
      op.emitError("RuntimePlan export does not support nested control flow");
      return failure();
    }
    if (op.getNumResults() != 1 ||
        (op.getName().getDialectNamespace() != "ckks" &&
         op.getName().getDialectNamespace() != "dist")) {
      op.emitError("unsupported operation in RuntimePlan export");
      return failure();
    }
    const uint64_t outputId = ids.lookup(op.getResult(0));
    if (auto encode = dyn_cast<ckks::EncodeOp>(op)) {
      auto payload = encodePayload(encode.getPayload(), &op, bundle,
                                   inlinePayloadMaxBytes);
      if (failed(payload))
        return failure();
      return Json{{"kind", "encode"},
                  {"payload", std::move(*payload)},
                  {"output", std::to_string(outputId)}};
    }
    if (auto transfer = dyn_cast<dist::TransferOp>(op)) {
      FailureOr<std::string> inputId = valueId(transfer.getInput(), &op);
      if (failed(inputId))
        return failure();
      auto type = ckks::getPolyType(transfer.getResult());
      Json instruction = {
          {"kind", "transfer"},
          {"transfer_id", std::to_string(transfer.getTransferId())},
          {"hint", "point_to_point"},
          {"inputs", Json::array({*inputId})},
          {"outputs", Json::array({std::to_string(outputId)})},
          {"sources",
           Json::array({placeJson(transfer.getSourceRankAttr().getInt(),
                                  transfer.getSourceDeviceAttr().getInt())})},
          {"destinations", Json::array({placeJson(
                               transfer.getDestinationRankAttr().getInt(),
                               transfer.getDestinationDeviceAttr().getInt())})},
          {"output_kinds",
           Json::array(
               {type.getComponents() == 1 ? "plaintext" : "ciphertext"})}};
      return instruction;
    }

    return buildCompute(&op, outputId);
  }

private:
  LogicalResult assignIds() {
    auto layout = ckks::buildRuntimePlanLayout(func);
    if (failed(layout))
      return failure();
    ids = std::move(layout->ids);
    valuesById = std::move(layout->values);
    instructions = std::move(layout->instructions);
    return success();
  }

  FailureOr<Json> valuePlace(Value value) {
    auto place = ckks::runtimePlace(value, func);
    if (failed(place))
      return failure();
    return placeJson(place->rank, place->device);
  }

  FailureOr<Json> buildValueDesc(Value value, uint64_t id) {
    auto type = ckks::getPolyType(value);
    if (!type) {
      if (Operation *definingOp = value.getDefiningOp())
        definingOp->emitError("expected a CKKS polynomial value");
      else
        func.emitError("expected a CKKS polynomial function argument");
      return failure();
    }
    constexpr unsigned maxRuntimeInteger = std::numeric_limits<int32_t>::max();
    if (type.getComponents() == 0 || type.getComponents() > maxRuntimeInteger ||
        type.getLevel() > maxRuntimeInteger ||
        type.getScaleLog2() > maxRuntimeInteger) {
      if (Operation *definingOp = value.getDefiningOp())
        definingOp->emitError("CKKS metadata is outside RuntimePlan V1 range");
      else
        func.emitError(
            "CKKS argument metadata is outside RuntimePlan V1 range");
      return failure();
    }
    FailureOr<Json> place = valuePlace(value);
    if (failed(place))
      return failure();
    return Json{
        {"id", std::to_string(id)},
        {"kind", type.getComponents() == 1 ? "plaintext" : "ciphertext"},
        {"place", std::move(*place)},
        {"context", contextId},
        {"level", type.getLevel()},
        {"scale_log2", type.getScaleLog2()},
        {"ntt", ntt},
        {"components", type.getComponents()}};
  }

  FailureOr<std::string> valueId(Value value, Operation *user) const {
    auto it = ids.find(value);
    if (it == ids.end()) {
      user->emitError("RuntimePlan input is not defined before use");
      return failure();
    }
    return std::to_string(it->second);
  }

  FailureOr<Json> makeCompute(Operation *op, uint64_t outputId,
                              llvm::StringRef name, ValueRange inputs) {
    Json inputIds = Json::array();
    for (Value input : inputs) {
      FailureOr<std::string> id = valueId(input, op);
      if (failed(id))
        return failure();
      inputIds.push_back(*id);
    }
    FailureOr<Json> place = valuePlace(op->getResult(0));
    if (failed(place))
      return failure();
    Json instruction = {{"kind", "compute"},
                        {"op", name.str()},
                        {"place", std::move(*place)},
                        {"inputs", std::move(inputIds)},
                        {"output", std::to_string(outputId)}};
    if (Attribute reuse = op->getAttr("runtime.reuse_input")) {
      auto index = dyn_cast<IntegerAttr>(reuse);
      if (!index || index.getInt() != 0) {
        op->emitError("runtime.reuse_input must be integer zero");
        return failure();
      }
      instruction["reuse_input"] = 0;
    }
    return instruction;
  }

  FailureOr<Json> buildCompute(Operation *op, uint64_t outputId) {
    FailureOr<Json> instruction = failure();
    if (auto value = dyn_cast<ckks::AddCCOp>(op))
      instruction =
          makeCompute(op, outputId, "add_cc", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::AddCPOp>(op))
      instruction =
          makeCompute(op, outputId, "add_cp", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::MulCCOp>(op))
      instruction =
          makeCompute(op, outputId, "mul_cc", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::MulCPOp>(op))
      instruction =
          makeCompute(op, outputId, "mul_cp", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::NegateCOp>(op))
      instruction = makeCompute(op, outputId, "negate", {value.getSrc()});
    else if (auto value = dyn_cast<ckks::RelinearizeOp>(op))
      instruction = makeCompute(op, outputId, "relinearize", {value.getSrc()});
    else if (auto value = dyn_cast<ckks::RotateCOp>(op)) {
      if (value.getOffset().size() != 1 || value.getOffset()[0] == 0 ||
          value.getOffset()[0] < std::numeric_limits<int32_t>::min() ||
          value.getOffset()[0] > std::numeric_limits<int32_t>::max()) {
        op->emitError("RuntimePlan Rotate requires exactly one nonzero step");
        return failure();
      }
      instruction = makeCompute(op, outputId, "rotate", {value.getSrc()});
      if (succeeded(instruction))
        (*instruction)["attrs"] = {{"steps", value.getOffset()[0]}};
    } else if (auto value = dyn_cast<ckks::RescaleCOp>(op)) {
      instruction = makeCompute(op, outputId, "rescale", {value.getSrc()});
      if (succeeded(instruction)) {
        auto outputType = ckks::getPolyType(op->getResult(0));
        (*instruction)["attrs"] = {
            {"target_level", outputType.getLevel()},
            {"target_scale_log2", outputType.getScaleLog2()}};
      }
    } else if (auto value = dyn_cast<ckks::ModswitchCOp>(op)) {
      instruction = makeCompute(op, outputId, "mod_switch", {value.getSrc()});
      if (succeeded(instruction)) {
        auto outputType = ckks::getPolyType(op->getResult(0));
        (*instruction)["attrs"] = {{"target_level", outputType.getLevel()}};
      }
    } else if (auto value = dyn_cast<ckks::BootstrapCOp>(op)) {
      if (bootProfile.empty()) {
        op->emitError("RuntimePlan Boot requires --boot-profile");
        return failure();
      }
      instruction = makeCompute(op, outputId, "boot", {value.getSrc()});
      if (succeeded(instruction)) {
        auto outputType = ckks::getPolyType(op->getResult(0));
        (*instruction)["attrs"] = {
            {"target_level", outputType.getLevel()},
            {"target_scale_log2", outputType.getScaleLog2()},
            {"target_components", outputType.getComponents()},
            {"operator_profile", bootProfile},
            {"implementation", bootImplementation}};
      }
    } else if (isa<ckks::UpscaleCOp>(op)) {
      op->emitError(
          "ckks.upscalec must be eliminated before RuntimePlan export");
      return failure();
    } else {
      op->emitError("unsupported CKKS operation in RuntimePlan export");
      return failure();
    }
    return instruction;
  }

  func::FuncOp func;
  std::string contextId;
  bool ntt;
  std::string bootProfile;
  std::string bootImplementation;
  PlaintextBundleWriter &bundle;
  uint64_t inlinePayloadMaxBytes;
  bool profile;
  llvm::DenseMap<Value, uint64_t> ids;
  std::map<uint64_t, Value> valuesById;
  std::vector<ckks::RuntimeInstruction> instructions;
};

struct EmitRuntimePlanPass
    : public hecate::ckks::impl::EmitRuntimePlanBase<EmitRuntimePlanPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (prefix.empty() || targetId.empty() || operatorSpecId.empty() ||
        operatorSpecSha256.empty() || contextId.empty()) {
      func.emitError("RuntimePlan export requires prefix, target-id, "
                     "operator-spec-id, operator-spec-sha256 and context-id");
      signalPassFailure();
      return;
    }
    if (!isCanonicalId(planId)) {
      func.emitError("RuntimePlan plan-id must be a canonical uint64 string");
      signalPassFailure();
      return;
    }
    if (!isSha256(operatorSpecSha256)) {
      func.emitError("operator-spec-sha256 must be sha256: followed by 64 "
                     "lowercase hexadecimal digits");
      signalPassFailure();
      return;
    }
    if (capabilityVersion < 1 ||
        capabilityVersion > std::numeric_limits<int32_t>::max() ||
        operatorSpecVersion < 1 ||
        operatorSpecVersion > std::numeric_limits<int32_t>::max() ||
        deviceCount < 0 || deviceCount > std::numeric_limits<int32_t>::max() ||
        inlinePayloadMaxBytes < 0 ||
        inlinePayloadMaxBytes > static_cast<int64_t>((1ULL << 53) - 1)) {
      func.emitError("RuntimePlan versions and device-count must fit the V1 "
                     "ranges; versions cannot be zero and the inline payload "
                     "limit must be a nonnegative safe JSON integer");
      signalPassFailure();
      return;
    }
    if (bootImplementation != "native" &&
        bootImplementation != "decrypt_reencrypt") {
      func.emitError("boot-implementation must be native or decrypt_reencrypt");
      signalPassFailure();
      return;
    }

    const bool binary = planFormat == "binary";
    if (planFormat != "json" && !binary) {
      func.emitError("plan-format must be binary or json");
      signalPassFailure();
      return;
    }
    if (binary && (bundleFormat != "pack" || pretty)) {
      func.emitError("binary plans require bundle-format=pack and pretty=false");
      signalPassFailure();
      return;
    }
    if (bundleFormat != "pack" && bundleFormat != "files") {
      func.emitError("bundle-format must be pack or files");
      signalPassFailure();
      return;
    }
    Json emittedDeviceCounts = Json::array();
    int64_t worldSize = 1;
    if (auto placedCounts =
            func->getAttrOfType<DenseI64ArrayAttr>("dist.device_counts")) {
      worldSize = static_cast<int64_t>(placedCounts.size());
      if (worldSize <= 0 || worldSize > std::numeric_limits<int32_t>::max()) {
        func.emitError("placed world size is outside RuntimePlan V1 range");
        signalPassFailure();
        return;
      }
      for (int64_t count : placedCounts.asArrayRef()) {
        if (count < 0 || count > std::numeric_limits<int32_t>::max()) {
          func.emitError("placed device count is outside RuntimePlan V1 range");
          signalPassFailure();
          return;
        }
        emittedDeviceCounts.push_back(count);
      }
    } else {
      emittedDeviceCounts.push_back(deviceCount.getValue());
    }
    const std::string artifactStem =
        prefix.getValue() + "." + func.getName().str();
    bool version2 = func->hasAttr("runtime.memory_planned");
    bool version3 = func->hasAttr("runtime.streaming");
    func.walk([&](Operation *op) {
      version2 |=
          isa<dist::ReleaseOp>(op) || op->hasAttr("runtime.reuse_input");
      version3 |= isa<dist::FenceOp>(op) || op->hasAttr("runtime.streaming");
    });
    const auto exportStart = Clock::now();
    const auto rssStart = residentBytes();
    JsonArtifact artifact(artifactStem + (binary ? ".runtime-plan.bin" : ".runtime-plan.json"),
                          reportIo.getValue(), !binary);
    if (failed(artifact.checkOpen(func))) {
      signalPassFailure();
      return;
    }
    PlaintextBundleWriter bundle(artifactStem + ".bundle",
                                 "runtime-plan-" + planId.getValue() + "-" +
                                     func.getName().str() + "-plaintexts",
                                 reportIo.getValue(), bundleFormat == "pack", binary);
    if (failed(bundle.initialize(func))) {
      signalPassFailure();
      return;
    }
    RuntimePlanBuilder builder(
        func, contextId.getValue(), ntt.getValue(), bootProfile.getValue(),
        bootImplementation.getValue(), bundle,
        static_cast<uint64_t>(inlinePayloadMaxBytes.getValue()),
        reportIo.getValue());
    double buildSeconds = 0, serializationSeconds = 0;
    auto emit = [&](auto &writer) -> LogicalResult {
      writer.field("format_version", version3 ? 3 : (version2 ? 2 : 1));
      writer.field("plan_id", planId.getValue());
      writer.field("target",
                   {{"target_id", targetId.getValue()},
                    {"capability_version", capabilityVersion.getValue()},
                    {"operator_spec",
                     {{"id", operatorSpecId.getValue()},
                      {"version", operatorSpecVersion.getValue()},
                      {"source_sha256", operatorSpecSha256.getValue()}}},
                    {"world_size", worldSize},
                    {"device_counts", emittedDeviceCounts}});
      const auto buildStart = Clock::now();
      if (failed(builder.write(writer))) {
        return failure();
      }
      buildSeconds = elapsed(buildStart);
      if (bundle.used()) {
        auto reference = bundle.writeManifest(func);
        if (failed(reference)) {
          return failure();
        }
        writer.field("plaintext_bundle", *reference);
      }
      writer.finish();
      if (!writer.good())
        return func.emitError(writer.error());
      serializationSeconds = writer.serializationSeconds;
      return success();
    };
    if (binary) {
      BinaryPlanWriter writer(artifact.output, artifactStem + ".records.tmp");
      if (failed(emit(writer))) { signalPassFailure(); return; }
    } else {
      JsonStreamWriter writer(artifact.output, pretty.getValue(), reportIo.getValue());
      if (failed(emit(writer))) { signalPassFailure(); return; }
    }
    if (failed(artifact.finish(func))) {
      signalPassFailure();
      return;
    }
    if (failed(bundle.publish(func))) {
      signalPassFailure();
      return;
    }
    if (failed(artifact.publish(func))) {
      bundle.rollback();
      signalPassFailure();
      return;
    }
    if (reportIo) {
      Json report = bundle.statistics();
      report.update({{"export_seconds", elapsed(exportStart)},
                     {"layout_seconds", builder.layoutSeconds},
                     {"build_seconds", buildSeconds},
                     {"plan_serialize_seconds", binary ? Json(nullptr) : Json(serializationSeconds)},
                     {"plan_write_seconds", artifact.buffer.writeSeconds},
                     {"plan_hash_seconds", artifact.buffer.hashSeconds},
                     {"plan_bytes", artifact.buffer.bytes},
                     {"plan_sha256", artifact.buffer.digestString},
                     {"values", builder.valueCount()},
                     {"instructions", builder.instructionCount()},
                     {"rss_start_bytes", rssStart},
                     {"rss_end_bytes", residentBytes()},
                     {"process_peak_rss_bytes", peakResidentBytes()},
                     {"pretty", pretty.getValue()}, {"plan_format", planFormat.getValue()}});
      llvm::errs() << "runtime-plan-io " << report.dump() << '\n';
    }
  }
};

} // namespace
