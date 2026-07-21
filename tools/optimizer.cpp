
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>
#include <iostream>

#include "hecate/Conversion/Passes.h"
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"

using namespace llvm;

using namespace mlir;
using namespace hecate;

void registerHecatePipeline(cl::opt<std::string> &outputFilename);

int main(int argc, char **argv) {

  // Options from MLIR Opt Main
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  static cl::opt<bool> splitInputFile(
      "split-input-file",
      cl::desc("Split the input file into pieces and process each "
               "chunk independently"),
      cl::init(false));

  static cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false));

  static cl::opt<bool> verifyPasses(
      "verify-each",
      cl::desc("Run the verifier after each transformation pass"),
      cl::init(true));

  static cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      cl::desc("Allow operation with no registered dialects"), cl::init(false));

  static cl::opt<bool> showDialects(
      "show-dialects", cl::desc("Print the list of registered dialects"),
      cl::init(false));

  static cl::opt<bool> emitBytecode(
      "emit-bytecode", cl::desc("Emit bytecode when generating output"),
      cl::init(false));

  static cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      cl::init(false)};

  static cl::opt<bool> dumpPassPipeline{
      "dump-pass-pipeline", cl::desc("Print the pipeline that will be run"),
      cl::init(false)};

  // Hecate Option
  //
  static cl::opt<std::string> ckks_config{
      "ckks-config", cl::desc("Set CKKS parameters from configuration file"),
      cl::init("./config.json")};

  InitLLVM y(argc, argv);

  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<earth::EarthDialect>();
  registry.insert<ckks::CKKSDialect>();
  registry.insert<dist::DistDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<tensor::TensorDialect>();
  context.getOrLoadDialect<earth::EarthDialect>();
  context.getOrLoadDialect<ckks::CKKSDialect>();
  context.getOrLoadDialect<dist::DistDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<tensor::TensorDialect>();

  // Uncomment the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  // Uncomment the following to make *all* MLIR core passes available.
  // This is only useful for experimenting with the command line to compose
  // registerAllPasses();

  registerLocationSnapshot();
  registerSymbolDCE();
  registerCanonicalizerPass();
  registerCSEPass();
  earth::registerEarthPasses();
  ckks::registerCKKSPasses();
  hecate::registerConversionPasses();

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  /* DebugCounter::registerCLOptions(); */
  registerHecatePipeline(outputFilename);

  PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");

  context.enableMultithreading();

  llvm::StringRef toolName("Hecate optimizer driver\n");

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);

  earth::EarthDialect::setCKKSParameters(ckks_config);

  if (showDialects) {
    llvm::outs() << "Available Dialects:\n";
    interleave(
        registry.getDialectNames(), llvm::outs(),
        [](auto name) { llvm::outs() << name; }, "\n");
    return asMainReturnCode(success());
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return asMainReturnCode(failure());
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return asMainReturnCode(failure());
  }

  MlirOptMainConfig config;
  config.setPassPipelineParser(passPipeline)
      .splitInputFile(splitInputFile)
      .verifyDiagnostics(verifyDiagnostics)
      .verifyPasses(verifyPasses)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(emitBytecode)
      .dumpPassPipeline(dumpPassPipeline);

  if (failed(MlirOptMain(output->os(), std::move(file), registry, config))) {
    return asMainReturnCode(failure());
  }

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return asMainReturnCode(success());
}

void registerHecatePipeline(cl::opt<std::string> &outputFilename) {
  // Options for hecate-opt

  static cl::opt<int64_t> waterline{
      "waterline", cl::desc("Waterline of scale management"), cl::init(20)};

  static cl::opt<int64_t> output_val{
      "output-val", cl::desc("Output value upper bound  of scale management"),
      cl::init(10)};

  static cl::opt<float> threshold{
      "threshold",
      cl::desc("scale coverage threshold of DaCapo, 0 <= threshold <= 1"),
      cl::init(0.5)};

  static cl::opt<bool> enable_printer{
      "enable-debug-printer",
      cl::desc(
          "Enable printing after scale management and earth-ckks conversion"),
      cl::init(false)};

  static cl::opt<bool> enable_check_smu{
      "enable-check-smu", cl::desc("Check SMU generation"), cl::init(false)};

  static cl::opt<int64_t> parallel_elasm{
      "parallel-elasm", cl::desc("Parallel # of ELASM"), cl::init(20)};
  static cl::opt<int64_t> num_iter_elasm{
      "num-iter-elasm", cl::desc("# Iters of ELASM"), cl::init(1000)};
  static cl::opt<int64_t> beta_elasm{"beta-elasm", cl::desc("Beta of ELASM"),
                                     cl::init(70)};
  static cl::opt<int64_t> gamma_elasm{"gamma-elasm", cl::desc("Gamma of ELASM"),
                                      cl::init(50)};

  static cl::opt<std::string> runtime_plan_id{
      "runtime-plan-id", cl::desc("RuntimePlan decimal plan id"),
      cl::init("0")};
  static cl::opt<std::string> runtime_plan_target_id{
      "runtime-plan-target-id", cl::desc("RuntimePlan target id"),
      cl::init("")};
  static cl::opt<int64_t> runtime_plan_capability_version{
      "runtime-plan-capability-version",
      cl::desc("RuntimePlan target capability version"), cl::init(1)};
  static cl::opt<std::string> runtime_plan_operator_spec_id{
      "runtime-plan-operator-spec-id", cl::desc("OperatorSpec id"),
      cl::init("")};
  static cl::opt<int64_t> runtime_plan_operator_spec_version{
      "runtime-plan-operator-spec-version", cl::desc("OperatorSpec version"),
      cl::init(1)};
  static cl::opt<std::string> runtime_plan_operator_spec_sha256{
      "runtime-plan-operator-spec-sha256",
      cl::desc("SHA-256 of the exact OperatorSpec bytes"), cl::init("")};
  static cl::opt<std::string> runtime_plan_context_id{
      "runtime-plan-context-id", cl::desc("CKKS context id"), cl::init("")};
  static cl::opt<int64_t> runtime_plan_device_count{
      "runtime-plan-device-count",
      cl::desc("Device count for the single RuntimePlan rank"), cl::init(0)};
  static cl::opt<std::string> runtime_plan_device_counts{
      "runtime-plan-device-counts",
      cl::desc("Per-rank device counts separated by x; enables placement"),
      cl::init("")};
  static cl::opt<std::string> runtime_plan_operator_spec_path{
      "runtime-plan-operator-spec-path",
      cl::desc("OperatorSpec V2 JSON used by placement"), cl::init("")};
  static cl::opt<std::string> runtime_plan_communication_profile_path{
      "runtime-plan-communication-profile-path",
      cl::desc("Payload-aware communication profile JSON used by placement"),
      cl::init("")};
  static cl::opt<std::string> runtime_plan_physical_level_operator_spec_path{
      "runtime-plan-physical-level-operator-spec-path",
      cl::desc("Lazy-rescale target OperatorSpec used to materialize physical "
               "levels"),
      cl::init("")};
  static cl::opt<int64_t> runtime_plan_levels_per_logical_level{
      "runtime-plan-levels-per-logical-level",
      cl::desc("Physical RNS levels consumed by one logical CKKS level"),
      cl::init(1)};
  static cl::opt<int64_t> runtime_plan_intra_rank_communication_cost{
      "runtime-plan-intra-rank-communication-cost",
      cl::desc("Fixed intra-rank point-to-point placement cost"),
      cl::init(1000)};
  static cl::opt<int64_t> runtime_plan_inter_rank_communication_cost{
      "runtime-plan-inter-rank-communication-cost",
      cl::desc("Fixed inter-rank point-to-point placement cost"),
      cl::init(10000)};
  static cl::opt<bool> runtime_plan_ntt{
      "runtime-plan-ntt", cl::desc("ValueDesc NTT state"), cl::init(true)};
  static cl::opt<std::string> runtime_plan_boot_profile{
      "runtime-plan-boot-profile", cl::desc("OperatorSpec boot profile id"),
      cl::init("")};
  static cl::opt<std::string> runtime_plan_boot_implementation{
      "runtime-plan-boot-implementation",
      cl::desc("Boot implementation: native or decrypt_reencrypt"),
      cl::init("native")};
  static cl::opt<int64_t> runtime_plan_inline_payload_max_bytes{
      "runtime-plan-inline-payload-max-bytes",
      cl::desc("Largest RuntimePlan Encode payload kept inline"),
      cl::init(4096)};

  auto addRuntimePlanExport = [&](OpPassManager &pm,
                                  const std::string &artifactPrefix) {
    if (!runtime_plan_physical_level_operator_spec_path.empty() ||
        runtime_plan_levels_per_logical_level != 1) {
      hecate::ckks::MaterializePhysicalLevelsOptions physicalLevels;
      physicalLevels.operatorSpecPath =
          runtime_plan_physical_level_operator_spec_path;
      physicalLevels.levelsPerLogicalLevel =
          runtime_plan_levels_per_logical_level;
      pm.addNestedPass<func::FuncOp>(
          hecate::ckks::createMaterializePhysicalLevels(physicalLevels));
    }
    if (!runtime_plan_device_counts.empty()) {
      hecate::ckks::AssignPlacementOptions placement;
      placement.deviceCounts = runtime_plan_device_counts;
      placement.operatorSpecPath = runtime_plan_operator_spec_path;
      placement.bootProfile = runtime_plan_boot_profile;
      placement.communicationProfilePath =
          runtime_plan_communication_profile_path;
      placement.intraRankCommunicationCost =
          runtime_plan_intra_rank_communication_cost;
      placement.interRankCommunicationCost =
          runtime_plan_inter_rank_communication_cost;
      pm.addNestedPass<func::FuncOp>(
          hecate::ckks::createAssignPlacement(placement));
      pm.addNestedPass<func::FuncOp>(
          hecate::ckks::createMaterializeCommunication());
    }
    hecate::ckks::EmitRuntimePlanOptions options;
    options.prefix = artifactPrefix;
    options.planId = runtime_plan_id;
    options.targetId = runtime_plan_target_id;
    options.capabilityVersion = runtime_plan_capability_version;
    options.operatorSpecId = runtime_plan_operator_spec_id;
    options.operatorSpecVersion = runtime_plan_operator_spec_version;
    options.operatorSpecSha256 = runtime_plan_operator_spec_sha256;
    options.contextId = runtime_plan_context_id;
    options.deviceCount = runtime_plan_device_count;
    options.ntt = runtime_plan_ntt;
    options.bootProfile = runtime_plan_boot_profile;
    options.bootImplementation = runtime_plan_boot_implementation;
    options.inlinePayloadMaxBytes = runtime_plan_inline_payload_max_bytes;
    pm.addNestedPass<func::FuncOp>(
        hecate::ckks::createEmitRuntimePlan(options));
  };

  PassPipelineRegistration<>(
      "eva", "Perform waterline rescaling and early modswitch",
      [&, addRuntimePlanExport](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createWaterlineRescaling({waterline, output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        addRuntimePlanExport(pm, (std::filesystem::path(dir) / stem).string());
      });

  PassPipelineRegistration<>(
      "snr", "Perform SNR rescaling and early modswitch",
      [&, addRuntimePlanExport](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createSNRRescaling({waterline, output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        addRuntimePlanExport(pm, (std::filesystem::path(dir) / stem).string());
      });

  PassPipelineRegistration<>(
      "elasm", "Perform ELLASM exploration ",
      [&, addRuntimePlanExport](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createELASMExplorer(
            {waterline, output_val, parallel_elasm, num_iter_elasm, beta_elasm,
             gamma_elasm}));
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createScaleManagementScheduler());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createSNRRescaling({waterline, output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createUpscaleBubbling());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createErrorEstimator());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        addRuntimePlanExport(pm, (std::filesystem::path(dir) / stem).string());
      });

  PassPipelineRegistration<>(
      "dacapo", "Perform automatic bootstrapping placement",
      [&, addRuntimePlanExport](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createRemoveBootstrap());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createBypassDetection({waterline, 0.5}));
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createCandidateSelection({waterline, output_val}));
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createDaCapoPlanner({waterline, output_val}));
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createBootstrapPlacement());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createProactiveRescaling({waterline, output_val}));
        /* pm.addNestedPass<func::FuncOp>( */
        /*     hecate::earth::createSNRRescaling({waterline, output_val})); */
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        addRuntimePlanExport(pm, (std::filesystem::path(dir) / stem).string());
      });
  PassPipelineRegistration<>(
      "pars", "Perform Scale Management on bootstrapped models",
      [&, addRuntimePlanExport](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createProactiveRescaling({waterline, output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());

        if (enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        addRuntimePlanExport(pm, (std::filesystem::path(dir) / stem).string());
      });
}
