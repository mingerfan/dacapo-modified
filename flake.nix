{
  description = "DaCapo compiler development environment and package";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          llvm = pkgs.llvmPackages_18;
        in
        rec {
          dacapo = llvm.stdenv.mkDerivation {
            pname = "dacapo";
            version = "0.1.0";
            src = self;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ninja
              llvm.tblgen
            ];

            buildInputs = [
              llvm.llvm
              llvm.mlir
            ];

            cmakeFlags = [
              "-DMLIR_DIR=${llvm.mlir.dev}/lib/cmake/mlir"
              "-DLLVM_DIR=${llvm.llvm.dev}/lib/cmake/llvm"
              "-DMLIR_TABLEGEN_EXE=${llvm.tblgen}/bin/mlir-tblgen"
              "-DLLVM_TABLEGEN_EXE=${llvm.tblgen}/bin/llvm-tblgen"
            ];

            installPhase = ''
              runHook preInstall
              mkdir -p $out/bin $out/lib
              cp -r bin/. $out/bin/
              cp lib/lib* $out/lib/
              runHook postInstall
            '';

            meta.mainProgram = "hecate-opt";
          };

          default = dacapo;
        });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          llvm = pkgs.llvmPackages_18;
        in
        {
          default = (pkgs.mkShell.override { stdenv = llvm.stdenv; }) {
            packages = [
              pkgs.cmake
              pkgs.ninja
              pkgs.git
              pkgs.python3
              llvm.clang
              llvm.clang-tools
              llvm.llvm
              llvm.mlir
              llvm.tblgen
            ];

            MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";
            LLVM_DIR = "${llvm.llvm.dev}/lib/cmake/llvm";
            MLIR_TABLEGEN_EXE = "${llvm.tblgen}/bin/mlir-tblgen";
            LLVM_TABLEGEN_EXE = "${llvm.tblgen}/bin/llvm-tblgen";
          };
        });
    };
}
