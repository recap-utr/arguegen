{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    systems,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        pkgs,
        lib,
        ...
      }: let
        python = pkgs.python311;
        poetry = pkgs.poetry;
        # https://github.com/NixOS/nixpkgs/pull/225937
        mkShell =
          if pkgs.stdenv.isDarwin
          then pkgs.mkShell.override {stdenv = pkgs.llvmPackages.libcxxStdenv;}
          else pkgs.mkShell;
      in {
        devShells.default = mkShell {
          packages = [poetry python];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = lib.makeLibraryPath [pkgs.stdenv.cc.cc];
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root --sync
          '';
        };
      };
    };
}
