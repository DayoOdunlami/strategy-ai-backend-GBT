{ pkgs }:

let
  python = pkgs.python310;
in
pkgs.mkShell {
  buildInputs = [
    python
    python.pkgs.setuptools
    python.pkgs.pip
    python.pkgs.wheel
    python.pkgs.numpy
    python.pkgs.requests
    python.pkgs.shapely
    python.pkgs.geopandas
  ];
}
