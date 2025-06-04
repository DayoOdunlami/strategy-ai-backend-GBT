{ pkgs }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    python312Packages.setuptools
    python312Packages.pip
    python312Packages.wheel
    python312Packages.numpy
    python312Packages.distutils-extra
    python312Packages.geopandas
    python312Packages.shapely
    python312Packages.requests
  ];
}
