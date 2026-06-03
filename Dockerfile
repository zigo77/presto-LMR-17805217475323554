# Use a pinned version for reproducibility
# continuumio/miniconda3 is based on Debian
FROM continuumio/miniconda3:24.7.1-0

# Set noninteractive to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install comprehensive system dependencies for scientific Python
# Organized by category for maintainability
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    # Runtime libraries for scientific computing
    libgfortran5 \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    # HDF5 and NetCDF (for climate/scientific data)
    libhdf5-dev \
    libhdf5-serial-dev \
    libnetcdf-dev \
    # Geospatial libraries (GEOS, PROJ, GDAL)
    libgeos-dev \
    libgeos++-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    libgdal-dev \
    libspatialindex-dev \
    # Graphics and rendering (for matplotlib, cartopy)
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libffi-dev \
    # XML/compression libraries
    libxml2-dev \
    libxslt1-dev \
    liblzma-dev \
    libbz2-dev \
    libzip-dev \
    zlib1g-dev \
    # SSL/TLS for secure downloads
    libssl-dev \
    ca-certificates \
    # Database libraries (MySQL in environment.yml)
    libmariadb-dev \
    # Version control
    git \
    curl \
    wget \
    # Utilities
    vim \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Set the working directory
WORKDIR /app

# Download the PAGES2kV2.nc file to the working directory
RUN curl -L -o PAGES2kV2.nc https://drive.google.com/uc?export=download&id=1XTNSfrajvw_3og1_1bP9BkcocqEMCmqM

# Copy environment file first for better layer caching
COPY environment.yml .

RUN printf "channels:\n - conda-forge\n - defaults\nchannel_priority: flexible\nalways_yes: true\n" > /opt/conda/.condarc && \
    conda install -n base conda-libmamba-solver && \
    conda env create -f environment.yml --solver=libmamba && \
    conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

# Initialize conda for shell usage
RUN conda init bash

# Set environment variables for the conda environment
ENV PATH=/opt/conda/envs/cfr-env/bin:$PATH \
    CONDA_DEFAULT_ENV=cfr-env \
    CONDA_PREFIX=/opt/conda/envs/cfr-env

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "cfr-env", "/bin/bash", "-c"]

# Comprehensive environment verification
# Tests core scientific libraries that the project depends on
RUN echo "Testing environment..." && \
    python -c "import sys; print(f'Python version: {sys.version}')" && \
    python -c "import numpy; print(f'NumPy {numpy.__version__}')" && \
    python -c "import pandas; print(f'Pandas {pandas.__version__}')" && \
    python -c "import xarray; print(f'xarray {xarray.__version__}')" && \
    python -c "import netCDF4; print(f'netCDF4 {netCDF4.__version__}')" && \
    python -c "import scipy; print(f'SciPy {scipy.__version__}')" && \
    python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')" && \
    python -c "import cartopy; print(f'Cartopy {cartopy.__version__}')" && \
    python -c "import cfr; print(f'CFR {cfr.__version__}')" && \
    echo "All core packages imported successfully!"

# Copy application code
# This is done last to maximize cache usage during development
COPY . .

# Create a non-root user for security (optional but recommended)
# Uncomment the following lines to run as non-root user
# RUN useradd -m -u 1000 appuser && \
#     chown -R appuser:appuser /app
# USER appuser

# Set the default command to activate environment and start bash
# Users can override this to run specific scripts
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate cfr-env && exec bash"]
