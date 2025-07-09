ARG BUILD_FROM
FROM $BUILD_FROM

# Use the published main project image as base
FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

# Copy add-on specific files
COPY generate_config.py /app/
COPY run.sh /app/

# Make run.sh executable
RUN chmod +x /app/run.sh

# Change the CMD to use the add-on's run script
CMD ["/app/run.sh"] 