# Use the published main project image as base
FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

# Switch to root temporarily to copy files
USER root

# Copy add-on specific files
COPY generate_config.py /app/
COPY run.sh /app/

# Make run.sh executable
RUN chmod +x /app/run.sh

# Switch back to non-root user
USER app

# Change the CMD to use the add-on's run script
CMD ["/app/run.sh"] 